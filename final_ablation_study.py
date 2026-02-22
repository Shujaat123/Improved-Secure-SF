"""
Ablation + MIA + Latency/Communication with protected-in decoders.

NEW CONCEPT IMPLEMENTED:
  • Clients send TRANSFORMED latents (T(z)) to the server.
  • Server INVERTS them (T^{-1}) to recover original latents z and TRAINS F: z_x -> z_y.
  • During inference, server maps in z-space, then RE-APPLIES the client’s forward transform T
    before sending back to the client, so links only carry transformed latents.
  • Client-side protected decoder still applies T^{-1} before D (unchanged), thus end-to-end is:
      client: E → T  ──> server: T^{-1} → F → T  ──> client: T^{-1} → D

Variants (UPDATED):
  1) full_model            : KLT + PPM + skip-connected AEs
  2) no_klt                : identity latent + PPM + skip-connected AEs
  3) no_skips_plain_ae     : KLT + PPM + plain AEs (no encoder–decoder skips)
  (REMOVED) no_ppm

Outputs:
  - ./ablation_results.csv (Dice/IoU/Sens/Spec + MIA + latency/comm)
  - ./results/ablations/<variant>_mia.csv
  - ./results/ablations/<variant>_latency_comm.csv
  - ./results/ablations/<variant>_viz_{1,2}.png
  - ./results/ablations/<variant>_tsne_images_before.png
  - ./results/ablations/<variant>_tsne_images_after.png
  - ./results/ablations/<variant>_tsne_masks_before.png
  - ./results/ablations/<variant>_tsne_masks_after.png
  - Models under ./saved_models/ablation/<variant>/

MIA METRICS (UPDATED):
  - auc, acc, sensitivity, specificity, jaccard

LATENCY (UPDATED):
  - encoder_ms (E only)
  - transform_ms (T forward: img + mask)
  - map_ms (UMN only)
  - inverse_transform_ms (T^{-1}: img + mask)
  - decoder_ms (D only)
  - total_ms = sum of the above
"""

import os, csv, json, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import HeNormal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import SimpleITK as sitk
import matplotlib.pyplot as plt


print('DATA LOADING STAGE')
os.chdir('/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION')
# -----------------------------
# Config
# -----------------------------
SEED = 42
NUM_CLIENTS = 3
AE_EPOCHS = 100
MAP_EPOCHS = 200
BATCH_SIZE = 16
RESULTS_CSV = "./results/ablations/ablation_results.csv"
RESULTS_DIR = "./results/ablations"
REUSE_FULL_AE_WEIGHTS = False

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)
he_normal = HeNormal

# -----------------------------
# Data loader
# -----------------------------
def load_mha_image_and_mask_data(folder_path):
    image_dir = os.path.join(folder_path, "image_mha")
    mask_dir  = os.path.join(folder_path, "label_mha")

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".mha")]
    imgs = []
    for fp in image_files:
        im = sitk.ReadImage(fp)
        ar = sitk.GetArrayFromImage(im)
        imgs.append(ar)
    images = np.array(imgs).astype(np.float32) / 255.0
    images = np.transpose(images, (0, 2, 3, 1))   # (N,H,W,C)

    mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".mha")]
    msks = []
    for fp in mask_files:
        im = sitk.ReadImage(fp)
        ar = sitk.GetArrayFromImage(im)
        msks.append(ar)
    masks = np.array(msks)
    masks = tf.one_hot(masks, depth=3).numpy()     # (N,H,W,C=3)
    print(f"Loaded {images.shape[0]} images {images.shape[1:]} and {masks.shape[0]} masks {masks.shape[1:]}")
    return images, masks

print("Loading data...")
img_train, mask_train = load_mha_image_and_mask_data("./datasets/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression")
img_test,  mask_test  = load_mha_image_and_mask_data("./datasets/testdataset")

# -----------------------------
# Latent transform (KLT-like)
# -----------------------------
def sample_orthogonal_tf(d, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
    A = tf.random.normal((d, d))
    Q, _ = tf.linalg.qr(A, full_matrices=False)
    return Q

class LatentWrapTF(layers.Layer):
    """
    Channels-last latent transform for (B,H,W,C).
      forward:  z' = z @ Q^T + b
      inverse:  z  = (z' - b) @ Q
    """
    def __init__(self, channels=None, Q=None, b=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self._channels = channels
        self._Q_init = Q
        self._b_init = b
        self._seed = seed
        self.Q = None
        self.b = None

    def build(self, input_shape):
        C = self._channels or int(input_shape[-1])
        Q_init = self._Q_init if self._Q_init is not None else sample_orthogonal_tf(C, seed=self._seed)
        b_init = self._b_init if self._b_init is not None else tf.random.normal((C,))
        self.Q = self.add_weight(
            name="Q", shape=(C, C),
            initializer=tf.constant_initializer(Q_init.numpy() if isinstance(Q_init, tf.Tensor) else Q_init),
            trainable=False
        )
        self.b = self.add_weight(
            name="b", shape=(C,),
            initializer=tf.constant_initializer(b_init.numpy() if isinstance(b_init, tf.Tensor) else b_init),
            trainable=False
        )
        super().build(input_shape)

    def call(self, z):
        z_lin = tf.einsum("bhwc,cd->bhwd", z, tf.transpose(self.Q))
        return z_lin + self.b

    @tf.function
    def inverse(self, z_prime):
        z_center = z_prime - self.b
        return tf.einsum("bhwc,cd->bhwd", z_center, self.Q)

# -----------------------------
# Losses & metrics
# -----------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    denom = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    dice  = (2. * inter + smooth) / (denom + smooth)
    return 1 - tf.reduce_mean(dice)

def combined_loss(alpha=0.5, beta=0.5):
    bce = tf.keras.losses.BinaryCrossentropy()
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        return alpha * bce(y_true, y_pred) + beta * dice_loss(y_true, y_pred)
    return loss

def dice_score(y_true, y_pred, eps=1e-6):
    y_true = y_true[..., 1:]
    y_pred = (y_pred[..., 1:] > 0.5).astype(np.float32)
    inter = np.sum(y_true * y_pred)
    denom = np.sum(y_true) + np.sum(y_pred)
    return (2.*inter + eps) / (denom + eps)

def iou(y_true, y_pred, eps=1e-6):
    y_true = y_true[..., 1:]
    y_pred = (y_pred[..., 1:] > 0.5).astype(np.float32)
    inter = np.sum(y_true * y_pred)
    union = np.sum(np.clip(y_true + y_pred, 0, 1))
    return (inter + eps) / (union + eps)

def sensitivity(y_true, y_pred, eps=1e-6):
    y_true = y_true[..., 1:]
    y_pred = (y_pred[..., 1:] > 0.5).astype(np.float32)
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return (tp + eps) / (tp + fn + eps)

def specificity(y_true, y_pred, eps=1e-6):
    y_true = y_true[..., 1:]
    y_pred = (y_pred[..., 1:] > 0.5).astype(np.float32)
    tn = np.sum((1 - y_true) * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    return (tn + eps) / (tn + fp + eps)

# -----------------------------
# AE builder (UPDATED: returns encoder_raw for latency breakdown)
# -----------------------------
def build_autoencoder_toggled(input_shape=(256,256,3),
                              use_skips=True,
                              use_klt=True,
                              seed=None):
    inp = layers.Input(shape=input_shape, name="ae_input")

    # Encoder (raw)
    e1 = layers.Conv2D(16, 3, 2, "same", activation="relu")(inp); e1 = layers.BatchNormalization()(e1)
    e2 = layers.Conv2D(32, 3, 2, "same", activation="relu")(e1);  e2 = layers.BatchNormalization()(e2)
    e3 = layers.Conv2D(64, 3, 2, "same", activation="relu")(e2);  e3 = layers.BatchNormalization()(e3)   # 32x32
    e4 = layers.Conv2D(128,3, 2, "same", activation="relu")(e3);  e4 = layers.BatchNormalization()(e4)   # 16x16
    b  = layers.Conv2D(256, 3, 2, "same", activation="relu", name="latent_b")(e4)                         # 8x8

    encoder_raw = models.Model(inp, [b, e3, e4], name=f"Encoder_raw_{'skips' if use_skips else 'plain'}")

    # KLT (or identity)
    if use_klt:
        T_b  = LatentWrapTF(channels=256, seed=None if seed is None else seed+1, name="T_b")
        T_s3 = LatentWrapTF(channels=64,  seed=None if seed is None else seed+2, name="T_s3")
        T_s4 = LatentWrapTF(channels=128, seed=None if seed is None else seed+3, name="T_s4")
        b_T, s3_T, s4_T = T_b(b), T_s3(e3), T_s4(e4)
    else:
        class IdentityWrap(layers.Layer):
            def call(self, x): return x
            def inverse(self, x): return x
        T_b, T_s3, T_s4 = IdentityWrap(), IdentityWrap(), IdentityWrap()
        b_T, s3_T, s4_T = b, e3, e4

    # Decoder inputs (raw domain)
    enc_in = layers.Input(shape=b.shape[1:],   name="dec_in_b")
    s3_in  = layers.Input(shape=e3.shape[1:],  name="dec_in_s3")
    s4_in  = layers.Input(shape=e4.shape[1:],  name="dec_in_s4")

    def up_block(x, filters, skip=None):
        x = layers.UpSampling2D(size=(2,2), interpolation="bilinear")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if use_skips and (skip is not None):
            x = layers.Concatenate()([x, skip])
        return x

    d1 = up_block(enc_in, 128, skip=(s4_in if use_skips else None))
    d2 = up_block(d1,     64,  skip=(s3_in if use_skips else None))
    d3 = up_block(d2,     32)
    d4 = up_block(d3,     16)
    out = layers.UpSampling2D(size=(2,2), interpolation="bilinear")(d4)
    out = layers.Conv2D(input_shape[-1], 3, padding="same", activation="sigmoid")(out)

    decoder_inputs = [enc_in, s3_in, s4_in] if use_skips else [enc_in]
    decoder = models.Model(decoder_inputs, out,
                           name=("Decoder_skips" if use_skips else "Decoder_plain"))

    # Full AE path: inverse KLT inside the graph BEFORE decoder
    if use_klt:
        b_inv  = layers.Lambda(lambda t: T_b.inverse(t),  name="inv_b")(b_T)
        s3_inv = layers.Lambda(lambda t: T_s3.inverse(t), name="inv_s3")(s3_T)
        s4_inv = layers.Lambda(lambda t: T_s4.inverse(t), name="inv_s4")(s4_T)
    else:
        b_inv, s3_inv, s4_inv = b_T, s3_T, s4_T

    if use_skips:
        recon = decoder([b_inv, s3_inv, s4_inv])
    else:
        recon = decoder([b_inv])

    autoencoder = models.Model(inp, recon, name=f"AE_{'skips' if use_skips else 'plain'}_{'klt' if use_klt else 'noklt'}")
    encoder_T  = models.Model(inp, [b_T, s3_T, s4_T], name=f"Encoder_T_{'klt' if use_klt else 'id'}")
    T_layers   = {"T_b": T_b, "T_s3": T_s3, "T_s4": T_s4}

    return autoencoder, encoder_raw, encoder_T, decoder, T_layers

# -----------------------------
# Mapping network (server)
# -----------------------------
class PyramidPoolingModule(layers.Layer):
    def __init__(self, in_channels, pool_sizes=(1,3,5,7), out_channels=None):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else max(1, in_channels // (len(pool_sizes)*2))
        self.convs = [layers.Conv2D(self.out_channels, 1, use_bias=False) for _ in pool_sizes]
        self.bns   = [layers.BatchNormalization() for _ in pool_sizes]
        self.relus = [layers.ReLU() for _ in pool_sizes]
        self.final_conv = layers.Conv2D(in_channels, 1, use_bias=False)
        self.final_bn   = layers.BatchNormalization()
        self.final_relu = layers.ReLU()

    def call(self, x, training=False):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        outs = [x]
        for i, s in enumerate(self.pool_sizes):
            y = tf.reduce_mean(x, axis=[1,2], keepdims=True)
            y = tf.image.resize(y, [s, s], method='bilinear')
            y = self.convs[i](y); y = self.bns[i](y, training=training); y = self.relus[i](y)
            y = tf.image.resize(y, [h, w], method='bilinear')
            outs.append(y)
        y = tf.concat(outs, axis=-1)
        y = self.final_conv(y); y = self.final_bn(y, training=training); y = self.final_relu(y)
        return y

def mapping_block(x, out_ch, use_ppm=True):
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer=he_normal())(x)
    x = layers.LeakyReLU(0.1)(x); x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer=he_normal())(x)
    x = layers.LeakyReLU(0.1)(x); x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer=he_normal())(x)
    x = layers.LeakyReLU(0.1)(x); x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    if use_ppm:
        x = PyramidPoolingModule(512)(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer=he_normal())(x)
    x = layers.LeakyReLU(0.1)(x); x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1024, 3, padding='same', kernel_initializer=he_normal())(x)
    x = layers.LeakyReLU(0.1)(x); x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    d1 = layers.MaxPooling2D(2)(x)
    d1 = layers.Conv2D(256, 3, padding='same', kernel_initializer=he_normal())(d1)
    d1 = layers.LeakyReLU(0.1)(d1); d1 = layers.BatchNormalization()(d1)
    d2 = layers.MaxPooling2D(2)(d1)
    d2 = layers.Conv2D(128, 3, padding='same', kernel_initializer=he_normal())(d2)
    d2 = layers.LeakyReLU(0.1)(d2); d2 = layers.BatchNormalization()(d2)
    d3 = layers.MaxPooling2D(2)(d2)
    d3 = layers.Conv2D(64, 3, padding='same', kernel_initializer=he_normal())(d3)
    d3 = layers.LeakyReLU(0.1)(d3); d3 = layers.BatchNormalization()(d3)
    u1 = layers.UpSampling2D(2, interpolation='bilinear')(d3)
    u1 = layers.Concatenate()([u1, d2])
    u1 = layers.Conv2D(128, 3, padding='same', kernel_initializer=he_normal())(u1)
    u1 = layers.LeakyReLU(0.1)(u1); u1 = layers.BatchNormalization()(u1); u1 = layers.Dropout(0.2)(u1)
    u2 = layers.UpSampling2D(2, interpolation='bilinear')(u1)
    u2 = layers.Concatenate()([u2, d1])
    u2 = layers.Conv2D(256, 3, padding='same', kernel_initializer=he_normal())(u2)
    u2 = layers.LeakyReLU(0.1)(u2); u2 = layers.BatchNormalization()(u2); u2 = layers.Dropout(0.2)(u2)
    u3 = layers.UpSampling2D(2, interpolation='bilinear')(u2)
    u3 = layers.Concatenate()([u3, x])
    u3 = layers.Conv2D(512, 3, padding='same', kernel_initializer=he_normal())(u3)
    u3 = layers.LeakyReLU(0.1)(u3); u3 = layers.BatchNormalization()(u3)
    if use_ppm:
        u3 = PyramidPoolingModule(512)(u3)
    out = layers.Conv2D(out_ch, 1, activation='relu', kernel_initializer=he_normal())(u3)
    return out

def build_unified_mapping_network_toggled(use_ppm=True):
    # UMN maps in ORIGINAL z-space
    in_b  = layers.Input(shape=(8, 8, 256),  name="input_b")
    in_s3 = layers.Input(shape=(32,32, 64),  name="input_s3")
    in_s4 = layers.Input(shape=(16,16,128),  name="input_s4")
    out_b  = mapping_block(in_b,  256, use_ppm)
    out_s3 = mapping_block(in_s3,  64, use_ppm)
    out_s4 = mapping_block(in_s4, 128, use_ppm)
    model = models.Model([in_b, in_s3, in_s4], [out_b, out_s3, out_s4], name=f"UMN_{'ppm' if use_ppm else 'noppm'}")
    model.compile(optimizer='adam', loss=['mse','mse','mse'])
    return model

# -----------------------------
# Protected decoder wrapper (client-side)
# -----------------------------
def make_protected_mask_decoder(mask_decoder, mask_Tlayers, use_skips=True, name="prot_dec"):
    pb_in  = tf.keras.Input(shape=mask_decoder.inputs[0].shape[1:], name="pb_protected")
    if use_skips and len(mask_decoder.inputs) == 3:
        ps3_in = tf.keras.Input(shape=mask_decoder.inputs[1].shape[1:], name="ps3_protected")
        ps4_in = tf.keras.Input(shape=mask_decoder.inputs[2].shape[1:], name="ps4_protected")

        pb_raw  = tf.keras.layers.Lambda(lambda t: mask_Tlayers["T_b"].inverse(t),  name="inv_T_b")(pb_in)
        ps3_raw = tf.keras.layers.Lambda(lambda t: mask_Tlayers["T_s3"].inverse(t), name="inv_T_s3")(ps3_in)
        ps4_raw = tf.keras.layers.Lambda(lambda t: mask_Tlayers["T_s4"].inverse(t), name="inv_T_s4")(ps4_in)

        out = mask_decoder([pb_raw, ps3_raw, ps4_raw])
        return tf.keras.Model([pb_in, ps3_in, ps4_in], out, name=name)

    pb_raw = tf.keras.layers.Lambda(lambda t: mask_Tlayers["T_b"].inverse(t), name="inv_T_b")(pb_in)
    out = mask_decoder(pb_raw)
    return tf.keras.Model(pb_in, out, name=name)

# -----------------------------
# Training helpers
# -----------------------------
def partition_clients(X, Y, num_clients=3, seed=SEED):
    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    X, Y = X[idx], Y[idx]
    splits = np.array_split(np.arange(len(X)), num_clients)
    return [(X[s], Y[s]) for s in splits]

def train_client_autoencoders(client_data, use_skips=True, use_klt=True, tag="", epochs=AE_EPOCHS):
    enc_raw_list, enc_T_list, dec_list, mask_Tlayers_list, img_Tlayers_list = [], [], [], [], []
    for ci, (X, Y) in enumerate(client_data, start=1):
        img_ae,  img_enc_raw,  img_encT,  img_dec,  img_T = build_autoencoder_toggled(
            input_shape=X.shape[1:], use_skips=use_skips, use_klt=use_klt, seed=SEED+ci)
        mask_ae, mask_enc_raw, mask_encT, mask_dec, mask_T = build_autoencoder_toggled(
            input_shape=Y.shape[1:], use_skips=use_skips, use_klt=use_klt, seed=SEED+100+ci)

        img_ae.compile(optimizer='adam',  loss=combined_loss(0.5,0.5))
        mask_ae.compile(optimizer='adam', loss=combined_loss(0.5,0.5))

        img_ae.fit(X, X, epochs=epochs, batch_size=BATCH_SIZE, validation_split=0.2,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)], verbose=1)
        mask_ae.fit(Y, Y, epochs=epochs, batch_size=BATCH_SIZE, validation_split=0.2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)], verbose=1)

        save_dir = f"./saved_models/ablation/{tag}/client_{ci}"
        os.makedirs(save_dir, exist_ok=True)
        img_ae.save_weights(os.path.join(save_dir, "imgAE.weights.h5"))
        mask_ae.save_weights(os.path.join(save_dir, "maskAE.weights.h5"))

        enc_raw_list.append((img_enc_raw, mask_enc_raw))
        enc_T_list.append((img_encT, mask_encT))
        dec_list.append((img_dec, mask_dec))
        mask_Tlayers_list.append(mask_T)
        img_Tlayers_list.append(img_T)

    return enc_raw_list, enc_T_list, dec_list, mask_Tlayers_list, img_Tlayers_list

@tf.function(jit_compile=False)
def _inverse_T_batch(zT, T_layer):
    return T_layer.inverse(zT)

def extract_latents_original_space(enc_T_list, client_data, img_Tlayers_list, mask_Tlayers_list):
    """
    Returns ORIGINAL z-space latents (after server inversion) for images and masks.
    """
    img_b, img_s3, img_s4 = [], [], []
    msk_b, msk_s3, msk_s4 = [], [], []
    for (img_encT, mask_encT), (X, Y), img_T, mask_T in zip(enc_T_list, client_data, img_Tlayers_list, mask_Tlayers_list):
        bT, s3T, s4T    = img_encT.predict(X, batch_size=BATCH_SIZE, verbose=0)
        mbT, ms3T, ms4T = mask_encT.predict(Y, batch_size=BATCH_SIZE, verbose=0)
        b   = _inverse_T_batch(tf.convert_to_tensor(bT),  img_T["T_b"]).numpy()
        s3  = _inverse_T_batch(tf.convert_to_tensor(s3T), img_T["T_s3"]).numpy()
        s4  = _inverse_T_batch(tf.convert_to_tensor(s4T), img_T["T_s4"]).numpy()
        mb  = _inverse_T_batch(tf.convert_to_tensor(mbT),  mask_T["T_b"]).numpy()
        ms3 = _inverse_T_batch(tf.convert_to_tensor(ms3T), mask_T["T_s3"]).numpy()
        ms4 = _inverse_T_batch(tf.convert_to_tensor(ms4T), mask_T["T_s4"]).numpy()
        img_b.append(b);   img_s3.append(s3);   img_s4.append(s4)
        msk_b.append(mb);  msk_s3.append(ms3);  msk_s4.append(ms4)
    return (img_b, img_s3, img_s4), (msk_b, msk_s3, msk_s4)

def train_umn_on_original_z(encoded_imgs_z, encoded_msks_z, use_ppm=True, tag=""):
    Xb  = np.concatenate(encoded_imgs_z[0], axis=0)
    Xs3 = np.concatenate(encoded_imgs_z[1], axis=0)
    Xs4 = np.concatenate(encoded_imgs_z[2], axis=0)
    Yb  = np.concatenate(encoded_msks_z[0], axis=0)
    Ys3 = np.concatenate(encoded_msks_z[1], axis=0)
    Ys4 = np.concatenate(encoded_msks_z[2], axis=0)
    model = build_unified_mapping_network_toggled(use_ppm=use_ppm)
    model.fit([Xb, Xs3, Xs4], [Yb, Ys3, Ys4],
              epochs=MAP_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)], verbose=1)
    os.makedirs(f"./saved_models/ablation/{tag}", exist_ok=True)
    model.save_weights(f"./saved_models/ablation/{tag}/umn.weights.h5")
    return model

def evaluate_variant_server_inverts_then_retransforms(umn,
                                                      img_encT, img_Tlayers,
                                                      mask_dec, mask_Tlayers,
                                                      use_skips, Xtest, Ytest):
    """
    Inference path implementing new concept:
      Client → (E,T) sends bT,s3T,s4T
      Server → T^{-1} -> UMN(z) -> T (mask-space) -> send back
      Client → Protected decoder T^{-1} -> D
    """
    bT, s3T, s4T = img_encT.predict(Xtest, batch_size=BATCH_SIZE, verbose=0)
    b   = _inverse_T_batch(tf.convert_to_tensor(bT),  img_Tlayers["T_b"]).numpy()
    s3  = _inverse_T_batch(tf.convert_to_tensor(s3T), img_Tlayers["T_s3"]).numpy()
    s4  = _inverse_T_batch(tf.convert_to_tensor(s4T), img_Tlayers["T_s4"]).numpy()

    pb_z, ps3_z, ps4_z = umn.predict([b, s3, s4], batch_size=BATCH_SIZE, verbose=0)

    pb  = mask_Tlayers["T_b"](tf.convert_to_tensor(pb_z)).numpy()
    ps3 = mask_Tlayers["T_s3"](tf.convert_to_tensor(ps3_z)).numpy()
    ps4 = mask_Tlayers["T_s4"](tf.convert_to_tensor(ps4_z)).numpy()

    prot_dec = make_protected_mask_decoder(mask_dec, mask_Tlayers, use_skips=use_skips)
    if use_skips and len(prot_dec.inputs) == 3:
        Yhat = prot_dec.predict([pb, ps3, ps4], batch_size=BATCH_SIZE, verbose=0)
    else:
        Yhat = prot_dec.predict([pb], batch_size=BATCH_SIZE, verbose=0)

    d   = dice_score(Ytest, Yhat)
    j   = iou(Ytest, Yhat)
    sen = sensitivity(Ytest, Yhat)
    spe = specificity(Ytest, Yhat)
    return d, j, sen, spe

# -----------------------------
# t-SNE utilities
# -----------------------------
def _latent_spatial_mean(z4d):
    """(B,H,W,C) -> (B,C) via spatial average pooling."""
    if isinstance(z4d, tf.Tensor):
        z4d = z4d.numpy()
    return np.mean(z4d, axis=(1, 2))

def _collect_bottleneck_vectors_before_after_T(encoder_T, Tlayers, X, n_samples=200, batch_size=32):
    """
    Returns two arrays of shape (N,C):
      - z_vec:   original bottleneck latent vectors BEFORE T   [via inverse(T)]
      - zT_vec:  transformed bottleneck latent vectors AFTER T [direct from encoder_T]
    Only the bottleneck stream (8x8x256) is used for visualization.
    """
    N = min(n_samples, X.shape[0])
    Xs = X[:N]
    bT, _, _ = encoder_T.predict(Xs, batch_size=batch_size, verbose=0)       # AFTER T
    b = Tlayers["T_b"].inverse(tf.convert_to_tensor(bT)).numpy()             # BEFORE T (z)
    zT_vec = _latent_spatial_mean(bT)  # (N,256)
    z_vec  = _latent_spatial_mean(b)   # (N,256)
    return z_vec, zT_vec

def plot_tsne_before_after_T_all_clients(enc_T_list,
                                         client_splits,
                                         img_Tlayers_list,
                                         mask_Tlayers_list,
                                         out_dir,
                                         seed=1337,
                                         n_per_client=200,
                                         tag="full_model"):
    """
    Build separate t-SNE embeddings and plots for BEFORE-T and AFTER-T latents.
    Generates 4 plots:
      - {tag}_tsne_images_before.png
      - {tag}_tsne_images_after.png
      - {tag}_tsne_masks_before.png
      - {tag}_tsne_masks_after.png
    """
    os.makedirs(out_dir, exist_ok=True)

    def _collect_all(kind="images"):
        all_before, all_after, all_labels = [], [], []
        for ci, ((img_encT, mask_encT), (Xi, Yi), img_T, mask_T) in enumerate(
            zip(enc_T_list, client_splits, img_Tlayers_list, mask_Tlayers_list), start=1
        ):
            if kind == "images":
                z_vec, zT_vec = _collect_bottleneck_vectors_before_after_T(
                    encoder_T=img_encT, Tlayers=img_T, X=Xi, n_samples=n_per_client
                )
            else:
                z_vec, zT_vec = _collect_bottleneck_vectors_before_after_T(
                    encoder_T=mask_encT, Tlayers=mask_T, X=Yi, n_samples=n_per_client
                )
            all_before.append(z_vec)
            all_after.append(zT_vec)
            all_labels.extend([ci] * z_vec.shape[0])
        return np.concatenate(all_before, axis=0), np.concatenate(all_after, axis=0), np.array(all_labels)

    def _tsne_and_plot(Z, labels, fname, title):
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca",
                    random_state=seed, n_iter=1500)
        emb = tsne.fit_transform(Z)
        plt.figure(figsize=(7, 6))
        for cid in np.unique(labels):
            sel = labels == cid
            plt.scatter(emb[sel, 0], emb[sel, 1], s=20, alpha=0.8, label=f"Client {cid}", edgecolor="none")
        plt.title(title)
        plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
        plt.legend(fontsize=8, frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(fname, dpi=220)
        plt.close()
        print(f"[t-SNE] Saved: {fname}")

    # Images
    Zb_img, Zt_img, L_img = _collect_all("images")
    _tsne_and_plot(Zb_img, L_img,
                   os.path.join(out_dir, f"{tag}_tsne_images_before.png"),
                   title="t-SNE (bottleneck) — Images BEFORE Transform")
    _tsne_and_plot(Zt_img, L_img,
                   os.path.join(out_dir, f"{tag}_tsne_images_after.png"),
                   title="t-SNE (bottleneck) — Images AFTER Transform")

    # Masks
    Zb_msk, Zt_msk, L_msk = _collect_all("masks")
    _tsne_and_plot(Zb_msk, L_msk,
                   os.path.join(out_dir, f"{tag}_tsne_masks_before.png"),
                   title="t-SNE (bottleneck) — Masks BEFORE Transform")
    _tsne_and_plot(Zt_msk, L_msk,
                   os.path.join(out_dir, f"{tag}_tsne_masks_after.png"),
                   title="t-SNE (bottleneck) — Masks AFTER Transform")

# -----------------------------
# Membership Inference Attack (z-space) (UPDATED metrics)
# -----------------------------
def mia_features_from_pairs_zspace(umn, Xb_z, Xs3_z, Xs4_z, Yb_z, Ys3_z, Ys4_z, batch_size=32):
    pb_z, ps3_z, ps4_z = umn.predict([Xb_z, Xs3_z, Xs4_z], batch_size=batch_size, verbose=0)
    def mse(a, b): return np.mean((a - b) ** 2, axis=(1,2,3))
    mse_b, mse_s3, mse_s4 = mse(pb_z, Yb_z), mse(ps3_z, Ys3_z), mse(ps4_z, Ys4_z)
    mse_all = mse_b + mse_s3 + mse_s4
    return np.stack([mse_b, mse_s3, mse_s4, mse_all], axis=1)

def make_member_nonmember_sets_zspace(encoded_imgs_z, encoded_msks_z, member_frac=0.8, seed=1337):
    img_b  = np.concatenate(encoded_imgs_z[0], axis=0)
    img_s3 = np.concatenate(encoded_imgs_z[1], axis=0)
    img_s4 = np.concatenate(encoded_imgs_z[2], axis=0)
    msk_b  = np.concatenate(encoded_msks_z[0], axis=0)
    msk_s3 = np.concatenate(encoded_msks_z[1], axis=0)
    msk_s4 = np.concatenate(encoded_msks_z[2], axis=0)
    N = img_b.shape[0]
    idx = np.arange(N); rng = np.random.default_rng(seed); rng.shuffle(idx)
    cut = int(member_frac * N)
    mem_idx, non_idx = idx[:cut], idx[cut:]
    member  = (img_b[mem_idx], img_s3[mem_idx], img_s4[mem_idx], msk_b[mem_idx], msk_s3[mem_idx], msk_s4[mem_idx])
    nonmem  = (img_b[non_idx], img_s3[non_idx], img_s4[non_idx], msk_b[non_idx], msk_s3[non_idx], msk_s4[non_idx])
    return member, nonmem

def _binary_confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def _mia_extra_metrics_from_preds(y_true, y_pred, eps=1e-12):
    tp, tn, fp, fn = _binary_confusion_counts(y_true, y_pred)
    sen = (tp + eps) / (tp + fn + eps)
    spe = (tn + eps) / (tn + fp + eps)
    jac = (tp + eps) / (tp + fp + fn + eps)
    return float(sen), float(spe), float(jac)

def run_membership_inference_attack_zspace(umn, encoded_imgs_z, encoded_msks_z, out_csv, seed=1337):
    member, nonmem = make_member_nonmember_sets_zspace(encoded_imgs_z, encoded_msks_z, member_frac=0.8, seed=seed)
    mem_Xb, mem_Xs3, mem_Xs4, mem_Yb, mem_Ys3, mem_Ys4 = member
    non_Xb, non_Xs3, non_Xs4, non_Yb, non_Ys3, non_Ys4 = nonmem

    X_mem = mia_features_from_pairs_zspace(umn, mem_Xb, mem_Xs3, mem_Xs4, mem_Yb, mem_Ys3, mem_Ys4)
    X_non = mia_features_from_pairs_zspace(umn, non_Xb, non_Xs3, non_Xs4, non_Yb, non_Ys3, non_Ys4)

    X = np.concatenate([X_mem, X_non], axis=0)
    y = np.concatenate([np.ones(len(X_mem), dtype=int), np.zeros(len(X_non), dtype=int)], axis=0)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    ])
    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_te, y_prob)
    acc = accuracy_score(y_te, y_pred)
    sen, spe, jac = _mia_extra_metrics_from_preds(y_te, y_pred)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["auc","acc","sensitivity","specificity","jaccard","num_members","num_nonmembers","seed"]
        )
        writer.writeheader()
        writer.writerow({
            "auc": float(auc),
            "acc": float(acc),
            "sensitivity": float(sen),
            "specificity": float(spe),
            "jaccard": float(jac),
            "num_members": int(len(X_mem)),
            "num_nonmembers": int(len(X_non)),
            "seed": int(seed)
        })

    return {"auc": float(auc), "acc": float(acc), "sensitivity": float(sen), "specificity": float(spe), "jaccard": float(jac)}

# -----------------------------
# Latency & Communication utils (UPDATED breakdown)
# -----------------------------
def payload_megabytes(latent_shapes, dtype="float32", directions=2):
    bytes_per = np.dtype(np.float32).itemsize if dtype == "float32" else np.dtype(np.float16).itemsize
    total_elems = sum([h*w*c for (h,w,c) in latent_shapes])
    total_bytes = directions * total_elems * bytes_per
    return total_bytes / (1024.0*1024.0)

@tf.function(jit_compile=False)
def _encode_raw_step(encoder_raw, x):
    return encoder_raw(x, training=False)

@tf.function(jit_compile=False)
def _forward_T_step(Tlayers, b, s3, s4):
    bT  = Tlayers["T_b"](b, training=False)  if hasattr(Tlayers["T_b"], "__call__") else Tlayers["T_b"](b)
    s3T = Tlayers["T_s3"](s3, training=False) if hasattr(Tlayers["T_s3"], "__call__") else Tlayers["T_s3"](s3)
    s4T = Tlayers["T_s4"](s4, training=False) if hasattr(Tlayers["T_s4"], "__call__") else Tlayers["T_s4"](s4)
    return bT, s3T, s4T

@tf.function(jit_compile=False)
def _inverse_T_step(Tlayers, bT, s3T, s4T):
    b  = Tlayers["T_b"].inverse(bT)
    s3 = Tlayers["T_s3"].inverse(s3T)
    s4 = Tlayers["T_s4"].inverse(s4T)
    return b, s3, s4

@tf.function(jit_compile=False)
def _map_step_z(umn, b, s3, s4):
    return umn([b, s3, s4], training=False)

@tf.function(jit_compile=False)
def _decode_step(mask_dec, use_skips, pb_raw, ps3_raw=None, ps4_raw=None):
    if use_skips:
        return mask_dec([pb_raw, ps3_raw, ps4_raw], training=False)
    return mask_dec([pb_raw], training=False)

def measure_inference_time_breakdown_new_concept(
    img_encoder_raw,           # E only
    img_Tlayers,               # img T / T^{-1}
    umn,                       # UMN only
    mask_Tlayers,              # mask T / T^{-1}
    mask_decoder,              # D only (raw decoder)
    sample_batch,
    use_skips=True,
    dtype="float32",
    warmup=10,
    repeats=50,
    out_csv="inference_breakdown.csv"
):
    """
    Measures:
      - encoder_ms: raw encoder only (E)
      - transform_ms: forward transforms only (T on img latents + T on mask latents)
      - map_ms: mapping network only (UMN)
      - inverse_transform_ms: inverse transforms only (T^{-1} on img latents at server + T^{-1} on mask latents at client)
      - decoder_ms: decoder only (D)
      - total_ms: sum of above
    """
    latent_shapes = [(8,8,256), (32,32,64), (16,16,128)] if use_skips else [(8,8,256)]

    # Warmup
    for _ in range(warmup):
        b, s3, s4 = _encode_raw_step(img_encoder_raw, sample_batch)
        bT, s3T, s4T = _forward_T_step(img_Tlayers, b, s3, s4)
        b_z, s3_z, s4_z = _inverse_T_step(img_Tlayers, bT, s3T, s4T)
        pb_z, ps3_z, ps4_z = _map_step_z(umn, b_z, s3_z, s4_z)
        pbT, ps3T, ps4T = _forward_T_step(mask_Tlayers, pb_z, ps3_z, ps4_z)
        pb_raw, ps3_raw, ps4_raw = _inverse_T_step(mask_Tlayers, pbT, ps3T, ps4T)
        _ = _decode_step(mask_decoder, use_skips, pb_raw, ps3_raw, ps4_raw)

    t_enc, t_T, t_map, t_Tinv, t_dec = [], [], [], [], []

    for _ in range(repeats):
        # Encoder only
        t0 = time.perf_counter()
        b, s3, s4 = _encode_raw_step(img_encoder_raw, sample_batch)
        t1 = time.perf_counter()

        # Forward img transform
        t2 = time.perf_counter()
        bT, s3T, s4T = _forward_T_step(img_Tlayers, b, s3, s4)
        t3 = time.perf_counter()

        # Inverse img transform (server)
        t4 = time.perf_counter()
        b_z, s3_z, s4_z = _inverse_T_step(img_Tlayers, bT, s3T, s4T)
        t5 = time.perf_counter()

        # Mapping only
        t6 = time.perf_counter()
        pb_z, ps3_z, ps4_z = _map_step_z(umn, b_z, s3_z, s4_z)
        t7 = time.perf_counter()

        # Forward mask transform (server)
        t8 = time.perf_counter()
        pbT, ps3T, ps4T = _forward_T_step(mask_Tlayers, pb_z, ps3_z, ps4_z)
        t9 = time.perf_counter()

        # Inverse mask transform (client)
        t10 = time.perf_counter()
        pb_raw, ps3_raw, ps4_raw = _inverse_T_step(mask_Tlayers, pbT, ps3T, ps4T)
        t11 = time.perf_counter()

        # Decoder only
        t12 = time.perf_counter()
        _ = _decode_step(mask_decoder, use_skips, pb_raw, ps3_raw, ps4_raw)
        t13 = time.perf_counter()

        enc_ms = (t1 - t0) * 1000.0
        transform_ms = ((t3 - t2) + (t9 - t8)) * 1000.0
        inv_ms = ((t5 - t4) + (t11 - t10)) * 1000.0
        map_ms = (t7 - t6) * 1000.0
        dec_ms = (t13 - t12) * 1000.0

        t_enc.append(enc_ms)
        t_T.append(transform_ms)
        t_Tinv.append(inv_ms)
        t_map.append(map_ms)
        t_dec.append(dec_ms)

    enc_ms  = float(np.mean(t_enc))
    T_ms    = float(np.mean(t_T))
    map_ms  = float(np.mean(t_map))
    Tinv_ms = float(np.mean(t_Tinv))
    dec_ms  = float(np.mean(t_dec))
    total_ms = enc_ms + T_ms + map_ms + Tinv_ms + dec_ms

    payload_mb = payload_megabytes(latent_shapes, dtype=dtype, directions=2)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "batch_size",
            "encoder_ms",
            "transform_ms",
            "map_ms",
            "inverse_transform_ms",
            "decoder_ms",
            "total_ms",
            "payload_mb",
            "dtype",
            "latent_shapes",
            "repeats",
            "warmup",
        ])
        writer.writeheader()
        writer.writerow({
            "batch_size": int(sample_batch.shape[0]),
            "encoder_ms": enc_ms,
            "transform_ms": T_ms,
            "map_ms": map_ms,
            "inverse_transform_ms": Tinv_ms,
            "decoder_ms": dec_ms,
            "total_ms": total_ms,
            "payload_mb": payload_mb,
            "dtype": dtype,
            "latent_shapes": str(latent_shapes),
            "repeats": int(repeats),
            "warmup": int(warmup),
        })

    return {
        "encoder_ms": enc_ms,
        "transform_ms": T_ms,
        "map_ms": map_ms,
        "inverse_transform_ms": Tinv_ms,
        "decoder_ms": dec_ms,
        "total_ms": total_ms,
        "payload_mb": payload_mb,
    }

# -----------------------------
# Variants (UPDATED: removed no_ppm)
# -----------------------------
variants = [
    {"name":"full_model",         "use_klt":True,  "use_ppm":True,  "use_skips":True},
    {"name":"no_klt",             "use_klt":False, "use_ppm":True,  "use_skips":True},
    {"name":"no_skips_plain_ae",  "use_klt":True,  "use_ppm":True,  "use_skips":False},
    {"name": "baseline", "use_klt": False, "use_ppm": True, "use_skips": False},
]

# -----------------------------
# Client partitions
# -----------------------------
client_splits = partition_clients(img_train, mask_train, num_clients=NUM_CLIENTS, seed=SEED)

# -----------------------------
# Run ablations (NEW concept)
# -----------------------------
rows = []
for v in variants:
    tag = v["name"]
    print(f"\n================= Variant (NEW CONCEPT): {tag} =================")

    # Train/reuse AEs
    if REUSE_FULL_AE_WEIGHTS and tag != "full_model":
        enc_raw_list, enc_T_list, dec_list, mask_Tlayers_list, img_Tlayers_list = [], [], [], [], []
        for ci, (X, Y) in enumerate(client_splits, start=1):
            img_ae,  img_enc_raw,  img_encT,  img_dec,  img_T = build_autoencoder_toggled(
                input_shape=X.shape[1:], use_skips=v["use_skips"], use_klt=v["use_klt"], seed=SEED+ci)
            mask_ae, mask_enc_raw, mask_encT, mask_dec, mask_T = build_autoencoder_toggled(
                input_shape=Y.shape[1:], use_skips=v["use_skips"], use_klt=v["use_klt"], seed=SEED+ci)
            fm_dir = f"./saved_models/ablation/full_model/client_{ci}"
            try:
                img_ae.load_weights(os.path.join(fm_dir, "imgAE.weights.h5"))
                mask_ae.load_weights(os.path.join(fm_dir, "maskAE.weights.h5"))
            except Exception:
                pass
            enc_raw_list.append((img_enc_raw, mask_enc_raw))
            enc_T_list.append((img_encT, mask_encT))
            dec_list.append((img_dec, mask_dec))
            mask_Tlayers_list.append(mask_T)
            img_Tlayers_list.append(img_T)
    else:
        enc_raw_list, enc_T_list, dec_list, mask_Tlayers_list, img_Tlayers_list = train_client_autoencoders(
            client_splits, use_skips=v["use_skips"], use_klt=v["use_klt"], tag=tag, epochs=AE_EPOCHS
        )

    # (Optional) t-SNE plots for FULL model before UMN training
    if tag == "full_model":
        plot_tsne_before_after_T_all_clients(
            enc_T_list=enc_T_list,
            client_splits=client_splits,
            img_Tlayers_list=img_Tlayers_list,
            mask_Tlayers_list=mask_Tlayers_list,
            out_dir=RESULTS_DIR,
            seed=SEED,
            n_per_client=200,
            tag=tag,
        )

    # SERVER: invert to original z and train UMN on z
    enc_imgs_z, enc_msks_z = extract_latents_original_space(enc_T_list, client_splits, img_Tlayers_list, mask_Tlayers_list)
    umn = train_umn_on_original_z(enc_imgs_z, enc_msks_z, use_ppm=v["use_ppm"], tag=tag)

    # Evaluate segmentation (avg over clients) with server invert→map→re-T
    dices, ious, sens, specs = [], [], [], []
    for (img_encT, _mask_encT), (_img_dec, mask_dec), mask_T, img_T in zip(enc_T_list, dec_list, mask_Tlayers_list, img_Tlayers_list):
        d, j, se, sp = evaluate_variant_server_inverts_then_retransforms(
            umn, img_encT, img_T, mask_dec, mask_T, v["use_skips"], img_test, mask_test
        )
        dices.append(d); ious.append(j); sens.append(se); specs.append(sp)

    dice_mean, dice_std = float(np.mean(dices)), float(np.std(dices))
    iou_mean,  iou_std  = float(np.mean(ious)),  float(np.std(ious))
    sen_mean,  sen_std  = float(np.mean(sens)),  float(np.std(sens))
    spe_mean,  spe_std  = float(np.mean(specs)), float(np.std(specs))

    # Predictions for visualization (three clients)
    preds_client_list = []
    for (img_encT, _mask_encT), (_img_dec, mask_dec), mask_T, img_T in zip(enc_T_list, dec_list, mask_Tlayers_list, img_Tlayers_list):
        bT_all, s3T_all, s4T_all = img_encT.predict(img_test, batch_size=BATCH_SIZE, verbose=0)
        b_all   = _inverse_T_batch(tf.convert_to_tensor(bT_all),  img_T["T_b"]).numpy()
        s3_all  = _inverse_T_batch(tf.convert_to_tensor(s3T_all), img_T["T_s3"]).numpy()
        s4_all  = _inverse_T_batch(tf.convert_to_tensor(s4T_all), img_T["T_s4"]).numpy()
        pb_z, ps3_z, ps4_z = umn.predict([b_all, s3_all, s4_all], batch_size=BATCH_SIZE, verbose=0)
        pb_all  = mask_T["T_b"](tf.convert_to_tensor(pb_z)).numpy()
        ps3_all = mask_T["T_s3"](tf.convert_to_tensor(ps3_z)).numpy()
        ps4_all = mask_T["T_s4"](tf.convert_to_tensor(ps4_z)).numpy()
        prot_dec = make_protected_mask_decoder(mask_dec, mask_T, use_skips=v["use_skips"])
        if v["use_skips"] and len(prot_dec.inputs) == 3:
            Yhat_all = prot_dec.predict([pb_all, ps3_all, ps4_all], batch_size=BATCH_SIZE, verbose=0)
        else:
            Yhat_all = prot_dec.predict([pb_all], batch_size=BATCH_SIZE, verbose=0)
        preds_client_list.append(Yhat_all)

    # Save qualitative grids
    def onehot_to_label(mask_onehot):
        if mask_onehot.ndim in (3, 4):
            return np.argmax(mask_onehot, axis=-1)
        raise ValueError("Unexpected mask shape.")

    def label_to_rgb(label2d, palette=None):
        if palette is None:
            palette = {0:(255,0,0), 1:(0,255,0), 2:(0,0,255)}
        h, w = label2d.shape
        rgb = np.zeros((h,w,3), dtype=np.uint8)
        for k, col in palette.items():
            rgb[label2d == k] = col
        return rgb

    def _prep_image_panel(x):
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[...,0]
        if x.max() > 1.0:
            x = x/255.0
        return np.clip(x, 0.0, 1.0)

    def _pred_to_rgb(pred_onehot):
        pred = np.clip(pred_onehot, 0.0, 1.0)
        lbl  = np.argmax(pred, axis=-1)
        return label_to_rgb(lbl)

    def _tile(axs, r, img, gt_rgb, p1_rgb, p2_rgb, p3_rgb):
        axs[r,0].imshow(_prep_image_panel(img), cmap='gray'); axs[r,0].set_title("Image", fontsize=9); axs[r,0].axis('off')
        axs[r,1].imshow(gt_rgb);  axs[r,1].set_title("Ground Truth", fontsize=9); axs[r,1].axis('off')
        axs[r,2].imshow(p1_rgb);  axs[r,2].set_title("Client-1", fontsize=9); axs[r,2].axis('off')
        axs[r,3].imshow(p2_rgb);  axs[r,3].set_title("Client-2", fontsize=9); axs[r,3].axis('off')
        axs[r,4].imshow(p3_rgb);  axs[r,4].set_title("Client-3", fontsize=9); axs[r,4].axis('off')

    def save_variant_figures(variant_name, Xtest, Ytest, preds_client_list, out_dir="./results", total_samples=12, rows_per_fig=6):
        os.makedirs(out_dir, exist_ok=True)
        assert len(preds_client_list) == 3
        N = Xtest.shape[0]
        sel = np.linspace(0, N-1, num=min(total_samples, N), dtype=int)
        groups = [sel[:rows_per_fig], sel[rows_per_fig:2*rows_per_fig]]
        for k, idxs in enumerate(groups, start=1):
            if len(idxs) == 0: continue
            fig, axs = plt.subplots(len(idxs), 5, figsize=(16, 3*len(idxs)), constrained_layout=True)
            if len(idxs) == 1:
                axs = np.expand_dims(axs, 0)
            for r, i in enumerate(idxs):
                img = Xtest[i]
                gt_rgb = label_to_rgb(onehot_to_label(Ytest[i]))
                p1_rgb = _pred_to_rgb(preds_client_list[0][i])
                p2_rgb = _pred_to_rgb(preds_client_list[1][i])
                p3_rgb = _pred_to_rgb(preds_client_list[2][i])
                _tile(axs, r, img, gt_rgb, p1_rgb, p2_rgb, p3_rgb)
            out_path = os.path.join(out_dir, f"{variant_name}_viz_{k}.png")
            plt.suptitle(f"{variant_name}: Image | GT | Client-1 | Client-2 | Client-3", fontsize=12)
            plt.savefig(out_path, dpi=200); plt.close(fig)
            print(f"Saved figure: {out_path}")

    save_variant_figures(
        variant_name=tag,
        Xtest=img_test, Ytest=mask_test,
        preds_client_list=preds_client_list,
        out_dir=RESULTS_DIR, total_samples=12, rows_per_fig=6
    )

    # MIA (in original z-space) (UPDATED)
    mia_csv   = os.path.join(RESULTS_DIR, f"{tag}_mia.csv")
    mia_stats = run_membership_inference_attack_zspace(umn, enc_imgs_z, enc_msks_z, out_csv=mia_csv, seed=SEED)

    # Latency/Comm breakdown (client-1 repr.) (UPDATED)
    img_enc_raw_1, _mask_enc_raw_1 = enc_raw_list[0]
    _img_dec_1, mask_dec_1         = dec_list[0]
    mask_T_1                       = mask_Tlayers_list[0]
    img_T_1                        = img_Tlayers_list[0]

    lat_csv   = os.path.join(RESULTS_DIR, f"{tag}_latency_comm.csv")
    sample_bs1 = tf.convert_to_tensor(img_test[:1], dtype=tf.float32)
    lat_stats = measure_inference_time_breakdown_new_concept(
        img_encoder_raw=img_enc_raw_1,
        img_Tlayers=img_T_1,
        umn=umn,
        mask_Tlayers=mask_T_1,
        mask_decoder=mask_dec_1,
        sample_batch=sample_bs1,
        use_skips=v["use_skips"],
        dtype="float32",
        warmup=10,
        repeats=50,
        out_csv=lat_csv
    )

    row = {
        "variant": tag, "use_klt": int(v["use_klt"]), "use_ppm": int(v["use_ppm"]), "use_skips": int(v["use_skips"]),
        "dice_mean": dice_mean, "dice_std": dice_std,
        "iou_mean":  iou_mean,  "iou_std":  iou_std,
        "sens_mean": sen_mean,  "sens_std": sen_std,
        "spec_mean": spe_mean,  "spec_std": spe_std,

        "mia_auc": float(mia_stats["auc"]),
        "mia_acc": float(mia_stats["acc"]),
        "mia_sensitivity": float(mia_stats["sensitivity"]),
        "mia_specificity": float(mia_stats["specificity"]),
        "mia_jaccard": float(mia_stats["jaccard"]),

        "lat_encoder_ms": float(lat_stats["encoder_ms"]),
        "lat_transform_ms": float(lat_stats["transform_ms"]),
        "lat_map_ms": float(lat_stats["map_ms"]),
        "lat_inverse_transform_ms": float(lat_stats["inverse_transform_ms"]),
        "lat_decoder_ms": float(lat_stats["decoder_ms"]),
        "lat_total_ms": float(lat_stats["total_ms"]),
        "payload_mb": float(lat_stats["payload_mb"]),

        "num_clients": NUM_CLIENTS, "seed": SEED
    }

    print(
        f"[{tag}] Dice={row['dice_mean']:.4f} IoU={row['iou_mean']:.4f} "
        f"Sens={row['sens_mean']:.4f} Spec={row['spec_mean']:.4f} | "
        f"MIA AUC={row['mia_auc']:.3f} Acc={row['mia_acc']:.3f} "
        f"Sen={row['mia_sensitivity']:.3f} Spe={row['mia_specificity']:.3f} Jac={row['mia_jaccard']:.3f} | "
        f"Latency total={row['lat_total_ms']:.2f}ms "
        f"(E={row['lat_encoder_ms']:.2f}, T={row['lat_transform_ms']:.2f}, "
        f"F={row['lat_map_ms']:.2f}, Tinv={row['lat_inverse_transform_ms']:.2f}, D={row['lat_decoder_ms']:.2f}) | "
        f"Payload={row['payload_mb']:.3f}MB"
    )

    rows.append(row)

# -----------------------------
# Write combined CSV (UPDATED fields)
# -----------------------------
fieldnames = [
    "variant","use_klt","use_ppm","use_skips",
    "dice_mean","dice_std","iou_mean","iou_std",
    "sens_mean","sens_std","spec_mean","spec_std",

    "mia_auc","mia_acc","mia_sensitivity","mia_specificity","mia_jaccard",

    "lat_encoder_ms","lat_transform_ms","lat_map_ms","lat_inverse_transform_ms","lat_decoder_ms","lat_total_ms",
    "payload_mb",

    "num_clients","seed"
]

with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nAll results written to {RESULTS_CSV}")
print(json.dumps(rows, indent=2))
