"""
FULL MODEL ONLY (KLT + PPM + skip-connected AEs) WITH:
1) Performance for BOTH reconstruction and segmentation
2) Cross prediction for BOTH tasks:
   - Cross Reconstruction: encoder_i + decoder_j (i != j)
   - Cross Segmentation  : encoder_i + decoder_j (i != j)  [server invert->UMN->re-T->protected decode]
3) Visualizations for ALL above tasks BUT EXCLUDING correct combinations (i==j) from visuals.
   (We still compute metrics for i==j, but we don't plot them.)

NOTES:
- Reconstruction is evaluated on image AEs (img_train/img_test) using Dice/IoU? No: reconstruction uses PSNR/SSIM + MSE.
- Segmentation is evaluated using Dice/IoU/Sensitivity/Specificity (same as your prior).
- Cross reconstruction uses the "new concept" transform-consistent routing:
    client i produces transformed bT,s3T,s4T
    server: invert with img_T_i^{-1} to z
    server: apply img_T_j forward to get protected latents for decoder_j
    client j: protected image decoder applies img_T_j^{-1} then D_j
- Cross segmentation uses:
    client i produces transformed image latents
    server: invert with img_T_i^{-1} to image z
    server: UMN maps to mask z
    server: apply mask_T_j forward to send protected mask latents
    client j: protected mask decoder applies mask_T_j^{-1} then D_j
"""

import os, csv, json, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import HeNormal
from sklearn.manifold import TSNE
import SimpleITK as sitk
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
SEED = 42
NUM_CLIENTS = 3
AE_EPOCHS = 100
MAP_EPOCHS = 200
BATCH_SIZE = 16

RESULTS_DIR = "./results/full_model"
MODELS_DIR  = "./saved_models/full_model"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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
# Partition clients
# -----------------------------
def partition_clients(X, Y, num_clients=3, seed=SEED):
    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    X, Y = X[idx], Y[idx]
    splits = np.array_split(np.arange(len(X)), num_clients)
    return [(X[s], Y[s]) for s in splits]

client_splits = partition_clients(img_train, mask_train, num_clients=NUM_CLIENTS, seed=SEED)

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

def seg_dice_score(y_true, y_pred, eps=1e-6):
    y_true = y_true[..., 1:]
    y_pred = (y_pred[..., 1:] > 0.5).astype(np.float32)
    inter = np.sum(y_true * y_pred)
    denom = np.sum(y_true) + np.sum(y_pred)
    return (2.*inter + eps) / (denom + eps)

def seg_iou(y_true, y_pred, eps=1e-6):
    y_true = y_true[..., 1:]
    y_pred = (y_pred[..., 1:] > 0.5).astype(np.float32)
    inter = np.sum(y_true * y_pred)
    union = np.sum(np.clip(y_true + y_pred, 0, 1))
    return (inter + eps) / (union + eps)

def seg_sensitivity(y_true, y_pred, eps=1e-6):
    y_true = y_true[..., 1:]
    y_pred = (y_pred[..., 1:] > 0.5).astype(np.float32)
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return (tp + eps) / (tp + fn + eps)

def seg_specificity(y_true, y_pred, eps=1e-6):
    y_true = y_true[..., 1:]
    y_pred = (y_pred[..., 1:] > 0.5).astype(np.float32)
    tn = np.sum((1 - y_true) * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    return (tn + eps) / (tn + fp + eps)

def recon_mse(x, xhat):
    x = np.asarray(x).astype(np.float32)
    xhat = np.asarray(xhat).astype(np.float32)
    return float(np.mean((x - xhat) ** 2))

def recon_psnr(x, xhat, max_val=1.0, eps=1e-12):
    mse = np.mean((x - xhat) ** 2)
    return float(10.0 * np.log10((max_val ** 2) / (mse + eps)))

def recon_ssim_batch(x, xhat, max_val=1.0):
    # Returns average SSIM over batch
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    xh_tf = tf.convert_to_tensor(xhat, dtype=tf.float32)
    s = tf.image.ssim(x_tf, xh_tf, max_val=max_val)  # (B,)
    return float(tf.reduce_mean(s).numpy())

# -----------------------------
# AE builder (FULL MODEL: use_skips=True, use_klt=True)
# Returns: autoencoder, encoder_raw, encoder_T, decoder, T_layers
# -----------------------------
def build_autoencoder_full(input_shape=(256,256,3), seed=None, name_prefix=""):
    use_skips = True
    use_klt   = True

    inp = layers.Input(shape=input_shape, name=f"{name_prefix}ae_input")

    # Encoder (raw)
    e1 = layers.Conv2D(16, 3, 2, "same", activation="relu")(inp); e1 = layers.BatchNormalization()(e1)
    e2 = layers.Conv2D(32, 3, 2, "same", activation="relu")(e1);  e2 = layers.BatchNormalization()(e2)
    e3 = layers.Conv2D(64, 3, 2, "same", activation="relu")(e2);  e3 = layers.BatchNormalization()(e3)   # 32x32
    e4 = layers.Conv2D(128,3, 2, "same", activation="relu")(e3);  e4 = layers.BatchNormalization()(e4)   # 16x16
    b  = layers.Conv2D(256, 3, 2, "same", activation="relu", name=f"{name_prefix}latent_b")(e4)           # 8x8

    encoder_raw = models.Model(inp, [b, e3, e4], name=f"{name_prefix}Encoder_raw")

    # KLT transforms
    T_b  = LatentWrapTF(channels=256, seed=None if seed is None else seed+1, name=f"{name_prefix}T_b")
    T_s3 = LatentWrapTF(channels=64,  seed=None if seed is None else seed+2, name=f"{name_prefix}T_s3")
    T_s4 = LatentWrapTF(channels=128, seed=None if seed is None else seed+3, name=f"{name_prefix}T_s4")
    b_T, s3_T, s4_T = T_b(b), T_s3(e3), T_s4(e4)

    # Decoder (raw domain)
    enc_in = layers.Input(shape=b.shape[1:],   name=f"{name_prefix}dec_in_b")
    s3_in  = layers.Input(shape=e3.shape[1:],  name=f"{name_prefix}dec_in_s3")
    s4_in  = layers.Input(shape=e4.shape[1:],  name=f"{name_prefix}dec_in_s4")

    def up_block(x, filters, skip=None):
        x = layers.UpSampling2D(size=(2,2), interpolation="bilinear")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if skip is not None:
            x = layers.Concatenate()([x, skip])
        return x

    d1 = up_block(enc_in, 128, skip=s4_in)
    d2 = up_block(d1,     64,  skip=s3_in)
    d3 = up_block(d2,     32)
    d4 = up_block(d3,     16)
    out = layers.UpSampling2D(size=(2,2), interpolation="bilinear")(d4)
    out = layers.Conv2D(input_shape[-1], 3, padding="same", activation="sigmoid")(out)

    decoder = models.Model([enc_in, s3_in, s4_in], out, name=f"{name_prefix}Decoder_skips")

    # Full AE path: inverse KLT inside the graph BEFORE decoder
    b_inv  = layers.Lambda(lambda t: T_b.inverse(t),  name=f"{name_prefix}inv_b")(b_T)
    s3_inv = layers.Lambda(lambda t: T_s3.inverse(t), name=f"{name_prefix}inv_s3")(s3_T)
    s4_inv = layers.Lambda(lambda t: T_s4.inverse(t), name=f"{name_prefix}inv_s4")(s4_T)

    recon = decoder([b_inv, s3_inv, s4_inv])

    autoencoder = models.Model(inp, recon, name=f"{name_prefix}AE_full")
    encoder_T   = models.Model(inp, [b_T, s3_T, s4_T], name=f"{name_prefix}Encoder_T_full")

    T_layers = {"T_b": T_b, "T_s3": T_s3, "T_s4": T_s4}
    return autoencoder, encoder_raw, encoder_T, decoder, T_layers

# -----------------------------
# Mapping network (server) - FULL MODEL uses PPM
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

def build_unified_mapping_network_full():
    in_b  = layers.Input(shape=(8, 8, 256),  name="input_b")
    in_s3 = layers.Input(shape=(32,32, 64),  name="input_s3")
    in_s4 = layers.Input(shape=(16,16,128),  name="input_s4")
    out_b  = mapping_block(in_b,  256, use_ppm=True)
    out_s3 = mapping_block(in_s3,  64, use_ppm=True)
    out_s4 = mapping_block(in_s4, 128, use_ppm=True)
    model = models.Model([in_b, in_s3, in_s4], [out_b, out_s3, out_s4], name="UMN_full_ppm")
    model.compile(optimizer='adam', loss=['mse','mse','mse'])
    return model

# -----------------------------
# Protected decoder wrappers (for CROSS reconstruction/segmentation)
# -----------------------------
def make_protected_decoder(decoder, Tlayers, name="prot_dec"):
    """
    Wrap a decoder so its inputs are *protected* (transformed) latents.
    Wrapper does: T^{-1} -> decoder(raw)
    Assumes skips (3 inputs): (b, s3, s4)
    """
    pb_in  = tf.keras.Input(shape=decoder.inputs[0].shape[1:], name="pb_protected")
    ps3_in = tf.keras.Input(shape=decoder.inputs[1].shape[1:], name="ps3_protected")
    ps4_in = tf.keras.Input(shape=decoder.inputs[2].shape[1:], name="ps4_protected")

    pb_raw  = tf.keras.layers.Lambda(lambda t: Tlayers["T_b"].inverse(t),  name="inv_T_b")(pb_in)
    ps3_raw = tf.keras.layers.Lambda(lambda t: Tlayers["T_s3"].inverse(t), name="inv_T_s3")(ps3_in)
    ps4_raw = tf.keras.layers.Lambda(lambda t: Tlayers["T_s4"].inverse(t), name="inv_T_s4")(ps4_in)

    out = decoder([pb_raw, ps3_raw, ps4_raw])
    return tf.keras.Model([pb_in, ps3_in, ps4_in], out, name=name)

@tf.function(jit_compile=False)
def inv_T_batch(zT, T_layer):
    return T_layer.inverse(zT)

# -----------------------------
# Train all clients AEs (image + mask) for FULL MODEL
# -----------------------------
def train_full_model_clients(client_data, epochs=AE_EPOCHS):
    img_autoencoders, img_enc_raws, img_encTs, img_decs, img_Ts = [], [], [], [], []
    msk_autoencoders, msk_enc_raws, msk_encTs, msk_decs, msk_Ts = [], [], [], [], []

    for ci, (X, Y) in enumerate(client_data, start=1):
        print(f"\n[Client {ci}] Training Image AE...")
        img_ae, img_enc_raw, img_encT, img_dec, img_T = build_autoencoder_full(
            input_shape=X.shape[1:], seed=SEED+ci, name_prefix=f"c{ci}_img_")
        img_ae.compile(optimizer='adam', loss=combined_loss(0.5,0.5))
        img_ae.fit(
            X, X,
            epochs=epochs, batch_size=BATCH_SIZE, validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
            verbose=1
        )

        print(f"[Client {ci}] Training Mask AE...")
        msk_ae, msk_enc_raw, msk_encT, msk_dec, msk_T = build_autoencoder_full(
            input_shape=Y.shape[1:], seed=SEED+100+ci, name_prefix=f"c{ci}_msk_")
        msk_ae.compile(optimizer='adam', loss=combined_loss(0.5,0.5))
        msk_ae.fit(
            Y, Y,
            epochs=epochs, batch_size=BATCH_SIZE, validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
            verbose=1
        )

        # Save weights
        cdir = os.path.join(MODELS_DIR, f"client_{ci}")
        os.makedirs(cdir, exist_ok=True)
        img_ae.save_weights(os.path.join(cdir, "imgAE.weights.h5"))
        msk_ae.save_weights(os.path.join(cdir, "maskAE.weights.h5"))

        img_autoencoders.append(img_ae); img_enc_raws.append(img_enc_raw); img_encTs.append(img_encT); img_decs.append(img_dec); img_Ts.append(img_T)
        msk_autoencoders.append(msk_ae); msk_enc_raws.append(msk_enc_raw); msk_encTs.append(msk_encT); msk_decs.append(msk_dec); msk_Ts.append(msk_T)

    return (img_autoencoders, img_enc_raws, img_encTs, img_decs, img_Ts,
            msk_autoencoders, msk_enc_raws, msk_encTs, msk_decs, msk_Ts)

# -----------------------------
# Extract ORIGINAL z-space latents (server inversion) to train UMN
# -----------------------------
def extract_original_z_for_umn(img_encTs, msk_encTs, client_data, img_Ts, msk_Ts):
    img_b, img_s3, img_s4 = [], [], []
    msk_b, msk_s3, msk_s4 = [], [], []

    for (img_encT, msk_encT), (X, Y), img_T, msk_T in zip(zip(img_encTs, msk_encTs), client_data, img_Ts, msk_Ts):
        bT, s3T, s4T    = img_encT.predict(X, batch_size=BATCH_SIZE, verbose=0)
        mbT, ms3T, ms4T = msk_encT.predict(Y, batch_size=BATCH_SIZE, verbose=0)

        b   = inv_T_batch(tf.convert_to_tensor(bT),  img_T["T_b"]).numpy()
        s3  = inv_T_batch(tf.convert_to_tensor(s3T), img_T["T_s3"]).numpy()
        s4  = inv_T_batch(tf.convert_to_tensor(s4T), img_T["T_s4"]).numpy()

        mb  = inv_T_batch(tf.convert_to_tensor(mbT),  msk_T["T_b"]).numpy()
        ms3 = inv_T_batch(tf.convert_to_tensor(ms3T), msk_T["T_s3"]).numpy()
        ms4 = inv_T_batch(tf.convert_to_tensor(ms4T), msk_T["T_s4"]).numpy()

        img_b.append(b);  img_s3.append(s3);  img_s4.append(s4)
        msk_b.append(mb); msk_s3.append(ms3); msk_s4.append(ms4)

    return (img_b, img_s3, img_s4), (msk_b, msk_s3, msk_s4)

def train_umn_full(enc_imgs_z, enc_msks_z):
    Xb  = np.concatenate(enc_imgs_z[0], axis=0)
    Xs3 = np.concatenate(enc_imgs_z[1], axis=0)
    Xs4 = np.concatenate(enc_imgs_z[2], axis=0)
    Yb  = np.concatenate(enc_msks_z[0], axis=0)
    Ys3 = np.concatenate(enc_msks_z[1], axis=0)
    Ys4 = np.concatenate(enc_msks_z[2], axis=0)

    umn = build_unified_mapping_network_full()
    umn.fit(
        [Xb, Xs3, Xs4], [Yb, Ys3, Ys4],
        epochs=MAP_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)],
        verbose=1
    )
    umn.save_weights(os.path.join(MODELS_DIR, "umn_full.weights.h5"))
    return umn

# -----------------------------
# Core inference routines (correct + cross)
# -----------------------------
def reconstruct_via_cross(img_encT_src, img_T_src, img_T_tgt, img_prot_dec_tgt, X):
    """
    Cross reconstruction:
      src: X -> (bT,s3T,s4T)  [protected by src T]
      server: invert with src T^{-1} => z
      server: apply target T (forward) => protected latents for target decoder
      tgt: protected decoder => recon
    """
    bT, s3T, s4T = img_encT_src.predict(X, batch_size=BATCH_SIZE, verbose=0)

    b  = inv_T_batch(tf.convert_to_tensor(bT),  img_T_src["T_b"])
    s3 = inv_T_batch(tf.convert_to_tensor(s3T), img_T_src["T_s3"])
    s4 = inv_T_batch(tf.convert_to_tensor(s4T), img_T_src["T_s4"])

    pb  = img_T_tgt["T_b"](b).numpy()
    ps3 = img_T_tgt["T_s3"](s3).numpy()
    ps4 = img_T_tgt["T_s4"](s4).numpy()

    Xhat = img_prot_dec_tgt.predict([pb, ps3, ps4], batch_size=BATCH_SIZE, verbose=0)
    return Xhat

def segment_via_cross(umn, img_encT_src, img_T_src, msk_T_tgt, msk_prot_dec_tgt, X):
    """
    Cross segmentation:
      src: X -> protected image latents via src encoder_T
      server: invert with src img T^{-1} to image z
      server: UMN maps to mask z
      server: apply target mask T forward to protect
      tgt: protected mask decoder => Yhat
    """
    bT, s3T, s4T = img_encT_src.predict(X, batch_size=BATCH_SIZE, verbose=0)

    b  = inv_T_batch(tf.convert_to_tensor(bT),  img_T_src["T_b"]).numpy()
    s3 = inv_T_batch(tf.convert_to_tensor(s3T), img_T_src["T_s3"]).numpy()
    s4 = inv_T_batch(tf.convert_to_tensor(s4T), img_T_src["T_s4"]).numpy()

    pb_z, ps3_z, ps4_z = umn.predict([b, s3, s4], batch_size=BATCH_SIZE, verbose=0)

    pb  = msk_T_tgt["T_b"](tf.convert_to_tensor(pb_z)).numpy()
    ps3 = msk_T_tgt["T_s3"](tf.convert_to_tensor(ps3_z)).numpy()
    ps4 = msk_T_tgt["T_s4"](tf.convert_to_tensor(ps4_z)).numpy()

    Yhat = msk_prot_dec_tgt.predict([pb, ps3, ps4], batch_size=BATCH_SIZE, verbose=0)
    return Yhat

# -----------------------------
# Visual helpers (EXCLUDE i==j)
# -----------------------------
def _to_gray(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 1:
        return img[..., 0]
    if img.ndim == 3 and img.shape[-1] == 3:
        # show first channel or mean
        return np.mean(img, axis=-1)
    return img

def onehot_to_label(mask_onehot):
    return np.argmax(mask_onehot, axis=-1)

def label_to_rgb(label2d, palette=None):
    if palette is None:
        palette = {0:(255,0,0), 1:(0,255,0), 2:(0,0,255)}
    h, w = label2d.shape
    rgb = np.zeros((h,w,3), dtype=np.uint8)
    for k, col in palette.items():
        rgb[label2d == k] = col
    return rgb

def save_cross_recon_visuals(X, Xhat, src_i, tgt_j, out_path, n=8):
    N = min(n, X.shape[0])
    sel = np.linspace(0, X.shape[0]-1, num=N, dtype=int)

    fig, axs = plt.subplots(N, 2, figsize=(8, 3*N), constrained_layout=True)
    if N == 1:
        axs = np.expand_dims(axs, 0)

    for r, idx in enumerate(sel):
        axs[r,0].imshow(_to_gray(X[idx]), cmap="gray")
        axs[r,0].set_title(f"Input (Enc {src_i})", fontsize=9)
        axs[r,0].axis("off")

        axs[r,1].imshow(_to_gray(Xhat[idx]), cmap="gray")
        axs[r,1].set_title(f"Recon (Dec {tgt_j})", fontsize=9)
        axs[r,1].axis("off")

    plt.suptitle(f"Cross Reconstruction: Encoder {src_i} -> Decoder {tgt_j}  (i!=j)", fontsize=12)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[VIS] Saved cross recon: {out_path}")

def save_cross_seg_visuals(X, Y, Yhat, src_i, tgt_j, out_path, n=8):
    N = min(n, X.shape[0])
    sel = np.linspace(0, X.shape[0]-1, num=N, dtype=int)

    fig, axs = plt.subplots(N, 3, figsize=(12, 3*N), constrained_layout=True)
    if N == 1:
        axs = np.expand_dims(axs, 0)

    for r, idx in enumerate(sel):
        axs[r,0].imshow(_to_gray(X[idx]), cmap="gray")
        axs[r,0].set_title(f"Image (Enc {src_i})", fontsize=9)
        axs[r,0].axis("off")

        gt_rgb = label_to_rgb(onehot_to_label(Y[idx]))
        axs[r,1].imshow(gt_rgb)
        axs[r,1].set_title("GT", fontsize=9)
        axs[r,1].axis("off")

        pr_rgb = label_to_rgb(onehot_to_label(Yhat[idx]))
        axs[r,2].imshow(pr_rgb)
        axs[r,2].set_title(f"Pred (Dec {tgt_j})", fontsize=9)
        axs[r,2].axis("off")

    plt.suptitle(f"Cross Segmentation: Encoder {src_i} -> Decoder {tgt_j}  (i!=j)", fontsize=12)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[VIS] Saved cross seg: {out_path}")

# -----------------------------
# MAIN: train full model, evaluate correct + cross, visualize cross only
# -----------------------------
print("\n=== Training FULL MODEL clients (image + mask AEs) ===")
(img_aes, img_enc_raws, img_encTs, img_decs, img_Ts,
 msk_aes, msk_enc_raws, msk_encTs, msk_decs, msk_Ts) = train_full_model_clients(client_splits, epochs=AE_EPOCHS)

# Build protected decoders for ALL clients (image + mask)
img_prot_decs = []
msk_prot_decs = []
for j in range(NUM_CLIENTS):
    img_prot_decs.append(make_protected_decoder(img_decs[j], img_Ts[j], name=f"prot_img_dec_{j+1}"))
    msk_prot_decs.append(make_protected_decoder(msk_decs[j], msk_Ts[j], name=f"prot_msk_dec_{j+1}"))

print("\n=== Training UMN (server) in ORIGINAL z-space ===")
enc_imgs_z, enc_msks_z = extract_original_z_for_umn(img_encTs, msk_encTs, client_splits, img_Ts, msk_Ts)
umn = train_umn_full(enc_imgs_z, enc_msks_z)

# -----------------------------
# Evaluate performance: Reconstruction (correct + cross) and Segmentation (correct + cross)
# -----------------------------
recon_rows = []
seg_rows   = []

for i in range(NUM_CLIENTS):
    for j in range(NUM_CLIENTS):
        src_i = i + 1
        tgt_j = j + 1

        # ---------- Reconstruction (image) ----------
        Xhat = reconstruct_via_cross(
            img_encT_src=img_encTs[i],
            img_T_src=img_Ts[i],
            img_T_tgt=img_Ts[j],
            img_prot_dec_tgt=img_prot_decs[j],
            X=img_test
        )
        mse  = recon_mse(img_test, Xhat)
        psnr = recon_psnr(img_test, Xhat)
        ssim = recon_ssim_batch(img_test, Xhat)

        recon_rows.append({
            "src_encoder": src_i,
            "tgt_decoder": tgt_j,
            "is_correct_pair": int(i == j),
            "mse": float(mse),
            "psnr": float(psnr),
            "ssim": float(ssim)
        })

        # ---------- Segmentation ----------
        Yhat = segment_via_cross(
            umn=umn,
            img_encT_src=img_encTs[i],
            img_T_src=img_Ts[i],
            msk_T_tgt=msk_Ts[j],
            msk_prot_dec_tgt=msk_prot_decs[j],
            X=img_test
        )

        d   = seg_dice_score(mask_test, Yhat)
        jcc = seg_iou(mask_test, Yhat)
        sen = seg_sensitivity(mask_test, Yhat)
        spe = seg_specificity(mask_test, Yhat)

        seg_rows.append({
            "src_encoder": src_i,
            "tgt_decoder": tgt_j,
            "is_correct_pair": int(i == j),
            "dice": float(d),
            "iou": float(jcc),
            "sensitivity": float(sen),
            "specificity": float(spe)
        })

        # ---------- Visuals (EXCLUDE i==j) ----------
        if i != j:
            out_recon = os.path.join(RESULTS_DIR, f"cross_recon_enc{src_i}_dec{tgt_j}.png")
            out_seg   = os.path.join(RESULTS_DIR, f"cross_seg_enc{src_i}_dec{tgt_j}.png")
            save_cross_recon_visuals(img_test, Xhat, src_i, tgt_j, out_recon, n=8)
            save_cross_seg_visuals(img_test, mask_test, Yhat, src_i, tgt_j, out_seg, n=8)

# Save metrics CSVs
recon_csv = os.path.join(RESULTS_DIR, "reconstruction_metrics_full_model.csv")
seg_csv   = os.path.join(RESULTS_DIR, "segmentation_metrics_full_model.csv")

with open(recon_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["src_encoder","tgt_decoder","is_correct_pair","mse","psnr","ssim"])
    writer.writeheader()
    writer.writerows(recon_rows)

with open(seg_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["src_encoder","tgt_decoder","is_correct_pair","dice","iou","sensitivity","specificity"])
    writer.writeheader()
    writer.writerows(seg_rows)

print(f"\n[OK] Saved reconstruction metrics: {recon_csv}")
print(f"[OK] Saved segmentation metrics:   {seg_csv}")

# Also dump a quick summary JSON
summary = {
    "reconstruction": recon_rows,
    "segmentation": seg_rows
}
with open(os.path.join(RESULTS_DIR, "full_model_cross_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"[OK] Saved summary JSON: {os.path.join(RESULTS_DIR, 'full_model_cross_summary.json')}")
