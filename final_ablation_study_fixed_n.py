"""
PSFH (MHA) — Collaborative vs Local UMN under Fixed Per-Client Budget (500 samples)
==================================================================================

Goal:
  - PSFH train set ~4000 samples, test set in separate folder.
  - Fix PER-CLIENT training samples = 500.
  - Sweep number of clients N in [1,2,3,4,5,6,7,8].
  - Compare:
      A) COLLAB UMN: one global UMN trained on aggregated z-pairs across clients (N*500 samples)
      B) LOCAL UMN : each client trains its own UMN on its own 500 samples
  - Evaluate:
      Reconstruction (IMAGE AE): SSIM, PSNR
      Segmentation (pipeline): Dice, IoU, Sensitivity, Specificity (ignore background for PSFH multi-class)
  - Run for 2 variants:
      1) full:            KLT + skips (AE) + PPM (UMN)
      2) no_klt_no_skips: identity transform + plain AE + PPM (UMN)

Notes:
  - No cross prediction, no visualization.
  - Uses the "new concept" inference path:
      client: E → T  -> server: T^{-1} → UMN(z) → T -> client: T^{-1} → D

Outputs:
  ./results/psfh_fixed500_collab_vs_local/
      summary.csv
      detailed.json
      per_setting/
        N{N}_{variant}_collab_clients.csv
        N{N}_{variant}_local_clients.csv

Make sure these folders exist:
  - PSFH_TRAIN_ROOT/image_mha, PSFH_TRAIN_ROOT/label_mha
  - PSFH_TEST_ROOT/image_mha,  PSFH_TEST_ROOT/label_mha
"""

import os, csv, json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import HeNormal
import SimpleITK as sitk
os.chdir('/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION')

# =============================================================================
# Config
# =============================================================================
SEED = 1337

PSFH_TRAIN_ROOT = "./datasets/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression"
PSFH_TEST_ROOT  = "./datasets/testdataset"

NUM_CLASSES = 3      # PSFH (one-hot depth=3 as in your earlier script)
IMG_CH = 3           # use 3-channel images (repeat if source is 1-channel)
TARGET_SIZE = (256, 256)

PER_CLIENT_TRAIN_SAMPLES = 500
CLIENT_SWEEP = [1, 2, 3, 4, 5, 6, 7, 8]

AE_EPOCHS = 100
MAP_EPOCHS = 200
BATCH_SIZE = 16

RESULTS_DIR = "./results/psfh_fixed500_collab_vs_local"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "per_setting"), exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)
he_normal = HeNormal


# =============================================================================
# PSFH Loader (train and test are separate folders)
# =============================================================================
def _resize_hw(img: np.ndarray, target_hw=(256,256), is_mask=False) -> np.ndarray:
    h, w = target_hw
    x = tf.convert_to_tensor(img, dtype=tf.float32)
    if x.ndim == 2:
        x = x[..., None]
    x = tf.expand_dims(x, 0)
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if is_mask else tf.image.ResizeMethod.BILINEAR
    x = tf.image.resize(x, [h, w], method=method)
    x = tf.squeeze(x, 0)
    return x.numpy()

def load_psfh(root_dir: str, num_classes: int = NUM_CLASSES, img_ch: int = IMG_CH, target_hw=(256,256)):
    image_dir = os.path.join(root_dir, "image_mha")
    mask_dir  = os.path.join(root_dir, "label_mha")

    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".mha")])
    mask_files  = sorted([os.path.join(mask_dir,  f) for f in os.listdir(mask_dir)  if f.endswith(".mha")])
    if len(image_files) == 0 or len(mask_files) == 0:
        raise FileNotFoundError(f"PSFH missing .mha under {root_dir}")

    imgs = []
    for fp in image_files:
        im = sitk.ReadImage(fp)
        ar = sitk.GetArrayFromImage(im)
        ar = ar[0] if ar.ndim == 3 else ar
        ar = ar.astype(np.float32)
        ar = ar / 255.0 if ar.max() > 1.0 else ar
        ar = np.clip(ar, 0.0, 1.0)
        ar = _resize_hw(ar, target_hw=target_hw, is_mask=False)
        if ar.ndim == 2:
            ar = ar[..., None]
        if img_ch == 3 and ar.shape[-1] == 1:
            ar = np.repeat(ar, 3, axis=-1)
        elif img_ch == 1 and ar.shape[-1] == 3:
            ar = np.mean(ar, axis=-1, keepdims=True)
        imgs.append(ar)

    msks = []
    for fp in mask_files:
        im = sitk.ReadImage(fp)
        ar = sitk.GetArrayFromImage(im)
        ar = ar[0] if ar.ndim == 3 else ar
        ar = ar.astype(np.float32)
        ar = _resize_hw(ar, target_hw=target_hw, is_mask=True)[..., 0]
        ar = np.rint(ar).astype(np.int32)
        msks.append(ar)

    X = np.asarray(imgs, np.float32)
    Y_lbl = np.asarray(msks, np.int32)
    Y = tf.one_hot(Y_lbl, depth=num_classes).numpy().astype(np.float32)

    print(f"[PSFH] Loaded {root_dir}: X={X.shape} Y(onehot)={Y.shape}")
    return X, Y


print("Loading PSFH train/test ...")
img_train_all, mask_train_all = load_psfh(PSFH_TRAIN_ROOT, num_classes=NUM_CLASSES, img_ch=IMG_CH, target_hw=TARGET_SIZE)
img_test,      mask_test      = load_psfh(PSFH_TEST_ROOT,  num_classes=NUM_CLASSES, img_ch=IMG_CH, target_hw=TARGET_SIZE)


# =============================================================================
# Fixed per-client sampling
# =============================================================================
def sample_clients_fixed_budget(X, Y, num_clients: int, per_client: int, seed: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns list of length num_clients.
    Each entry is (Xc, Yc) with exactly per_client samples.
    Sampling is WITHOUT replacement globally (clients disjoint) to reflect "more clients => more total data".

    Requires: num_clients * per_client <= len(X)
    """
    N = len(X)
    need = num_clients * per_client
    if need > N:
        raise ValueError(f"Need {need} samples but train set has {N}. Reduce clients or per_client budget.")

    rng = np.random.default_rng(seed + 17 * num_clients)
    idx = np.arange(N)
    rng.shuffle(idx)
    chosen = idx[:need]
    splits = np.split(chosen, num_clients)

    out = []
    for s in splits:
        out.append((X[s], Y[s]))
    return out


# =============================================================================
# Latent transform (KLT-like) and Identity
# =============================================================================
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

class IdentityWrap(layers.Layer):
    def call(self, x): return x
    @tf.function
    def inverse(self, x): return x


# =============================================================================
# Loss
# =============================================================================
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


# =============================================================================
# AE builder (toggles)
# =============================================================================
def build_autoencoder_toggled(input_shape,
                              use_skips: bool,
                              use_klt: bool,
                              seed=None,
                              name_prefix=""):
    inp = layers.Input(shape=input_shape, name=f"{name_prefix}ae_input")

    # Encoder
    e1 = layers.Conv2D(16, 3, 2, "same", activation="relu")(inp); e1 = layers.BatchNormalization()(e1)
    e2 = layers.Conv2D(32, 3, 2, "same", activation="relu")(e1);  e2 = layers.BatchNormalization()(e2)
    e3 = layers.Conv2D(64, 3, 2, "same", activation="relu")(e2);  e3 = layers.BatchNormalization()(e3)   # 32x32
    e4 = layers.Conv2D(128,3, 2, "same", activation="relu")(e3);  e4 = layers.BatchNormalization()(e4)   # 16x16
    b  = layers.Conv2D(256, 3, 2, "same", activation="relu", name=f"{name_prefix}latent_b")(e4)           # 8x8

    # KLT or Identity
    if use_klt:
        T_b  = LatentWrapTF(channels=256, seed=None if seed is None else seed+1, name=f"{name_prefix}T_b")
        T_s3 = LatentWrapTF(channels=64,  seed=None if seed is None else seed+2, name=f"{name_prefix}T_s3")
        T_s4 = LatentWrapTF(channels=128, seed=None if seed is None else seed+3, name=f"{name_prefix}T_s4")
    else:
        T_b  = IdentityWrap(name=f"{name_prefix}T_b_id")
        T_s3 = IdentityWrap(name=f"{name_prefix}T_s3_id")
        T_s4 = IdentityWrap(name=f"{name_prefix}T_s4_id")

    b_T, s3_T, s4_T = T_b(b), T_s3(e3), T_s4(e4)

    # Decoder inputs (raw domain)
    enc_in = layers.Input(shape=b.shape[1:],  name=f"{name_prefix}dec_in_b")
    s3_in  = layers.Input(shape=e3.shape[1:], name=f"{name_prefix}dec_in_s3")
    s4_in  = layers.Input(shape=e4.shape[1:], name=f"{name_prefix}dec_in_s4")

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
    decoder = models.Model(decoder_inputs, out, name=f"{name_prefix}Decoder_{'skips' if use_skips else 'plain'}")

    # inverse before decoder for full AE graph
    b_inv  = layers.Lambda(lambda t: T_b.inverse(t),  name=f"{name_prefix}inv_b")(b_T)
    s3_inv = layers.Lambda(lambda t: T_s3.inverse(t), name=f"{name_prefix}inv_s3")(s3_T)
    s4_inv = layers.Lambda(lambda t: T_s4.inverse(t), name=f"{name_prefix}inv_s4")(s4_T)

    recon = decoder([b_inv, s3_inv, s4_inv]) if use_skips else decoder([b_inv])
    autoencoder = models.Model(inp, recon, name=f"{name_prefix}AE_{'skips' if use_skips else 'plain'}_{'klt' if use_klt else 'id'}")
    encoder_T = models.Model(inp, [b_T, s3_T, s4_T], name=f"{name_prefix}EncoderT_{'klt' if use_klt else 'id'}")

    T_layers = {"T_b": T_b, "T_s3": T_s3, "T_s4": T_s4}
    return autoencoder, encoder_T, decoder, T_layers


# =============================================================================
# UMN (PPM always ON)
# =============================================================================
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

def mapping_block(x, out_ch):
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer=he_normal())(x)
    x = layers.LeakyReLU(0.1)(x); x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer=he_normal())(x)
    x = layers.LeakyReLU(0.1)(x); x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer=he_normal())(x)
    x = layers.LeakyReLU(0.1)(x); x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

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

    u3 = PyramidPoolingModule(512)(u3)

    out = layers.Conv2D(out_ch, 1, activation='relu', kernel_initializer=he_normal())(u3)
    return out

def build_umn():
    in_b  = layers.Input(shape=(8, 8, 256),  name="input_b")
    in_s3 = layers.Input(shape=(32,32, 64),  name="input_s3")
    in_s4 = layers.Input(shape=(16,16,128),  name="input_s4")
    out_b  = mapping_block(in_b,  256)
    out_s3 = mapping_block(in_s3,  64)
    out_s4 = mapping_block(in_s4, 128)
    model = models.Model([in_b, in_s3, in_s4], [out_b, out_s3, out_s4], name="UMN_ppm")
    model.compile(optimizer='adam', loss=['mse','mse','mse'])
    return model


# =============================================================================
# Protected decoder wrapper (client-side)
# =============================================================================
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


# =============================================================================
# Metrics: Reconstruction
# =============================================================================
def recon_ssim(x, xhat, max_val=1.0):
    x_tf = tf.convert_to_tensor(x, tf.float32)
    xh_tf = tf.convert_to_tensor(xhat, tf.float32)
    s = tf.image.ssim(x_tf, xh_tf, max_val=max_val)
    return float(tf.reduce_mean(s).numpy())

def recon_psnr(x, xhat, max_val=1.0):
    x_tf = tf.convert_to_tensor(x, tf.float32)
    xh_tf = tf.convert_to_tensor(xhat, tf.float32)
    p = tf.image.psnr(x_tf, xh_tf, max_val=max_val)
    return float(tf.reduce_mean(p).numpy())


# =============================================================================
# Metrics: Segmentation (ignore background for PSFH multi-class)
# =============================================================================
def _fg_select(y_true, y_pred):
    return y_true[..., 1:], y_pred[..., 1:]

def dice_score(y_true, y_pred, eps=1e-6):
    y_true, y_pred = _fg_select(y_true, y_pred)
    y_pred = (y_pred > 0.5).astype(np.float32)
    inter = np.sum(y_true * y_pred)
    denom = np.sum(y_true) + np.sum(y_pred)
    return float((2.*inter + eps) / (denom + eps))

def iou_score(y_true, y_pred, eps=1e-6):
    y_true, y_pred = _fg_select(y_true, y_pred)
    y_pred = (y_pred > 0.5).astype(np.float32)
    inter = np.sum(y_true * y_pred)
    union = np.sum(np.clip(y_true + y_pred, 0, 1))
    return float((inter + eps) / (union + eps))

def sensitivity(y_true, y_pred, eps=1e-6):
    y_true, y_pred = _fg_select(y_true, y_pred)
    y_pred = (y_pred > 0.5).astype(np.float32)
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return float((tp + eps) / (tp + fn + eps))

def specificity(y_true, y_pred, eps=1e-6):
    y_true, y_pred = _fg_select(y_true, y_pred)
    y_pred = (y_pred > 0.5).astype(np.float32)
    tn = np.sum((1 - y_true) * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    return float((tn + eps) / (tn + fp + eps))


# =============================================================================
# Inference (new concept)
# =============================================================================
@tf.function(jit_compile=False)
def _inverse_T_batch(zT, T_layer):
    return T_layer.inverse(zT)

def evaluate_segmentation_new_concept(umn,
                                      img_encT, img_Tlayers,
                                      mask_dec, mask_Tlayers,
                                      use_skips,
                                      Xtest, Ytest):
    # client encode+T
    bT, s3T, s4T = img_encT.predict(Xtest, batch_size=BATCH_SIZE, verbose=0)

    # server invert to z
    b  = _inverse_T_batch(tf.convert_to_tensor(bT),  img_Tlayers["T_b"]).numpy()
    s3 = _inverse_T_batch(tf.convert_to_tensor(s3T), img_Tlayers["T_s3"]).numpy()
    s4 = _inverse_T_batch(tf.convert_to_tensor(s4T), img_Tlayers["T_s4"]).numpy()

    # server map in z
    pb_z, ps3_z, ps4_z = umn.predict([b, s3, s4], batch_size=BATCH_SIZE, verbose=0)

    # server apply target mask transform T
    pb  = mask_Tlayers["T_b"](tf.convert_to_tensor(pb_z)).numpy()
    ps3 = mask_Tlayers["T_s3"](tf.convert_to_tensor(ps3_z)).numpy()
    ps4 = mask_Tlayers["T_s4"](tf.convert_to_tensor(ps4_z)).numpy()

    # client protected decode
    prot_dec = make_protected_mask_decoder(mask_dec, mask_Tlayers, use_skips=use_skips)
    Yhat = prot_dec.predict([pb, ps3, ps4], batch_size=BATCH_SIZE, verbose=0) if use_skips else prot_dec.predict([pb], batch_size=BATCH_SIZE, verbose=0)

    return {
        "dice": dice_score(Ytest, Yhat),
        "iou": iou_score(Ytest, Yhat),
        "sensitivity": sensitivity(Ytest, Yhat),
        "specificity": specificity(Ytest, Yhat),
    }


# =============================================================================
# Experiment variants
# =============================================================================
@dataclass
class VariantSpec:
    name: str
    use_klt: bool
    use_skips: bool

VARIANTS = [
    VariantSpec(name="full", use_klt=True, use_skips=True),
    VariantSpec(name="no_klt_no_skips", use_klt=False, use_skips=False),
]


# =============================================================================
# Training blocks
# =============================================================================
def train_client_aes(client_splits, variant: VariantSpec):
    """
    For each client:
      - train image AE
      - train mask AE
    Returns lists:
      img_encT_list, img_ae_list, img_dec_list, img_T_list,
      msk_encT_list, msk_ae_list, msk_dec_list, msk_T_list
    """
    img_encT_list, img_ae_list, img_dec_list, img_T_list = [], [], [], []
    msk_encT_list, msk_ae_list, msk_dec_list, msk_T_list = [], [], [], []

    for ci, (Xc, Yc) in enumerate(client_splits, start=1):
        img_ae, img_encT, img_dec, img_T = build_autoencoder_toggled(
            input_shape=Xc.shape[1:],
            use_skips=variant.use_skips,
            use_klt=variant.use_klt,
            seed=SEED + ci,
            name_prefix=f"c{ci}_img_"
        )
        msk_ae, msk_encT, msk_dec, msk_T = build_autoencoder_toggled(
            input_shape=Yc.shape[1:],
            use_skips=variant.use_skips,
            use_klt=variant.use_klt,
            seed=SEED + 100 + ci,
            name_prefix=f"c{ci}_msk_"
        )

        img_ae.compile(optimizer="adam", loss=combined_loss(0.5, 0.5))
        msk_ae.compile(optimizer="adam", loss=combined_loss(0.5, 0.5))

        print(f"\n[Variant={variant.name}] Client {ci}: train IMAGE AE (n={len(Xc)})")
        img_ae.fit(
            Xc, Xc,
            epochs=AE_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
            verbose=1
        )

        print(f"[Variant={variant.name}] Client {ci}: train MASK AE (n={len(Yc)})")
        msk_ae.fit(
            Yc, Yc,
            epochs=AE_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
            verbose=1
        )

        img_encT_list.append(img_encT); img_ae_list.append(img_ae); img_dec_list.append(img_dec); img_T_list.append(img_T)
        msk_encT_list.append(msk_encT); msk_ae_list.append(msk_ae); msk_dec_list.append(msk_dec); msk_T_list.append(msk_T)

    return img_encT_list, img_ae_list, img_dec_list, img_T_list, msk_encT_list, msk_ae_list, msk_dec_list, msk_T_list


def extract_z_pairs_per_client(img_encT_list, msk_encT_list, client_splits, img_T_list, msk_T_list):
    """
    Returns per-client ORIGINAL z-space latents:
      per_client_pairs[k] = (Xb, Xs3, Xs4, Yb, Ys3, Ys4)
    """
    per_client_pairs = []
    for (Xc, Yc), img_encT, msk_encT, img_T, msk_T in zip(client_splits, img_encT_list, msk_encT_list, img_T_list, msk_T_list):
        bT, s3T, s4T = img_encT.predict(Xc, batch_size=BATCH_SIZE, verbose=0)
        mbT, ms3T, ms4T = msk_encT.predict(Yc, batch_size=BATCH_SIZE, verbose=0)

        b  = _inverse_T_batch(tf.convert_to_tensor(bT),  img_T["T_b"]).numpy()
        s3 = _inverse_T_batch(tf.convert_to_tensor(s3T), img_T["T_s3"]).numpy()
        s4 = _inverse_T_batch(tf.convert_to_tensor(s4T), img_T["T_s4"]).numpy()

        mb  = _inverse_T_batch(tf.convert_to_tensor(mbT),  msk_T["T_b"]).numpy()
        ms3 = _inverse_T_batch(tf.convert_to_tensor(ms3T), msk_T["T_s3"]).numpy()
        ms4 = _inverse_T_batch(tf.convert_to_tensor(ms4T), msk_T["T_s4"]).numpy()

        per_client_pairs.append((b, s3, s4, mb, ms3, ms4))

    return per_client_pairs


def train_umn_collaborative(per_client_pairs):
    """
    Aggregate all clients' data to train ONE UMN.
    """
    Xb  = np.concatenate([p[0] for p in per_client_pairs], axis=0)
    Xs3 = np.concatenate([p[1] for p in per_client_pairs], axis=0)
    Xs4 = np.concatenate([p[2] for p in per_client_pairs], axis=0)
    Yb  = np.concatenate([p[3] for p in per_client_pairs], axis=0)
    Ys3 = np.concatenate([p[4] for p in per_client_pairs], axis=0)
    Ys4 = np.concatenate([p[5] for p in per_client_pairs], axis=0)

    umn = build_umn()
    umn.fit(
        [Xb, Xs3, Xs4], [Yb, Ys3, Ys4],
        epochs=MAP_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)],
        verbose=1
    )
    return umn


def train_umn_local(per_client_pairs):
    """
    Train one UMN per client on that client's data.
    Returns list of umn_models length = num_clients
    """
    umns = []
    for k, (Xb, Xs3, Xs4, Yb, Ys3, Ys4) in enumerate(per_client_pairs, start=1):
        umn = build_umn()
        umn.fit(
            [Xb, Xs3, Xs4], [Yb, Ys3, Ys4],
            epochs=MAP_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)],
            verbose=1
        )
        umns.append(umn)
    return umns


# =============================================================================
# Evaluation
# =============================================================================
def eval_reconstruction_per_client(img_ae_list, Xtest):
    """
    Evaluate SSIM+PSNR for each client's IMAGE AE on Xtest.
    """
    stats = []
    for ci, img_ae in enumerate(img_ae_list, start=1):
        Xhat = img_ae.predict(Xtest, batch_size=BATCH_SIZE, verbose=0)
        stats.append({
            "client": ci,
            "ssim": recon_ssim(Xtest, Xhat),
            "psnr": recon_psnr(Xtest, Xhat),
        })
    return stats

def eval_segmentation_collab(umn_global,
                             img_encT_list, img_T_list,
                             msk_dec_list, msk_T_list,
                             use_skips,
                             Xtest, Ytest):
    """
    Evaluate segmentation for each client decoder using the SAME global UMN.
    """
    stats = []
    for ci, (img_encT, img_T, msk_dec, msk_T) in enumerate(zip(img_encT_list, img_T_list, msk_dec_list, msk_T_list), start=1):
        met = evaluate_segmentation_new_concept(
            umn=umn_global,
            img_encT=img_encT, img_Tlayers=img_T,
            mask_dec=msk_dec, mask_Tlayers=msk_T,
            use_skips=use_skips,
            Xtest=Xtest, Ytest=Ytest
        )
        met["client"] = ci
        stats.append(met)
    return stats

def eval_segmentation_local(umn_list,
                            img_encT_list, img_T_list,
                            msk_dec_list, msk_T_list,
                            use_skips,
                            Xtest, Ytest):
    """
    Evaluate segmentation for each client using its OWN UMN.
    """
    stats = []
    for ci, (umn, img_encT, img_T, msk_dec, msk_T) in enumerate(zip(umn_list, img_encT_list, img_T_list, msk_dec_list, msk_T_list), start=1):
        met = evaluate_segmentation_new_concept(
            umn=umn,
            img_encT=img_encT, img_Tlayers=img_T,
            mask_dec=msk_dec, mask_Tlayers=msk_T,
            use_skips=use_skips,
            Xtest=Xtest, Ytest=Ytest
        )
        met["client"] = ci
        stats.append(met)
    return stats

def _mean_std(stats: List[Dict[str, float]], key: str):
    vals = [float(d[key]) for d in stats]
    return float(np.mean(vals)), float(np.std(vals))


# =============================================================================
# Main sweep
# =============================================================================
def run_sweep():
    summary_rows = []
    detailed = []

    for variant in VARIANTS:
        for n_clients in CLIENT_SWEEP:
            print("\n" + "="*100)
            print(f"RUN: variant={variant.name} | clients={n_clients} | per_client_train={PER_CLIENT_TRAIN_SAMPLES}")
            print("="*100)

            # Sample exactly 500 per client (disjoint pools)
            client_splits = sample_clients_fixed_budget(
                img_train_all, mask_train_all,
                num_clients=n_clients,
                per_client=PER_CLIENT_TRAIN_SAMPLES,
                seed=SEED
            )

            # Train client AEs
            img_encT_list, img_ae_list, img_dec_list, img_T_list, msk_encT_list, msk_ae_list, msk_dec_list, msk_T_list = \
                train_client_aes(client_splits, variant=variant)

            # Reconstruction evaluation
            recon_stats = eval_reconstruction_per_client(img_ae_list, img_test)
            recon_ssim_mean, recon_ssim_std = _mean_std(recon_stats, "ssim")
            recon_psnr_mean, recon_psnr_std = _mean_std(recon_stats, "psnr")

            # Extract per-client z pairs
            per_client_pairs = extract_z_pairs_per_client(img_encT_list, msk_encT_list, client_splits, img_T_list, msk_T_list)

            # A) Collaborative UMN
            print(f"\nTraining COLLABORATIVE UMN: total_samples={n_clients*PER_CLIENT_TRAIN_SAMPLES}")
            umn_global = train_umn_collaborative(per_client_pairs)
            seg_stats_collab = eval_segmentation_collab(
                umn_global,
                img_encT_list, img_T_list,
                msk_dec_list, msk_T_list,
                use_skips=variant.use_skips,
                Xtest=img_test, Ytest=mask_test
            )

            # B) Local UMNs
            print(f"\nTraining LOCAL UMNs: {n_clients} networks each with {PER_CLIENT_TRAIN_SAMPLES} samples")
            umn_list = train_umn_local(per_client_pairs)
            seg_stats_local = eval_segmentation_local(
                umn_list,
                img_encT_list, img_T_list,
                msk_dec_list, msk_T_list,
                use_skips=variant.use_skips,
                Xtest=img_test, Ytest=mask_test
            )

            # Aggregate means/stds for segmentation
            dice_mean_c, dice_std_c = _mean_std(seg_stats_collab, "dice")
            iou_mean_c,  iou_std_c  = _mean_std(seg_stats_collab, "iou")
            sen_mean_c,  sen_std_c  = _mean_std(seg_stats_collab, "sensitivity")
            spe_mean_c,  spe_std_c  = _mean_std(seg_stats_collab, "specificity")

            dice_mean_l, dice_std_l = _mean_std(seg_stats_local, "dice")
            iou_mean_l,  iou_std_l  = _mean_std(seg_stats_local, "iou")
            sen_mean_l,  sen_std_l  = _mean_std(seg_stats_local, "sensitivity")
            spe_mean_l,  spe_std_l  = _mean_std(seg_stats_local, "specificity")

            # Summary rows (collab + local)
            summary_rows.append({
                "variant": variant.name,
                "num_clients": n_clients,
                "per_client_train": PER_CLIENT_TRAIN_SAMPLES,
                "total_train_used": n_clients * PER_CLIENT_TRAIN_SAMPLES,
                "strategy": "collab",
                "recon_ssim_mean": recon_ssim_mean, "recon_ssim_std": recon_ssim_std,
                "recon_psnr_mean": recon_psnr_mean, "recon_psnr_std": recon_psnr_std,
                "dice_mean": dice_mean_c, "dice_std": dice_std_c,
                "iou_mean": iou_mean_c,   "iou_std": iou_std_c,
                "sens_mean": sen_mean_c,  "sens_std": sen_std_c,
                "spec_mean": spe_mean_c,  "spec_std": spe_std_c,
                "seed": SEED
            })
            summary_rows.append({
                "variant": variant.name,
                "num_clients": n_clients,
                "per_client_train": PER_CLIENT_TRAIN_SAMPLES,
                "total_train_used": n_clients * PER_CLIENT_TRAIN_SAMPLES,
                "strategy": "local",
                "recon_ssim_mean": recon_ssim_mean, "recon_ssim_std": recon_ssim_std,
                "recon_psnr_mean": recon_psnr_mean, "recon_psnr_std": recon_psnr_std,
                "dice_mean": dice_mean_l, "dice_std": dice_std_l,
                "iou_mean": iou_mean_l,   "iou_std": iou_std_l,
                "sens_mean": sen_mean_l,  "sens_std": sen_std_l,
                "spec_mean": spe_mean_l,  "spec_std": spe_std_l,
                "seed": SEED
            })

            detailed.append({
                "variant": variant.name,
                "num_clients": n_clients,
                "per_client_train": PER_CLIENT_TRAIN_SAMPLES,
                "total_train_used": n_clients * PER_CLIENT_TRAIN_SAMPLES,
                "reconstruction_per_client": recon_stats,
                "segmentation_per_client_collab": seg_stats_collab,
                "segmentation_per_client_local": seg_stats_local
            })

            # Save per-setting client-level CSVs
            per_dir = os.path.join(RESULTS_DIR, "per_setting")

            collab_csv = os.path.join(per_dir, f"N{n_clients}_{variant.name}_collab_clients.csv")
            with open(collab_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["client","recon_ssim","recon_psnr","dice","iou","sensitivity","specificity"])
                writer.writeheader()
                for r, s in zip(recon_stats, seg_stats_collab):
                    writer.writerow({
                        "client": r["client"],
                        "recon_ssim": r["ssim"],
                        "recon_psnr": r["psnr"],
                        "dice": s["dice"],
                        "iou": s["iou"],
                        "sensitivity": s["sensitivity"],
                        "specificity": s["specificity"],
                    })

            local_csv = os.path.join(per_dir, f"N{n_clients}_{variant.name}_local_clients.csv")
            with open(local_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["client","recon_ssim","recon_psnr","dice","iou","sensitivity","specificity"])
                writer.writeheader()
                for r, s in zip(recon_stats, seg_stats_local):
                    writer.writerow({
                        "client": r["client"],
                        "recon_ssim": r["ssim"],
                        "recon_psnr": r["psnr"],
                        "dice": s["dice"],
                        "iou": s["iou"],
                        "sensitivity": s["sensitivity"],
                        "specificity": s["specificity"],
                    })

            print(f"\nSaved per-setting CSVs:")
            print(f"  {collab_csv}")
            print(f"  {local_csv}")

    # Write summary CSV
    summary_csv = os.path.join(RESULTS_DIR, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "variant","num_clients","per_client_train","total_train_used","strategy",
            "recon_ssim_mean","recon_ssim_std",
            "recon_psnr_mean","recon_psnr_std",
            "dice_mean","dice_std",
            "iou_mean","iou_std",
            "sens_mean","sens_std",
            "spec_mean","spec_std",
            "seed"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Write detailed JSON
    detailed_json = os.path.join(RESULTS_DIR, "detailed.json")
    with open(detailed_json, "w") as f:
        json.dump(detailed, f, indent=2)

    print("\nDONE.")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Detailed JSON: {detailed_json}")


if __name__ == "__main__":
    run_sweep()
