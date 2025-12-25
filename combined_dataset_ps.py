

"""
UNIFIED 4-DATASET PIPELINE (FIXED PSFH TEST FOLDER) + FULL MODEL
===============================================================

Fix requested:
- PSFH test data is in a DIFFERENT folder (e.g. ./datasets/testdataset).
- This script now supports PSFH with:
    train_root = ".../Pubic Symphysis-Fetal Head Segmentation and Angle of Progression"
    test_root  = ".../testdataset"

What this script does (same as before):
- Unified dataset interface for: PSFH, FUMPE, MRI, NERVE
- Trains FULL model (KLT + SKIPS) for NUM_CLIENTS clients:
    * Image AE (reconstruction)
    * Mask  AE (reconstruction)
    * UMN mapping network (server, PPM) in ORIGINAL z-space
- Evaluates:
    * Reconstruction: SSIM, PSNR, MSE (image AE)
    * Segmentation: Dice, IoU, Sensitivity, Specificity (image->mask via UMN + protected decoder)
- Writes outputs under:
    ./results/unified/<dataset_name>/
    ./saved_models/unified/<dataset_name>/

No cross prediction, no visualization (as per your later preference for that phase you can re-enable if you want).
"""

import os, csv, json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import HeNormal

# Optional imports per dataset
try:
    import SimpleITK as sitk
except Exception:
    sitk = None

try:
    import pydicom
    from scipy.io import loadmat
    from glob import glob
    from tqdm import tqdm
    from PIL import Image
except Exception:
    pydicom = None
    loadmat = None
    glob = None
    tqdm = None
    Image = None

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


# =============================================================================
# Global config
# =============================================================================
SEED = 42
NUM_CLIENTS = 3
AE_EPOCHS = 100
MAP_EPOCHS = 200
BATCH_SIZE = 16

np.random.seed(SEED)
tf.random.set_seed(SEED)
he_normal = HeNormal


# =============================================================================
# Unified Dataset Interface (PSFH now supports separate test_root)
# =============================================================================
@dataclass
class DatasetConfig:
    name: str

    # If dataset uses single root, use root.
    root: Optional[str] = None

    # If dataset uses separate train/test roots (PSFH), set train_root/test_root.
    train_root: Optional[str] = None
    test_root: Optional[str] = None

    target_size: Tuple[int, int] = (256, 256)
    seed: int = 42
    test_ratio: float = 0.1
    normalize: bool = True

    # channels/classes
    num_classes: int = 2
    image_channels: int = 1

    # splitting: "random" | "patient" | "predefined"
    split_strategy: str = "random"

    extra: Optional[Dict[str, Any]] = None


class UnifiedDataset:
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def load(self):
        name = self.cfg.name.lower().strip()
        if name in ("psfh", "psfh_mha", "pubic_symphysis_fetal_head"):
            return self._load_psfh_train_test_separate()
        if name in ("fumpe", "fumpe_new", "fumpe_ct"):
            return self._load_fumpe()
        if name in ("mri", "mri_segmentation", "nii_slices_png"):
            return self._load_mri_png_slices()
        if name in ("nerve", "ultrasound_nerve", "ultrasound-nerve-segmentation"):
            return self._load_nerve_tif_masks()
        raise ValueError(f"Unknown dataset name: {self.cfg.name}")

    # -----------------------------
    # Common utils
    # -----------------------------
    def _random_split_indices(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.arange(n)
        self._rng.shuffle(idx)
        n_test = int(round(self.cfg.test_ratio * n))
        te = idx[:n_test]
        tr = idx[n_test:]
        return tr, te

    def _patient_split(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        unique_ids = np.unique(ids)
        self._rng.shuffle(unique_ids)
        n_test = int(round(self.cfg.test_ratio * len(unique_ids)))
        test_ids = set(unique_ids[:n_test].tolist())
        te = np.where(np.isin(ids, list(test_ids)))[0]
        tr = np.where(~np.isin(ids, list(test_ids)))[0]
        return tr, te

    def _as_channels_last(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return x[..., None]
        return x

    def _resize_hw(self, img: np.ndarray, is_mask: bool) -> np.ndarray:
        h, w = self.cfg.target_size
        x = tf.convert_to_tensor(img, dtype=tf.float32)
        x = tf.expand_dims(x, 0)
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if is_mask else tf.image.ResizeMethod.BILINEAR
        x = tf.image.resize(x, [h, w], method=method)
        return tf.squeeze(x, 0).numpy()

    def _ensure_float01(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.cfg.normalize:
            if x.max() > 1.0:
                mn, mx = float(x.min()), float(x.max())
                if mx > mn:
                    x = (x - mn) / (mx - mn)
                else:
                    x = np.zeros_like(x, dtype=np.float32)
            x = np.clip(x, 0.0, 1.0)
        return x.astype(np.float32)

    def _standardize_image_channels(self, img: np.ndarray) -> np.ndarray:
        img = self._as_channels_last(img).astype(np.float32)
        if self.cfg.image_channels == 1:
            if img.shape[-1] == 1:
                return img
            if img.shape[-1] >= 3:
                return np.mean(img[..., :3], axis=-1, keepdims=True).astype(np.float32)
            return img[..., :1]
        else:
            if img.shape[-1] == 3:
                return img
            if img.shape[-1] == 1:
                return np.repeat(img, 3, axis=-1).astype(np.float32)
            if img.shape[-1] > 3:
                return img[..., :3].astype(np.float32)
            return np.repeat(img[..., :1], 3, axis=-1).astype(np.float32)

    # -----------------------------
    # PSFH (MHA + multi-class) with separate train/test roots
    # -----------------------------
    def _load_psfh_train_test_separate(self):
        if sitk is None:
            raise ImportError("SimpleITK required for PSFH")

        train_root = self.cfg.train_root or self.cfg.root
        test_root  = self.cfg.test_root
        if train_root is None or test_root is None:
            raise ValueError("PSFH requires cfg.train_root (or cfg.root) AND cfg.test_root")

        def load_one(root_dir: str):
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
                ar2 = ar[0] if ar.ndim == 3 else ar
                imgs.append(ar2)
            images = np.array(imgs).astype(np.float32)
            images = images / 255.0 if images.max() > 1.0 else images
            images = self._ensure_float01(images)
            images = self._as_channels_last(images)
            images = self._standardize_image_channels(images)
            images = np.array([self._resize_hw(im, is_mask=False) for im in images], dtype=np.float32)

            msks = []
            for fp in mask_files:
                im = sitk.ReadImage(fp)
                ar = sitk.GetArrayFromImage(im)
                ar2 = ar[0] if ar.ndim == 3 else ar
                msks.append(ar2)
            masks_label = np.array(msks).astype(np.int32)  # (N,H,W)

            # resize labels (nearest)
            resized = []
            for m in masks_label:
                m_ = self._resize_hw(m[..., None].astype(np.float32), is_mask=True)[..., 0]
                resized.append(np.rint(m_).astype(np.int32))
            masks_label = np.stack(resized, axis=0)

            K = int(self.cfg.num_classes)
            if K <= 1:
                raise ValueError("PSFH requires cfg.num_classes >= 2 (typically 3).")
            masks = tf.one_hot(masks_label, depth=K).numpy().astype(np.float32)

            return images, masks

        Xtr, Ytr = load_one(train_root)
        Xte, Yte = load_one(test_root)

        meta = {
            "name": "PSFH",
            "num_classes": int(self.cfg.num_classes),
            "image_channels": int(Xtr.shape[-1]),
            "mask_channels": int(Ytr.shape[-1]),
            "split_strategy": "predefined_separate_folders",
            "train_root": train_root,
            "test_root": test_root
        }
        return (Xtr, Ytr), (Xte, Yte), meta

    # -----------------------------
    # FUMPE (DICOM + .mat, binary)
    # -----------------------------
    def _load_fumpe(self):
        if pydicom is None or loadmat is None or glob is None or tqdm is None or Image is None:
            raise ImportError("FUMPE requires pydicom, scipy, tqdm, pillow")

        root_dir = self.cfg.root
        if root_dir is None:
            raise ValueError("FUMPE requires cfg.root")

        scan_root = os.path.join(root_dir, "CT_scans")
        gt_root   = os.path.join(root_dir, "GroundTruth")
        patient_dirs = sorted([d for d in os.listdir(scan_root) if d.startswith("PAT")])

        def load_dicom_volume(slice_paths):
            slices = []
            for path in slice_paths:
                try:
                    ds = pydicom.dcmread(path)
                    slices.append((ds, path))
                except Exception as e:
                    print(f"❌ Failed {path}: {e}")
            try:
                slices.sort(key=lambda x: int(x[0].InstanceNumber))
            except Exception:
                slices.sort(key=lambda x: x[1])
            volume = []
            for ds, _ in slices:
                img = ds.pixel_array.astype(np.int16)
                slope = getattr(ds, "RescaleSlope", 1)
                intercept = getattr(ds, "RescaleIntercept", 0)
                img = img * slope + intercept
                volume.append(img)
            return np.stack(volume, axis=0)  # (D,H,W)

        def load_gt_mask(mat_path, volume_shape):
            mat = loadmat(mat_path)
            for key in ["GT", "Mask", "seg", "volume"]:
                if key in mat:
                    mask = mat[key].astype(np.float32)
                    if mask.shape != volume_shape and sorted(mask.shape) == sorted(volume_shape):
                        mask = np.transpose(mask, (2, 0, 1))  # (H,W,D)->(D,H,W)
                    return mask
            for v in mat.values():
                if isinstance(v, np.ndarray):
                    return v.astype(np.float32)
            raise ValueError(f"No mask in {mat_path}")

        X_list, Y_list, ID_list = [], [], []

        for pid in tqdm(patient_dirs, desc="Processing patients"):
            dcm_paths = sorted(glob(os.path.join(scan_root, pid, "*.dcm")))
            if len(dcm_paths) == 0:
                continue

            try:
                vol = load_dicom_volume(dcm_paths)
            except Exception as e:
                print(f"⚠️ Skip {pid} DICOM error: {e}")
                continue

            if self.cfg.normalize:
                vol = np.clip(vol, -1000, 1000)
                vol = (vol + 1000) / 2000.0  # [0,1]

            mat_path = os.path.join(gt_root, f"{pid}.mat")
            if not os.path.exists(mat_path):
                print(f"⚠️ No GT for {pid}")
                continue

            try:
                mask = load_gt_mask(mat_path, vol.shape)
            except Exception as e:
                print(f"⚠️ Skip {pid} mask error: {e}")
                continue

            if vol.shape != mask.shape:
                print(f"⚠️ Shape mismatch {pid}")
                continue

            for i in range(vol.shape[0]):
                img_slice = Image.fromarray((vol[i] * 255).astype(np.uint8)).resize(self.cfg.target_size)
                msk_slice = Image.fromarray((mask[i] > 0).astype(np.uint8) * 255).resize(self.cfg.target_size)

                img_arr = np.array(img_slice).astype(np.float32) / 255.0
                msk_arr = (np.array(msk_slice).astype(np.float32) > 127).astype(np.float32)

                img_arr = img_arr[..., None]
                msk_arr = msk_arr[..., None]
                img_arr = self._standardize_image_channels(img_arr)

                X_list.append(img_arr)
                Y_list.append(msk_arr)
                ID_list.append(pid)

        X = np.asarray(X_list, np.float32)
        Y = np.asarray(Y_list, np.float32)
        IDs = np.asarray(ID_list)

        if self.cfg.split_strategy == "patient":
            tr, te = self._patient_split(IDs)
        else:
            tr, te = self._random_split_indices(len(X))

        meta = {
            "name": "FUMPE",
            "num_classes": 2,
            "image_channels": int(X.shape[-1]),
            "mask_channels": 1,
            "split_strategy": self.cfg.split_strategy,
        }
        return (X[tr], Y[tr]), (X[te], Y[te]), meta

    # -----------------------------
    # MRI (PNG slices + labels) predefined split
    # -----------------------------
    def _load_mri_png_slices(self):
        if PILImage is None:
            raise ImportError("MRI requires pillow")

        root = self.cfg.root
        if root is None:
            raise ValueError("MRI requires cfg.root")

        train_img_dir = os.path.join(root, "train_slices")
        train_lbl_dir = os.path.join(root, "train_labels")
        test_img_dir  = os.path.join(root, "test_slices")
        test_lbl_dir  = os.path.join(root, "test_labels")

        def load_pairs(img_dir, lbl_dir):
            img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))])
            lbl_paths = sorted([os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir)
                                if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))])
            if len(img_paths) == 0 or len(lbl_paths) == 0:
                raise FileNotFoundError(f"MRI missing files under {img_dir} / {lbl_dir}")
            if len(img_paths) != len(lbl_paths):
                raise ValueError("MRI mismatch between images and labels count")

            X, Y = [], []
            for ip, mp in zip(img_paths, lbl_paths):
                im = PILImage.open(ip).convert("L").resize(self.cfg.target_size)
                mk = PILImage.open(mp).convert("L").resize(self.cfg.target_size)

                im = np.array(im).astype(np.float32) / 255.0
                mk = (np.array(mk).astype(np.float32) > 127).astype(np.float32)

                im = im[..., None]
                mk = mk[..., None]
                im = self._standardize_image_channels(im)

                X.append(im); Y.append(mk)
            return np.asarray(X, np.float32), np.asarray(Y, np.float32)

        Xtr, Ytr = load_pairs(train_img_dir, train_lbl_dir)
        Xte, Yte = load_pairs(test_img_dir, test_lbl_dir)

        meta = {
            "name": "MRI",
            "num_classes": 2,
            "image_channels": int(Xtr.shape[-1]),
            "mask_channels": 1,
            "split_strategy": "predefined",
        }
        return (Xtr, Ytr), (Xte, Yte), meta

    # -----------------------------
    # NERVE (tif + *_mask.tif) random split
    # -----------------------------
    def _load_nerve_tif_masks(self):
        if PILImage is None:
            raise ImportError("NERVE requires pillow")

        root = self.cfg.root
        if root is None:
            raise ValueError("NERVE requires cfg.root")

        files = sorted(os.listdir(root))
        imgs, masks = [], []
        for fn in files:
            if fn.lower().endswith(".tif") and (not fn.lower().endswith("_mask.tif")):
                mask_fn = fn[:-4] + "_mask.tif"
                if mask_fn in files:
                    imgs.append(os.path.join(root, fn))
                    masks.append(os.path.join(root, mask_fn))
        if len(imgs) == 0:
            raise FileNotFoundError(f"NERVE: no pairs in {root}")

        X_list, Y_list = [], []
        for ip, mp in zip(imgs, masks):
            im = PILImage.open(ip).convert("L")
            mk = PILImage.open(mp).convert("L")

            # replicate pad(14) then resize
            im_np = np.array(im)
            im_np = np.pad(im_np, ((14,14),(14,14)), mode="constant", constant_values=0)
            im = PILImage.fromarray(im_np).resize(self.cfg.target_size)
            mk = mk.resize(self.cfg.target_size)

            im = np.array(im).astype(np.float32) / 255.0
            mk = (np.array(mk).astype(np.float32) > 127).astype(np.float32)

            im = im[..., None]
            mk = mk[..., None]
            im = self._standardize_image_channels(im)

            X_list.append(im); Y_list.append(mk)

        X = np.asarray(X_list, np.float32)
        Y = np.asarray(Y_list, np.float32)

        tr, te = self._random_split_indices(len(X))

        meta = {
            "name": "NERVE",
            "num_classes": 2,
            "image_channels": int(X.shape[-1]),
            "mask_channels": 1,
            "split_strategy": "random",
        }
        return (X[tr], Y[tr]), (X[te], Y[te]), meta


def load_dataset(cfg: DatasetConfig):
    return UnifiedDataset(cfg).load()


# =============================================================================
# FULL MODEL COMPONENTS (KLT + SKIPS) + UMN (PPM)
# =============================================================================
def sample_orthogonal_tf(d, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
    A = tf.random.normal((d, d))
    Q, _ = tf.linalg.qr(A, full_matrices=False)
    return Q

class LatentWrapTF(layers.Layer):
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

def build_autoencoder_full(input_shape, seed=None, name_prefix=""):
    inp = layers.Input(shape=input_shape, name=f"{name_prefix}ae_input")
    e1 = layers.Conv2D(16, 3, 2, "same", activation="relu")(inp); e1 = layers.BatchNormalization()(e1)
    e2 = layers.Conv2D(32, 3, 2, "same", activation="relu")(e1);  e2 = layers.BatchNormalization()(e2)
    e3 = layers.Conv2D(64, 3, 2, "same", activation="relu")(e2);  e3 = layers.BatchNormalization()(e3)
    e4 = layers.Conv2D(128,3, 2, "same", activation="relu")(e3);  e4 = layers.BatchNormalization()(e4)
    b  = layers.Conv2D(256, 3, 2, "same", activation="relu", name=f"{name_prefix}latent_b")(e4)

    T_b  = LatentWrapTF(channels=256, seed=None if seed is None else seed+1, name=f"{name_prefix}T_b")
    T_s3 = LatentWrapTF(channels=64,  seed=None if seed is None else seed+2, name=f"{name_prefix}T_s3")
    T_s4 = LatentWrapTF(channels=128, seed=None if seed is None else seed+3, name=f"{name_prefix}T_s4")

    b_T, s3_T, s4_T = T_b(b), T_s3(e3), T_s4(e4)
    encoder_T = models.Model(inp, [b_T, s3_T, s4_T], name=f"{name_prefix}Encoder_T_full")

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

    b_inv  = layers.Lambda(lambda t: T_b.inverse(t),  name=f"{name_prefix}inv_b")(b_T)
    s3_inv = layers.Lambda(lambda t: T_s3.inverse(t), name=f"{name_prefix}inv_s3")(s3_T)
    s4_inv = layers.Lambda(lambda t: T_s4.inverse(t), name=f"{name_prefix}inv_s4")(s4_T)

    recon = decoder([b_inv, s3_inv, s4_inv])
    autoencoder = models.Model(inp, recon, name=f"{name_prefix}AE_full")

    T_layers = {"T_b": T_b, "T_s3": T_s3, "T_s4": T_s4}
    return autoencoder, encoder_T, decoder, T_layers


class PyramidPoolingModule(layers.Layer):
    def __init__(self, in_channels, pool_sizes=(1,3,5,7), out_channels=None):
        super().__init__()
        self.pool_sizes = pool_sizes
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

def build_unified_mapping_network():
    in_b  = layers.Input(shape=(8, 8, 256),  name="input_b")
    in_s3 = layers.Input(shape=(32,32, 64),  name="input_s3")
    in_s4 = layers.Input(shape=(16,16,128),  name="input_s4")
    out_b  = mapping_block(in_b,  256)
    out_s3 = mapping_block(in_s3,  64)
    out_s4 = mapping_block(in_s4, 128)
    model = models.Model([in_b, in_s3, in_s4], [out_b, out_s3, out_s4], name="UMN_full_ppm")
    model.compile(optimizer='adam', loss=['mse','mse','mse'])
    return model


# =============================================================================
# Metrics
# =============================================================================
def recon_mse(x, xhat):
    x = np.asarray(x, np.float32)
    xhat = np.asarray(xhat, np.float32)
    return float(np.mean((x - xhat) ** 2))

def recon_psnr(x, xhat, max_val=1.0):
    x_tf = tf.convert_to_tensor(x, tf.float32)
    xh_tf = tf.convert_to_tensor(xhat, tf.float32)
    p = tf.image.psnr(x_tf, xh_tf, max_val=max_val)
    return float(tf.reduce_mean(p).numpy())

def recon_ssim(x, xhat, max_val=1.0):
    x_tf = tf.convert_to_tensor(x, tf.float32)
    xh_tf = tf.convert_to_tensor(xhat, tf.float32)
    s = tf.image.ssim(x_tf, xh_tf, max_val=max_val)
    return float(tf.reduce_mean(s).numpy())

def _seg_select_fg(y_true, y_pred, num_classes: int):
    if num_classes > 2:
        return y_true[..., 1:], y_pred[..., 1:]
    return y_true, y_pred

def seg_dice(y_true, y_pred, num_classes: int, eps=1e-6):
    yt, yp = _seg_select_fg(y_true, y_pred, num_classes)
    yp = (yp > 0.5).astype(np.float32)
    inter = np.sum(yt * yp)
    denom = np.sum(yt) + np.sum(yp)
    return float((2.*inter + eps) / (denom + eps))

def seg_iou(y_true, y_pred, num_classes: int, eps=1e-6):
    yt, yp = _seg_select_fg(y_true, y_pred, num_classes)
    yp = (yp > 0.5).astype(np.float32)
    inter = np.sum(yt * yp)
    union = np.sum(np.clip(yt + yp, 0, 1))
    return float((inter + eps) / (union + eps))

def seg_sensitivity(y_true, y_pred, num_classes: int, eps=1e-6):
    yt, yp = _seg_select_fg(y_true, y_pred, num_classes)
    yp = (yp > 0.5).astype(np.float32)
    tp = np.sum(yt * yp)
    fn = np.sum(yt * (1 - yp))
    return float((tp + eps) / (tp + fn + eps))

def seg_specificity(y_true, y_pred, num_classes: int, eps=1e-6):
    yt, yp = _seg_select_fg(y_true, y_pred, num_classes)
    yp = (yp > 0.5).astype(np.float32)
    tn = np.sum((1 - yt) * (1 - yp))
    fp = np.sum((1 - yt) * yp)
    return float((tn + eps) / (tn + fp + eps))


# =============================================================================
# Routing helpers (new concept)
# =============================================================================
@tf.function(jit_compile=False)
def inv_T(zT, T_layer):
    return T_layer.inverse(zT)

def make_protected_decoder(decoder, Tlayers, name="prot_dec"):
    pb_in  = tf.keras.Input(shape=decoder.inputs[0].shape[1:], name="pb_protected")
    ps3_in = tf.keras.Input(shape=decoder.inputs[1].shape[1:], name="ps3_protected")
    ps4_in = tf.keras.Input(shape=decoder.inputs[2].shape[1:], name="ps4_protected")
    pb_raw  = tf.keras.layers.Lambda(lambda t: Tlayers["T_b"].inverse(t),  name="inv_T_b")(pb_in)
    ps3_raw = tf.keras.layers.Lambda(lambda t: Tlayers["T_s3"].inverse(t), name="inv_T_s3")(ps3_in)
    ps4_raw = tf.keras.layers.Lambda(lambda t: Tlayers["T_s4"].inverse(t), name="inv_T_s4")(ps4_in)
    out = decoder([pb_raw, ps3_raw, ps4_raw])
    return tf.keras.Model([pb_in, ps3_in, ps4_in], out, name=name)

def segment_with_new_concept(umn, img_encT, img_T, msk_T_tgt, msk_prot_dec_tgt, X):
    bT, s3T, s4T = img_encT.predict(X, batch_size=BATCH_SIZE, verbose=0)
    b  = inv_T(tf.convert_to_tensor(bT),  img_T["T_b"]).numpy()
    s3 = inv_T(tf.convert_to_tensor(s3T), img_T["T_s3"]).numpy()
    s4 = inv_T(tf.convert_to_tensor(s4T), img_T["T_s4"]).numpy()
    pb_z, ps3_z, ps4_z = umn.predict([b, s3, s4], batch_size=BATCH_SIZE, verbose=0)
    pb  = msk_T_tgt["T_b"](tf.convert_to_tensor(pb_z)).numpy()
    ps3 = msk_T_tgt["T_s3"](tf.convert_to_tensor(ps3_z)).numpy()
    ps4 = msk_T_tgt["T_s4"](tf.convert_to_tensor(ps4_z)).numpy()
    Yhat = msk_prot_dec_tgt.predict([pb, ps3, ps4], batch_size=BATCH_SIZE, verbose=0)
    return Yhat


# =============================================================================
# Training helpers
# =============================================================================
def partition_clients(X, Y, num_clients=3, seed=SEED):
    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    X, Y = X[idx], Y[idx]
    splits = np.array_split(np.arange(len(X)), num_clients)
    return [(X[s], Y[s]) for s in splits]

def train_full_clients(client_splits, dataset_tag, img_shape, msk_shape, save_root):
    img_encTs, img_decs, img_Ts = [], [], []
    msk_encTs, msk_decs, msk_Ts = [], [], []

    for ci, (Xc, Yc) in enumerate(client_splits, start=1):
        print(f"\n[{dataset_tag}] Client {ci} training IMAGE AE...")
        img_ae, img_encT, img_dec, img_T = build_autoencoder_full(
            input_shape=img_shape, seed=SEED + ci, name_prefix=f"c{ci}_img_"
        )
        img_ae.compile(optimizer="adam", loss=combined_loss(0.5, 0.5))
        img_ae.fit(
            Xc, Xc,
            epochs=AE_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
            verbose=1
        )

        print(f"[{dataset_tag}] Client {ci} training MASK AE...")
        msk_ae, msk_encT, msk_dec, msk_T = build_autoencoder_full(
            input_shape=msk_shape, seed=SEED + 100 + ci, name_prefix=f"c{ci}_msk_"
        )
        msk_ae.compile(optimizer="adam", loss=combined_loss(0.5, 0.5))
        msk_ae.fit(
            Yc, Yc,
            epochs=AE_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
            verbose=1
        )

        cdir = os.path.join(save_root, f"client_{ci}")
        os.makedirs(cdir, exist_ok=True)
        img_ae.save_weights(os.path.join(cdir, "imgAE.weights.h5"))
        msk_ae.save_weights(os.path.join(cdir, "maskAE.weights.h5"))

        img_encTs.append(img_encT); img_decs.append(img_dec); img_Ts.append(img_T)
        msk_encTs.append(msk_encT); msk_decs.append(msk_dec); msk_Ts.append(msk_T)

    return img_encTs, img_decs, img_Ts, msk_encTs, msk_decs, msk_Ts

def extract_original_z_for_umn(img_encTs, msk_encTs, client_splits, img_Ts, msk_Ts):
    img_b, img_s3, img_s4 = [], [], []
    msk_b, msk_s3, msk_s4 = [], [], []

    for (Xc, Yc), img_encT, msk_encT, img_T, msk_T in zip(client_splits, img_encTs, msk_encTs, img_Ts, msk_Ts):
        bT, s3T, s4T = img_encT.predict(Xc, batch_size=BATCH_SIZE, verbose=0)
        mbT, ms3T, ms4T = msk_encT.predict(Yc, batch_size=BATCH_SIZE, verbose=0)

        b  = inv_T(tf.convert_to_tensor(bT),  img_T["T_b"]).numpy()
        s3 = inv_T(tf.convert_to_tensor(s3T), img_T["T_s3"]).numpy()
        s4 = inv_T(tf.convert_to_tensor(s4T), img_T["T_s4"]).numpy()

        mb  = inv_T(tf.convert_to_tensor(mbT),  msk_T["T_b"]).numpy()
        ms3 = inv_T(tf.convert_to_tensor(ms3T), msk_T["T_s3"]).numpy()
        ms4 = inv_T(tf.convert_to_tensor(ms4T), msk_T["T_s4"]).numpy()

        img_b.append(b);  img_s3.append(s3);  img_s4.append(s4)
        msk_b.append(mb); msk_s3.append(ms3); msk_s4.append(ms4)

    return (img_b, img_s3, img_s4), (msk_b, msk_s3, msk_s4)

def train_umn(enc_imgs_z, enc_msks_z, save_root):
    Xb  = np.concatenate(enc_imgs_z[0], axis=0)
    Xs3 = np.concatenate(enc_imgs_z[1], axis=0)
    Xs4 = np.concatenate(enc_imgs_z[2], axis=0)

    Yb  = np.concatenate(enc_msks_z[0], axis=0)
    Ys3 = np.concatenate(enc_msks_z[1], axis=0)
    Ys4 = np.concatenate(enc_msks_z[2], axis=0)

    umn = build_unified_mapping_network()
    umn.fit(
        [Xb, Xs3, Xs4], [Yb, Ys3, Ys4],
        epochs=MAP_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)],
        verbose=1
    )
    umn.save_weights(os.path.join(save_root, "umn.weights.h5"))
    return umn


# =============================================================================
# Runner per dataset
# =============================================================================
def run_dataset(cfg: DatasetConfig):
    (img_train, msk_train), (img_test, msk_test), meta = load_dataset(cfg)

    dataset_tag = cfg.name.lower()
    res_root = os.path.join("./results/unified", dataset_tag)
    mdl_root = os.path.join("./saved_models/unified", dataset_tag)
    os.makedirs(res_root, exist_ok=True)
    os.makedirs(mdl_root, exist_ok=True)

    client_splits = partition_clients(img_train, msk_train, num_clients=NUM_CLIENTS, seed=cfg.seed)

    img_shape = img_train.shape[1:]
    msk_shape = msk_train.shape[1:]

    img_encTs, img_decs, img_Ts, msk_encTs, msk_decs, msk_Ts = train_full_clients(
        client_splits, dataset_tag, img_shape, msk_shape, mdl_root
    )

    img_prot_decs = [make_protected_decoder(img_decs[j], img_Ts[j], name=f"prot_img_dec_{j+1}") for j in range(NUM_CLIENTS)]
    msk_prot_decs = [make_protected_decoder(msk_decs[j], msk_Ts[j], name=f"prot_msk_dec_{j+1}") for j in range(NUM_CLIENTS)]

    enc_imgs_z, enc_msks_z = extract_original_z_for_umn(img_encTs, msk_encTs, client_splits, img_Ts, msk_Ts)
    umn = train_umn(enc_imgs_z, enc_msks_z, mdl_root)

    # Reconstruction (image AE) + Segmentation (via UMN + protected decoder)
    num_classes = int(meta["num_classes"])

    recon_rows = []
    seg_rows = []

    for i in range(NUM_CLIENTS):
        # recon per client AE
        Xhat = models.Model(img_encTs[i].inputs, img_prot_decs[i].outputs) if False else None  # unused
        Xrec = None

        # easiest: rebuild AE output via saved AE? We didn't keep img_ae models here.
        # Instead evaluate reconstruction by composing encoder_T + inv + decoder.
        # We'll just do: decode(protected latents) == AE's recon, but we need bT/s3T/s4T from encoder_T:
        bT, s3T, s4T = img_encTs[i].predict(img_test, batch_size=BATCH_SIZE, verbose=0)
        # server NOT needed; for recon, client protected decoder expects protected latents and does inv inside.
        Xrec = img_prot_decs[i].predict([bT, s3T, s4T], batch_size=BATCH_SIZE, verbose=0)

        recon_rows.append({
            "client": i+1,
            "mse": recon_mse(img_test, Xrec),
            "psnr": recon_psnr(img_test, Xrec),
            "ssim": recon_ssim(img_test, Xrec),
        })

        # segmentation per client decoder (same UMN, but each client has its own transforms/decoder)
        Yhat = segment_with_new_concept(
            umn=umn,
            img_encT=img_encTs[i],
            img_T=img_Ts[i],
            msk_T_tgt=msk_Ts[i],
            msk_prot_dec_tgt=msk_prot_decs[i],
            X=img_test
        )
        seg_rows.append({
            "client": i+1,
            "dice": seg_dice(msk_test, Yhat, num_classes),
            "iou": seg_iou(msk_test, Yhat, num_classes),
            "sensitivity": seg_sensitivity(msk_test, Yhat, num_classes),
            "specificity": seg_specificity(msk_test, Yhat, num_classes),
        })

    # Save outputs
    recon_csv = os.path.join(res_root, "reconstruction_metrics.csv")
    seg_csv   = os.path.join(res_root, "segmentation_metrics.csv")

    with open(recon_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["client","mse","psnr","ssim"])
        writer.writeheader()
        writer.writerows(recon_rows)

    with open(seg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["client","dice","iou","sensitivity","specificity"])
        writer.writeheader()
        writer.writerows(seg_rows)

    summary = {"dataset_meta": meta, "dataset_config": cfg.__dict__, "reconstruction": recon_rows, "segmentation": seg_rows}
    with open(os.path.join(res_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{dataset_tag}] DONE.")
    print(f"  Results: {res_root}")
    print(f"  Models : {mdl_root}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    DATASET_RUN_LIST = [
        # PSFH: TRAIN and TEST are DIFFERENT folders (FIXED)
        DatasetConfig(
            name="psfh",
            train_root="./datasets/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression",
            test_root="./datasets/testdataset",
            target_size=(256, 256),
            num_classes=3,
            image_channels=3,
            split_strategy="predefined_separate_folders",
            seed=SEED
        ),

        # FUMPE
        DatasetConfig(
            name="fumpe",
            root="./datasets/FUMPE_NEW",
            target_size=(256, 256),
            num_classes=2,
            image_channels=1,
            split_strategy="patient",
            test_ratio=0.1,
            seed=SEED
        ),

        # MRI
        DatasetConfig(
            name="mri",
            root="./datasets/MRI_Segmentation",
            target_size=(256, 256),
            num_classes=2,
            image_channels=1,
            split_strategy="predefined",
            seed=SEED
        ),

        # NERVE
        DatasetConfig(
            name="nerve",
            root="./datasets/ultrasound-nerve-segmentation/train",
            target_size=(256, 256),
            num_classes=2,
            image_channels=1,
            split_strategy="random",
            test_ratio=0.1,
            seed=SEED
        ),
    ]

    for cfg in DATASET_RUN_LIST:
        run_dataset(cfg)
