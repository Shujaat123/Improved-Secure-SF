#!/usr/bin/env bash
# ============================================================
# FILE: run_psfh_stage1_arch_ablation_1client.sh
#
# PURPOSE:
#   Architectural ablation study (STAGE-1 only: AEs + UMN) for PSFH
#   using *ONE client* to isolate architecture effects.
#
# VARIANTS (matches your table idea):
#   1) Full model (PPCMI-SF):       skips=1, klt=1, ppm=1
#   2) No KLT:                      skips=1, klt=0, ppm=1
#   3) No Skips (plain AE):         skips=0, klt=1, ppm=1
#   4) Baseline (no KLT, no skips): skips=0, klt=0, ppm=0 (you can set ppm=1 if baseline still uses UMN-PPM)
#
# OUTPUT:
#   RESULTS_BASE/arch_ablation_1client/logs_YYYYmmdd_HHMMSS/
#     ├── run.log (master)
#     ├── full_ppcmi_sf/
#     ├── no_klt/
#     ├── no_skips_plain_ae/
#     └── baseline_no_klt_no_skips/
#
# USAGE:
#   chmod +x run_psfh_stage1_arch_ablation_1client.sh
#   ./run_psfh_stage1_arch_ablation_1client.sh
#
# OVERRIDES:
#   PYTHON=python DEVICE=cuda:0 AE_EPOCHS=50 UMN_EPOCHS=80 ./run_psfh_stage1_arch_ablation_1client.sh
# ============================================================

set -euo pipefail

# ----------------------------
# User-configurable paths
# ----------------------------
PYTHON="${PYTHON:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/validation_code/ppms_psfh/main_train_stage1.py}"

DATA_TRAIN="${DATA_TRAIN:-/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/datasets/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression}"  # <-- change to your PSFH root
RESULTS_BASE="${RESULTS_BASE:-/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/validation_code/ppms_psfh/runs/a_ablation}"

# ----------------------------
# Common hyperparams
# ----------------------------
DEVICE="${DEVICE:-cuda:1}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"

IMG_SIZE="${IMG_SIZE:-256}"
NO_RESIZE="${NO_RESIZE:-0}"          # 1 = pass --no-resize
NUM_CLASSES="${NUM_CLASSES:-3}"

NUM_CLIENTS=1                        # fixed per your request
VAL_FRAC="${VAL_FRAC:-0.2}"

AE_EPOCHS="${AE_EPOCHS:-100}"
AE_BATCH="${AE_BATCH:-16}"
AE_LR="${AE_LR:-1e-3}"
AE_PATIENCE="${AE_PATIENCE:-15}"

UMN_EPOCHS="${UMN_EPOCHS:-150}"
UMN_BATCH="${UMN_BATCH:-16}"
UMN_LR="${UMN_LR:-1e-3}"
UMN_PATIENCE="${UMN_PATIENCE:-20}"
UMN_VAL_FRAC="${UMN_VAL_FRAC:-0.2}"

SKIP_INVERSE_CHECK="${SKIP_INVERSE_CHECK:-0}"  # 1 = pass --skip-inverse-check

# ----------------------------
# Timestamped log folder
# ----------------------------
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${RESULTS_BASE}/arch_ablation_1client/logs_${TS}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/run.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "============================================================"
echo "PSFH STAGE-1 ARCH ABLATION (1 client)"
echo "START     : $(date)"
echo "LOG_DIR   : ${LOG_DIR}"
echo "DEVICE    : ${DEVICE}"
echo "DATA_TRAIN: ${DATA_TRAIN}"
echo "TRAIN_SCRIPT: ${TRAIN_SCRIPT}"
echo "============================================================"

# ----------------------------
# Helper to run one variant
# ----------------------------
run_variant () {
  local name="$1"
  local use_skips="$2"
  local use_klt="$3"
  local use_ppm="$4"

  local out_dir="${LOG_DIR}/${name}"
  mkdir -p "${out_dir}"

  echo "------------------------------------------------------------"
  echo "VARIANT: ${name}"
  echo "  use_skips=${use_skips}  use_klt=${use_klt}  use_ppm=${use_ppm}"
  echo "  results_dir=${out_dir}"
  echo "------------------------------------------------------------"

  # build command
  cmd=(
    "${PYTHON}" "${TRAIN_SCRIPT}"
    --data-train "${DATA_TRAIN}"
    --results-dir "${out_dir}"
    --log-dirname "logs"
    --seed "${SEED}"
    --device "${DEVICE}"
    --num-workers "${NUM_WORKERS}"
    --img-size "${IMG_SIZE}"
    --num-classes "${NUM_CLASSES}"
    --num-clients "${NUM_CLIENTS}"
    --val-frac "${VAL_FRAC}"
    --ae-epochs "${AE_EPOCHS}"
    --ae-batch "${AE_BATCH}"
    --ae-lr "${AE_LR}"
    --ae-patience "${AE_PATIENCE}"
    --use-skips "${use_skips}"
    --use-klt "${use_klt}"
    --use-ppm "${use_ppm}"
    --umn-epochs "${UMN_EPOCHS}"
    --umn-batch "${UMN_BATCH}"
    --umn-lr "${UMN_LR}"
    --umn-patience "${UMN_PATIENCE}"
    --umn-val-frac "${UMN_VAL_FRAC}"
  )

  if [ "${NO_RESIZE}" = "1" ]; then
    cmd+=(--no-resize)
  fi
  if [ "${SKIP_INVERSE_CHECK}" = "1" ]; then
    cmd+=(--skip-inverse-check)
  fi

  echo "CMD: ${cmd[*]}"
  "${cmd[@]}"
}

# ----------------------------
# Run the ablation variants
# ----------------------------

# 1) Full model (PPCMI-SF): skips=1, klt=1, ppm=1
run_variant "full_ppcmi_sf" 1 1 1

# 2) No KLT: skips=1, klt=0, ppm=1
run_variant "no_klt" 1 0 1

# 3) No Skips (plain AE): skips=0, klt=1, ppm=1
run_variant "no_skips_plain_ae" 0 1 1

# 4) Baseline (no KLT, no skips): skips=0, klt=0, ppm=0
run_variant "baseline_no_klt_no_skips" 0 0 1

echo "============================================================"
echo "✅ DONE: $(date)"
echo "ALL RESULTS IN: ${LOG_DIR}"
echo "MASTER LOG    : ${MASTER_LOG}"
echo "============================================================"


# Example usage
# chmod +x run_ablation.sh



# Run
# ./run_ablation.sh
