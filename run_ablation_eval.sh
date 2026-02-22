#!/usr/bin/env bash
# ============================================================
# FILE: run_ablation_eval.sh
#
# PURPOSE:
#   Automate Stage-2 evaluation for *PSFH only* (architectural ablation),
#   using YOUR PSFH evaluation.py (which takes NO --dataset and NO --surface-unit).
#
#   Each run writes EVERYTHING into one timestamped folder:
#
#     EVAL_BASE/logs_YYYYmmdd_HHMMSS/
#       ├── run.log
#       ├── full_ppcmi_sf/s1_t1/...
#       ├── no_klt/s1_t1/...
#       ├── no_skips_plain_ae/s1_t1/...
#       └── baseline_no_klt_no_skips/s1_t1/...
#
# EXPECTS (per variant ckpt-dir):
#   umn.pt
#   clientK_img_ae.pt
#   clientK_mask_ae.pt
#
# USAGE:
#   chmod +x run_psfh_federated_eval_stage2_arch_ablation.sh
#   ./run_psfh_federated_eval_stage2_arch_ablation.sh
#
# OPTIONAL OVERRIDES:
#   PYTHON=python DEVICE=cuda:1 AMP=1 BATCH=4 SAVE_SAMPLES=0 ./run_psfh_federated_eval_stage2_arch_ablation.sh
# ============================================================

set -euo pipefail

# ----------------------------
# Runtime
# ----------------------------
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
AMP="${AMP:-1}"
BATCH="${BATCH:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

SAVE_SAMPLES="${SAVE_SAMPLES:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"

SURFACE_METRICS="${SURFACE_METRICS:-1}"   # 1 compute, 0 skip

LAT_ITERS="${LAT_ITERS:-100}"
LAT_WARMUP="${LAT_WARMUP:-20}"

# ----------------------------
# Paths (EDIT THESE)
# ----------------------------
EVAL_SCRIPT="${EVAL_SCRIPT:-/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/validation_code/ppms_psfh/evaluation.py}"

# PSFH data root
PSFH_ROOT="${PSFH_ROOT:-/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/datasets/testdataset}"

# Where the stage-1 ckpt folders live for each variant
# expected structure:
#   CKPT_ROOT/logs_YYYYmmdd_HHMMSS/<variant_name>/{umn.pt,client*_img_ae.pt,client*_mask_ae.pt}
CKPT_ROOT="${CKPT_ROOT:-/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/validation_code/ppms_psfh/runs/a_ablation/arch_ablation_1client}"

# Where to save evaluation logs/results
EVAL_BASE="${EVAL_BASE:-/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/validation_code/ppms_psfh/eval_runs/a_ablation}"

# ----------------------------
# PSFH model/data settings (must match training)
# ----------------------------
IMG_SIZE="${IMG_SIZE:-256}"
IN_CH="${IN_CH:-3}"
N_CLASSES="${N_CLASSES:-3}"
RESIZE="${RESIZE:-1}"

# If you trained 1-client ablation, keep only 1:1
# If you later do multi-client, change to: PAIRS=("1:1" "1:2" "2:1" ...)
PAIRS=("1:1")

# ----------------------------
# Timestamped folder
# ----------------------------
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${EVAL_BASE}/logs_${TS}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/run.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "============================================================"
echo "PSFH STAGE-2 EVAL (ARCH ABLATION) START: $(date)"
echo "LOG_DIR      : ${LOG_DIR}"
echo "MASTER_LOG   : ${MASTER_LOG}"
echo "DEVICE       : ${DEVICE} AMP=${AMP} BATCH=${BATCH} WORKERS=${NUM_WORKERS}"
echo "SURFACE      : metrics=${SURFACE_METRICS}"
echo "SAVE_SAMPLES : ${SAVE_SAMPLES}  NUM_SAMPLES=${NUM_SAMPLES}"
echo "PSFH_ROOT    : ${PSFH_ROOT}"
echo "CKPT_ROOT    : ${CKPT_ROOT}"
echo "EVAL_SCRIPT  : ${EVAL_SCRIPT}"
echo "============================================================"

# ============================================================
# Helper
# ============================================================
run_variant () {
  local variant_name="$1"
  local ckpt_dir="$2"
  shift 2

  local extra_flags=("$@")

  # Basic checkpoint presence
  if [ ! -f "${ckpt_dir}/umn.pt" ]; then
    echo "⚠️  Missing umn.pt in ${ckpt_dir} → skipping ${variant_name}"
    return 0
  fi

  # We don't hardcode client1 only; we will validate per pair below.
  for pair in "${PAIRS[@]}"; do
    local src="${pair%%:*}"
    local tgt="${pair##*:}"

    local img_ckpt="${ckpt_dir}/client${src}_img_ae.pt"
    local msk_ckpt="${ckpt_dir}/client${tgt}_mask_ae.pt"

    if [ ! -f "${img_ckpt}" ]; then
      echo "⚠️  Missing ${img_ckpt} → skipping ${variant_name} src=${src} tgt=${tgt}"
      continue
    fi
    if [ ! -f "${msk_ckpt}" ]; then
      echo "⚠️  Missing ${msk_ckpt} → skipping ${variant_name} src=${src} tgt=${tgt}"
      continue
    fi

    local out_dir="${LOG_DIR}/${variant_name}/s${src}_t${tgt}"
    mkdir -p "${out_dir}"

    echo "------------------------------------------------------------"
    echo "VARIANT: ${variant_name}"
    echo "CKPT   : ${ckpt_dir}"
    echo "PAIR   : src=${src} tgt=${tgt}"
    echo "OUT    : ${out_dir}"
    echo "FLAGS  : ${extra_flags[*]:-(none)}"
    echo "------------------------------------------------------------"

    cmd=(
      "${PYTHON}" "${EVAL_SCRIPT}"
      --data-root "${PSFH_ROOT}"
      --ckpt-dir "${ckpt_dir}"
      --src-client "${src}"
      --tgt-client "${tgt}"
      --outdir "${out_dir}"
      --img-size "${IMG_SIZE}"
      --resize "${RESIZE}"
      --in-ch "${IN_CH}"
      --n-classes "${N_CLASSES}"
      --device "${DEVICE}"
      --batch-size "${BATCH}"
      --num-workers "${NUM_WORKERS}"
      --amp "${AMP}"
      --seed "${SEED}"
      --surface-metrics "${SURFACE_METRICS}"
      --save-samples "${SAVE_SAMPLES}"
      --num-samples "${NUM_SAMPLES}"
      --lat-iters "${LAT_ITERS}"
      --lat-warmup "${LAT_WARMUP}"
    )

    # append architecture flags (must match evaluation.py: --no-skips/--no-klt/--no-ppm)
    cmd+=("${extra_flags[@]}")

    echo "CMD: ${cmd[*]}"
    "${cmd[@]}"
  done
}

# ============================================================
# Find latest stage-1 logs folder (so you don't have to edit it)
# ============================================================
LATEST_LOGS_DIR="$(ls -dt "${CKPT_ROOT}"/logs_* 2>/dev/null | head -n 1 || true)"
if [ -z "${LATEST_LOGS_DIR}" ]; then
  echo "❌ Could not find any ${CKPT_ROOT}/logs_* folders. Check CKPT_ROOT."
  exit 1
fi

echo "Using latest CKPT logs folder: ${LATEST_LOGS_DIR}"

# Variant folders inside the latest logs dir
FULL_DIR="${LATEST_LOGS_DIR}/full_ppcmi_sf"
NO_KLT_DIR="${LATEST_LOGS_DIR}/no_klt"
NO_SKIPS_DIR="${LATEST_LOGS_DIR}/no_skips_plain_ae"
BASELINE_DIR="${LATEST_LOGS_DIR}/baseline_no_klt_no_skips"

# ============================================================
# Run ablation evals (PSFH only)
# ============================================================

# Full model (skips ON, klt ON, ppm ON)
run_variant "full_ppcmi_sf" "${FULL_DIR}"

# No KLT (skips ON, klt OFF)
run_variant "no_klt" "${NO_KLT_DIR}" --no-klt

# No Skips (skips OFF, klt ON)
run_variant "no_skips_plain_ae" "${NO_SKIPS_DIR}" --no-skips

# Baseline (skips OFF, klt OFF, ppm OFF)
run_variant "baseline_no_klt_no_skips" "${BASELINE_DIR}" --no-skips --no-klt 

echo "============================================================"
echo "✅ PSFH STAGE-2 EVALUATIONS COMPLETED"
echo "END: $(date)"
echo "RESULTS   : ${LOG_DIR}"
echo "MASTERLOG : ${MASTER_LOG}"
echo "============================================================"

# Run
# chmod +x run_ablation_eval.sh
# ./run_ablation_eval.sh

