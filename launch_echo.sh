#!/usr/bin/env bash
# Launch LeWorldModel training on iCardio echocardiograms.
# Run from the LeWorldModel directory inside the echojepav2 conda env:
#
#   conda activate echojepav2
#   cd /home/mashrafimonon/iCardio/LeWorldModel
#   bash launch_echo.sh [5pct|10pct|30pct] [device1] [device2]
#   e.g.: bash launch_echo.sh 5pct cuda:1 cuda:3

set -euo pipefail

FRACTION=${1:-5pct}
DEVICE1=${2:-cuda:1}
DEVICE2=${3:-}

REPO_DIR=/home/mashrafimonon/iCardio
LEWM_DIR=${REPO_DIR}/LeWorldModel

SHARD_INDEX=${REPO_DIR}/EchoJEPAv2/evaluation/shard_index.pkl
TRAIN_UUIDS=${REPO_DIR}/EchoJEPAv2/training/train_dicoms_${FRACTION}.txt
HOLDOUT=${REPO_DIR}/EchoJEPAv2/training/holdout_dicoms.txt
OUT_DIR=${REPO_DIR}/checkpoints/lewm/echo_vitT_${FRACTION}

if [ -n "${DEVICE2}" ]; then
    DEVICES="${DEVICE1} ${DEVICE2}"
    BATCH=128
    echo "=== LeWorldModel  fraction=${FRACTION}  devices=${DEVICES} (2-GPU, batch=128) ==="
else
    DEVICES="${DEVICE1}"
    BATCH=64
    echo "=== LeWorldModel  fraction=${FRACTION}  device=${DEVICE1} (1-GPU, batch=64) ==="
fi
echo "    uuids  : ${TRAIN_UUIDS}"
echo "    output : ${OUT_DIR}"

python "${LEWM_DIR}/train_echo.py" \
    --shard-index    "${SHARD_INDEX}" \
    --train-uuids    "${TRAIN_UUIDS}" \
    --holdout-uuids  "${HOLDOUT}" \
    --output-dir     "${OUT_DIR}" \
    --devices        ${DEVICES} \
    --epochs         20 \
    --batch-size     ${BATCH} \
    --lr             5e-5 \
    --warmup-epochs  5 \
    --history-size   8 \
    --num-preds      1 \
    --img-size       224 \
    --num-workers    6 \
    --wandb-name     "lewm-echo-vitT-${FRACTION}" \
    --wandb-entity   anaatef9-mbzuai
