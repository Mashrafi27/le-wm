#!/bin/bash
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --job-name=lewm_echo_100pct
#SBATCH -t 72:00:00
#SBATCH --output=/vast/users/mohammad.yaqub/project/LeWorldModel/logs/lewm_%j.out
#SBATCH --error=/vast/users/mohammad.yaqub/project/LeWorldModel/logs/lewm_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate echojepav2

PROJECT="/vast/users/mohammad.yaqub/project"
LEWM_DIR="$PROJECT/LeWorldModel"
ECHOJEPAV2_DIR="$PROJECT/EchoJEPAv2"

export MIOPEN_USER_DB_PATH="$HOME/.cache/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$HOME/.cache/miopen_cache"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
mkdir -p "$LEWM_DIR/logs"

# Force implicit GEMM — avoids miopenStatusUnknownError on Conv1d with MI300X
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
export PYTORCH_TUNABLEOP_ENABLED=0

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPUs:    $ROCR_VISIBLE_DEVICES"
echo "Started: $(date)"
echo ""

python "$LEWM_DIR/train_echo.py" \
    --shard-index   "$ECHOJEPAV2_DIR/evaluation/shard_index_amd.pkl" \
    --train-uuids   "$ECHOJEPAV2_DIR/training/train_dicoms.txt" \
    --holdout-uuids "$ECHOJEPAV2_DIR/training/holdout_dicoms.txt" \
    --output-dir    "$PROJECT/checkpoints/lewm/echo_vitT_100pct" \
    --devices       cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --epochs        20 \
    --batch-size    512 \
    --lr            4e-4 \
    --warmup-epochs 5 \
    --history-size  8 \
    --num-preds     1 \
    --img-size      224 \
    --num-workers   8 \
    --wandb-name    "lewm-echo-vitT-100pct-amd" \
    --wandb-entity  anaatef9-mbzuai

echo ""
echo "Finished: $(date)"
