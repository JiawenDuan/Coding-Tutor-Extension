#!/bin/bash
#SBATCH --job-name=tutor-opd
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
set -e
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jiawen
set -u

# Qwen3-8B experiment settings. Edit the defaults here or override them in the environment.
OPD_MODE="${OPD_MODE:-}"  # Required: task or level
BACKBONE_PATH="${BACKBONE_PATH:-/data/models/Qwen3-8B}"
TEACHER_ADAPTER_PATH="${TEACHER_ADAPTER_PATH:-checkpoints/sft-Qwen3-8B/checkpoint-200}"
DATA_PATH="${DATA_PATH:-}"  # Required: path to the OPD training dataset

case "$OPD_MODE" in
    task|level) ;;
    *) echo "Set OPD_MODE to task or level." >&2; exit 1 ;;
esac
[[ -f "$DATA_PATH" ]] || { echo "Training dataset not found: $DATA_PATH" >&2; exit 1; }
[[ -f "$TEACHER_ADAPTER_PATH/adapter_config.json" ]] || { echo "SFT adapter not found: $TEACHER_ADAPTER_PATH" >&2; exit 1; }
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/OPD/$(basename "$BACKBONE_PATH")-opd-$OPD_MODE}"

python3 opd/train_opd.py \
    --trainer sdft \
    --separate_teacher_gpu \
    --opd_mode "$OPD_MODE" \
    --model_name_or_path "$BACKBONE_PATH" \
    --teacher_model_name_or_path "$TEACHER_ADAPTER_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_prompt_length 4096 \
    "$@"
