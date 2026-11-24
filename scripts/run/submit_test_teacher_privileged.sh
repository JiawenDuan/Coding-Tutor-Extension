#!/usr/bin/env bash
# 评测 base teacher 作为 tutor，使用 OPD 训练时的 privileged prompt 结构。
#   prompt = student_instruction (无 Reference Knowledge) + privileged_context (Profile + Prior + Reference Knowledge)
#   这与 SDFT teacher_prompt_template "{prompt}\n\n{privileged_context}" 完全一致。
#
# 输出目录：
#   dialogue/.../vanilla/backbone/Qwen3-8B-opd-teacher/
#   student_posttest/.../vanilla/backbone/Qwen3-8B-opd-teacher/
set -euo pipefail

ROOT="/data/wangjian/Coding-Tutor-Extension"
export PROJECT_ROOT="$ROOT"
export OUTPUT_DIALOGUE_BASE="$ROOT/validate_reward/output/dialogue"
export OUTPUT_STUDENT_POSTTEST_BASE="$ROOT/validate_reward/output/student_posttest"
export ITERATION="${ITERATION:-backbone}"
export RUN_DIALOGUE_JOB=1

export TUTOR_USE_BASE=1
export TUTOR_SETTING=vanilla
export TUTOR_PROMPT_SETTING=opd_teacher
export TUTOR_VLLM_PORT="${TUTOR_VLLM_PORT:-8001}"
export STUDENT_VLLM_PORT="${STUDENT_VLLM_PORT:-8002}"
export STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ}"
export TUTOR_GPU_MEMORY_UTILIZATION="${TUTOR_GPU_MEMORY_UTILIZATION:-0.55}"
export STUDENT_GPU_MEMORY_UTILIZATION="${STUDENT_GPU_MEMORY_UTILIZATION:-0.75}"

unset EXTRA_DEP_JOB

echo "========== OPD Teacher eval: Qwen3-8B (base, no LoRA, with privileged context) =========="
export TUTOR_VLLM_BACKEND=qwen
export TUTOR_MODEL="Qwen3-8B-opd-teacher"
export TUTOR_VLLM_MODEL_NAME="Qwen3-8B"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data_old/models/Qwen3-8B}"
export TUTOR_CHAT_TEMPLATE="$ROOT/scripts/run/qwen3_nonthinking.jinja"
unset LORA_ADAPTER

bash "$ROOT/scripts/run/submit_test.sh"
