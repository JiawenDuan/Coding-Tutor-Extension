#!/usr/bin/env bash
# 提交 OPD 验证实验的评测（dialogue -> code_gen -> coding_test），结果写到 vanilla_base 下：
#   output/dialogue/vanilla_base/<组>/<运行名>/<level>/
#   output/student_posttest/vanilla_base/<组>/<运行名>/<level>/
#
# 用法（在集群上执行）：
#   bash scripts/run/submit_opd_eval.sh backbone teacher   # 不依赖训练，现在就能跑
#   bash scripts/run/submit_opd_eval.sh opd                # OPD 训练完之后
#   bash scripts/run/submit_opd_eval.sh Qwen3-8B           # 只跑某一个运行名
#   DRY_RUN=1 bash scripts/run/submit_opd_eval.sh backbone teacher opd   # 只打印，不提交
#
# 参数是组名（backbone / teacher / opd）或下表里的运行名，可以混着写。
#
# 测任意一个 LoRA checkpoint（不在下表里的也行）：
#   bash scripts/run/submit_opd_eval.sh --adapter <adapter 目录> --base <base 模型目录>
# 可选：
#   --name <运行名>     默认取 adapter 目录名；目录名是 checkpoint-N 时取 "<上一级目录名>-checkpoint-N"
#   --group <组>        默认 opd
#   --prompt <setting>  默认 base；也可以是 task_teacher / level_teacher 等
#   --backend qwen|llama  默认从 base 模型目录名判断
#
# coding_test 会原地改写 EvoCodeBench 的 Source_Code，两个 post 作业同时跑会互相污染结果，
# 所以本脚本把所有 post 作业串成一条链，并且排在本用户已有的 post 作业之后。
# dialogue 作业之间不互相等待，每个运行用各自的端口。
set -euo pipefail

ROOT="/data/wangjian/Coding-Tutor-Extension"
SETTING="vanilla_base"

QWEN_BASE="${QWEN_BASE:-/data_old/models/Qwen3-8B}"
LLAMA_BASE="${LLAMA_BASE:-/data_old/models/Llama-3.1-8B-Instruct}"
QWEN_SFT="${QWEN_SFT:-/data/wangjian/Coding-Tutor-Extension-old/checkpoints/sft-Qwen3-8B/checkpoint-200}"
LLAMA_SFT="${LLAMA_SFT:-/data/wangjian/Coding-Tutor-Extension-old/checkpoints/sft-Llama-3.1-8B-Instruct/checkpoint-200}"
OPD_DIR="${OPD_DIR:-$ROOT/checkpoints/OPD}"
STUDENT_MODEL="${STUDENT_MODEL_PATH:-/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ}"
DRY_RUN="${DRY_RUN:-0}"

# 组 | 运行名 | backend | tutor prompt | LoRA adapter（空 = 原始模型）
RUNS="
backbone|Qwen3-8B|qwen|base|
backbone|Qwen3-8B-task-prompt|qwen|task_teacher|
backbone|Qwen3-8B-level-prompt|qwen|level_teacher|
backbone|Llama-3.1-8B-Instruct|llama|base|
backbone|Llama-3.1-8B-Instruct-task-prompt|llama|task_teacher|
backbone|Llama-3.1-8B-Instruct-level-prompt|llama|level_teacher|
teacher|Qwen3-8B-sft-200-task-prompt|qwen|task_teacher|$QWEN_SFT
teacher|Qwen3-8B-sft-200-level-prompt|qwen|level_teacher|$QWEN_SFT
teacher|Llama-3.1-8B-Instruct-sft-200-task-prompt|llama|task_teacher|$LLAMA_SFT
teacher|Llama-3.1-8B-Instruct-sft-200-level-prompt|llama|level_teacher|$LLAMA_SFT
opd|Qwen3-8B-opd-task|qwen|base|$OPD_DIR/Qwen3-8B-opd-task
opd|Qwen3-8B-opd-level|qwen|base|$OPD_DIR/Qwen3-8B-opd-level
opd|Llama-3.1-8B-Instruct-opd-task|llama|base|$OPD_DIR/Llama-3.1-8B-Instruct-opd-task
opd|Llama-3.1-8B-Instruct-opd-level|llama|base|$OPD_DIR/Llama-3.1-8B-Instruct-opd-level
"

die() { echo "错误: $*" >&2; exit 1; }

ADAPTER="" BASE="" NAME="" GROUP="opd" PROMPT="base" BACKEND=""
SELECTED=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --adapter|--base|--name|--group|--prompt|--backend)
      [ "$#" -ge 2 ] || die "$1 后面需要一个值"
      case "$1" in
        --adapter) ADAPTER="${2%/}" ;;
        --base) BASE="${2%/}" ;;
        --name) NAME="$2" ;;
        --group) GROUP="$2" ;;
        --prompt) PROMPT="$2" ;;
        --backend) BACKEND="$2" ;;
      esac
      shift 2 ;;
    --*) die "不认识的选项 $1" ;;
    *) SELECTED+=("$1"); shift ;;
  esac
done

# 表里的第一个运行用序号 1；--adapter 模式从 INDEX_START 之后取序号，避开表里已占的端口。
INDEX_START=0
if [ -n "$ADAPTER" ] || [ -n "$BASE" ]; then
  [ -n "$ADAPTER" ] && [ -n "$BASE" ] || die "--adapter 和 --base 要一起给"
  [ "${#SELECTED[@]}" -eq 0 ] || die "--adapter 模式下不能再写组名或运行名: ${SELECTED[*]}"

  base_name="$(basename "$BASE")"
  if [ -z "$BACKEND" ]; then
    case "$base_name" in
      *[Qq]wen*) BACKEND=qwen ;;
      *[Ll]lama*) BACKEND=llama ;;
      *) die "从 '$base_name' 判断不出 backend，请加 --backend qwen 或 --backend llama" ;;
    esac
  fi
  case "$BACKEND" in qwen|llama) ;; *) die "--backend 只能是 qwen 或 llama，当前: $BACKEND" ;; esac
  # 和 traver/run_base.py 的 --tutor_prompt_setting 可选值保持一致，拼错时在提交前就报错。
  case "$PROMPT" in
    base|student|privileged|teacher|opd_teacher|task_teacher|level_teacher) ;;
    *) die "--prompt 不是 run_base.py 支持的值: $PROMPT" ;;
  esac

  if [ -z "$NAME" ]; then
    NAME="$(basename "$ADAPTER")"
    case "$NAME" in checkpoint-*) NAME="$(basename "$(dirname "$ADAPTER")")-$NAME" ;; esac
  fi
  # 运行名和组名会直接用作目录名和 vLLM 里的模型名。
  for value in "$NAME" "$GROUP"; do
    case "$value" in ""|*[!A-Za-z0-9._-]*) die "运行名/组名只能包含字母、数字、点、下划线和连字符: '$value'" ;; esac
  done

  if [ "$DRY_RUN" != 1 ]; then
    [ -d "$BASE" ] || die "base 模型目录不存在: $BASE"
    [ -f "$ADAPTER/adapter_config.json" ] || die "找不到 $ADAPTER/adapter_config.json"
    # Qwen3-8B 和 Llama-3.1-8B 的 LoRA 形状相同，adapter 挂错 base 也能加载，只会悄悄得到错的模型。
    trained_on="$(sed -n 's/.*"base_model_name_or_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$ADAPTER/adapter_config.json" | head -n 1)"
    if [ -n "$trained_on" ] && [ "$(basename "$trained_on")" != "$base_name" ] && [ "${ALLOW_BASE_MISMATCH:-0}" != 1 ]; then
      die "adapter 是在 '$trained_on' 上训练的，和 --base '$BASE' 的目录名不一致。确认是同一个模型的话，加 ALLOW_BASE_MISMATCH=1 再提交"
    fi
  fi

  if [ "$BACKEND" = qwen ]; then QWEN_BASE="$BASE"; else LLAMA_BASE="$BASE"; fi
  RUNS="$GROUP|$NAME|$BACKEND|$PROMPT|$ADAPTER"
  SELECTED=("$GROUP")
  INDEX_START=$((14 + $(printf '%s' "$NAME" | cksum | cut -d' ' -f1) % 400))
fi

[ "${#SELECTED[@]}" -gt 0 ] || die "请给出要提交的组名（backbone / teacher / opd）或运行名，例如: bash $0 backbone teacher；或者用 --adapter <目录> --base <目录> 测单个 checkpoint"

# 每个参数都必须能匹配到至少一个运行，防止拼错后什么都没提交。
for arg in "${SELECTED[@]}"; do
  matched=0
  while IFS='|' read -r group name _rest; do
    [ -n "$group" ] || continue
    if [ "$arg" = "$group" ] || [ "$arg" = "$name" ]; then matched=1; fi
  done <<< "$RUNS"
  [ "$matched" = 1 ] || die "参数 '$arg' 既不是组名也不是运行名"
done

selected() {  # group name
  local arg
  for arg in "${SELECTED[@]}"; do
    if [ "$arg" = "$1" ] || [ "$arg" = "$2" ]; then return 0; fi
  done
  return 1
}

# 提交一个运行：打印 "<dialogue 作业号> <post 作业号>"。
# 在子 shell 里设置环境变量，避免上一个运行的设置漏到下一个。
submit_run() {  # group name backend prompt adapter index prev_post
  local group="$1" name="$2" backend="$3" prompt="$4" adapter="$5" index="$6" prev_post="$7"
  (
    export PROJECT_ROOT="$ROOT"
    export TUTOR_SETTING="$SETTING"
    export ITERATION="$group"
    export TUTOR_MODEL="$name"
    export TUTOR_PROMPT_SETTING="$prompt"
    export TUTOR_VLLM_BACKEND="$backend"
    export STUDENT_MODEL_PATH="$STUDENT_MODEL"
    # 每个运行固定用自己的一对端口，多个 dialogue 作业落在同一节点也不会冲突。
    export TUTOR_VLLM_PORT=$((8100 + 2 * index - 1))
    export STUDENT_VLLM_PORT=$((8100 + 2 * index))
    # 这些变量会让下游脚本改用远程服务或别的模型名，这里一律用本作业自己起的 vLLM。
    unset TUTOR_VLLM_MODEL_NAME TUTOR_VLLM_ENDPOINT TUTOR_VLLM_API_KEY TUTOR_VLLM_SCRIPT
    unset STUDENT_VLLM_ENDPOINT USE_REMOTE_VLLM

    if [ "$backend" = qwen ]; then
      export BASE_MODEL_PATH="$QWEN_BASE"
      export TUTOR_CHAT_TEMPLATE="$ROOT/scripts/run/qwen3_nonthinking.jinja"
    else
      export BASE_MODEL_PATH="$LLAMA_BASE"
      unset TUTOR_CHAT_TEMPLATE
    fi
    if [ -n "$adapter" ]; then
      export TUTOR_USE_BASE=0
      export LORA_ADAPTER="$adapter"
    else
      export TUTOR_USE_BASE=1
      unset LORA_ADAPTER
    fi

    # shellcheck source=scripts/run/configure_dialogue_gpus.sh
    source "$(dirname "${BASH_SOURCE[0]}")/configure_dialogue_gpus.sh" >/dev/null || exit 1

    local post_dep="afterok:JOB1"
    [ -z "$prev_post" ] || post_dep="$post_dep,afterany:$prev_post"

    if [ "$DRY_RUN" = 1 ]; then
      {
        echo "  TUTOR_SETTING=$TUTOR_SETTING ITERATION=$ITERATION TUTOR_MODEL=$TUTOR_MODEL"
        echo "  TUTOR_PROMPT_SETTING=$TUTOR_PROMPT_SETTING TUTOR_VLLM_BACKEND=$TUTOR_VLLM_BACKEND TUTOR_USE_BASE=$TUTOR_USE_BASE"
        echo "  BASE_MODEL_PATH=$BASE_MODEL_PATH LORA_ADAPTER=${LORA_ADAPTER:-<无>}"
        echo "  TUTOR_VLLM_PORT=$TUTOR_VLLM_PORT STUDENT_VLLM_PORT=$STUDENT_VLLM_PORT gres=$DIALOGUE_SBATCH_GRES"
        echo "  post 依赖: ${post_dep/JOB1/<本运行的 dialogue 作业>}"
      } >&2
      echo "dry-dlg-$index dry-post-$index"
      exit 0
    fi

    # 命令替换里 set -e 不生效，所以每一步都显式检查。
    local job1 job2
    job1=$(sbatch --parsable --job-name="dlg-$name" --gres="$DIALOGUE_SBATCH_GRES" --export=ALL \
      --gpu-bind="$SLURM_GPU_BIND" "$ROOT/scripts/run/sbatch_job_dialogue.sh" </dev/null) || exit 1
    [ -n "$job1" ] || exit 1
    # dialogue 失败时取消对应的 post，而不是让它永远挂着挡住后面的 post。
    job2=$(sbatch --parsable --job-name="post-$name" --export=ALL \
      --dependency="${post_dep/JOB1/$job1}" --kill-on-invalid-dep=yes \
      "$ROOT/scripts/run/sbatch_job_post.sh" </dev/null) || job2=""
    if [ -z "$job2" ]; then
      echo "post 作业提交失败；dialogue 作业 $job1 已经提交，需要手动处理" >&2
      exit 1
    fi
    echo "$job1 $job2"
  )
}

# 新的 post 链排在本用户已经在队列里的 post 作业后面（也可以用 EXTRA_DEP_JOB 手动指定）。
prev_post="${EXTRA_DEP_JOB:-}"
if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "/data/${USER}/slurm-logs"
  queued=$(squeue -h -u "$USER" -o '%i %j' 2>/dev/null | awk '$2 ~ /^(test-post|post-)/ {print $1}' | paste -sd: -) || queued=""
  if [ -n "$queued" ]; then
    prev_post="${prev_post:+$prev_post:}$queued"
    echo "已有 post 作业在队列里，新的 post 会排在它们之后: $queued"
  fi
fi

index=$INDEX_START
submitted=0
while IFS='|' read -r group name backend prompt adapter; do
  [ -n "$group" ] || continue
  index=$((index + 1))
  selected "$group" "$name" || continue

  if [ -n "$adapter" ] && [ "$DRY_RUN" != 1 ] && [ ! -f "$adapter/adapter_config.json" ]; then
    echo "跳过 $group/$name: 找不到 adapter $adapter/adapter_config.json" >&2
    continue
  fi

  echo "== $group/$name =="
  # 下游脚本是断点续跑的：已有的对话和测试结果不会重新生成。
  existing="${OUTPUT_DIALOGUE_BASE:-$ROOT/output/dialogue}/$SETTING/$group/$name"
  if [ -d "$existing" ]; then
    echo "  注意: $existing 已存在，里面已有的结果会被沿用而不是重跑。换了 checkpoint 的话请先改运行名或移走旧目录。"
  fi
  jobs=$(submit_run "$group" "$name" "$backend" "$prompt" "$adapter" "$index" "$prev_post") \
    || die "$group/$name 提交失败，后面的运行没有提交"
  job1="${jobs%% *}"
  job2="${jobs##* }"
  echo "  dialogue 作业: $job1   post 作业: $job2"
  echo "  结果: output/dialogue/$SETTING/$group/$name/  和  output/student_posttest/$SETTING/$group/$name/"
  prev_post="$job2"
  submitted=$((submitted + 1))
done <<< "$RUNS"

if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN：以上 $submitted 个运行只是打印，没有提交。"
else
  echo "共提交 $submitted 个运行。"
fi
