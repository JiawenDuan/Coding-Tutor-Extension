# Coding-Tutor-Extension

这份说明只覆盖两件事：怎么提交 OPD 训练，怎么提交评测（dialogue → code generation → 跑测试）。

所有命令都在集群上、项目根目录 `/data/wangjian/Coding-Tutor-Extension` 下执行。

## 1. 提交 OPD 训练

脚本：`opd/opd_training.sh`。teacher 是冻结的 SFT-200 adapter 加上 privileged 信息，student 是 backbone 加一个新的 LoRA。

一共四个模型，各提交一次：

```bash
# Qwen3-8B
OPD_MODE=task  DATA_PATH=output/opd/Qwen3-8B/train.jsonl sbatch opd/opd_training.sh
OPD_MODE=level DATA_PATH=output/opd/Qwen3-8B/train.jsonl sbatch opd/opd_training.sh

# Llama-3.1-8B-Instruct
OPD_MODE=task \
BACKBONE_PATH=/data_old/models/Llama-3.1-8B-Instruct \
TEACHER_ADAPTER_PATH=/data/wangjian/Coding-Tutor-Extension-old/checkpoints/sft-Llama-3.1-8B-Instruct/checkpoint-200 \
DATA_PATH=output/opd/Llama-3.1-8B-Instruct/train.jsonl \
sbatch opd/opd_training.sh

OPD_MODE=level \
BACKBONE_PATH=/data_old/models/Llama-3.1-8B-Instruct \
TEACHER_ADAPTER_PATH=/data/wangjian/Coding-Tutor-Extension-old/checkpoints/sft-Llama-3.1-8B-Instruct/checkpoint-200 \
DATA_PATH=output/opd/Llama-3.1-8B-Instruct/train.jsonl \
sbatch opd/opd_training.sh
```

### 变量

| 变量 | 必填 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `OPD_MODE` | 是 | 无 | `task`（teacher 看到任务相关的 privileged 信息）或 `level`（teacher 看到学生水平信息） |
| `DATA_PATH` | 是 | 无 | 训练数据，`output/opd/<backbone>/train.jsonl` |
| `BACKBONE_PATH` | 否 | `/data/models/Qwen3-8B` | student 的 base 模型 |
| `TEACHER_ADAPTER_PATH` | 否 | `/data/wangjian/Coding-Tutor-Extension-old/checkpoints/sft-Qwen3-8B/checkpoint-200` | teacher 的 SFT adapter，必须和 backbone 配套 |
| `OUTPUT_DIR` | 否 | `checkpoints/OPD/<backbone 目录名>-opd-<OPD_MODE>` | 模型保存位置 |

`sbatch opd/opd_training.sh` 后面写的参数会原样传给 `opd/train_opd.py`。先用少量数据试跑时，记得换一个输出目录：

```bash
OPD_MODE=task DATA_PATH=output/opd/Qwen3-8B/train.jsonl OUTPUT_DIR=checkpoints/OPD/debug \
sbatch opd/opd_training.sh --max_train_samples 16
```

### 产出

```text
checkpoints/OPD/
  Qwen3-8B-opd-task/
  Qwen3-8B-opd-level/
  Llama-3.1-8B-Instruct-opd-task/
  Llama-3.1-8B-Instruct-opd-level/
```

每个目录的根下是训练结束时的最终 adapter，另外每 100 步存一个 `checkpoint-N/`。

### 注意

- 必须在项目根目录下提交，脚本里用的是相对路径。
- 每个作业申请 2 张卡：student 在第一张，teacher 在第二张。
- 脚本没有指定日志位置，Slurm 日志是提交目录下的 `slurm-<作业号>.out`。
- 同名目录已存在时重新提交，不会接着训练，而是从头再训一遍并写进同一个目录。
- `output/` 不进 git，所以 `output/opd/` 要单独拷到集群上。重新生成用 `python opd/prepare_dialogues.py`，它读的是 `output/dialogue/vanilla/first_iter/` 下的 SFT-200 对话；如果那个目录已经改名成 `vanilla0`，要先改脚本里的 `FIRST_ITER`。
- 训练数据只包含 80 个任务，另外 20 个任务在 `test.jsonl` 里，没有参与训练。

## 2. 提交评测

脚本：`scripts/run/submit_opd_eval.sh`。每个运行提交两个 Slurm 作业：

1. `dlg-<运行名>`：起 tutor 和 student 的 vLLM，生成对话（2 张卡）。
2. `post-<运行名>`：student 根据对话写代码，然后跑测试（1 张卡）。对话作业成功后才开始。

评测跑全部 100 个任务、三个学生水平（`low_level`、`med_level`、`high_level`）。

### 测一个 checkpoint

```bash
bash scripts/run/submit_opd_eval.sh \
  --adapter checkpoints/OPD/Qwen3-8B-opd-task \
  --base /data_old/models/Qwen3-8B
```

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `--adapter` | 必填 | LoRA adapter 目录，里面要有 `adapter_config.json` |
| `--base` | 必填 | adapter 对应的 base 模型目录 |
| `--name` | adapter 目录名 | 运行名，也是结果目录名。adapter 是 `…/X/checkpoint-100` 时默认为 `X-checkpoint-100` |
| `--group` | `opd` | 结果放在哪个组下 |
| `--prompt` | `base` | tutor 用的 prompt：`base`、`task_teacher`、`level_teacher` 等 |
| `--backend` | 从 base 目录名判断 | `qwen` 或 `llama` |

### 跑预先定义好的实验

```bash
bash scripts/run/submit_opd_eval.sh backbone teacher   # 不依赖训练，随时可以跑
bash scripts/run/submit_opd_eval.sh opd                # 四个 OPD 模型训练完之后
bash scripts/run/submit_opd_eval.sh Qwen3-8B           # 只跑某一个运行
```

参数是组名或运行名，可以混着写。两个 backbone（Qwen3-8B、Llama-3.1-8B-Instruct）各有下面 7 个运行，共 14 个：

| 组 | 运行名 | tutor 模型 | tutor prompt |
| --- | --- | --- | --- |
| `backbone` | `<backbone>` | 原始模型 | `base` |
| `backbone` | `<backbone>-task-prompt` | 原始模型 | `task_teacher` |
| `backbone` | `<backbone>-level-prompt` | 原始模型 | `level_teacher` |
| `teacher` | `<backbone>-sft-200-task-prompt` | SFT-200 | `task_teacher` |
| `teacher` | `<backbone>-sft-200-level-prompt` | SFT-200 | `level_teacher` |
| `opd` | `<backbone>-opd-task` | OPD student（task） | `base` |
| `opd` | `<backbone>-opd-level` | OPD student（level） | `base` |

`opd` 组里 adapter 还不存在的运行会被跳过并提示。

### 先看会提交什么

```bash
DRY_RUN=1 bash scripts/run/submit_opd_eval.sh backbone teacher opd
DRY_RUN=1 bash scripts/run/submit_opd_eval.sh --adapter <adapter 目录> --base <base 模型目录>
```

只打印每个运行的设置，不提交作业。第一次用建议先这样看一眼，再只提交一个运行试跑。

### 结果在哪

```text
output/dialogue/vanilla_base/<组>/<运行名>/<level>/simulated_dialogs.jsonl
output/student_posttest/vanilla_base/<组>/<运行名>/<level>/
  completion_lm.jsonl        student 的原始输出
  completion.jsonl           提取出来的代码
  dependency_results.jsonl   Recall 结果
  test_results.jsonl         Pass 结果
output/student_posttest/vanilla_base/<组>/<运行名>/test_progress.log
```

目录不用提前建。Slurm 日志在 `/data/$USER/slurm-logs/`，文件名是 `dlg-<运行名>-<作业号>.out` 和 `post-<运行名>-<作业号>.out`；同一目录下的 `<作业号>-tutor.log` 和 `<作业号>-student.log` 是两个 vLLM 的日志。

### 注意

- **已有结果不会重跑。** 对话、代码生成、测试都是断点续跑的。同一个运行名下如果已经有结果，会直接沿用。重训后复测要换运行名（`--name`），或者先移走 `output/dialogue` 和 `output/student_posttest` 下的同名目录。脚本发现同名目录时会提示。
- **post 作业是排队跑的。** 跑测试会原地改写 EvoCodeBench 的源码目录，两个 post 作业同时跑会互相污染结果。脚本会把新的 post 作业排在你队列里已有的 post 作业之后。不要绕过脚本手动同时跑两个。
- **adapter 和 base 必须配套。** Qwen3-8B 和 Llama-3.1-8B 的 LoRA 形状相同，挂错了也能加载，只会得到一个错的模型。`--adapter` 模式下，adapter 记录的 base 和 `--base` 的目录名不一致时会拒绝提交；确认是同一个模型只是目录不同，加 `ALLOW_BASE_MISMATCH=1`。

### 路径设置

脚本里的默认路径可以用环境变量覆盖：

| 变量 | 默认值 |
| --- | --- |
| `QWEN_BASE` | `/data_old/models/Qwen3-8B` |
| `LLAMA_BASE` | `/data_old/models/Llama-3.1-8B-Instruct` |
| `QWEN_SFT` | `/data/wangjian/Coding-Tutor-Extension-old/checkpoints/sft-Qwen3-8B/checkpoint-200` |
| `LLAMA_SFT` | `/data/wangjian/Coding-Tutor-Extension-old/checkpoints/sft-Llama-3.1-8B-Instruct/checkpoint-200` |
| `OPD_DIR` | `checkpoints/OPD` |
| `STUDENT_MODEL_PATH` | `/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ` |

项目根目录 `/data/wangjian/Coding-Tutor-Extension` 写死在脚本开头的 `ROOT` 里，换位置要改这一行。

训练脚本的 Qwen 默认路径是 `/data/models/Qwen3-8B`，评测脚本是 `/data_old/models/Qwen3-8B`。提交前确认集群上实际用的是哪一个，另一个用环境变量改掉。
