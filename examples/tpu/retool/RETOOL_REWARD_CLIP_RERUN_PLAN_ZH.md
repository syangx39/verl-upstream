# TPU v7x ReTool 对齐 Askhat 运行与重跑方案（TPU v7x Only）

状态：**待 review（rev 4）**（2026-11-04）。本文为 TPU v7x 专项对齐方案，尚未提交新的集群运行。

- **rev 4**：改为 **TPU v7x Only**（GPU 重跑待 TPU v7x 对齐完成后再按需启动）；补入对 Askhat WandB（`google-trellis/verl-torchtpu-grpo-test`）全部 12 个 Qwen3-30B-A3B ReTool 运行的完整审计（原始汇总见 [`results/askhat-wandb-audit/askhat_runs_summary.json`](results/askhat-wandb-audit/askhat_runs_summary.json)）；明确 Askhat 两代配置（Regime 1：PR #108 v6e 8K；Regime 2：v5p / v7x 16K + thinking + shaped scorer）的真实差异；加入 TPU v7x 接入 WandB 的安全配置方案。
- **rev 3 / rev 2**：补齐固定样例 shaping 校验、双口径（默认 vs strict）评估、strict scorer 100 字符限制说明与统计区间边界。

相关证据：
- 我们的 TPU v7x 基线 `p3-retool-rl-11` 结果与 scorer 审计：[`REPRO_RESULTS_TPU7X.md`](REPRO_RESULTS_TPU7X.md)、[`results/p3-retool-rl-11/`](results/p3-retool-rl-11/)
- Askhat WandB 运行审计数据：[`results/askhat-wandb-audit/askhat_runs_summary.json`](results/askhat-wandb-audit/askhat_runs_summary.json)

---

## 0. 目标与边界

1. **只聚焦 TPU v7x（`2 × 2x2x4 = 32 chips`）**：本阶段不重跑 GPU，集中定位并消除我们 TPU v7x 运行（`p3-retool-rl-11`）与 Askhat 运行之间的训练爬升和 AIME 评估差距。
2. **以 Askhat WandB 实测证据为准**：通过直接核对 Askhat 在 `google-trellis/verl-torchtpu-grpo-test` 中的 `config.yaml`、逐步训练指标、`val-core` / `val-aux` 和 `val/generations` 抽样表，区分哪些参数在 Askhat 运行中真正生效、哪些没有生效。
3. **接入 WandB 实时对齐**：后续所有 TPU v7x 运行同步上报到 `google-trellis/verl-torchtpu-grpo-test`（`LOGGER_LIST="['console','tensorboard','wandb']"`），在同一面板下与 Askhat 的曲线逐项对比。

---

## 1. Askhat WandB 完整审计：两代配置与三大核心发现

我们对 `google-trellis/verl-torchtpu-grpo-test` 中 Askhat 的全部 Qwen3-30B-A3B 运行进行了完整拉取与核对（见 [`askhat_runs_summary.json`](results/askhat-wandb-audit/askhat_runs_summary.json)）。审计表明，Askhat 的实验分为**两代截然不同的配置（Regime 1 vs Regime 2）**。

### 1.1 跨运行完整对照表

| 运行 ID（实验名） | 硬件 | 配置代际 | 实际生效 Scorer | `enable_thinking` | 长度 / 轮数 / `n` / `lr` | 有效 `max_norm` | 训练 Reward（Steps 1–5 → 21–25） | Step 0 AIME 2024 / 2025 | Step 25 AIME 2024 / 2025 | 后续步数 AIME 2024 / 2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| **`dj6g4b0i`**（`...-p5b-400steps`，14 步） | v6e-32 × 2 | **Regime 1**（PR #108） | 默认 `math_dapo`（无 shaping） | `False` | 8K / 8 轮 / `n=8` / `1.4e-6` | `1.0` | `-0.305 → -0.233`（Step 14） | `23.00%` / `15.89%`（`@30`） | — | — |
| **`lb5cdzuq`**（`...-p5b-400steps-v2`，26 步） | v6e-32 × 2 | **Regime 1**（PR #108） | 默认 `math_dapo`（无 shaping） | `False` | 8K / 8 轮 / `n=8` / `1.4e-6` | `1.0` | **`-0.273 → -0.157`** | **`25.44%` / `14.44%`**（`@30`） | **`30.00%` / `18.33%`**（`@30`） | — |
| **`1qhymrv4`** → **`39othwaq`**（续跑至 ~340 步） | v6e-32 × 2 | **Regime 1**（PR #108） | 默认 `math_dapo`（无 shaping） | `False` | 8K / 8 轮 / `n=8` / `1.4e-6` | `1.0` | `-0.145 → -0.049`（首段） | —（续跑） | `30.89%` / `23.67%`（`1qh` `@30`） | **200–340 步**：`50.78%–51.89%` / `37.67%–39.11%`（`@30`） |
| **我们的 `p3-retool-rl-11`**（25 步） | **v7x-16 × 2** | **Regime 1**（PR #108） | 默认 `math_dapo`（无 shaping） | `False` | 8K / 8 轮 / `n=8` / `1.4e-6` | `1.0` | **`-0.270 → -0.247`** | **`25.56%` / `17.33%`**（`@30`） | **`26.78%` / `17.78%`**（`@30`） | — |
| **`yjowd03u`**（`retool-dapo-...-v5p`，仅 Step 0） | v5p | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / 16 轮 / `n=16` / `2e-6` | `1.0` | — | **`48.75%` / `30.83%`**（`@8`） | — | — |
| **`fcze3xlx`**（`...-v5p-cp2`，23 步） | v5p-32 × 2 | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / 16 轮 / `n=16` / `2e-6` | `1.0` | **`-0.364 → +0.456`**（Step 23 `+0.656`） | —（`val_before_train=False`） | —（停于 Step 23） | — |
| **`ci0k54pl`**（`...-v5p-cp2`，150 步） | v5p-32 × 2 | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / 16 轮 / `n=16` / `2e-6` | `1.0` | **`-0.348 → +0.340`** | —（同 `yjowd03u` 起点） | **`58.33%` / `38.75%`**（`@8`） | **Step 50**：`63.33%` / `53.75%`；**Step 150**：`70.83%` / `50.83%`（`@8`） |
| **`hsodvfgr`**（`...-v5p-cp1-opt`，150 步） | v5p-32 × 2 | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / 16 轮 / `n=16` / `2e-6` | `1.0` | **`-0.344 → +0.367`** | — | `52.08%` / `39.17%`（`@8`） | **Step 125–150**：`66.67%–67.92%` / `55.00%`（`@8`） |
| **`yrcsp4z6`**（`...-v5p-cp1-pr115`，108+ 步） | v5p-32 × 2 | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / 16 轮 / `n=16` / `2e-6` | `1.0` | **`-0.305 → +0.483`** | — | **`60.00%` / `46.67%`**（`@8`） | **Step 75–100**：`68.33%–70.00%` / `51.67%–55.00%`（`@8`） |
| **`unch8eca`**（`...-think16k-tpu7x-opt1`，5 步） | **v7x-8 × 2** | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / 8 轮 / `n=16` / `2e-6` | `1.0` | **Step 1 `-0.339` → Step 5 `+0.219`** | — | —（第 5 步后崩溃） | — |

---

### 1.2 发现一：PR #108 的 TPU v6e 8K 系列（`lb5cdzuq` 等）同样回退到了默认 `math_dapo` scorer，且 `lb5cdzuq` 自身 Step 0 为 `25.44% / 14.44%`

1. **PR #108 的 v6e 运行并没有启用 ReTool shaped scorer**：
   - **配置证据**：在 `ozkz5d07`、`dj6g4b0i`、`lb5cdzuq`、`1qhymrv4`、`39othwaq` 的 WandB `config.yaml` 中，`trainer.use_v1: true`，顶层 `custom_reward_function.path: /data/verl/recipe/retool/retool.py`，而 V1 reward loader 实际读取的 **`reward.custom_reward_function.path: null`**——与我们的 `p3-retool-rl-11` 完全一致。
   - **数值恒等式证据**：在这 5 个运行的每一个验证步上，
     $$\text{val-aux/aime\_2024/score/mean@30} - \bigl(2 \times \text{val-core/aime\_2024/acc/mean@30} - 1\bigr) \equiv 0.000000$$
     例如 `lb5cdzuq` Step 0：`score = -0.491111 = 2 × 0.254444 - 1`；Step 25：`score = -0.400000 = 2 × 0.300000 - 1`。
   - **样本表证据**：`lb5cdzuq` 的 `val/generations` 表（每步 20 条）中，所有样本 `score` 严格为 `-1.0` 或 `+1.0`，没有任何 `-0.9 / -0.8 / -0.7 / -0.6` 的 tool-use shaping 分数。
2. **PR #108 中 `22.44% → 30.00%` / `14.17% → 18.33%` 的真实来源**：
   - `30.00% / 18.33%` 来自 `lb5cdzuq` 的 Step 25（`val-core/aime_2024/acc/mean@30 = 0.3000`、`aime_2025 = 0.1833`）。
   - 但 **`lb5cdzuq` 自身的 Step 0（`val_before_train=True`）是 `25.44% / 14.44%`**（与我们 `p3-retool-rl-11` 的 Step 0 `25.56% / 17.33%` 几乎一致），单次 run 内部的 Step 0 → Step 25 变化为 **`+4.56 pp`（`25.44% → 30.00%`，其中 `maj@30` 仅从 `46.10%` 变为 `46.58%`，`+0.48 pp`）** 和 **`+3.89 pp`（`14.44% → 18.33%`）**。
   - PR #108 报告将更早一次独立评估的 Step 0（`22.44% / 14.17%`）与 `lb5cdzuq` 的 Step 25 拼在了一起。实际上，同一个 v6e SFT checkpoint（`bjb9xppo`）在 Askhat 的 4 次独立 Step-0 `n=30` 评估中，AIME 2024 在 **`22.44%–25.67%`** 之间波动（`dj6g4b0i` `23.00%`、`lb5cdzuq` `25.44%`、`lx3stkgp` `25.67%`、`ozkz5d07` `25.67%`）。
3. **Regime 1（8K 无 thinking）下 25 步本来就处于慢爬升期**：
   - `lb5cdzuq` 前 25 步训练 reward 从 `-0.2727`（Steps 1–5）缓步升到 `-0.1570`（Steps 21–25），平均回复长度 `894` tokens、平均轮数 `6.01`；我们的 `p3-retool-rl-11` 从 `-0.2695` 到 `-0.2469`，平均回复长度 `758 → 822` tokens、平均轮数 `5.86 → 6.08`。
   - 在这套 8K 配置下，Askhat 在 v6e 上接力训练了 **200–400 步**（`lb5cdzuq` → `1qhymrv4` → `hdc2m3lf` → `39othwaq`）才达到 AIME 2024 `50.78%–51.89%`、AIME 2025 `37.67%–39.11%`。

---

### 1.3 发现二：Askhat 的 TPU v5p 快速爬升系列（Regime 2）同时改动了 5 项配置，而有效 clipping 始终是 `1.0`

在 PR #108 之后（对应上游 PR #111 / #115），Askhat 在 TPU v5p 系列（`fcze3xlx`、`ci0k54pl`、`rq1p980n`、`hsodvfgr`、`yrcsp4z6`）中**同时调整了 5 个核心参数**：

1. **修复 V1 Reward 路径**：设置 `reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu`（顶层 `custom_reward_function.path=null`）。在 `ci0k54pl` / `yrcsp4z6` 中，$\text{score} - (2 \times \text{acc} - 1) = +0.036 \sim +0.150$，`val/generations` 中明确出现 `-0.9, -0.7, -0.6` 等多轮工具调用 shaping 分数。
2. **开启 Qwen3 Thinking 模式（影响最大）**：`+data.apply_chat_template_kwargs.enable_thinking=True`（PR #108 脚本为 `False`）。
   - 开启 `<think>` 后，模型在调用 Python 工具前后进行显式推理，平均回复长度从 `~800` tokens 增至 `~2,400` tokens，平均轮数从 `~6.0` 增至 `~8.4`。
   - **仅在 Step 0（未经任何 RL 训练）**，同一个 SFT 模型在开启 thinking + 16K 后（`yjowd03u`），AIME 2024 `mean@8` 直接达到 **`48.75%`**（`maj@8 = 58.12%`），AIME 2025 `mean@8` 达到 **`30.83%`**（`maj@8 = 39.96%`）——比关闭 thinking 时的 Step 0（`~25% / ~15%`）高出约 **+23 pp / +15 pp**！
3. **输出长度与轮数翻倍（8K / 8 轮 → 16K / 16 轮）**：`data.max_response_length=16384`（`max_model_len=18432`、`torchtitan.max_seq_len=18432`），`max_user_turns=16`、`max_assistant_turns=16`。这把回复截断率（`response_length/clip_ratio`）从我们 8K 运行的 **`8.8%–10.7%`** 降到了 **`0.1%`**（`ci0k54pl` 为 `0.00098`）。
4. **GRPO 每题采样数翻倍（`n=8 → n=16`）**：`actor_rollout_ref.rollout.n=16`，每步总轨迹数从 `512` 增至 `1,024`，减小了组内均值/标准差估计方差。
5. **Actor 学习率上调 +43%（`1.4e-6 → 2e-6`）**：`actor_rollout_ref.actor.optim.lr=2e-6`。

> [!IMPORTANT]
> **关于 Gradient Clipping（D7）的核实结论**：
> Askhat 在 v6e 配置了 `optim.clip_grad=0.088`，在 v5p / v7x 配置了 `optim.clip_grad=0.0625`。但无论在 `tpu-main`、上游 PR #108 还是 PR #115 中，`verl/workers/engine/torchtitan/transformer_impl.py` 均未将 `optim.clip_grad` 传入 `TrainingConfig(max_norm=...)`（Askhat 的 `ci0k54pl_output.log` 中同样打印了 `WARNING: TorchTitan grad clipping uses training.max_norm=1.0; optim.clip_grad=0.0625 is not passed to TorchTitan`）。此外，Askhat 所有运行中的 `actor/grad_norm` 均在 `0.05–0.22` 之间，从未触发 `1.0` 裁剪。
> 因此，**保持 TorchTitan 有效 `training.max_norm=1.0` 已经与 Askhat 的所有实际运行 100% 一致**。为了与 Askhat 对齐，**不需要、也不应当**把有效 clipping 改成 `0.088` 或 `0.0625`。

---

### 1.4 发现三：Askhat 已在 TPU v7x 上跑过 Regime 2（`unch8eca`），前 5 步爬升速度与 TPU v5p 完全一致

在 `google-trellis/verl-torchtpu-grpo-test` 中，Askhat 于 10 月 30 日在 **TPU v7x** 上启动过一次 Regime 2 运行 **`unch8eca`**（`qwen3-30b-a3b-retool-dapo-150step-think16k-tpu7x-opt1`，`2 × 8` trainer chips + `2 × 8` rollout chips，起点 `/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu-full`），启用了上述全部 5 项改动（仅 `max_turns=8`）。

对比 `unch8eca`（TPU v7x）与 `fcze3xlx` / `ci0k54pl` / `yrcsp4z6`（TPU v5p）以及 `p3-retool-rl-11`（TPU v7x，Regime 1）的前 5 步 `critic/rewards/mean`：

| 步数 | `p3-retool-rl-11`（v7x，Regime 1：8K 无 thinking） | `lb5cdzuq`（v6e，Regime 1：8K 无 thinking） | `fcze3xlx`（v5p，Regime 2） | `ci0k54pl`（v5p，Regime 2） | `yrcsp4z6`（v5p，Regime 2） | **`unch8eca`（TPU v7x，Regime 2）** |
|---|---|---|---|---|---|---|
| **Step 1** | `-0.2734` | `-0.3008` | `-0.5179` | `-0.4896` | `-0.4081` | **`-0.3391`** |
| **Step 2** | `-0.3242` | `-0.2930` | `-0.4158` | `-0.4139` | `-0.3292` | **`-0.2549`** |
| **Step 3** | `-0.1836` | `-0.2422` | `-0.3693` | `-0.4175` | `-0.3578` | **`-0.1377`** |
| **Step 4** | `-0.2930` | `-0.2227` | `-0.3547` | `-0.3407` | `-0.3910` | **`-0.0276`** |
| **Step 5** | `-0.2734` | `-0.3047` | `-0.1623` | `-0.0762` | `-0.0402`（Step 6 `+0.1195`） | **`+0.2191`** |
| **Steps 1–5 均值** | `-0.2695` | `-0.2727` | `-0.3640` | `-0.3476` | `-0.3053` | **`-0.1080`** |

- **为什么 `unch8eca` 在第 5 步后崩溃**：`unch8eca` 在 TPU v7x 上设置了 `release_buffers_after_sync: True`（且仅用了 2 台 host = 16 TensorCores 跑 rollout，`gpu_memory_utilization: 0.75`），触发了我们在 PR #10 中诊断并修复的 `tpu_sync_torch` `unbind_buffer` 运行时缺陷。我们的 `tpu-main` 已通过 `retool_tpu.py --check-raiden-unbind` 自动将该环境下的 `RELEASE_SYNC_BUFFERS` 设为 `False`，并在 `p3-retool-rl-11` 中稳定跑完 25 步同步。
- **核心结论**：**TPU v7x 硬件和训练栈本身完全能够快速收敛**；我们 `p3-retool-rl-11` 与 Askhat v5p 运行的差距，全部来自 **Regime 1（8K 无 thinking + 默认 scorer）与 Regime 2（16K + thinking + ReTool shaped scorer + `n=16` + `lr=2e-6`）的配方差异**。

---

## 2. TPU v7x 对齐实验矩阵（只跑 TPU v7x）

我们在 TPU v7x（`2 × 2x2x4 = 32 chips`）上设计三个配置层级，全部从同一个 SFT checkpoint（`p2-retool-sft-01/global_step_372/huggingface`）出发，使用同一个容器镜像（`verl-tpu@sha256:d92a2b3d...`）和 `tpu-main` 代码（`05e4398a`，训练与 rollout 代码与 `aa10dfed` 完全一致）：

| 实验 ID | 对齐目标 | Scorer（`reward.custom_reward_function.path`） | `enable_thinking` | `max_response_length` / `MAX_TURNS` | `N_RESP_PER_PROMPT` | `ACTOR_LR` | 有效 `max_norm` | 状态 / 用途 |
|---|---|---|---|---|---|---|---|---|
| **A**（`p3-retool-rl-11`） | Askhat Regime 1（PR #108 `lb5cdzuq`） | `None`（默认 `math_dapo`，`strict=False`，无 shaping） | `False` | `8192` / `8` | `8` | `1.4e-6` | `1.0` | **已完成**。作为 Regime 1 基线保留，不覆盖 |
| **B**（`p3b-retool-rl-*`，可选消融） | Regime 1 + 修复 Scorer | `pkg://examples.tpu.retool.retool_tpu`（`strict=True` + shaping） | `False` | `8192` / `8` | `8` | `1.4e-6` | `1.0` | **控制变量消融**：同代码、同镜像下仅替换 scorer，隔离检验 `retool_tpu` 在 8K 无 thinking 下的单独作用 |
| **C**（`p3c-retool-rl-*`，**核心推荐**） | **Askhat Regime 2（`unch8eca` / `ci0k54pl` / `yrcsp4z6`）** | `pkg://examples.tpu.retool.retool_tpu`（`strict=True` + shaping） | **`True`** | **`16384` / `16`** | **`16`** | **`2e-6`** | `1.0` | **完整对齐 Askhat 快速爬升配方**：验证 5 步内训练 reward 从 `~ -0.35` 升至 `> 0`，并在 25 步复现高水平 AIME 表现 |

> [!TIP]
> **执行路径建议（见第 7 节）**：
> 因为审计已证实 PR #108 的 `lb5cdzuq` 也是默认 scorer，而 Askhat 真正快速爬升的运行（`fcze3xlx` / `ci0k54pl` / `yrcsp4z6` / `unch8eca`）全部是 **配置 C（Regime 2）**，建议**优先直接跑 C（先跑 5 步 smoke 对齐 `unch8eca` 的 `-0.34 → +0.22` 曲线，通过后直跑 C 的 25 步）**；B 作为可选消融，仅在需要单独拆分“8K 无 thinking 下只修 scorer 的边际影响”时运行。

---

## 3. 配置 C（Regime 2）在 TPU v7x 上的技术预检与启动参数

### 3.1 为什么现有 `run_qwen3_30b_a3b_dapo_torchtitan.sh` 在 `2 × 2x2x4` 上天然支持 16K（配置 C）

1. **Trainer 序列长度与动态 batch 预算已按 18,432 tokens 编译**：
   - 在 [`run_qwen3_30b_a3b_dapo_torchtitan.sh` L145-L149](run_qwen3_30b_a3b_dapo_torchtitan.sh#L145-L149) 中，默认 `ACTOR_MAX_TOKEN_LEN_PER_GPU=4608`、`CONTEXT_PARALLEL_SIZE=4`。
   - `TorchTitanEngine.forward_backward_batch`（[`transformer_impl.py` L508-L511](../../verl/workers/engine/torchtitan/transformer_impl.py#L508-L511)）设置 `sp_size = tensor_parallel_size * context_parallel_size = 4`，因此 `prepare_micro_batches` 的单条序列上限为：
     $$\text{max\_token\_len} = 4608 \times 4 = 18432 = \text{max\_prompt\_length}(2048) + \text{max\_response\_length}(16384)$$
   - 同时 `torchtitan_max_seq_len = 4608 * 4 = 18432`。也就是说，`p3-retool-rl-11` 在 8K 运行时，TorchTitan 实际上已经按 `18,432` 的最大序列长度完成了 XLA/SplashAttention 编译与显存分配，切到 `MAX_RESPONSE_LENGTH=16384` **不会增加 Trainer 的单 micro-batch 编译序列上限**。
2. **Rollout 显存与 KV cache 容量（相比 `unch8eca` 翻倍）**：
   - Askhat 的 `unch8eca` 只用了 `1 × 2x2x2`（2 台 host = 16 TensorCores，`TP=16, EP=16`）跑 16K rollout，且开了 `release_buffers_after_sync=True`。
   - 我们使用 `1 × 2x2x4`（4 台 host = 32 TensorCores，`TP=32, EP=32`），每个 TensorCore（96 GB HBM）仅分担 `1/32` 的模型权重（约 `1.9 GB`），即使保持 `RELEASE_SYNC_BUFFERS=False`（规避 `unbind_buffer` bug）和 `ROLLOUT_GPU_MEMORY_UTILIZATION=0.62`（或上调至 `0.70`），32 张卡的 KV cache 容量也远超 `max_model_len=18432`。
3. **单步耗时与 `DRIVER_TIMEOUT_MIN` 预估**：
   - 配置 C 将每步轨迹数从 `64 × 8 = 512` 增至 `64 × 16 = 1,024`（2×），开启 thinking 后平均回复长度从 `~800` tokens 增至 `~2,400` tokens（~3×），因此每步训练+生成耗时预计从 `p3-retool-rl-11` 的 `~118s/step` 增至 `~300–420s/step`。
   - 25 步训练加 Step 0 / Step 25 两次评估（16K 下每次验证约 `25–35 min`）预计总耗时约 `3.0–3.5 小时`，因此 25 步正式运行时将 `DRIVER_TIMEOUT_MIN` 从默认 `180` 上调至 **`360`（6 小时）**。

### 3.2 启动参数对照（无需改镜像，直接通过环境变量 + Hydra Override 启动）

同时我们会在 [`run_qwen3_30b_a3b_dapo_torchtitan.sh`](run_qwen3_30b_a3b_dapo_torchtitan.sh) 中补齐 `REWARD_FN`（默认设置 `reward.custom_reward_function.path`）和 `ENABLE_THINKING` 环境变量，并在 `main_ppo.py` 中加入 V1 顶层 `custom_reward_function.path` 非空且 `reward.custom_reward_function.path` 为空时的显式报错防呆。即使在未改脚本的 `tpu-main` 上，配置 C 也只需传入以下参数：

```bash
# 配置 C（对齐 Askhat Regime 2：unch8eca / ci0k54pl / yrcsp4z6）
JOBENV_MAX_RESPONSE_LENGTH=16384 \
JOBENV_MAX_TURNS=16 \
JOBENV_N_RESP_PER_PROMPT=16 \
JOBENV_ACTOR_LR=2e-6 \
JOBENV_LOGGER_LIST="['console','tensorboard','wandb']" \
JOBENV_PROJECT_NAME="verl-torchtpu-grpo-test" \
bash examples/tpu/gke/run_ray_jobset_v7x_nap.sh \
  examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh \
  "reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu" \
  "reward.custom_reward_function.name=compute_score" \
  "+data.apply_chat_template_kwargs.enable_thinking=True"
```

（若运行配置 B，则保持 `8192 / 8 / 8 / 1.4e-6` 和 `enable_thinking=False`，仅传 `reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu`。）

---

## 4. Smoke 验收与 25 步评估口径

### 4.1 5 步 Smoke 门禁（配置 C 先跑 5 步对标 `unch8eca`）

在跑 25 步完整实验前，先跑一次 **5 步 Smoke**（`TOTAL_TRAINING_STEPS=5`、`TEST_FREQ=100`、`VAL_BEFORE_TRAIN=True`、`N_RESP_PER_PROMPT_VAL=2`）：

1. **固定样例 Shaping 检查（必须通过）**：
   用解析后的配置调用 [`get_custom_reward_fn`](../../verl/trainer/ppo/reward.py#L50) 并通过 `NaiveRewardManager`（带 `__num_turns__`）执行固定样例集，确认：
   - `\boxed{42}`（正确，`num_turns=2, 6`）→ `1.0`
   - `\boxed{41}`（错误，`num_turns=2, 3, 4, 6, 10`）→ `-1.0, -0.95, -0.9, -0.8, -0.6`
   - 仅有 `Answer: 42` 无 `\boxed{}`（`num_turns=2`）→ `-1.0`（`strict_box_verify=True` 判错，区别于默认 scorer 的 `+1.0`）
   - `\boxed{42}` 后缀超过 100 字符（`num_turns=2`）→ `-1.0`
2. **生效配置与同步检查（必须通过）**：
   - Resolved config 中 `reward.custom_reward_function.path == "pkg://examples.tpu.retool.retool_tpu"`，`data.apply_chat_template_kwargs.enable_thinking == True`。
   - TorchTitan 有效 clipping 为 `training.max_norm=1.0`。
   - Step-0 Raiden float32 逐元素比较 `0 / 13,920` mismatch，后续 5 次 numel/L1 校验全部通过。
   - 验证集 `val-aux/aime_2024/score/mean@2 > 2 * val-core/aime_2024/acc/mean@2 - 1`（直接证明多轮 shaping 在验证集生效）。
3. **5 步训练轨迹对标 `unch8eca`（观察门禁）**：
   - 检查 `response_length/mean` 是否从 `~800` tokens 跃升至 `~2,000–2,600` tokens，`response_length/clip_ratio` 是否从 `~10%` 降至 `< 1%`。
   - 检查 `critic/rewards/mean` 在 Steps 1–5 是否呈现类似 `unch8eca`（`-0.339 → +0.219`）和 `ci0k54pl` / `yrcsp4z6`（`-0.40 → -0.04`）的快速上升趋势。

### 4.2 25 步正式运行与评估口径

1. **评估采样数与完整性校验**：
   - Step 0 与 Step 25 在 AIME 2024 / 2025 上统一采样 **`n=30`**（`N_RESP_PER_PROMPT_VAL=30`），用 `summarize_validation_dump` 校验每个数据集 30 道题、每题 30 个独立 session。
   - **同时报告 `mean@30` 与前 8 个样本的 `mean@8` / `maj@8`**：
     - `mean@30` 用于与 PR #108（`lb5cdzuq`）及我们的 `p3-retool-rl-11` 直接对比；
     - `mean@8` / `maj@8` 用于与 Askhat Regime 2 的 v5p 运行（`yjowd03u` Step 0 `48.75% / 30.83%`、`ci0k54pl` Step 25 `58.33% / 38.75%`、`yrcsp4z6` Step 25 `60.00% / 46.67%`，均使用 `val_kwargs.n=8`）同口径对比。
2. **双 Scorer 口径报告**：
   每份验证 dump 同时输出：
   - **Strict scorer（`strict_box_verify=True`）准确率**：运行时 `val-core/.../acc` 记录值（只检查末尾 100 字符内的 `\boxed{}` 并做字符串精确匹配）；
   - **默认 scorer（`strict_box_verify=False`）准确率**：离线重算值（支持 `Answer:` 提取与归一化），便于与旧基线 A 对比；
   - **Shaped reward 均值（`val-aux/.../score`）**：单独列出，不与准确率混用。
3. **统计边界说明**：
   - 逐题配对 bootstrap 95% 区间仅刻画固定 checkpoint 在这 30+30 道题上的评估抽样与生成采样波动，不覆盖单次训练 run 的随机性。
   - 从 Askhat 的 4 次独立 Step-0 `n=30` 评估（`22.44%–25.67%`）和我们的 3 次 Step-0 `n=30` 评估（`23.78%–25.56%`）可知，单次 `n=30` 评估本身的采样波动约为 `±1.5–2.0 pp`。

---

## 5. WandB 接入与密钥安全方案

为了将后续 TPU v7x 运行直接暴露到 WandB 并与 Askhat 的运行同屏对比，采用以下配置与安全规范：

1. **目标 Project 与 Entity**：
   - `WANDB_ENTITY=google-trellis`（已通过 WandB API 校验当前账号默认团队为 `google-trellis`，用户名为 `xiaotongyang`）
   - `PROJECT_NAME=verl-torchtpu-grpo-test`（与 Askhat 的 v6e / v5p / v7x 运行在同一 Project 下，或按需设为 `verl_tpu_retool`）
   - `LOGGER_LIST="['console','tensorboard','wandb']"`
2. **Ray Head → `TaskRunner` 的环境变量转发链路**：
   - [`PlatformTPU.get_ray_init_kwargs()`](../../verl/plugin/platform/platform_tpu.py#L616-L630) 已经原生支持从进程环境变量读取 `WANDB_API_KEY`、`WANDB_ENTITY`、`WANDB_PROJECT`、`WANDB_MODE`，并通过 `runner_env_vars` 注入 `TaskRunner` Actor（[`main_ppo.py` L111-L113](../../verl/trainer/main_ppo.py#L111-L113)）。
   - 该路径**不会**将 `WANDB_API_KEY` 写入 Hydra `config` 对象，因此日志打印的 `OmegaConf.to_yaml(config)` 和上传到 WandB 的 `config.yaml` 都不会包含密钥。
3. **防止密钥落盘到代码库或公开日志**：
   - **绝不**将 `WANDB_API_KEY` 写入任何 Git 跟踪文件、PR 描述或文档。
   - [`pod_env_snapshot.py`](../gke/v7x_nap/pod_env_snapshot.py#L31) 和 [`run_ray_jobset_v7x_nap.sh`](../gke/run_ray_jobset_v7x_nap.sh#L332) 已将包含 `_KEY` 的变量名列入 `SECRET_MARKERS` 自动脱敏（打印为 `<redacted>`）。
   - 为避免 `JOBENV_WANDB_API_KEY` 明文出现在 GCS 的 `job_spec.json` 中，我们在 [`run_qwen3_30b_a3b_dapo_torchtitan.sh`](run_qwen3_30b_a3b_dapo_torchtitan.sh) 中支持从挂载盘私有文件 `WANDB_API_KEY_FILE`（例如 `/data/xiaotongyang/.secrets/wandb_api_key`，仅限本人 bucket 访问）读取并 `export WANDB_API_KEY`，从而让 `job_spec.json` 只记录文件路径而不记录密钥本身。

---

## 6. 交付物与执行步骤

1. **脚本与防呆小幅增强（合入 `tpu-main`）**：
   - `examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh`：
     - 将 `custom_reward_function.*` 迁至 `reward.custom_reward_function.*`，新增 `REWARD_FN=${REWARD_FN:-retool}`（`retool` 对应 `pkg://examples.tpu.retool.retool_tpu`，`default` 对应空路径以复现基线 A）；
     - 新增 `ENABLE_THINKING=${ENABLE_THINKING:-False}` 开关（设为 `True` 时传 `+data.apply_chat_template_kwargs.enable_thinking=True`）；
     - 支持 `WANDB_API_KEY_FILE` 自动加载 `WANDB_API_KEY`。
   - `verl/trainer/main_ppo.py`：当 `trainer.use_v1=True` 且顶层 `custom_reward_function.path` 非空而 `reward.custom_reward_function.path` 为空时直接抛错，防止再次静默回退（附单元测试与第 4.1 节固定样例测试）。
2. **配置 C（Regime 2）5 步 Smoke（`p3c-retool-smoke-01`）**：
   - 在 TPU v7x 上跑 5 步 + Step-0 `n=2` 验证，连接 WandB，验证固定样例、shaped reward 生效以及前 5 步 reward 爬升（对标 `unch8eca` 的 `-0.339 → +0.219`）。
3. **配置 C（Regime 2）25 步正式运行（`p3c-retool-rl-01`）**：
   - 在 TPU v7x 上跑完 25 步 + Step 0 / Step 25 `n=30` AIME 评估，产出完整验证汇总与 WandB 链接。

---

## 7. 待确认决策项

1. **实验顺序**：是否同意**直接优先跑配置 C（对齐 Askhat `unch8eca` / `ci0k54pl` 的 16K + `enable_thinking=True` + ReTool shaped scorer + `n=16` + `lr=2e-6`）**，先跑 5 步 smoke 确认爬升，再跑 25 步？（配置 B 的 8K 仅换 scorer 消融作为可选补充。）
2. **配置 C 的 `MAX_TURNS` 取值**：Askhat 在 v5p（`ci0k54pl` / `yrcsp4z6`）上用了 `MAX_TURNS=16`（实际平均轮数 `8.4`），在 v7x（`unch8eca`）上用了 `MAX_TURNS=8`（实际平均轮数 `8.0`）。推荐设为 **`MAX_TURNS=16`**（与 `ci0k54pl` / `yrcsp4z6` 完全对齐，避免多轮长思考被 8 轮截断）。
3. **WandB Project 名称**：直接上报到 Askhat 所在的 `google-trellis/verl-torchtpu-grpo-test`（推荐，便于在同一张图上直接叠加 `unch8eca` / `ci0k54pl` / `lb5cdzuq` 曲线），还是新建独立 project `google-trellis/verl-tpu7x-retool`？
