# TPU v7x ReTool 对齐 Askhat 运行与重跑方案（TPU v7x Only）

状态：**rev 7（已补齐 V1 validation dump `num_turns` 落盘与 CPU 回归测试、更新 `CODE_REF=00298962` 并重新生成 dry-run manifest、补充本地 launcher checkout 校验）**（2026-11-04）。本文为 TPU v7x 专项对齐方案，尚未提交集群训练运行。

- **rev 7**：
  1. **补齐 V1 Validation Dump 的 `num_turns` 字段并新增 CPU 回归测试（第 4.1 节第 3 项）**：在 [`verl/trainer/ppo/v1/trainer_base.py` L1229-L1232](../../verl/trainer/ppo/v1/trainer_base.py#L1229-L1232) 的 `_validate()` 中按 `session_final_indices` 将 `sample_turns` 写入 `validation/{step}.jsonl` 的 `num_turns` 字段（`{"num_turns": [sample_turns[i] for i in session_final_indices]}`），并在 [`tests/trainer/ppo/v1/test_rollout_audit_on_cpu.py`](../../tests/trainer/ppo/v1/test_rollout_audit_on_cpu.py#L722-L1035) 中补充了多输出 session 的 `num_turns` 落盘与 `retool_tpu.compute_score` 离线重算对账 CPU 回归测试；
  2. **将 `CODE_REF` 与实验名更新到包含上述修复的 commit `00298962e462b2e260ad4c305d3181c49f01fe7d`（短 SHA `00298962`），并重新生成 `DRY_RUN=1` 的 `run_record.json`、`jobset.yaml` 和 `job_spec.json`（第 2 / 3.2 节）**；
  3. **在启动命令中加入本地 launcher checkout 强校验（第 3.2 节）**：由于 `CODE_REF` 只决定打包进 `code.tar.gz` 的代码版本、不会替换当前 shell 执行的本地 `examples/tpu/gke/run_ray_jobset_v7x_nap.sh` 与 `v7x_nap/render.py`，启动命令前显式校验当前工作树包含 `00298962e462b2e260ad4c305d3181c49f01fe7d`（`git merge-base --is-ancestor 00298962e462b2e260ad4c305d3181c49f01fe7d HEAD`）。
- **rev 6**：
  1. **落实 W&B Kubernetes Secret 注入并验证凭证权限（第 3.2 / 5 节）**：在 [`ray-jobset-v7x-nap.yaml`](../gke/ray-jobset-v7x-nap.yaml)、[`v7x_nap/render.py`](../gke/v7x_nap/render.py) 与 [`run_ray_jobset_v7x_nap.sh`](../gke/run_ray_jobset_v7x_nap.sh) 中实现 `WANDB_SECRET_NAME` / `WANDB_SECRET_KEY` 注入（向 `ray-head` 容器注入 `WANDB_API_KEY` 的 `secretKeyRef`，并在启动前由 launcher 校验集群 Secret 及 key 存在且非空）；已在 `bodaborg-tpu7x-nap` 的 `priority-dev` 与 `default` 命名空间创建 `secret/xiaotongyang-wandb`，并通过从集群 Secret 读取密钥调用 W&B GraphQL API 验证登录身份（`viewer.username=xiaotongyang, entity=google-trellis`）及目标项目 `google-trellis/verl-torchtpu-grpo-test` 写权限（`readOnly=False`）；
  2. **固定 `CODE_REF` 并同步实验名 SHA 与 `run_record.json`（第 2 / 3.2 节）**；
  3. **修正离线验收的 session 最终输出与解码口径（第 4.1 节第 3 项）**。
- **rev 5**：
  1. 修正第 1.1 / 1.4 节表格，使其与 [`askhat_runs_summary.json`](results/askhat-wandb-audit/askhat_runs_summary.json) 及 [`p3-retool-rl-11/train_metrics.json`](results/p3-retool-rl-11/train_metrics.json) 逐位一致，并标明 `ci0k54pl`（从 `fcze3xlx` Step 20 续跑）与 `hsodvfgr`（从 `rq1p980n` Step 80 续跑）的起止步数；
  2. 修正第 3.2 节启动命令：显式传 `JOB_SCRIPT=examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh`、`JOBENV_VERIFY_PARITY=all`、`JOBENV_RELEASE_SYNC_BUFFERS=False`、`JOBENV_VERL_RAIDEN_RELEASE_BUFFERS=0` 与 `++data.apply_chat_template_kwargs.enable_thinking=True`，并附上 `DRY_RUN=1` 生成的 `job_spec.json` 核验结果；
  3. 明确 `MAX_TURNS=16` 同时设置 `max_user_turns=16` 和 `max_assistant_turns=16`（对齐 v5p `ci0k54pl` / `rq1p980n` / `yrcsp4z6`，而 `unch8eca` 为各 8 轮且切片拓扑不同，不称与 `unch8eca` 完全同配置）；
  4. 将固定样例作为 shaping 硬门槛，不要求 `n=2` 随机验证集必现“答错且多轮”，并将 `unch8eca` 等前 5 步 reward 曲线仅作参考而非“Step 5 必须 > 0”的硬门槛，补充学习率 schedule 恒定性核验；
  5. 补入集群 `kueue-system/time-limit-controller` 的实测队列时长上限（`priority-dev=120 min`、`default=180 min`、预约队列 `r-*` 按预约窗口）及 25 步运行的队列选择策略。
- **rev 4**：改为 **TPU v7x Only**；补入对 Askhat WandB（`google-trellis/verl-torchtpu-grpo-test`）12 个 Qwen3-30B-A3B 运行的审计与 WandB 安全接入方案。

相关证据：
- 我们的 TPU v7x 基线 `p3-retool-rl-11` 结果与 scorer 审计：[`REPRO_RESULTS_TPU7X.md`](REPRO_RESULTS_TPU7X.md)、[`results/p3-retool-rl-11/`](results/p3-retool-rl-11/)
- Askhat WandB 运行审计数据：[`results/askhat-wandb-audit/askhat_runs_summary.json`](results/askhat-wandb-audit/askhat_runs_summary.json)

---

## 0. 目标与边界

1. **只聚焦 TPU v7x（`2 × 2x2x4 = 32 chips`）**：本阶段不重跑 GPU，集中验证在 TPU v7x 上启用 Askhat 后续 16K + thinking + shaped reward 配方（配置 C）后的训练与评估表现；单变量 scorer 消融（配置 B）留作后续可选实验。
2. **以 Askhat WandB 原始记录为准**：所有对照数字均直接取自 `google-trellis/verl-torchtpu-grpo-test` 的 `config.yaml`、原始 history、验证表和我们已提交的 [`p3-retool-rl-11/train_metrics.json`](results/p3-retool-rl-11/train_metrics.json)。
3. **接入 WandB 实时对齐**：新运行上报至 `google-trellis/verl-torchtpu-grpo-test`（`LOGGER_LIST="['console','tensorboard','wandb']"`），实验名显式标注 `v7x`、`16k`、`think`、`shaped`、`16t`（`max_user_turns=16, max_assistant_turns=16`）和代码 SHA，避免与旧运行混淆。

---

## 1. Askhat WandB 完整审计：两代配置与三大核心发现

我们对 `google-trellis/verl-torchtpu-grpo-test` 中 Askhat 的全部 12 个 Qwen3-30B-A3B ReTool 运行进行了拉取与核对（见 [`askhat_runs_summary.json`](results/askhat-wandb-audit/askhat_runs_summary.json)）。审计表明，Askhat 的实验分为**两代配置（Regime 1 vs Regime 2）**。

### 1.1 跨运行完整对照表

| 运行 ID（实验名与记录步数范围） | 硬件 | 配置代际 | 实际生效 Scorer | `enable_thinking` | 长度 / `max_user,assistant_turns` / `n` / `lr` | 有效 `max_norm` | 训练 Reward（Steps 1–5 均值 → Steps 21–25 均值） | Step 0 AIME 2024 / 2025 | Step 25 AIME 2024 / 2025 | 后续步数 AIME 2024 / 2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| **`dj6g4b0i`**（`...-p5b-400steps`，Steps 1–48） | v6e-32 × 2 | **Regime 1**（PR #108） | 默认 `math_dapo`（无 shaping） | `False` | 8K / `8, 8` / `n=8` / `1.4e-6` | `1.0` | `-0.3156 → -0.1156` | `23.00%` / `15.89%`（`@30`） | `25.33%` / `20.56%`（`@30`） | — |
| **`lb5cdzuq`**（`...-p5b-400steps-v2`，Steps 1–29） | v6e-32 × 2 | **Regime 1**（PR #108） | 默认 `math_dapo`（无 shaping） | `False` | 8K / `8, 8` / `n=8` / `1.4e-6` | `1.0` | **`-0.2727 → -0.1570`** | **`25.44%` / `14.44%`**（`@30`） | **`30.00%` / `18.33%`**（`@30`） | — |
| **`1qhymrv4`**（从 `lb5cdzuq` s20 续跑，本地 Steps 1–219）→ **`39othwaq`**（从 s300 续跑，本地 Steps 1–100） | v6e-32 × 2 | **Regime 1**（PR #108） | 默认 `math_dapo`（无 shaping） | `False` | 8K / `8, 8` / `n=8` / `1.4e-6` | `1.0` | `1qh`: `-0.2195 → -0.0133`<br>`39o`: `+0.5273 → +0.4242` | —（续跑） | `30.89%` / `23.67%`（`1qh` `@30`） | **累计 200–400 步**：`50.78%–51.89%` / `37.67%–39.11%`（`@30`） |
| **我们的 `p3-retool-rl-11`**（Steps 1–25） | **v7x-16 × 2** | **Regime 1**（PR #108） | 默认 `math_dapo`（无 shaping） | `False` | 8K / `8, 8` / `n=8` / `1.4e-6` | `1.0` | **`-0.2695 → -0.2469`** | **`25.56%` / `17.33%`**（`@30`） | **`26.78%` / `17.78%`**（`@30`） | — |
| **`yjowd03u`**（`retool-dapo-...-v5p`，仅 Step 0 验证） | v5p | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / `16, 16` / `n=16` / `2e-6` | `1.0` | — | **`48.75%` / `30.83%`**（`@8`） | — | — |
| **`fcze3xlx`**（`...-v5p-cp2`，Steps 1–23） | v5p-32 × 2 | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / `16, 16` / `n=16` / `2e-6` | `1.0` | **`-0.3640 → +0.4563`**（Steps 21–23，Step 23 为 `+0.6562`） | —（`val_before_train=False`） | —（停于 Step 23） | — |
| **`ci0k54pl`**（`...-v5p-cp2`，从 `fcze3xlx` Step 20 续跑，Steps 21–150） | v5p-32 × 2 | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / `16, 16` / `n=16` / `2e-6` | `1.0` | `— → +0.4786`（仅含 Steps 21–25） | —（从 Step 20 续跑） | **`58.33%` / `38.75%`**（`@8`） | **Step 50**：`63.33%` / `53.75%`；**Step 150**：`70.83%` / `50.83%`（`@8`） |
| **`rq1p980n`**（`...-v5p-cp1-opt`，Steps 1–80） | v5p-32 × 2 | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / `16, 16` / `n=16` / `2e-6` | `1.0` | **`-0.3041 → +0.4835`** | — | **`56.67%` / `46.25%`**（`@8`） | **Step 75**：`72.50%` / `57.08%`（`@8`） |
| **`hsodvfgr`**（`...-v5p-cp1-opt`，从 `rq1p980n` Step 80 续跑，Steps 81–150） | v5p-32 × 2 | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / `16, 16` / `n=16` / `2e-6` | `1.0` | —（从 Step 81 开始记录） | — | — | **Step 125–150**：`67.92% → 66.67%` / `55.00%`（`@8`） |
| **`yrcsp4z6`**（`...-v5p-cp1-pr115`，Steps 1–108） | v5p-32 × 2 | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / `16, 16` / `n=16` / `2e-6` | `1.0` | **`-0.3056 → +0.4825`** | — | **`60.00%` / `46.67%`**（`@8`） | **Step 75–100**：`68.33% → 70.00%` / `51.67% → 55.00%`（`@8`） |
| **`unch8eca`**（`...-think16k-tpu7x-opt1`，Steps 1–5） | **v7x-8 × 2** | **Regime 2**（16K+think） | ReTool `retool_tpu`（有 shaping） | **`True`** | 16K / `8, 8` / `n=16` / `2e-6` | `1.0` | **`-0.1080`（Steps 1–5 均值；Step 1 `-0.3391` → Step 5 `+0.2191`）** | — | —（第 5 步后崩溃） | — |

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
   - 但 **`lb5cdzuq` 自身的 Step 0（`val_before_train=True`）是 `25.44% / 14.44%`**（与我们 `p3-retool-rl-11` 的 Step 0 `25.56% / 17.33%` 高度一致），单次运行内部的 Step 0 → Step 25 变化为 **`+4.56 pp`（`25.44% → 30.00%`，其中 `maj@30` 仅从 `46.10%` 变为 `46.58%`，`+0.48 pp`）** 和 **`+3.89 pp`（`14.44% → 18.33%`）**。另一个同配置的 `dj6g4b0i` 在单次运行内 Step 0 → Step 25 为 `23.00% → 25.33%`（`+2.33 pp`）和 `15.89% → 20.56%`（`+4.67 pp`）。
   - PR #108 正文将更早一次独立评估的 Step 0（`22.44% / 14.17%`）与 `lb5cdzuq` 的 Step 25 拼在了一起。同一个 v6e SFT checkpoint（`bjb9xppo`）在 Askhat 的 4 次独立 Step-0 `n=30` 评估中，AIME 2024 在 **`22.44%–25.67%`** 之间波动。
3. **Regime 1（8K 无 thinking）下 25 步本来就处于慢爬升期**：
   - `lb5cdzuq` 前 25 步训练 reward 从 `-0.2727`（Steps 1–5 均值）缓步升到 `-0.1570`（Steps 21–25 均值），平均回复长度 `894` tokens、平均轮数 `6.01`；我们的 `p3-retool-rl-11` 从 `-0.2695` 到 `-0.2469`，平均回复长度 `758 → 822` tokens、平均轮数 `5.86 → 6.08`。
   - 在这套 8K 配置下，Askhat 在 v6e 上接力训练了 **200–400 步**（`lb5cdzuq` → `1qhymrv4` → `hdc2m3lf` → `39othwaq`）才达到 AIME 2024 `50.78%–51.89%`、AIME 2025 `37.67%–39.11%`。

---

### 1.3 发现二：Askhat 的 TPU v5p 快速爬升系列（Regime 2）同时改动了 5 项配置，而有效 clipping 始终是 `1.0`

在 PR #108 之后（对应上游 PR #111 / #115），Askhat 在 TPU v5p 系列（`fcze3xlx` → `ci0k54pl`、`rq1p980n` → `hsodvfgr`、`yrcsp4z6`）中**同时调整了 5 个核心参数**：

1. **修复 V1 Reward 路径**：设置 `reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu`（顶层 `custom_reward_function.path=null`）。在 `ci0k54pl` / `rq1p980n` / `yrcsp4z6` 中，$\text{score} - (2 \times \text{acc} - 1) = +0.036 \sim +0.150$，`val/generations` 中明确出现 `-0.9, -0.7, -0.6` 等多轮工具调用 shaping 分数。
2. **开启 Qwen3 Thinking 模式（影响最大）**：`+data.apply_chat_template_kwargs.enable_thinking=True`（PR #108 脚本为 `False`）。
   - 开启 `<think>` 后，模型在调用 Python 工具前后进行显式推理，平均回复长度从 `~800` tokens 增至 `~2,400` tokens，平均轮数从 `~6.0` 增至 `~8.4`。
   - **仅在 Step 0（未经任何 RL 训练）**，同一个 SFT 模型在开启 thinking + 16K 后（`yjowd03u`），AIME 2024 `mean@8` 直接达到 **`48.75%`**（`maj@8 = 58.12%`），AIME 2025 `mean@8` 达到 **`30.83%`**（`maj@8 = 39.96%`）。
3. **输出长度与轮数翻倍（8K / 各 8 轮 → 16K / 各 16 轮）**：`data.max_response_length=16384`（`max_model_len=18432`、`torchtitan.max_seq_len=18432`），`actor_rollout_ref.rollout.multi_turn.max_user_turns=16`、`max_assistant_turns=16`（由脚本中的 `MAX_TURNS=16` 同时设置）。这把回复截断率（`response_length/clip_ratio`）从我们 8K 运行的 **`8.8%–10.7%`** 降到了 **`0.1%`**（`ci0k54pl` 为 `0.00098`）。
4. **GRPO 每题采样数翻倍（`n=8 → n=16`）**：`actor_rollout_ref.rollout.n=16`，每步总轨迹数从 `512` 增至 `1,024`。
5. **Actor 学习率上调 +43%（`1.4e-6 → 2e-6`）**：`actor_rollout_ref.actor.optim.lr=2e-6`。

> [!IMPORTANT]
> **关于 Gradient Clipping（D7）的核实结论**：
> Askhat 在 v6e 配置了 `optim.clip_grad=0.088`，在 v5p / v7x 配置了 `optim.clip_grad=0.0625`。但无论在 `tpu-main`、上游 PR #108 还是 PR #115 中，`verl/workers/engine/torchtitan/transformer_impl.py` 均未将 `optim.clip_grad` 传入 `TrainingConfig(max_norm=...)`（Askhat 的 `ci0k54pl_output.log` 中同样打印了 `WARNING: TorchTitan grad clipping uses training.max_norm=1.0; optim.clip_grad=0.0625 is not passed to TorchTitan`）。此外，Askhat 所有运行中的 `actor/grad_norm` 均在 `0.05–0.22` 之间，从未触发 `1.0` 裁剪。
> 因此，**保持 TorchTitan 有效 `training.max_norm=1.0` 已经与 Askhat 的所有实际运行 100% 一致**。

---

### 1.4 发现三：从 Step 1 起跑的各运行前 5 步精确对照（含 Askhat 在 TPU v7x 上的 `unch8eca`）

下表列出所有从 Step 1 起跑（非中间步续跑）的运行在 Steps 1–5 和 Steps 21–25 的 `critic/rewards/mean` 原始记录值（与 [`p3-retool-rl-11/train_metrics.json`](results/p3-retool-rl-11/train_metrics.json) 和 [`askhat_runs_summary.json`](results/askhat-wandb-audit/askhat_runs_summary.json) 完全一致；注：`ci0k54pl` 从 Step 21 开始记录，`hsodvfgr` 从 Step 81 开始记录，因此不在 Steps 1–5 列中）：

| 步数 | `p3-retool-rl-11`（v7x，Regime 1：8K, `8/8` 轮） | `lb5cdzuq`（v6e，Regime 1：8K, `8/8` 轮） | `dj6g4b0i`（v6e，Regime 1：8K, `8/8` 轮） | `fcze3xlx`（v5p，Regime 2：16K, `16/16` 轮） | `rq1p980n`（v5p，Regime 2：16K, `16/16` 轮） | `yrcsp4z6`（v5p，Regime 2：16K, `16/16` 轮） | **`unch8eca`（TPU v7x，Regime 2：16K, `8/8` 轮）** |
|---|---|---|---|---|---|---|---|
| **Step 1** | `-0.1367` | `+0.0625` | `-0.1172` | `-0.5233` | `-0.4293` | `-0.4081` | **`-0.3391`** |
| **Step 2** | `-0.3633` | `-0.5352` | `-0.5117` | `-0.4161` | `-0.4190` | `-0.4352` | **`-0.2549`** |
| **Step 3** | `-0.5078` | `-0.2617` | `-0.4453` | `-0.4460` | `-0.3954` | `-0.3391` | **`-0.1377`** |
| **Step 4** | `-0.1211` | `-0.3320` | `-0.2422` | `-0.1520` | `-0.2165` | `-0.3052` | **`-0.0276`** |
| **Step 5** | `-0.2188` | `-0.2969` | `-0.2617` | `-0.2827` | `-0.0601` | `-0.0402` | **`+0.2191`** |
| **Steps 1–5 均值** | **`-0.2695`** | **`-0.2727`** | **`-0.3156`** | **`-0.3640`** | **`-0.3041`** | **`-0.3056`** | **`-0.1080`** |
| **Steps 21–25 均值** | **`-0.2469`** | **`-0.1570`** | **`-0.1156`** | **`+0.4563`**（Steps 21–23） | **`+0.4835`** | **`+0.4825`** | —（第 5 步后崩溃） |

- **关于 `unch8eca` 的边界说明**：
  - `unch8eca` 在 TPU v7x 上使用的是 `max_user_turns=8`、`max_assistant_turns=8`，且拓扑为 `2x2x2`（`nnodes=2, n_gpus_per_node=8`，trainer `dp_shard=4, cp=4`，rollout `tp=16, ep=16`），起点为 `/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu-full`（尚未证明与我们的 `p2-retool-sft-01/global_step_372` 逐位一致）。
  - 我们的主配置 C 采用 `MAX_TURNS=16`（`max_user_turns=16, max_assistant_turns=16`，与 v5p 的 `fcze3xlx` / `ci0k54pl` / `rq1p980n` / `yrcsp4z6` 对齐）和 `2x2x4` 切片（32 TensorCores/切片），因此**不称配置 C 与 `unch8eca` 完全同配置**。
  - 同时也看到，在 `16/16` 轮的 v5p 运行（`fcze3xlx`、`rq1p980n`、`yrcsp4z6`）中，Step 5 的 reward 仍在 `-0.2827 ~ -0.0402` 之间，直到 Steps 6–10 才跨过 `0`、并在 Steps 21–25 升至 `+0.46 ~ +0.48`。因此 5 步 smoke 中**不把“Step 5 reward > 0”设为硬门槛**，上表仅作为趋势参考。

---

## 2. TPU v7x 对齐实验矩阵（只跑 TPU v7x）

全部运行从共享的 SFT checkpoint（`p2-retool-sft-01/checkpoint/global_step_372/huggingface`）出发，使用同一容器镜像（`verl-tpu@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e`），并将运行代码版本显式固定到包含 launcher W&B Secret 注入与 V1 validation dump `num_turns` 落盘修复的 commit（`CODE_REF=00298962e462b2e260ad4c305d3181c49f01fe7d`，短 SHA `00298962`；若本 PR 采用 squash merge 合入 `tpu-main`，启动前将 `CODE_REF` 与 `EXPERIMENT_NAME` 后缀同步更新为合入后的 `tpu-main` commit SHA）：

| 实验 ID | 对齐目标 | Scorer（`reward.custom_reward_function.path`） | `enable_thinking` | `max_response_length` / `max_user,assistant_turns` | `N_RESP_PER_PROMPT` | `ACTOR_LR` | 有效 `max_norm` | 状态 / 执行优先级 |
|---|---|---|---|---|---|---|---|---|
| **A**（`p3-retool-rl-11`） | Askhat Regime 1（PR #108 `lb5cdzuq`） | `None`（默认 `math_dapo`，`strict=False`，无 shaping） | `False` | `8192` / `8, 8` | `8` | `1.4e-6` | `1.0` | **已完成**。作为 Regime 1 基线保留，不覆盖 |
| **C**（`p3c-retool-*`，**第一优先级**） | **Askhat Regime 2（v5p `ci0k54pl` / `rq1p980n` / `yrcsp4z6`）** | `pkg://examples.tpu.retool.retool_tpu`（`strict=True` + shaping） | **`True`** | **`16384` / `16, 16`**（`MAX_TURNS=16`） | **`16`** | **`2e-6`** | `1.0` | **优先执行**：先跑 5 步 smoke（`p3c-retool-smoke-01`），验收通过后跑 25 步（`p3c-retool-rl-01`） |
| **B**（`p3b-retool-*`，后续可选消融） | Regime 1 + 修复 Scorer | `pkg://examples.tpu.retool.retool_tpu`（`strict=True` + shaping） | `False` | `8192` / `8, 8` | `8` | `1.4e-6` | `1.0` | **留作后续单变量消融**：仅在 C 完成后、需要单独拆分 8K 下 scorer 边际影响时运行 |

---

## 3. 配置 C（Regime 2）在 TPU v7x 上的技术预检、启动命令与集群时长限制

### 3.1 序列长度、学习率 Schedule 与 Raiden 参数预检

1. **Trainer 序列长度与动态 batch 预算已按 18,432 tokens 编译**：
   - [`run_qwen3_30b_a3b_dapo_torchtitan.sh` L145-L149](run_qwen3_30b_a3b_dapo_torchtitan.sh#L145-L149) 默认 `ACTOR_MAX_TOKEN_LEN_PER_GPU=4608`、`CONTEXT_PARALLEL_SIZE=4`。
   - `TorchTitanEngine.forward_backward_batch`（[`transformer_impl.py` L508-L511](../../verl/workers/engine/torchtitan/transformer_impl.py#L508-L511)）设置 `sp_size = tensor_parallel_size * context_parallel_size = 4`，因此 `prepare_micro_batches` 的单序列上限为 `4608 × 4 = 18432 = 2048 + 16384`，且 `torchtitan_max_seq_len = 18432`。切到 `MAX_RESPONSE_LENGTH=16384` 不会改变 Trainer 编译的最大序列长度。
2. **Smoke（5 步）与正式运行（25 步 / 150 步）的学习率 Schedule 完全一致（恒定学习率）**：
   - 在 [`run_qwen3_30b_a3b_dapo_torchtitan.sh` L109-L110, L239-L240](run_qwen3_30b_a3b_dapo_torchtitan.sh#L109-L110) 中，`min_lr_factor="${MIN_LR_FACTOR:-1.0}"`，且 `TorchtitanOptimizerConfig` 默认 `lr_warmup_steps=-1`、`lr_warmup_steps_ratio=0.0`。
   - 在 [`transformer_impl.py` L214-L223](../../verl/workers/engine/torchtitan/transformer_impl.py#L214-L223) 中，`warmup_steps = int(0.0 * total_steps) = 0` 且 `min_lr_factor = 1.0`，因此无论 `TOTAL_TRAINING_STEPS` 设为 `5`、`25` 还是 `150`，每一步的实际学习率恒为常数 `2.0e-6`（已核实 Askhat 的 `fcze3xlx`、`rq1p980n`、`yrcsp4z6`、`unch8eca` 在 WandB 中每步 `actor/lr` 均为恒定 `2.0e-6`）。Smoke 验收时将逐歩核对 `actor/lr == 2.0e-6`。
3. **必须显式设置 `JOBENV_RELEASE_SYNC_BUFFERS=False` 与 `JOBENV_VERIFY_PARITY=all`**：
   - 脚本默认 `VERIFY_PARITY="${VERIFY_PARITY:-off}"`（[L159](run_qwen3_30b_a3b_dapo_torchtitan.sh#L159)），因此必须显式传 `JOBENV_VERIFY_PARITY=all`。
   - 当 `RELEASE_SYNC_BUFFERS` 未设置时，脚本调用 `retool_tpu.py --check-raiden-unbind`（[L152-L158](run_qwen3_30b_a3b_dapo_torchtitan.sh#L152-L158)），该函数仅检查 `hasattr(tpu_sync_torch, "unbind_weights")`。在当前镜像 `verl-tpu@sha256:d92a2b3d...`（`tpu_sync_torch 0.1.2.dev20260331`）中该属性存在，因此自动检测会选 `True` 并触发运行时 bug！所以在启动命令中**必须与 `p3-retool-rl-11` 一样显式传入 `JOBENV_RELEASE_SYNC_BUFFERS=False` 和 `JOBENV_VERL_RAIDEN_RELEASE_BUFFERS=0`**，并在启动后核对 resolved config。
4. **Hydra 覆盖 `enable_thinking` 需使用 `++`**：
   - 因为 [`run_qwen3_30b_a3b_dapo_torchtitan.sh` L222](run_qwen3_30b_a3b_dapo_torchtitan.sh#L222) 已经写了 `+data.apply_chat_template_kwargs.enable_thinking=False`，在命令行末尾通过 `"$@"` 覆盖时必须使用 `++data.apply_chat_template_kwargs.enable_thinking=True`（若用单个 `+` 会触发 Hydra 重复 append 报错）。

### 3.2 配置 C 5 步 Smoke 启动命令与 `DRY_RUN=1` 核验过的 `run_record.json`、`job_spec.json`、`jobset.yaml`

`run_ray_jobset_v7x_nap.sh` 通过环境变量 `JOB_SCRIPT` 指定脚本（默认是 0.6B/30B GRPO 脚本），通过 `CODE_REF` 固定打包进 `code.tar.gz` 的 git commit（默认 `HEAD`，因此必须显式传入 `CODE_REF`），并通过 `WANDB_SECRET_NAME` / `WANDB_SECRET_KEY` 向 `ray-head` 容器注入 Kubernetes Secret（不经过 `JOBENV_*` 或 `job_spec.json`）。
**注意**：`CODE_REF` 只决定打包上传到 GCS `code.tar.gz` 的代码版本，**不会替换当前 shell 执行的本地 `examples/tpu/gke/run_ray_jobset_v7x_nap.sh` 和 `v7x_nap/render.py`**。因此启动命令开头显式执行 `git merge-base --is-ancestor 00298962e462b2e260ad4c305d3181c49f01fe7d HEAD && test -z "$(git status --porcelain --untracked-files=no)"` 校验当前工作目录 checkout 已包含修复且没有未提交修改。配置 C 的 5 步 smoke 完整启动命令如下：

```bash
git merge-base --is-ancestor 00298962e462b2e260ad4c305d3181c49f01fe7d HEAD && \
test -z "$(git status --porcelain --untracked-files=no)" && \
RUN_NAME=p3c-retool-smoke-01 \
GCS_BUCKET=ubench-logs \
RUN_ROOT=xiaotongyang/retool-v7x/runs \
INPUTS_ROOT=xiaotongyang/retool-v7x/inputs \
RAY_NAMESPACE=priority-dev \
PREWARM=1 \
JOBSET_MAX_RESTARTS=1 \
ALLOW_REQUEUE=0 \
CODE_REF=00298962e462b2e260ad4c305d3181c49f01fe7d \
WANDB_SECRET_NAME=xiaotongyang-wandb \
WANDB_SECRET_KEY=WANDB_API_KEY \
IMAGE="us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e" \
JOB_SCRIPT="examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh" \
JOBENV_LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32" \
JOBENV_MODEL_PATH="/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x" \
JOBENV_MODEL_SRC_PATH="/data/xiaotongyang/retool-v7x/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface" \
JOBENV_NNODES_TRAINER="4" \
JOBENV_N_CHIPS_TRAINER="8" \
JOBENV_NNODES_ROLLOUT="4" \
JOBENV_N_CHIPS_ROLLOUT="8" \
JOBENV_TRAIN_BATCH_SIZE="64" \
JOBENV_PPO_MINI_BATCH_SIZE="64" \
JOBENV_MAX_RESPONSE_LENGTH="16384" \
JOBENV_MAX_TURNS="16" \
JOBENV_N_RESP_PER_PROMPT="16" \
JOBENV_ACTOR_LR="2e-6" \
JOBENV_NUM_WARMUP_BATCHES="2" \
JOBENV_RELEASE_SYNC_BUFFERS="False" \
JOBENV_VERL_RAIDEN_RELEASE_BUFFERS="0" \
JOBENV_VERIFY_PARITY="all" \
JOBENV_VAL_BEFORE_TRAIN="True" \
JOBENV_N_RESP_PER_PROMPT_VAL="2" \
JOBENV_TEST_FREQ="100" \
JOBENV_TOTAL_TRAINING_STEPS="5" \
JOBENV_SAVE_FREQ="-1" \
JOBENV_LOGGER_LIST="['console','tensorboard','wandb']" \
JOBENV_PROJECT_NAME="verl-torchtpu-grpo-test" \
JOBENV_WANDB_ENTITY="google-trellis" \
JOBENV_EXPERIMENT_NAME="qwen3-30b-a3b-retool-dapo-v7x-16k-think-shaped-16t-smoke-01-sha00298962" \
JOBENV_TENSORBOARD_DIR="@RUN_DIR@/tensorboard" \
bash examples/tpu/gke/run_ray_jobset_v7x_nap.sh \
  "trainer.rollout_data_dir=@RUN_DIR@/rollout" \
  "trainer.validation_data_dir=@RUN_DIR@/validation" \
  "+trainer.rollout_audit_dir=@RUN_DIR@/rollout_audit" \
  "+trainer.rollout_audit_sessions=8" \
  "reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu" \
  "reward.custom_reward_function.name=compute_score" \
  "++data.apply_chat_template_kwargs.enable_thinking=True"
```

已从包含修复的本地 checkout 执行 `DRY_RUN=1`，在 `bodaborg-tpu7x-nap` API server 上通过集群 Secret 校验（`verified Secret xiaotongyang-wandb[WANDB_API_KEY] exists in namespace priority-dev`）以及 `jobset.yaml`、`prewarm.yaml` server-side dry-run 校验：

1. **生成的 `run_record.json`（确认 `code_sha` 与 `EXPERIMENT_NAME` 后缀 `sha00298962` 完全一致，且记录了 Secret 引用名）**：

```json
{
 "code_sha": "00298962e462b2e260ad4c305d3181c49f01fe7d",
 "created_at": "2026-11-04T03:30:58Z",
 "driver_timeout_s": "6600",
 "gcs_bucket": "ubench-logs",
 "image": "us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e",
 "inputs_root": "xiaotongyang/retool-v7x/inputs",
 "jobset_max_restarts": "1",
 "kube_context": "gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap",
 "kueue_queue": "multislice-queue",
 "node_rank_source": "job-index",
 "node_wait_timeout_s": "1200",
 "pod_tpu_accelerator_type": "tpu7x",
 "priority_class": "medium",
 "ray_namespace": "priority-dev",
 "ray_tpu_per_host": "8",
 "require_node_rank": "1",
 "rollout_slices": "1",
 "run_dir": "/data/xiaotongyang/retool-v7x/runs/p3c-retool-smoke-01",
 "run_name": "p3c-retool-smoke-01",
 "run_owner": "xiaotongyang",
 "tpu_topology": "2x2x4",
 "trainer_slices": "1",
 "wandb_secret_key": "WANDB_API_KEY",
 "wandb_secret_name": "xiaotongyang-wandb"
}
```

2. **生成的 `jobset.yaml` 中 `ray-head` 容器环境变量片段（确认 `VERL_NAP_CODE_SHA` 与 `WANDB_API_KEY` `secretKeyRef` 注入）**：

```yaml
                    - name: VERL_NAP_CODE_SHA
                      value: "00298962e462b2e260ad4c305d3181c49f01fe7d"
                    ...
                    - name: WANDB_API_KEY
                      valueFrom:
                        secretKeyRef:
                          name: "xiaotongyang-wandb"
                          key: "WANDB_API_KEY"
```

3. **生成的 `job_spec.json`（确认不包含明文 `WANDB_API_KEY`，且与 `p3-retool-rl-11/job_spec.json` 逐项核对无误）**：

```json
{
 "entrypoint": [
  "bash",
  "examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh",
  "trainer.rollout_data_dir=/data/xiaotongyang/retool-v7x/runs/p3c-retool-smoke-01/rollout",
  "trainer.validation_data_dir=/data/xiaotongyang/retool-v7x/runs/p3c-retool-smoke-01/validation",
  "+trainer.rollout_audit_dir=/data/xiaotongyang/retool-v7x/runs/p3c-retool-smoke-01/rollout_audit",
  "+trainer.rollout_audit_sessions=8",
  "reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu",
  "reward.custom_reward_function.name=compute_score",
  "++data.apply_chat_template_kwargs.enable_thinking=True"
 ],
 "env_vars": {
  "ACTOR_LR": "2e-6",
  "EXPERIMENT_NAME": "qwen3-30b-a3b-retool-dapo-v7x-16k-think-shaped-16t-smoke-01-sha00298962",
  "LIBTPU_INIT_ARGS": "--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32",
  "LOGGER_LIST": "['console','tensorboard','wandb']",
  "MAX_RESPONSE_LENGTH": "16384",
  "MAX_TURNS": "16",
  "MODEL_PATH": "/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x",
  "MODEL_SRC_PATH": "/data/xiaotongyang/retool-v7x/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface",
  "NNODES_ROLLOUT": "4",
  "NNODES_TRAINER": "4",
  "NUM_WARMUP_BATCHES": "2",
  "N_CHIPS_ROLLOUT": "8",
  "N_CHIPS_TRAINER": "8",
  "N_RESP_PER_PROMPT": "16",
  "N_RESP_PER_PROMPT_VAL": "2",
  "PPO_MINI_BATCH_SIZE": "64",
  "PROJECT_NAME": "verl-torchtpu-grpo-test",
  "PYTHONPATH": ".",
  "PYTHONUNBUFFERED": "1",
  "RAY_DATA_HOME": "/data/xiaotongyang/retool-v7x/inputs",
  "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
  "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
  "RAY_memory_monitor_refresh_ms": "0",
  "RAY_memory_usage_threshold": "0.99",
  "RELEASE_SYNC_BUFFERS": "False",
  "SAVE_FREQ": "-1",
  "TENSORBOARD_DIR": "/data/xiaotongyang/retool-v7x/runs/p3c-retool-smoke-01/tensorboard",
  "TEST_FREQ": "100",
  "TOTAL_TRAINING_STEPS": "5",
  "TPU_ACCELERATOR_TYPE": "tpu7x",
  "TRAIN_BATCH_SIZE": "64",
  "VAL_BEFORE_TRAIN": "True",
  "VERIFY_PARITY": "all",
  "VERL_NAP_RUN_DIR": "/data/xiaotongyang/retool-v7x/runs/p3c-retool-smoke-01",
  "VERL_PLATFORM": "tpu",
  "VERL_RAIDEN_RELEASE_BUFFERS": "0",
  "WANDB_ENTITY": "google-trellis"
 }
}
```

### 3.3 集群队列时长限制核查（`time-limit-controller`）与 25 步运行策略

我们直接核查了 `bodaborg-tpu7x-nap` 集群 `kueue-system` 命名空间下的 `time-limit-controller` Deployment 配置：
- 控制器对共享命名空间从 `PodsReady` 起算的硬上限（`TEAM_LIMITS`）为：
  - **`priority-dev`：`120` 分钟**（超过即触发 `[EVICT RUNNING]` 挂起 Workload）；
  - **`default`：`180` 分钟**（但 `default` 的 `nominalQuota=0`，仅靠借用空闲配额）；
  - **预约命名空间（`r-*`）**：不在 `TEAM_LIMITS` 中，由 `calendar_schedule_daemon.py` 按预约时间窗口（`start_time` 至 `end_time`）放行。
- 单纯调大客户端脚本的 `DRIVER_TIMEOUT_MIN` **不会**延长集群 `time-limit-controller` 的驱逐上限。
- **执行策略**：
  1. **5 步 Smoke（`p3c-retool-smoke-01`）**：虽然 `TEST_FREQ=100`，[`trainer_base.py` L507-L516](../../verl/trainer/ppo/v1/trainer_base.py#L507-L516) 在 `is_last_step` 时仍会触发评估，因此包含 Step 0 和 Step 5 两次 `n=2` validation，预计总耗时 `~35–45` 分钟，完全在 `priority-dev` 的 120 分钟上限之内。同时从 smoke 实测配置 C 在 `2 × 2x2x4` 上的准确单步耗时（`timing_s/step`）与验证耗时。
  2. **25 步正式运行（`p3c-retool-rl-01`）**：根据 smoke 实测的单步耗时计算 25 步 + 评估所需总时长：
     - 若单次运行在 110 分钟以内，可在 `priority-dev` 直接运行；
     - 若需要 120–170 分钟，可在 `default`（180 分钟上限，视集群空闲借用情况）或申请 `r-*` 预约窗口运行；
     - 若超过 170 分钟，则在 **预约队列 `r-*`（>= 4 小时窗口）** 中单次跑完，或将 Step-0 `n=30` 评估拆为独立评估任务（正式训练设 `VAL_BEFORE_TRAIN=False`，与 Askhat 的 `yjowd03u` + `ci0k54pl` / `rq1p980n` / `yrcsp4z6` 做法一致）以缩短训练作业时长。在 P4 checkpoint resume 验证完成前，**不假设被超时中断的作业可无损分段续跑**。

---

## 4. Smoke 验收与 25 步评估口径

### 4.1 5 步 Smoke 验收标准（`p3c-retool-smoke-01`）

**硬性放行门槛（全部通过后才启动 25 步正式运行）**：

1. **Resolved Config 逐项核对**：
   - `reward.custom_reward_function.path == "pkg://examples.tpu.retool.retool_tpu"` 且 `name == "compute_score"`；
   - `data.apply_chat_template_kwargs.enable_thinking == True`；
   - `data.max_response_length == 16384`，`actor_rollout_ref.rollout.max_model_len == 18432`；
   - `actor_rollout_ref.rollout.multi_turn.max_user_turns == 16` 且 `max_assistant_turns == 16`；
   - `actor_rollout_ref.rollout.n == 16`，`actor_rollout_ref.actor.optim.lr == 2e-6`；
   - `actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.release_buffers_after_sync == False` 且 `verify_parity == "all"`；
   - TorchTitan 日志确认有效 clipping 为 `training.max_norm=1.0`，且 5 步每一步记录的 `actor/lr == 2.0e-6`。
2. **固定样例 Shaping 确定性检查（硬门槛）**：
   用该 run 解析后的配置调用 [`get_custom_reward_fn`](../../verl/trainer/ppo/reward.py#L50) 并通过 `NaiveRewardManager`（构造带 `__num_turns__` 的 batch）跑固定样例集，确认：
   - 结尾 `\boxed{42}`（正确，`num_turns=2, 6`）→ `1.0, 1.0`
   - 结尾 `\boxed{41}`（错误，`num_turns=2, 3, 4, 6, 10`）→ `-1.0, -0.95, -0.9, -0.8, -0.6`
   - 仅有 `Answer: 42` 无 `\boxed{}`（`num_turns=2`）→ `-1.0`（`strict_box_verify=True` 判错，区别于默认 scorer 的 `+1.0`）
   - `\boxed{42}` 后缀超过 100 字符（`num_turns=2`）→ `-1.0`
3. **实际运行样本与验证集一致性检查（硬门槛，按每个独立 session 的最终 output 对账）**：
   - **验证集（`validation/0.jsonl` 与最后一歩触发的 `validation/5.jsonl`，`expected_n=2`）**：多轮 agent 的一个 session（`{uid}_{session_id}`）可能产生多条 output（`{uid}_{session_id}_{output_index}`），并由最终 output（最大 `output_index`）承载该 session 的 reward，因此验证集 JSONL 行数不写死为 120 行。[`verl/trainer/ppo/v1/trainer_base.py` L1229-L1232](../../verl/trainer/ppo/v1/trainer_base.py#L1229-L1232) 已按 `session_final_indices` 将每个 session 的 `sample_turns` 写入 `validation/{step}.jsonl` 的 `num_turns` 字段（回归测试见 [`test_rollout_audit_on_cpu.py`](../../tests/trainer/ppo/v1/test_rollout_audit_on_cpu.py#L722-L1035)）。验收时对 `validation/0.jsonl`（及 `validation/5.jsonl`）使用 [`summarize_validation_dump`](../../verl/trainer/ppo/v1/rollout_audit.py#L992)（传入 `expected_n=2` 及 60 道预期题目的 `expected_questions`）校验 `complete == True` 且 `score_complete == True`（即 60 道预期题目 × 每题 2 个独立 session = 120 个完整 session），并取每个 session 的**最终 output**（`max(output_index)`），确认其记录的 `row["acc"]` 和 `row["score"]` 与离线调用 `retool_tpu.compute_score(data_source, row["output"], ground_truth, extra_info={"num_turns": row["num_turns"]})` 重算结果完全一致（`validation/{step}.jsonl` 的 `row["output"]` 已由 [`trainer_base.py` L1151](../../verl/trainer/ppo/v1/trainer_base.py#L1151) 以 `skip_special_tokens=True` 解码）。
   - **训练审计样本（`rollout_audit/{1..5}.jsonl`，每步 8 个 session、共 40 个独立 session）**：按 `record["session"]` 分组并取每个 session 的**最终 output**（`max(output_index)`）。由于 `rollout_audit.py` 生成的 `record["response_text"]` 使用了 `skip_special_tokens=False`（保留了末尾 `<|im_end|>` 等特殊 token），而运行时 [`NaiveRewardManager`](../../verl/workers/reward_manager/naive.py#L119) 使用 `tokenizer.decode(valid_response_ids, skip_special_tokens=True)` 解码，因此离线重算必须先对 `record["response_ids"]` 执行完全相同的 `tokenizer.decode(record["response_ids"], skip_special_tokens=True)`，再调用 `retool_tpu.compute_score(..., extra_info={"num_turns": record["num_turns"]})`，确认重算的 `acc`、`pred`、`score` 与日志中的 `record["reward_extra_info"]["acc"]`、`record["reward_extra_info"]["pred"]` 及 `record["score"]` 100% 一致。
   - 注：`n=2` 随机验证集不一定恰好抽到“答错且多轮”的样本，因此**不要求验证集随机样本一定出现 shaped 分数**；实际验证集（120 个 session 最终 output）与训练抽样（40 个 session 最终 output）中命中 shaping（`-0.95 ~ -0.6`）的条数单独统计记录。
4. **Raiden 权重同步与训练稳定性检查（硬门槛）**：
   - Step-0 Raiden float32 逐元素比较 `0 / 13,920` mismatch（`env/raiden_exact_parity_step_0.json`），Steps 1–5 的 5 次 numel/L1 校验全部通过；
   - 40 条审计轨迹中 `0` 条 `response_mask` 违规；无异常或非有限 `grad_norm` 跳过更新告警；
   - WandB 面板（`google-trellis/verl-torchtpu-grpo-test`）正常收到全部 5 步指标。
5. **前 5 步行为观察项（参考项，不设“Step 5 > 0”硬门槛）**：
   - 记录 `response_length/mean`（预期从 8K 无 thinking 的 `~800` tokens 显著增加）、`response_length/clip_ratio`、`num_turns/mean` 以及 Steps 1–5 的 `critic/rewards/mean` 轨迹，并与第 1.4 节的 `fcze3xlx` / `rq1p980n` / `yrcsp4z6` / `unch8eca` 并列对比。

### 4.2 25 步正式运行与评估口径（`p3c-retool-rl-01`）

1. **评估采样数与完整性校验**：
   - Step 0 与 Step 25 在 AIME 2024 / 2025 上采样 **`n=30`**（`N_RESP_PER_PROMPT_VAL=30`），用 `summarize_validation_dump` 校验每个数据集 30 道题、每题 30 个独立 session。
   - **同时报告 `mean@30` 与每题前 8 个 session 的 `mean@8` / `maj@8`**：
     - `mean@30` 用于与 PR #108（`lb5cdzuq`）及我们的 `p3-retool-rl-11` 同口径对比；
     - `mean@8` / `maj@8` 用于与 Askhat Regime 2 的 v5p 运行（`yjowd03u` Step 0 `48.75% / 30.83%`、`rq1p980n` Step 25 `56.67% / 46.25%`、`ci0k54pl` Step 25 `58.33% / 38.75%`、`yrcsp4z6` Step 25 `60.00% / 46.67%`，均使用 `val_kwargs.n=8`）同口径对比。
2. **双 Scorer 口径报告**：
   每份验证 dump 同时输出：
   - **Strict scorer（`strict_box_verify=True`）准确率**：运行时 `val-core/.../acc` 记录值（只检查末尾 100 字符内的 `\boxed{}` 并做字符串精确匹配）；
   - **默认 scorer（`strict_box_verify=False`）准确率**：离线重算值（支持 `Answer:` 提取与归一化），便于与旧基线 A 对比；
   - **Shaped reward 均值（`val-aux/.../score`）**：单独列出，不与准确率混用。
3. **统计边界说明**：
   - 逐题配对 bootstrap 95% 区间仅刻画固定 checkpoint 在这 30+30 道题上的评估抽样与生成采样波动，不覆盖单次训练 run 的随机性。

---

## 5. WandB 接入与密钥安全方案（已实现并通过集群与 W&B 权限核验）

1. **目标 Project 与命名规范**：
   - `WANDB_ENTITY=google-trellis`、`PROJECT_NAME=verl-torchtpu-grpo-test`、`LOGGER_LIST="['console','tensorboard','wandb']"`。
   - `EXPERIMENT_NAME` 统一带平台、配置与代码版本标签：
     - Smoke：`qwen3-30b-a3b-retool-dapo-v7x-16k-think-shaped-16t-smoke-01-sha00298962`
     - 25 步：`qwen3-30b-a3b-retool-dapo-v7x-16k-think-shaped-16t-25step-01-sha<short_sha>`
2. **模板与 Launcher 注入实现（不落盘到代码库、`config.yaml`、`job_spec.json` 或 `driver.log`）**：
   - **容器层注入**：在 [`ray-jobset-v7x-nap.yaml`](../gke/ray-jobset-v7x-nap.yaml#L139)、[`v7x_nap/render.py`](../gke/v7x_nap/render.py#L70-L92) 和 [`run_ray_jobset_v7x_nap.sh`](../gke/run_ray_jobset_v7x_nap.sh#L33-L34) 中新增 `WANDB_SECRET_NAME`（默认空）与 `WANDB_SECRET_KEY`（默认 `WANDB_API_KEY`）。当设置 `WANDB_SECRET_NAME=xiaotongyang-wandb` 时，渲染器向 `ray-head` 容器注入 `valueFrom.secretKeyRef: {name: "xiaotongyang-wandb", key: "WANDB_API_KEY"}`；`run_record.json` 仅记录 Secret 名称与键名（`wandb_secret_name` / `wandb_secret_key`），**绝不通过 `JOBENV_WANDB_API_KEY` 写入 `job_spec.json`**。
   - **提交前集群强校验**：[`run_ray_jobset_v7x_nap.sh`](../gke/run_ray_jobset_v7x_nap.sh#L179-L192) 在渲染 manifest 后、提交或 dry-run 前，执行 `kubectl get secret "${WANDB_SECRET_NAME}" -n "${RAY_NAMESPACE}" -o jsonpath="{.data.${WANDB_SECRET_KEY}}"`，若目标命名空间不存在该 Secret 或对应 key 为空则立即报错终止。
   - **Ray Driver / TaskRunner 传递链路**：`ray-head` 容器环境变量中的 `WANDB_API_KEY` 由 `head_supervisor.py` 启动的本地 Ray head 继承；Ray job driver（`run_qwen3_30b_a3b_dapo_torchtitan.sh` → `verl.trainer.main_ppo`）在 head 节点运行并直接继承 `WANDB_API_KEY`（不在 shell 中赋值，因此 `set -x` 不会在 `driver.log` 中打印密钥）；随后 [`PlatformTPU.get_ray_init_kwargs()`](../../verl/plugin/platform/platform_tpu.py#L616-L630) 通过 `runner_env_vars` 将 `WANDB_API_KEY` 和 `WANDB_ENTITY` 转发给 `TaskRunner` Actor（[`main_ppo.py` L111-L125](../../verl/trainer/main_ppo.py#L111-L125)），由 [`Tracking`](../../verl/utils/tracking.py#L76-L86) 完成 `wandb.init(project="verl-torchtpu-grpo-test", entity="google-trellis", ...)`。同时 [`pod_env_snapshot.py`](../gke/v7x_nap/pod_env_snapshot.py#L31) 已对含 `_KEY` 的变量自动脱敏为 `<redacted>`。
3. **集群 Secret 与 W&B 目标 Project 写权限实测结果**：
   - 已在 `gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap` 的 `priority-dev` 与 `default` 命名空间创建 `Secret/xiaotongyang-wandb`（key 为 `WANDB_API_KEY`）。
   - 已从集群 `priority-dev` 命名空间读取 `Secret/xiaotongyang-wandb` 中的 `WANDB_API_KEY`，直接向 `https://api.wandb.ai/graphql` 发起鉴权与项目权限核验，返回：
     - `viewer`：`{"username": "xiaotongyang", "email": "xiaotongyang@google.com", "entity": "google-trellis"}`
     - `project(name: "verl-torchtpu-grpo-test", entityName: "google-trellis")`：`{"id": "UHJvamVjdDp2MTp2ZXJsLXRvcmNodHB1LWdycG8tdGVzdDpnb29nbGUtdHJlbGxpcw==", "name": "verl-torchtpu-grpo-test", "readOnly": false, "access": "PRIVATE"}`
   - 确认该集群 Secret 凭证有效且对 `google-trellis/verl-torchtpu-grpo-test` 具备写权限（`readOnly: false`）。

---

## 6. 已确认决策汇总

1. **优先配置 C**：先跑配置 C 的 5 步 smoke（`p3c-retool-smoke-01`），验证配置生效、工具调用、权重同步和训练稳定性，并测量 16K 下的实际单步耗时；通过后选择时长充足的队列运行配置 C 的 25 步实验（`p3c-retool-rl-01`）。配置 B 留作后续单变量消融。
2. **`MAX_TURNS=16` 作为主配置**：明确设置 `max_user_turns=16`、`max_assistant_turns=16`，与 v5p 的 `ci0k54pl` / `rq1p980n` / `yrcsp4z6` 对齐。
3. **WandB Project**：使用现有 `google-trellis/verl-torchtpu-grpo-test`，通过明确的 `EXPERIMENT_NAME` 标签区分 v7x 新配置与旧运行。
