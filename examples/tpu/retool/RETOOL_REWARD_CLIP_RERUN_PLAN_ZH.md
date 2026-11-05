# TPU v7x ReTool 对齐 Askhat 运行与重跑方案（TPU v7x Only）

状态：**rev 13（64 卡 `p3c-retool-25s-02` 已完整跑完 25/25 步训练与 Step 0 / Step 25 `n=30` AIME 评估；本版补齐评审指出的证据缺口：整段掩码审计并从原始会话离线重算、运行代码推到 GitHub、耗时拆分与 checkpoint 状态；W&B [`wopup8dq`](https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/wopup8dq)）**（2026-11-05）。

- **rev 13**（评审意见修正，详见第 8.5.2、8.5.6–8.5.8 节）：
  1. **掩码审计改为整段审计**：新的 [`rollout_mask_audit.py`](rollout_mask_audit.py) 在整条 response 上配对 `<tool_response>` / `<tool_call>` 块，并要求每段 `mask=0` 恰好是 agent loop 注入的工具轮次；旧脚本逐段查找完整标签，会漏掉跨 mask segment 的块。200 个原始审计会话已归档为 [`results/p3c-retool-25s-02/rollout_audit/*.jsonl.gz`](results/p3c-retool-25s-02/rollout_audit/)（大小与 MD5 和 GCS 对象一致），`verify_evidence.py` 改为从原始会话重算，不再读汇总：`0` 违规。
  2. **运行代码可复现**：`p3c-retool-25s-01` / `-02` 运行的 `348bbff9` 原先只在本地，现已推到 GitHub tag [`run-code/p3c-retool-25s-01-02`](https://github.com/syangx39/verl-upstream/tree/run-code/p3c-retool-25s-01-02)；GCS 上 launcher 打包的 `src-348bbff9f4d3.tar.gz` 与该 commit 的 `1,350` 个文件逐一一致（[`verify_code_snapshot.py`](results/p3c-retool-25s-02/verify_code_snapshot.py)）。
  3. **Checkpoint 状态写明**：`-02` 用 `SAVE_FREQ=-1`，整个 run 没有保存任何 checkpoint（包括 Step 25）；checkpoint 改写 GCS 路径后的写入没有被这次 run 实测。
  4. **耗时复核与拆分**：Steps 2–25 `timing_s/step` 均值 `287.3` s 与 W&B 历史一致；用 TensorBoard 事件的 `wall_time` 拆分端到端墙钟，发现每步约 `46.7` s 花在计时之外的 TensorBoard 写入（`TENSORBOARD_DIR` 在 GCS Fuse 上），且 `timing_s/step` 中平均 `194.7` s 是等 rollout；删去“25 步纯训练墙钟 9,108.2 s”的说法（tqdm 总耗时含 TensorBoard 写入与 Step 25 验证）。
  5. **Raiden 表述**：Step 0 的逐元素比对（96 rank / 41,760 分片）与 Steps 1–25 的范数检查分开写；后者是范数级检查，不等于每步逐元素一致。

- **rev 12**：
  1. **补入 64 卡 TPU v7x 正式运行 `p3c-retool-25s-02`（W&B [`wopup8dq`](https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/wopup8dq)）25/25 步完整训练与 Step 0 → Step 25 `n=30` 评估归档（第 8.5 节，产物目录 [`results/p3c-retool-25s-02/`](results/p3c-retool-25s-02/)）**：
     - **AIME 2024（`n=30`，900 条样本）**：Step 0 `399/900`（**`44.33%`**）→ Step 25 `492/900`（**`54.67%`**），提升 **`+10.33` pp**（二层配对 bootstrap 95% CI **`[+3.78, +17.22]` pp**；30 题中 18 题上升 / 5 题下降 / 7 题持平；`maj@30` `63.78%` → `71.59%`）；
     - **AIME 2025（`n=30`，900 条样本）**：Step 0 `293/900`（**`32.56%`**）→ Step 25 `378/900`（**`42.00%`**），提升 **`+9.44` pp**（二层配对 bootstrap 95% CI **`[+3.67, +15.56]` pp**；30 题中 19 题上升 / 3 题下降 / 8 题持平；`maj@30` `48.77%` → `61.67%`，`best@30` `69.77%` → `81.21%`）；
     - **训练与放行门禁**：训练 reward `critic/score/mean` 5 步均值从 Steps 1–5 的 `-0.108` 升至 Steps 21–25 的 **`+0.516`**；25 步共 `25,600` 条轨迹中 `25,451` 条（`99.42%`）含 `<tool_response>`（共 `84,914` 次工具返回），`-2.0` 截断惩罚 `0` 条；200 个审计会话的整段掩码审计 `0` 违规（rev 13 从原始会话重算）；Raiden：Step 0 三副本 96 rank / `41,760` 个分片逐元素比对 `0` 不一致，Steps 1–25 每步 3 个副本（`75 / 75` 次）的分布式范数检查通过（范数级检查）；
     - **跨平台（TPU v7x vs GB200 GPU）与 Askhat v5p 对照**：在同一 SFT 起点（`p2-retool-sft-01/checkpoint/global_step_372`）与同口径 `n=30` 评估下，TPU v7x `p3c-retool-25s-02` 与 GB200 64-GPU `g64-retool-25s-01` 的 Step 25 准确率高度一致（AIME 2024 `54.67%` vs `54.00%`，AIME 2025 `42.00%` vs `41.78%`）。

- **rev 11**：
  1. **补入 64 卡 TPU v7x 正式运行 `p3c-retool-25s-01`（W&B [`smcnyyoh`](https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/smcnyyoh)）Steps 0–3 在线放行检查全项通过报告（第 8 节）**：
     - **节点池坏 Slice 隔离（[`run_ray_jobset_v7x_nap.sh`](../gke/run_ray_jobset_v7x_nap.sh) `EXCLUDE_NODEPOOLS`）**：排查 `r-b0f80` 命名空间下各 `2x2x4` 节点池，排除 1 个坏盘池（`nap-tpu7x-stand-4t-13ryoomz` `mkdir /data/...: Input/output error`）与 6 个半占用池，在 3 个空闲存量健康池 + 1 个新扩容 NAP 池上实现 `17/17` Pod 一次性全部 Running；
     - **Step 0 三副本（`3 × 32 = 96` ranks）逐元素 Raiden 权重同步核验（[`results/p3c-retool-25s-01/raiden_exact_parity_step_0.json`](results/p3c-retool-25s-01/raiden_exact_parity_step_0.json)）**：`num_results == 96`、`expected_ranks == 96`、`expected_workers_per_replica == [32, 32, 32]`、`checked_per_rank == [435]`（合计 **`41,760`** 个张量分片逐元素校验），**`num_mismatched == 0`、`num_unresolved == 0`、`coverage_errors == []`**，耗时 `38.2628s`；
     - **Step 0 `n=30`（每题 30 采样，共 `1,800` 条 16K 多轮轨迹）AIME 2024 / 2025 基线与 `num_turns` 离线对账（[`results/p3c-retool-25s-01/val0_gate3_summary.json`](results/p3c-retool-25s-01/val0_gate3_summary.json)）**：AIME 2025 **`acc/mean@30 = 31.78%` (`286/900`)**、`maj@30 = 45.04%`、`best@30 = 71.74%`、`reward/mean@30 = -0.2046`；AIME 2024 **`acc/mean@30 = 44.78%` (`403/900`)**、`maj@30 = 62.49%`、`best@30 = 79.25%`、`reward/mean@30 = +0.0343`；全部 `1,800 / 1,800` 条样本离线调用 `retool_tpu.compute_score` 按真实 `num_turns` 复算 **`0` 误差**；
     - **Steps 1–3 三副本分布式范数校验、梯度更新、掩码审计与稳态吞吐（[`results/p3c-retool-25s-01/steps_1_to_3_release_gate_summary.json`](results/p3c-retool-25s-01/steps_1_to_3_release_gate_summary.json)）**：Steps 1–3 每步均完整打印 `Replica 0, 1, 2` 共 3 个副本的 `100% DISTRIBUTED NORM PARITY CONFIRMED!`（每副本 `30,532,122,624` 参数 across `579` tensors，`L1 delta <= 0.415`、`L2 delta <= 0.000794`），跨 3 切片权重同步耗时 `22.89s -> 16.55s -> 15.06s`；Steps 1–3 共 `3,072` 条训练轨迹（`99.15%` 含 `<tool_response>`，共 `10,166` 次工具调用），`response_length/clip_ratio` 仅 **`0.10% / 0.10% / 0.20%`（远低于 `< 5%` 门槛）**，`-2.0` `reward_CLIP` 计数为 **`0`**，`24 / 24` 审计会话掩码违规为 **`0`**；Step 2–3 稳态 `timing_s/gen = 0.04s`（100% 异步掩盖），单步总耗时仅 **`156.77s`（Step 2）/ `144.33s`（Step 3）**，已放行继续跑满 25 步！
- **rev 10**：
  1. **校准 §6.5.5 中进程内自动校验与人工早期监控的真实边界（不虚称已入库自动 watcher 或 Step 3 阻塞屏障）**：
     - 明确在固定代码版本 `CODE_REF=53bc215c8880483142caee0b4d25d57066a0a54f` 中，仅 Raiden 的 Step 0 逐元素对账与每步 `numel + L1` 校验具备进程内抛出 `RaidenParityError` 的自动中止机制；仓库中并未合入独立的后台 watcher 守护脚本，Trainer 也没有在 Step 3 结束时等待外部离线对账的同步暂停屏障（若外部检查延迟，训练可能已进入 Step 4）。因此将 `validation/0.jsonl`、`rollout_audit/{1..3}.jsonl` 及 `head.log` 检查明确表述为**人工/Agent 早期跟跑监控（发现异常后尽快手动执行 `kubectl delete jobset` 停止作业）**。
  2. **校准 §6.4 / §6.5.5 对 `CODE_REF=53bc215c` Raiden 同步校验（Step 0 `RaidenParityCheck.report_exact` 与 Step 1+ `_verify_weight_parity` / `_compare_replica_norms`）覆盖范围的描述**：
     - **Step 0**：[`raiden_checkpoint_engine.py` L1480–L1489](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py#L1480-L1489) 调用 [`RaidenParityCheck.report_exact`](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py)，按运行时**实际创建的 `manager.replicas`** 动态计算预期 rank 数并校验实际 replicas 的完整性（而非在代码中写死固定常数 `96`）；`expected_ranks == 96` 与 `expected_workers_per_replica == [32, 32, 32]` 由后续人工核验 `raiden_exact_parity_step_0.json` 确认。
     - **Step 1+**：在 [`raiden_checkpoint_engine.py` L806-L919](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py#L806-L919) 中，`_verify_weight_parity` 对每个 replica 仅检查返回的 `sampler_workers` 非空（未核验 worker rank 的身份与唯一性），且 `_compare_replica_norms` 的自动通过条件只判定逐张量与全局的 **`numel`（`delta_numel == 0`）和 `L1` 范数（`rel_tol = 1e-3`）**，`Global L2 Norm` 仅打印日志、不参与自动判定。文档明确说明：**聚合 `numel` / `L1` 能检测造成聚合值不一致的异常，`ray status` 可辅助确认 TPU 资源在线，但二者不能单独证明每副本 32 个不同 rank 集合完整无重**。
  3. **修正 §6.3 表中 Askhat `yjowd03u` Step 0 与 `ci0k54pl` Step 25 的 `mean_score`**：与 [`askhat_runs_summary.json`](results/askhat-wandb-audit/askhat_runs_summary.json) 逐位对齐（`yjowd03u` Step 0 为 **`+0.10625 / -0.22125`**；`ci0k54pl` Step 25 为 **`+0.249167 / -0.075`**）。
- **rev 9**：
  1. **修正 §6.5 中 64 卡正式运行（`p3c-retool-25s-01`）启动命令，使其与 `DRY_RUN=1` 生成的 `run_record.json` / `job_spec.json` 100% 同源一致**：
     - 显式设置 `JOB_SCRIPT="examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh"` 并移除位置参数中的 `-- bash ...`（避免 launcher 误选默认 GRPO 脚本）；
     - 显式设置 `GCS_BUCKET=ubench-logs`、`RUN_ROOT=xiaotongyang/retool-v7x/runs`、`INPUTS_ROOT=xiaotongyang/retool-v7x/inputs`，并用 `@RUN_DIR@/{rollout,validation,rollout_audit,tensorboard}` 统一生成产物路径；
     - 显式传入 `JOBENV_LIBTPU_INIT_ARGS` 与 `JOBENV_NUM_WARMUP_BATCHES="2"`；将无效变量 `TOPOLOGY` / `WAIT_FOR_COMPLETION=0` 更正为 launcher 实际读取的 `TPU_TOPOLOGY="2x2x4"` 与 `KEEP_JOBSET=1`；
     - 统一使用 `p3c-retool-smoke-01` 实测 Pod 运行的同一镜像 digest `verl-tpu@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e`（已由 `.pods.json` 的 `imageID` 与 `tests/special_tpu/gke/tpu-ci-image.env` 交叉核实）；
     - 补充动态预约命名空间 `r-b0f80` 的 W&B Secret（`Secret/xiaotongyang-wandb`）创建与启动前 `DRY_RUN=1 RAY_NAMESPACE=r-b0f80` 校验流程。
  2. **按真实 `raiden_exact_parity_step_0.json` Schema 与浮点容差 `abs_tol=1e-6` 完善 §6.5.5 检查项**：
     - 对齐 [`raiden_checkpoint_engine.py`](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py) Step 0 实际落盘字段：`num_results == 96`、`expected_ranks == 96`、`expected_workers_per_replica == [32, 32, 32]`、`checked_per_rank == [435]`、`replica_rank_pairs` 完整覆盖 `0..2 × 0..31`、`num_mismatched == 0`、`num_unresolved == 0`、`coverage_errors == []` 及 `sum(r["checked"] for r in ranks) == 41,760`；
     - 离线 `score` 复算采用 `math.isclose(..., abs_tol=1e-6)` 避免 float32 舍入误报。
  3. **更正 §6.3 中 Askhat `ci0k54pl` Step 25 准确率（`58.33% / 38.75%`）并同步全篇关于 32 卡“早期部分验证”的表述**。
- **rev 8**：
  1. **补入 `p3c-retool-smoke-01`（W&B [`uinjk8rs`](https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/uinjk8rs)）32 卡早期部分验证结果（第 6 节）**：
     - **完成范围说明**：已完成 Step 0 验证（`n=2`，120 个 session）、Step 0 单副本 13,920 分片逐元素校验、Step 1 完整训练（含 8 条 session 审计与 1 次分布式范数校验）及 Step 2 前向/反向更新；Step 2 在权重同步阶段因 `r-db36b` 动态预约到期被回收，Steps 3–5 未运行（原计划的 5 次同步检查、40 条审计轨迹与末次验证因此未在 32 卡上全部跑完，转由 64 卡正式 run 的前 3 步执行三副本放行检查）；
     - **Gate 1（Resolved Config & W&B）**：确认 `retool_tpu.compute_score`、`enable_thinking=True`、`16384/18432`、两侧 `16` turns、`training.max_norm=1.0`、`release_buffers_after_sync=False`、`verify_parity=all`、`actor/lr=2.0e-6` 全部生效，`WANDB_API_KEY` 已在 `pod_env.json` 脱敏且 W&B 实时收到指标；
     - **Gate 2（固定样例经 `NaiveRewardManager` 完整链路检查）**：9/9 固定样例通过 `get_custom_reward_fn` + `NaiveRewardManager.__call__` 端到端校验；
     - **Gate 3（部分完成：`validation/0.jsonl` 120 个 session + `rollout_audit/1.jsonl` 8 个 session 离线对账）**：`summarize_validation_dump` 校验 `complete=True, score_complete=True`；`120/120` 验证 session 与 `8/8` 训练审计 session 的 `acc`、`pred`、`score` 离线复算 **100% 一致（0 mismatch）**，`0` 条 `response_mask` 违规；Step 0 小样本（`n=2`，每集 60 样本）初测为 **AIME 2024 `41.67%`（`mean_score = -0.0200`）、AIME 2025 `28.33%`（`mean_score = -0.2383`）**，正式基线对比以 64 卡 Step 0 / Step 25 的 `n=30`（每集 900 样本）为准；
     - **Gate 4（部分完成：单副本 Raiden 权重同步）**：Step 0 逐元素 float32 对账 `0 / 13,920` mismatch（`1.79s`），Step 1 分布式范数对账 `100% DISTRIBUTED NORM PARITY CONFIRMED`（`30,532,122,624` 参数 across `579` tensors，端到端 `15.59s`）；
  2. **补充 64 卡预约队列 `r-b0f80`（`Wed Nov 4, 8:00 AM – 2:00 PM`）运行与前 3 步在线放行方案（第 6.5 节）**。
- **rev 7**：
  1. **补齐 V1 Validation Dump 的 `num_turns` 字段并新增 CPU 回归测试（第 4.1 节第 3 项）**：在 [`verl/trainer/ppo/v1/trainer_base.py` L1229-L1232](../../../verl/trainer/ppo/v1/trainer_base.py#L1229-L1232) 的 `_validate()` 中按 `session_final_indices` 将 `sample_turns` 写入 `validation/{step}.jsonl` 的 `num_turns` 字段（`{"num_turns": [sample_turns[i] for i in session_final_indices]}`），并在 [`tests/trainer/ppo/v1/test_rollout_audit_on_cpu.py`](../../../tests/trainer/ppo/v1/test_rollout_audit_on_cpu.py#L722-L1035) 中补充了多输出 session 的 `num_turns` 落盘与 `retool_tpu.compute_score` 离线重算对账 CPU 回归测试；
  2. **将 `CODE_REF` 与实验名更新到包含上述修复的 commit `53bc215c8880483142caee0b4d25d57066a0a54f`（短 SHA `53bc215c`），并重新生成 `DRY_RUN=1` 的 `run_record.json`、`jobset.yaml` 和 `job_spec.json`（第 2 / 3.2 节）**；
  3. **在启动命令中加入本地 launcher checkout 强校验（第 3.2 节）**：由于 `CODE_REF` 只决定打包进 `code.tar.gz` 的代码版本、不会替换当前 shell 执行的本地 `examples/tpu/gke/run_ray_jobset_v7x_nap.sh` 与 `v7x_nap/render.py`，启动命令前显式校验当前工作树包含 `53bc215c8880483142caee0b4d25d57066a0a54f`（`git merge-base --is-ancestor 53bc215c8880483142caee0b4d25d57066a0a54f HEAD`）。
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
- 本轮配置 C Smoke `p3c-retool-smoke-01` 实测验收数据：[`results/p3c-retool-smoke-01/`](results/p3c-retool-smoke-01/)（[`step1_audit_and_metrics.json`](results/p3c-retool-smoke-01/step1_audit_and_metrics.json)、[`val0_gate3_summary.json`](results/p3c-retool-smoke-01/val0_gate3_summary.json)、[`fixed_fixture_shaping_verification.json`](results/p3c-retool-smoke-01/fixed_fixture_shaping_verification.json)、[`raiden_exact_parity_step_0.json`](results/p3c-retool-smoke-01/raiden_exact_parity_step_0.json)）
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

全部运行从共享的 SFT checkpoint（`p2-retool-sft-01/checkpoint/global_step_372/huggingface`）出发，使用同一容器镜像（`verl-tpu@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e`），并将运行代码版本显式固定到包含 launcher W&B Secret 注入与 V1 validation dump `num_turns` 落盘修复的 commit（`CODE_REF=53bc215c8880483142caee0b4d25d57066a0a54f`，短 SHA `53bc215c`；若本 PR 采用 squash merge 合入 `tpu-main`，启动前将 `CODE_REF` 与 `EXPERIMENT_NAME` 后缀同步更新为合入后的 `tpu-main` commit SHA）：

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
   - `TorchTitanEngine.forward_backward_batch`（[`transformer_impl.py` L508-L511](../../../verl/workers/engine/torchtitan/transformer_impl.py#L508-L511)）设置 `sp_size = tensor_parallel_size * context_parallel_size = 4`，因此 `prepare_micro_batches` 的单序列上限为 `4608 × 4 = 18432 = 2048 + 16384`，且 `torchtitan_max_seq_len = 18432`。切到 `MAX_RESPONSE_LENGTH=16384` 不会改变 Trainer 编译的最大序列长度。
2. **Smoke（5 步）与正式运行（25 步 / 150 步）的学习率 Schedule 完全一致（恒定学习率）**：
   - 在 [`run_qwen3_30b_a3b_dapo_torchtitan.sh` L109-L110, L239-L240](run_qwen3_30b_a3b_dapo_torchtitan.sh#L109-L110) 中，`min_lr_factor="${MIN_LR_FACTOR:-1.0}"`，且 `TorchtitanOptimizerConfig` 默认 `lr_warmup_steps=-1`、`lr_warmup_steps_ratio=0.0`。
   - 在 [`transformer_impl.py` L214-L223](../../../verl/workers/engine/torchtitan/transformer_impl.py#L214-L223) 中，`warmup_steps = int(0.0 * total_steps) = 0` 且 `min_lr_factor = 1.0`，因此无论 `TOTAL_TRAINING_STEPS` 设为 `5`、`25` 还是 `150`，每一步的实际学习率恒为常数 `2.0e-6`（已核实 Askhat 的 `fcze3xlx`、`rq1p980n`、`yrcsp4z6`、`unch8eca` 在 WandB 中每步 `actor/lr` 均为恒定 `2.0e-6`）。Smoke 验收时将逐歩核对 `actor/lr == 2.0e-6`。
3. **必须显式设置 `JOBENV_RELEASE_SYNC_BUFFERS=False` 与 `JOBENV_VERIFY_PARITY=all`**：
   - 脚本默认 `VERIFY_PARITY="${VERIFY_PARITY:-off}"`（[L159](run_qwen3_30b_a3b_dapo_torchtitan.sh#L159)），因此必须显式传 `JOBENV_VERIFY_PARITY=all`。
   - 当 `RELEASE_SYNC_BUFFERS` 未设置时，脚本调用 `retool_tpu.py --check-raiden-unbind`（[L152-L158](run_qwen3_30b_a3b_dapo_torchtitan.sh#L152-L158)），该函数仅检查 `hasattr(tpu_sync_torch, "unbind_weights")`。在当前镜像 `verl-tpu@sha256:d92a2b3d...`（`tpu_sync_torch 0.1.2.dev20260331`）中该属性存在，因此自动检测会选 `True` 并触发运行时 bug！所以在启动命令中**必须与 `p3-retool-rl-11` 一样显式传入 `JOBENV_RELEASE_SYNC_BUFFERS=False` 和 `JOBENV_VERL_RAIDEN_RELEASE_BUFFERS=0`**，并在启动后核对 resolved config。
4. **Hydra 覆盖 `enable_thinking` 需使用 `++`**：
   - 因为 [`run_qwen3_30b_a3b_dapo_torchtitan.sh` L222](run_qwen3_30b_a3b_dapo_torchtitan.sh#L222) 已经写了 `+data.apply_chat_template_kwargs.enable_thinking=False`，在命令行末尾通过 `"$@"` 覆盖时必须使用 `++data.apply_chat_template_kwargs.enable_thinking=True`（若用单个 `+` 会触发 Hydra 重复 append 报错）。

### 3.2 配置 C 5 步 Smoke 启动命令与 `DRY_RUN=1` 核验过的 `run_record.json`、`job_spec.json`、`jobset.yaml`

`run_ray_jobset_v7x_nap.sh` 通过环境变量 `JOB_SCRIPT` 指定脚本（默认是 0.6B/30B GRPO 脚本），通过 `CODE_REF` 固定打包进 `code.tar.gz` 的 git commit（默认 `HEAD`，因此必须显式传入 `CODE_REF`），并通过 `WANDB_SECRET_NAME` / `WANDB_SECRET_KEY` 向 `ray-head` 容器注入 Kubernetes Secret（不经过 `JOBENV_*` 或 `job_spec.json`）。
**注意**：`CODE_REF` 只决定打包上传到 GCS `code.tar.gz` 的代码版本，**不会替换当前 shell 执行的本地 `examples/tpu/gke/run_ray_jobset_v7x_nap.sh` 和 `v7x_nap/render.py`**。因此启动命令开头显式执行 `git merge-base --is-ancestor 53bc215c8880483142caee0b4d25d57066a0a54f HEAD && test -z "$(git status --porcelain --untracked-files=no)"` 校验当前工作目录 checkout 已包含修复且没有未提交修改。配置 C 的 5 步 smoke 完整启动命令如下：

```bash
git merge-base --is-ancestor 53bc215c8880483142caee0b4d25d57066a0a54f HEAD && \
test -z "$(git status --porcelain --untracked-files=no)" && \
RUN_NAME=p3c-retool-smoke-01 \
GCS_BUCKET=ubench-logs \
RUN_ROOT=xiaotongyang/retool-v7x/runs \
INPUTS_ROOT=xiaotongyang/retool-v7x/inputs \
RAY_NAMESPACE=priority-dev \
PREWARM=1 \
JOBSET_MAX_RESTARTS=1 \
ALLOW_REQUEUE=0 \
CODE_REF=53bc215c8880483142caee0b4d25d57066a0a54f \
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
JOBENV_EXPERIMENT_NAME="qwen3-30b-a3b-retool-dapo-v7x-16k-think-shaped-16t-smoke-01-sha53bc215c" \
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

1. **生成的 `run_record.json`（确认 `code_sha` 与 `EXPERIMENT_NAME` 后缀 `sha53bc215c` 完全一致，且记录了 Secret 引用名）**：

```json
{
 "code_sha": "53bc215c8880483142caee0b4d25d57066a0a54f",
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
                      value: "53bc215c8880483142caee0b4d25d57066a0a54f"
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
  "EXPERIMENT_NAME": "qwen3-30b-a3b-retool-dapo-v7x-16k-think-shaped-16t-smoke-01-sha53bc215c",
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
  1. **5 步 Smoke（`p3c-retool-smoke-01`）**：虽然 `TEST_FREQ=100`，[`trainer_base.py` L507-L516](../../../verl/trainer/ppo/v1/trainer_base.py#L507-L516) 在 `is_last_step` 时仍会触发评估，因此包含 Step 0 和 Step 5 两次 `n=2` validation，预计总耗时 `~35–45` 分钟，完全在 `priority-dev` 的 120 分钟上限之内。同时从 smoke 实测配置 C 在 `2 × 2x2x4` 上的准确单步耗时（`timing_s/step`）与验证耗时。
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
   用该 run 解析后的配置调用 [`get_custom_reward_fn`](../../../verl/trainer/ppo/reward.py#L50) 并通过 `NaiveRewardManager`（构造带 `__num_turns__` 的 batch）跑固定样例集，确认：
   - 结尾 `\boxed{42}`（正确，`num_turns=2, 6`）→ `1.0, 1.0`
   - 结尾 `\boxed{41}`（错误，`num_turns=2, 3, 4, 6, 10`）→ `-1.0, -0.95, -0.9, -0.8, -0.6`
   - 仅有 `Answer: 42` 无 `\boxed{}`（`num_turns=2`）→ `-1.0`（`strict_box_verify=True` 判错，区别于默认 scorer 的 `+1.0`）
   - `\boxed{42}` 后缀超过 100 字符（`num_turns=2`）→ `-1.0`
3. **实际运行样本与验证集一致性检查（硬门槛，按每个独立 session 的最终 output 对账）**：
   - **验证集（`validation/0.jsonl` 与最后一歩触发的 `validation/5.jsonl`，`expected_n=2`）**：多轮 agent 的一个 session（`{uid}_{session_id}`）可能产生多条 output（`{uid}_{session_id}_{output_index}`），并由最终 output（最大 `output_index`）承载该 session 的 reward，因此验证集 JSONL 行数不写死为 120 行。[`verl/trainer/ppo/v1/trainer_base.py` L1229-L1232](../../../verl/trainer/ppo/v1/trainer_base.py#L1229-L1232) 已按 `session_final_indices` 将每个 session 的 `sample_turns` 写入 `validation/{step}.jsonl` 的 `num_turns` 字段（回归测试见 [`test_rollout_audit_on_cpu.py`](../../../tests/trainer/ppo/v1/test_rollout_audit_on_cpu.py#L722-L1035)）。验收时对 `validation/0.jsonl`（及 `validation/5.jsonl`）使用 [`summarize_validation_dump`](../../../verl/trainer/ppo/v1/rollout_audit.py#L992)（传入 `expected_n=2` 及 60 道预期题目的 `expected_questions`）校验 `complete == True` 且 `score_complete == True`（即 60 道预期题目 × 每题 2 个独立 session = 120 个完整 session），并取每个 session 的**最终 output**（`max(output_index)`），确认其记录的 `row["acc"]` 和 `row["score"]` 与离线调用 `retool_tpu.compute_score(data_source, row["output"], ground_truth, extra_info={"num_turns": row["num_turns"]})` 重算结果完全一致（`validation/{step}.jsonl` 的 `row["output"]` 已由 [`trainer_base.py` L1151](../../../verl/trainer/ppo/v1/trainer_base.py#L1151) 以 `skip_special_tokens=True` 解码）。
   - **训练审计样本（`rollout_audit/{1..5}.jsonl`，每步 8 个 session、共 40 个独立 session）**：按 `record["session"]` 分组并取每个 session 的**最终 output**（`max(output_index)`）。由于 `rollout_audit.py` 生成的 `record["response_text"]` 使用了 `skip_special_tokens=False`（保留了末尾 `<|im_end|>` 等特殊 token），而运行时 [`NaiveRewardManager`](../../../verl/workers/reward_manager/naive.py#L119) 使用 `tokenizer.decode(valid_response_ids, skip_special_tokens=True)` 解码，因此离线重算必须先对 `record["response_ids"]` 执行完全相同的 `tokenizer.decode(record["response_ids"], skip_special_tokens=True)`，再调用 `retool_tpu.compute_score(..., extra_info={"num_turns": record["num_turns"]})`，确认重算的 `acc`、`pred`、`score` 与日志中的 `record["reward_extra_info"]["acc"]`、`record["reward_extra_info"]["pred"]` 及 `record["score"]` 100% 一致。
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
     - Smoke：`qwen3-30b-a3b-retool-dapo-v7x-16k-think-shaped-16t-smoke-01-sha53bc215c`
     - 25 步：`qwen3-30b-a3b-retool-dapo-v7x-16k-think-shaped-16t-25step-01-sha<short_sha>`
2. **模板与 Launcher 注入实现（不落盘到代码库、`config.yaml`、`job_spec.json` 或 `driver.log`）**：
   - **容器层注入**：在 [`ray-jobset-v7x-nap.yaml`](../gke/ray-jobset-v7x-nap.yaml#L139)、[`v7x_nap/render.py`](../gke/v7x_nap/render.py#L70-L92) 和 [`run_ray_jobset_v7x_nap.sh`](../gke/run_ray_jobset_v7x_nap.sh#L33-L34) 中新增 `WANDB_SECRET_NAME`（默认空）与 `WANDB_SECRET_KEY`（默认 `WANDB_API_KEY`）。当设置 `WANDB_SECRET_NAME=xiaotongyang-wandb` 时，渲染器向 `ray-head` 容器注入 `valueFrom.secretKeyRef: {name: "xiaotongyang-wandb", key: "WANDB_API_KEY"}`；`run_record.json` 仅记录 Secret 名称与键名（`wandb_secret_name` / `wandb_secret_key`），**绝不通过 `JOBENV_WANDB_API_KEY` 写入 `job_spec.json`**。
   - **提交前集群强校验**：[`run_ray_jobset_v7x_nap.sh`](../gke/run_ray_jobset_v7x_nap.sh#L179-L192) 在渲染 manifest 后、提交或 dry-run 前，执行 `kubectl get secret "${WANDB_SECRET_NAME}" -n "${RAY_NAMESPACE}" -o jsonpath="{.data.${WANDB_SECRET_KEY}}"`，若目标命名空间不存在该 Secret 或对应 key 为空则立即报错终止。
   - **Ray Driver / TaskRunner 传递链路**：`ray-head` 容器环境变量中的 `WANDB_API_KEY` 由 `head_supervisor.py` 启动的本地 Ray head 继承；Ray job driver（`run_qwen3_30b_a3b_dapo_torchtitan.sh` → `verl.trainer.main_ppo`）在 head 节点运行并直接继承 `WANDB_API_KEY`（不在 shell 中赋值，因此 `set -x` 不会在 `driver.log` 中打印密钥）；随后 [`PlatformTPU.get_ray_init_kwargs()`](../../../verl/plugin/platform/platform_tpu.py#L616-L630) 通过 `runner_env_vars` 将 `WANDB_API_KEY` 和 `WANDB_ENTITY` 转发给 `TaskRunner` Actor（[`main_ppo.py` L111-L125](../../../verl/trainer/main_ppo.py#L111-L125)），由 [`Tracking`](../../../verl/utils/tracking.py#L76-L86) 完成 `wandb.init(project="verl-torchtpu-grpo-test", entity="google-trellis", ...)`。同时 [`pod_env_snapshot.py`](../gke/v7x_nap/pod_env_snapshot.py#L31) 已对含 `_KEY` 的变量自动脱敏为 `<redacted>`。
3. **集群 Secret 与 W&B 目标 Project 写权限实测结果**：
   - 已在 `gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap` 的 `priority-dev` 与 `default` 命名空间创建 `Secret/xiaotongyang-wandb`（key 为 `WANDB_API_KEY`）。
   - 已从集群 `priority-dev` 命名空间读取 `Secret/xiaotongyang-wandb` 中的 `WANDB_API_KEY`，直接向 `https://api.wandb.ai/graphql` 发起鉴权与项目权限核验，返回：
     - `viewer`：`{"username": "xiaotongyang", "email": "xiaotongyang@google.com", "entity": "google-trellis"}`
     - `project(name: "verl-torchtpu-grpo-test", entityName: "google-trellis")`：`{"id": "UHJvamVjdDp2MTp2ZXJsLXRvcmNodHB1LWdycG8tdGVzdDpnb29nbGUtdHJlbGxpcw==", "name": "verl-torchtpu-grpo-test", "readOnly": false, "access": "PRIVATE"}`
   - 确认该集群 Secret 凭证有效且对 `google-trellis/verl-torchtpu-grpo-test` 具备写权限（`readOnly: false`）。

---

## 6. `p3c-retool-smoke-01`（32 卡）早期部分验证结果与 64 卡（`r-b0f80`）正式运行放行方案

我们在 `bodaborg-tpu7x-nap` 集群（`r-db36b` 队列，`2 × 2x2x4` = 32 chips，1 Trainer Slice + 1 Rollout Slice）上以固定代码版本 `53bc215c8880483142caee0b4d25d57066a0a54f` 启动并运行了 `p3c-retool-smoke-01`：
- **GCS 产物目录**：`gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3c-retool-smoke-01/`
- **WandB Run**：[`google-trellis/verl-torchtpu-grpo-test/runs/uinjk8rs`](https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/uinjk8rs)（`qwen3-30b-a3b-retool-dapo-v7x-16k-think-shaped-16t-smoke-01-sha53bc215c`）
- **验证完成度定位（早期部分验证，尚未跑完原计划全部 5 步门槛）**：
  - **已完成项**：Step 0 验证（`validation/0.jsonl`，`n=2` 共 120 个 session）、Step 0 单副本逐元素 Raiden 校验（`32 × 435 = 13,920` 分片，`0` mismatch）、Step 1 完整训练与 8 条 session 审计（`rollout_audit/1.jsonl`）、Step 1 单副本分布式范数校验、以及 Step 2 前向/反向更新（`actor/grad_norm = 0.4596`）。
  - **未完成项与中断原因**：在 `09:25:08Z`（Step 2 Raiden 权重同步阶段）因 `r-db36b` 动态预约窗口到期被集群回收（`Evicted=True(Preempted), Preempted=True(InCohortReclamation)`），**Steps 3–5 未运行**。因此原计划 §4.1 要求的 **5 次权重同步检查、40 条训练审计轨迹（8 × 5 步）以及末次 Step 5 验证（`validation/5.jsonl`）尚未在 32 卡上跑完**。
  - **后续放行策略**：跳过额外的独立 32 卡重跑，直接转入 `r-b0f80` 预约队列的 **64 卡正式运行（`p3c-retool-25s-01`，从 Step 0 起跑）**，并将 **64 卡运行的前 3 步（Step 0 + Steps 1–3）作为在线放行门槛（In-Flight Release Gate，见 §6.5）**：若 Steps 0–3 全部通过三副本同步与训练审计检查，则同一次 run 连续运行至 Step 25；若任一检查未通过则立即停止排查。

以下为 `p3c-retool-smoke-01`（32 卡）已完成的实测核验项与后续 64 卡放行规范：

### 6.1 Gate 1：Resolved Config 与 W&B / Secret 注入核验（32 卡已验证通过）

从 `head.log` 打印的 Hydra resolved config 及 `gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3c-retool-smoke-01/env/pod_env.json` 逐项核实：
- `reward.custom_reward_function.path == "pkg://examples.tpu.retool.retool_tpu"` 且 `name == "compute_score"`；
- `data.apply_chat_template_kwargs.enable_thinking == True`；
- `data.max_response_length == 16384`，`actor_rollout_ref.rollout.max_model_len == 18432`；
- `actor_rollout_ref.rollout.multi_turn.max_user_turns == 16` 且 `max_assistant_turns == 16`；
- `actor_rollout_ref.rollout.n == 16`，`actor_rollout_ref.actor.optim.lr == 2e-6`（Step 1 实测 `actor/lr == 2.0e-6`）；
- `actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.release_buffers_after_sync == False` 且 `verify_parity == "all"`；
- TorchTitan 日志确认：`TorchTitan grad clipping uses training.max_norm=1.0`；
- `pod_env.json` 中 `WANDB_API_KEY` 自动脱敏为 `"<redacted>"`，`job_spec.json` 不含明文密钥，W&B Run [`uinjk8rs`](https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/uinjk8rs) 在线实时接收指标。

### 6.2 Gate 2：固定样例经 `get_custom_reward_fn` + `NaiveRewardManager` 完整链路确定性检查（9 / 9 全部通过）

按 §4.1 Gate 2 要求，使用 [`get_custom_reward_fn`](../../../verl/trainer/ppo/reward.py#L50) 加载 `pkg://examples.tpu.retool.retool_tpu:compute_score`，并构造包含 `prompts`、`responses`、`attention_mask` 及 `non_tensor_batch["__num_turns__"]` 的 `DataProto` 批次，完整经过 [`NaiveRewardManager.__call__`](../../../verl/workers/reward_manager/naive.py#L90-L175)（验证 `non_tensor_batch["__num_turns__"]` → `extra_info["num_turns"]` 注入、`reward_tensor[i, valid_response_length - 1]` 末位有效 token 落点、非末位 token 全为 `0.0`、以及 `reward_extra_info` 字典输出），9 组固定样例全部通过（结果落盘于 [`results/p3c-retool-smoke-01/fixed_fixture_shaping_verification.json`](results/p3c-retool-smoke-01/fixed_fixture_shaping_verification.json)）：
- 结尾 `\boxed{42}`（正确，`num_turns=2, 8`）→ `reward_tensor` 末位有效 token 及 `reward_extra_info["score"]` 均为 `1.0, 1.0`（`acc = True`）
- 结尾 `\boxed{41}`（错误，`num_turns=2, 3, 4, 6, 10`）→ `reward_tensor` 末位有效 token 及 `reward_extra_info["score"]` 均为 `-1.0, -0.95, -0.9, -0.8, -0.6`（`acc = False`）
- 仅有 `Answer: 42` 无 `\boxed{}`（`num_turns=2`）→ `-1.0`（`strict_box_verify=True` 判错）
- `\boxed{42}` 后缀超过 100 字符（`num_turns=2`）→ `-1.0`（`acc = False`）

### 6.3 Gate 3（部分完成）：`validation/0.jsonl` 与 `rollout_audit/1.jsonl` 离线逐 Session 复算对账

1. **Step 0 验证集（`validation/0.jsonl`，`expected_n=2`）完整性与离线对账**：
   - 使用 [`summarize_validation_dump`](../../../verl/trainer/ppo/v1/rollout_audit.py#L992)（传入 `expected_n=2` 及 AIME 2024 / 2025 各 30 道预期题目的 `question_id`）校验通过：**`complete == True` 且 `score_complete == True`**（共 60 道题 × 2 个独立 session = **120 个完整 session**，`missing_num_turns = 0`）。
   - 对全部 120 个 session 调用 `retool_tpu.compute_score(data_source, row["output"], row["gts"], extra_info={"num_turns": row["num_turns"]})` 离线复算，与 `validation/0.jsonl` 中记录的 `row["acc"]` 和 `row["score"]` **`120 / 120` 完全一致（`0` 处 mismatch）**。
   - **78 / 120 个 session**（`aime_2024` 35 条、`aime_2025` 43 条，即全部答错样本）均触发了多轮工具调用（`num_turns > 2`），命中 ReTool shaped reward（`[-0.9, -0.6]`）；120 个 session 的 `num_turns` 分布为 `{4: 44, 6: 20, 8: 14, 10: 13, 12: 12, 14: 6, 16: 2, 18: 3, 20: 2, 22: 2, 24: 1, 28: 1}`（`min=4, mean=8.23, max=28`）。
   - **Step 0 小样本（`n=2`）初步验证结果与历史参考对比**：
     - **更正说明**：与 §1.1 及 [`askhat_runs_summary.json`](results/askhat-wandb-audit/askhat_runs_summary.json) 对齐——Askhat Regime 2 的 `yjowd03u`（`16K, thinking=True, n=8`）在 **Step 0** 的记录值为 **`48.75% / 30.83%`**（`mean_score = +0.10625 / -0.22125`）；`ci0k54pl` 从 `fcze3xlx` 的 `global_step_20` 续训，其首个记录的 validation 位于 **Step 25**（`n=8`），记录值为 **`58.33% / 38.75%`**（`mean_score = +0.249167 / -0.075`）；Askhat Regime 1（`8K, thinking=False, n=30`）的 `lb5cdzuq` Step 0 为 **`25.44% / 14.44%`**（`mean_score = -0.491111 / -0.711111`；更早一次独立 Step 0 评估为 `22.44% / 14.17%`）。
     - 本轮 `p3c-retool-smoke-01` 的 Step 0 仅采样 **`n=2`（每数据集 60 个样本）**，样本量较小、抽样方差较大，仅作为 `16K + thinking=True + shaped reward` 管线生效的初步冒烟参考，**不能单凭 `n=2` 结论声称与 Askhat `n=8` / `n=30` 完全对齐或独立证明显著提升**；正式基线对比以 64 卡运行的 Step 0 / Step 25 **`n=30`（每数据集 900 个样本，同时导出前 8 个 session 的 `mean@8`）** 为准。

| 数据集 | 旧 v7x 基线 `p3-retool-rl-11` Step 0 (`8K`, `thinking=False`, `n=30`) | Askhat `lb5cdzuq` Step 0 (`8K`, `thinking=False`, `n=30`) | Askhat `yjowd03u` Step 0 (`16K`, `thinking=True`, `n=8`) | Askhat `ci0k54pl` **Step 25** (`from step 20`, `16K`, `thinking=True`, `n=8`) | **本轮 `p3c-retool-smoke-01` Step 0 (`16K`, `thinking=True`, `n=2` 小样本初测) Strict / Default Scorer** | **本轮 Step 0 (`n=2`) Mean Shaped Score** |
| --- | --- | --- | --- | --- | --- | --- |
| **AIME 2024** | `25.56%` (`-0.4889`) | `25.44%` (`-0.491111`) | **`48.75%`** (`+0.10625`) | **`58.33%`** (`+0.249167`) | **`41.67%` (`25 / 60`)** | **`-0.0200`** |
| **AIME 2025** | `17.33%` (`-0.6533`) | `14.44%` (`-0.711111`) | **`30.83%`** (`-0.22125`) | **`38.75%`** (`-0.075`) | **`28.33%` (`17 / 60`)** | **`-0.2383`** |

2. **Step 1 训练审计样本（`rollout_audit/1.jsonl`，已完成 8 / 40 个计划 session）对账**：
   - 对 Step 1 的 8 个 session 的 `response_ids` 执行 `tokenizer.decode(record["response_ids"], skip_special_tokens=True)` 后调用 `retool_tpu.compute_score(..., extra_info={"num_turns": record["num_turns"]})`，重算的 `acc`、`pred`、`score`（`abs_tol=1e-6`）与日志记录 **`8 / 8` 100% 一致**，且 **`0` 条 `response_mask` 违规**（注：Steps 2–5 的其余 32 条审计样本因作业在 Step 2 权重同步阶段被回收而未生成）：

| Session Key | `num_turns` | `acc` | `pred` | 记录 `score` | 离线重算 `score` | Response Tokens | `response_mask == 1` Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `02f06182-..._13` | `4` | `False` | `""` | `-0.9` | `-0.9` | `489` | `471` |
| `21f5ddda-..._0` | `4` | `False` | `""` | `-0.9` | `-0.9` | `3,077` | `3,038` |
| `2835a9e4-..._2` | `6` | `False` | `""` | `-0.8` | `-0.8` | `1,788` | `1,727` |
| `2835a9e4-..._12` | `4` | `True` | `"36"` | `+1.0` | `+1.0` | `3,310` | `3,280` |
| `89417b37-..._7` | `8` | `True` | `"171"` | `+1.0` | `+1.0` | `1,062` | `911` |
| `92061a0a-..._8` | `10` | `False` | `""` | `-0.6` | `-0.6` | `1,053` | `972` |
| `bf8dc400-..._11` | `4` | `False` | `""` | `-0.9` | `-0.9` | `12,638` | `12,622` |
| `f1c30eec-..._7` | `4` | `True` | `"55"` | `+1.0` | `+1.0` | `6,686` | `6,671` |

### 6.4 Gate 4（部分完成）：单副本（32 卡）Raiden 权重同步与 Step 1–2 训练指标

- **Step 0 单副本逐元素 float32 校验（`env/raiden_exact_parity_step_0.json`）**：`num_results = 32`、`expected_ranks = 32`、`expected_workers_per_replica = [32]`、`checked_per_rank = [435]`（合计 `32 × 435 = 13,920` 个张量分片），**`num_mismatched = 0`、`num_unresolved = 0`、`coverage_errors = []`**，耗时 `1.7911s`（`H2D DMA = 0.0993s`）。
- **Step 1 单副本分布式范数校验（已完成 1 / 5 次计划同步校验）**：`[RAIDEN PARITY VERIFIED | Step 1] 100% DISTRIBUTED NORM PARITY CONFIRMED!`
  - `Global L1 Norm = 576,240,144.43`（`Trainer = 576,240,144.62, delta = 0.1904`）
  - `Global L2 Norm = 4,206.4321`（`Trainer = 4,206.4329, delta = 0.000793`）
  - `Total Parameters = 30,532,122,624` across `579` tensors，端到端同步耗时 `15.5899s`。
  - Step 2 在完成 actor 反向传播与梯度裁剪（`grad_norm = 0.4596`）后、进入 Raiden 权重同步时被预约回收，因此 Steps 2–5 的同步校验未完成。
- **Step 1 关键训练指标（已同步至 WandB `uinjk8rs`）**：
  - `critic/score/mean = -0.31992`
  - `response_length/mean = 1,843.52`（`max = 16,384.0`，`clip_ratio = 0.0254`）
  - `training/num_turns/mean = 6.21875`（`min = 2, max = 16`）
  - `actor/grad_norm = 0.15484`（Step 2 `grad_norm = 0.4596`，无非有限梯度或跳过更新），`actor/ppo_kl = 0.01182`，`actor/pg_clipfrac = 0.02232`，`actor/lr = 2.0e-6`。

---

### 6.5 64 Chips（`r-b0f80`：1 Trainer Slice + 3 Rollout Slices）启动规范、超时设置与前 3 步在线放行门槛

#### 1. 为什么 64 卡启动命令必须显式指定 `JOBENV_ROLLOUT_TP=32` 与 `JOBENV_ROLLOUT_EP=32`

在 [`examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh` L135-L153](run_qwen3_30b_a3b_dapo_torchtitan.sh#L135-L153) 中：
```bash
NNODES_TRAINER="${NNODES_TRAINER:-2}"
NNODES_ROLLOUT="${NNODES_ROLLOUT:-2}"
N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-8}"
N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-8}"
TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))
ROLLOUT_TP="${ROLLOUT_TP:-${TOTAL_ROLLOUT_CHIPS}}"
ROLLOUT_EP="${ROLLOUT_EP:-${ROLLOUT_TP}}"
```
当 64 卡配置为 `1 Trainer Slice (4 nodes × 8 chips = 32 logical chips / 16 physical chips) + 3 Rollout Slices (12 nodes × 8 chips = 96 logical chips / 48 physical chips)` 时，`TOTAL_ROLLOUT_CHIPS` 为 `12 × 8 = 96`。若只修改 `JOBENV_NNODES_ROLLOUT=12` 而不显式覆盖 `JOBENV_ROLLOUT_TP` 和 `JOBENV_ROLLOUT_EP`，脚本会默认设置 `tensor_model_parallel_size=96`、`expert_parallel_size=96`（尝试跨 3 个 slice 组建单个 96-way TP replica，而非预期的 **3 个独立 `TP=32, EP=32` Rollout Replica**）。

因此，64 卡启动命令**必须同时显式传入**：
```bash
TRAINER_SLICES=1 \
ROLLOUT_SLICES=3 \
JOBENV_NNODES_TRAINER=4 \
JOBENV_N_CHIPS_TRAINER=8 \
JOBENV_NNODES_ROLLOUT=12 \
JOBENV_N_CHIPS_ROLLOUT=8 \
JOBENV_ROLLOUT_TP=32 \
JOBENV_ROLLOUT_EP=32
```

#### 2. 为什么必须显式设置 `RUN_LIMIT_MIN` 与 `DRIVER_TIMEOUT_MIN`

在 [`examples/tpu/gke/run_ray_jobset_v7x_nap.sh` L137-L145](../gke/run_ray_jobset_v7x_nap.sh#L137-L145) 中：
```bash
if [[ "${RAY_NAMESPACE}" == "default" ]]; then
  DEFAULT_RUN_LIMIT_MIN=180
else
  DEFAULT_RUN_LIMIT_MIN=120
fi
RUN_LIMIT_MIN="${RUN_LIMIT_MIN:-${DEFAULT_RUN_LIMIT_MIN}}"
DRIVER_TIMEOUT_MIN="${DRIVER_TIMEOUT_MIN:-$((RUN_LIMIT_MIN - 10))}"
```
对于 `RAY_NAMESPACE=r-b0f80`（6 小时预约窗口：`8:00 AM – 2:00 PM PST`，共 `360` 分钟），脚本默认仍会落入 `else` 分支并设置 `RUN_LIMIT_MIN=120`、`DRIVER_TIMEOUT_MIN=110`（1 小时 50 分钟后由 `head_supervisor.py` 主动终止 driver！）。
因此在提交时，**必须根据 `r-b0f80` 预约窗口的实际剩余分钟数显式设置 `RUN_LIMIT_MIN` 和 `DRIVER_TIMEOUT_MIN`**（例如 8:00 AM 准时启动时设为 `RUN_LIMIT_MIN=355`、`DRIVER_TIMEOUT_MIN=345`，为结束清理预留 buffer）。

#### 3. 镜像 Digest 核实与 `r-b0f80` 动态命名空间 W&B Secret 准备

1. **镜像 Digest 核实（统一使用 `@sha256:d92a2b3d...`）**：
   - 从 `p3c-retool-smoke-01` 运行时的 9 个 Pod 状态快照（`.pods.json` 中 `status.containerStatuses[0].imageID`）与仓库 [`tests/special_tpu/gke/tpu-ci-image.env`](../../../tests/special_tpu/gke/tpu-ci-image.env) 交叉核实，标签 `us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20261006-tsync1006` 对应的真实镜像摘要为 **`sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e`**。
   - 64 卡命令显式固定 `IMAGE="us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e"`，确保与 32 卡 smoke 使用完全相同的容器镜像与软件栈。
2. **动态预约命名空间 `r-b0f80` 的 W&B Secret 创建与预检流程**：
   - 在 `bodaborg-tpu7x-nap` 集群上，`r-*` 预约命名空间及其 `LocalQueue`（`multislice-queue`）由 `calendar-schedule-daemon` 在预约开始时刻（`2026-11-04T16:00:00Z` / `8:00 AM PST`）动态创建，预约生效前集群中尚不存在 `namespace/r-b0f80`。
   - 因此：
     - **预约生效前（Pre-window Dry-Run）**：使用 `DRY_RUN=1 RAY_NAMESPACE=priority-dev` 对下列 64 卡完整字面命令执行预渲染与 API server dry-run，核验 `run_record.json`、`job_spec.json`、`jobset.yaml` 和 `prewarm.yaml`；
     - **预约生效时（`8:00 AM PST`）**：确认 `namespace/r-b0f80` 与 `LocalQueue/multislice-queue` 就绪后，首先在 `r-b0f80` 创建同名 `Secret/xiaotongyang-wandb`（含 `WANDB_API_KEY`），随后先执行一次 `DRY_RUN=1 RAY_NAMESPACE=r-b0f80` 确认输出 `verified Secret xiaotongyang-wandb[WANDB_API_KEY] exists in namespace r-b0f80` 且 `jobset.yaml` 通过 `r-b0f80` server-side dry-run，再以 `DRY_RUN=0` 正式提交。

#### 4. 64 卡正式运行（`p3c-retool-25s-01`）完整字面启动命令与 `DRY_RUN=1` 同源产物

> **说明**：
> - 通过 `JOB_SCRIPT="examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh"` 指定训练入口脚本，位置参数直接传 Hydra 覆盖项（不传 `-- bash ...`）；
> - 显式指定 `GCS_BUCKET=ubench-logs`、`RUN_ROOT=xiaotongyang/retool-v7x/runs`、`INPUTS_ROOT=xiaotongyang/retool-v7x/inputs`，并用 `@RUN_DIR@` 占位符生成产物目录；
> - 使用 launcher 实际识别的 `TPU_TOPOLOGY="2x2x4"`（而非 `TOPOLOGY`）和 `KEEP_JOBSET=1`（防止客户端进程退出时 `EXIT` trap 误删运行中的 JobSet；注：`run_ray_jobset_v7x_nap.sh` 本身为前台监控脚本，不存在 `WAIT_FOR_COMPLETION` 变量）。

```bash
git merge-base --is-ancestor 53bc215c8880483142caee0b4d25d57066a0a54f HEAD && \
test -z "$(git status --porcelain --untracked-files=no)" && \
DRY_RUN=0 \
KEEP_JOBSET=1 \
RUN_NAME="p3c-retool-25s-01" \
GCS_BUCKET="ubench-logs" \
RUN_ROOT="xiaotongyang/retool-v7x/runs" \
INPUTS_ROOT="xiaotongyang/retool-v7x/inputs" \
RAY_NAMESPACE="r-b0f80" \
KUEUE_QUEUE="multislice-queue" \
PRIORITY_CLASS="medium" \
PREWARM=1 \
JOBSET_MAX_RESTARTS=1 \
ALLOW_REQUEUE=0 \
RUN_LIMIT_MIN=355 \
DRIVER_TIMEOUT_MIN=345 \
TPU_TOPOLOGY="2x2x4" \
TRAINER_SLICES=1 \
ROLLOUT_SLICES=3 \
CODE_REF="53bc215c8880483142caee0b4d25d57066a0a54f" \
WANDB_SECRET_NAME="xiaotongyang-wandb" \
WANDB_SECRET_KEY="WANDB_API_KEY" \
IMAGE="us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e" \
JOB_SCRIPT="examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh" \
JOBENV_LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32" \
JOBENV_MODEL_PATH="/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x" \
JOBENV_MODEL_SRC_PATH="/data/xiaotongyang/retool-v7x/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface" \
JOBENV_NNODES_TRAINER="4" \
JOBENV_N_CHIPS_TRAINER="8" \
JOBENV_NNODES_ROLLOUT="12" \
JOBENV_N_CHIPS_ROLLOUT="8" \
JOBENV_ROLLOUT_TP="32" \
JOBENV_ROLLOUT_EP="32" \
JOBENV_TRAIN_BATCH_SIZE="64" \
JOBENV_PPO_MINI_BATCH_SIZE="64" \
JOBENV_MAX_RESPONSE_LENGTH="16384" \
JOBENV_MAX_TURNS="16" \
JOBENV_N_RESP_PER_PROMPT="16" \
JOBENV_N_RESP_PER_PROMPT_VAL="30" \
JOBENV_ACTOR_LR="2e-6" \
JOBENV_NUM_WARMUP_BATCHES="2" \
JOBENV_RELEASE_SYNC_BUFFERS="False" \
JOBENV_VERL_RAIDEN_RELEASE_BUFFERS="0" \
JOBENV_VERIFY_PARITY="all" \
JOBENV_VAL_BEFORE_TRAIN="True" \
JOBENV_TEST_FREQ="25" \
JOBENV_TOTAL_TRAINING_STEPS="25" \
JOBENV_SAVE_FREQ="5" \
JOBENV_LOGGER_LIST="['console','tensorboard','wandb']" \
JOBENV_PROJECT_NAME="verl-torchtpu-grpo-test" \
JOBENV_WANDB_ENTITY="google-trellis" \
JOBENV_EXPERIMENT_NAME="qwen3-30b-a3b-retool-dapo-v7x-64c-16k-think-shaped-16t-25s-01-sha53bc215c" \
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

我们对上述**完全相同的字面命令**执行了 `DRY_RUN=1`（预检时仅将 `RAY_NAMESPACE` 指向已存在的 `priority-dev` 以通过 API server admission 校验），生成的同源 `run_record.json`、`job_spec.json` 与 `jobset.yaml` 如下：
- **JobSet Slices（共 4 个 `2x2x4` slice = 16 hosts / 64 physical chips）**：
  - `trainer-0`（`tpu-group-0`，4 pods × 4 chips）
  - `rollout-0`、`rollout-1`、`rollout-2`（`tpu-group-1`、`tpu-group-2`、`tpu-group-3`，共 12 pods × 4 chips）
- **该字面命令生成的 `run_record.json`**：
```json
{
 "code_sha": "53bc215c8880483142caee0b4d25d57066a0a54f",
 "created_at": "2026-11-04T11:37:13Z",
 "driver_timeout_s": "20700",
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
 "rollout_slices": "3",
 "run_dir": "/data/xiaotongyang/retool-v7x/runs/p3c-retool-25s-01",
 "run_name": "p3c-retool-25s-01",
 "run_owner": "xiaotongyang",
 "tpu_topology": "2x2x4",
 "trainer_slices": "1",
 "wandb_secret_key": "WANDB_API_KEY",
 "wandb_secret_name": "xiaotongyang-wandb"
}
```
- **该字面命令生成的 64 卡 `job_spec.json`**：
```json
{
 "entrypoint": [
  "bash",
  "examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh",
  "trainer.rollout_data_dir=/data/xiaotongyang/retool-v7x/runs/p3c-retool-25s-01/rollout",
  "trainer.validation_data_dir=/data/xiaotongyang/retool-v7x/runs/p3c-retool-25s-01/validation",
  "+trainer.rollout_audit_dir=/data/xiaotongyang/retool-v7x/runs/p3c-retool-25s-01/rollout_audit",
  "+trainer.rollout_audit_sessions=8",
  "reward.custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu",
  "reward.custom_reward_function.name=compute_score",
  "++data.apply_chat_template_kwargs.enable_thinking=True"
 ],
 "env_vars": {
  "ACTOR_LR": "2e-6",
  "EXPERIMENT_NAME": "qwen3-30b-a3b-retool-dapo-v7x-64c-16k-think-shaped-16t-25s-01-sha53bc215c",
  "LIBTPU_INIT_ARGS": "--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32",
  "LOGGER_LIST": "['console','tensorboard','wandb']",
  "MAX_RESPONSE_LENGTH": "16384",
  "MAX_TURNS": "16",
  "MODEL_PATH": "/dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x",
  "MODEL_SRC_PATH": "/data/xiaotongyang/retool-v7x/runs/p2-retool-sft-01/checkpoint/global_step_372/huggingface",
  "NNODES_ROLLOUT": "12",
  "NNODES_TRAINER": "4",
  "NUM_WARMUP_BATCHES": "2",
  "N_CHIPS_ROLLOUT": "8",
  "N_CHIPS_TRAINER": "8",
  "N_RESP_PER_PROMPT": "16",
  "N_RESP_PER_PROMPT_VAL": "30",
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
  "ROLLOUT_EP": "32",
  "ROLLOUT_TP": "32",
  "SAVE_FREQ": "5",
  "TENSORBOARD_DIR": "/data/xiaotongyang/retool-v7x/runs/p3c-retool-25s-01/tensorboard",
  "TEST_FREQ": "25",
  "TOTAL_TRAINING_STEPS": "25",
  "TPU_ACCELERATOR_TYPE": "tpu7x",
  "TRAIN_BATCH_SIZE": "64",
  "VAL_BEFORE_TRAIN": "True",
  "VERIFY_PARITY": "all",
  "VERL_NAP_RUN_DIR": "/data/xiaotongyang/retool-v7x/runs/p3c-retool-25s-01",
  "VERL_PLATFORM": "tpu",
  "VERL_RAIDEN_RELEASE_BUFFERS": "0",
  "WANDB_ENTITY": "google-trellis"
 }
}
```

#### 5. 64 卡正式运行前 3 步（Steps 0–3）早期监控与放行检查规范

此前 32 卡（单 rollout replica）的验证结果不能直接证明 64 卡（三 rollout replica）同步通过。因此在 `p3c-retool-25s-01` 启动后，对 **Step 0 及 Steps 1–3** 执行以下多副本检查：

1. **进程内自动校验与人工早期监控的真实边界（不虚称已入库自动暂停屏障）**：
   - **已实现的进程内自动中止项（`CODE_REF=53bc215c` 内置）**：
     1. **Step 0 逐元素与实际创建的 replicas 完整性校验**：[`RaidenParityCheck.report_exact`（`raiden_checkpoint_engine.py` L1480–L1489）](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py#L1480-L1489) 根据运行时实际创建的 `manager.replicas` 动态计算预期 rank 数（`expected_workers_per_replica` 与 `expected_ranks`），并在进程内强制核对实际 replicas 的完整性（`num_results == expected_ranks`、`replica_rank_pairs` 无缺失、`num_mismatched == 0`、`num_unresolved == 0`、`coverage_errors == []`）；若不满足直接抛出 `RaidenParityError` 中止训练。至于实际创建的拓扑是否确实等于 **`expected_ranks == 96` 且 `expected_workers_per_replica == [32, 32, 32]`**，由下方第 3 项对 `raiden_exact_parity_step_0.json` 的人工核验确认。
     2. **Step 1+ 每步 `update_weights` 的聚合 `numel` 与 `L1` 范数校验**：[`_verify_weight_parity` / `_compare_replica_norms`（L806–L919）](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py#L806-L919) 对每个 replica 要求返回的 `sampler_workers` 非空，并自动校验逐张量及全局的 **`numel`（`delta_numel == 0`）与 `L1` 范数（`rel_tol = 1e-3`）**；若不满足直接抛出 `RaidenParityError`。
   - **未内置自动阻塞屏障、需人工/Agent 早期跟跑监控并手动停止的项**：
     - 当前 `CODE_REF=53bc215c` 代码库中**并未合入**独立的后台 watcher 守护脚本，Trainer 也没有在 Step 3 结束后暂停等待外部离线对账的阻塞机制；若外部检查存在延迟，训练进程可能已进入 Step 4。
     - 因此，下述关于 Resolved Config、`head.log` 中 `Global L2 Norm` delta、`validation/0.jsonl` 与 `rollout_audit/{1..3}.jsonl` 离线 reward/mask 对账均定位为**人工/Agent 早期跟跑监控（Manual Early-Stage Monitoring）**：在作业启动后由操作员/Agent 紧盯 Step 0–3 产物落盘并立即执行离线核验，**一旦发现任何异常，尽快手动执行 `kubectl delete jobset p3c-retool-25s-01 -n r-b0f80` 停止作业**；全部通过则保持作业继续运行至 Step 25。
2. **Multi-Replica 拓扑与 Resolved Config 检查（Step 0 启动时）**：
   - 确认实际创建了 **3 个独立的 vLLM Rollout Replica（`sampler0`、`sampler1`、`sampler2`）**，每个 replica 为 `TP=32, EP=32`，合计 **96 个 rollout worker ranks**。
   - 确认 Hydra resolved config 与 §6.1 所有字段完全一致。
3. **Step 0 三副本 Raiden 逐元素同步检查（按 `env/raiden_exact_parity_step_0.json` 真实 Schema 校验）**：
   - 校验顶层字段：
     - `step == 0`
     - `num_results == 96` 且 `expected_ranks == 96`
     - `expected_workers_per_replica == [32, 32, 32]`
     - `expected_checked_per_rank == 435` 且 `checked_per_rank == [435]`（或实际每 rank 分片数 `S`，满足所有 rank 一致）
     - `replica_rank_pairs` 精确覆盖全部 96 个 `[replica_idx, rank_idx]` 组合（`replica_idx ∈ {0, 1, 2}`、`rank_idx ∈ {0..31}`，无缺失或重复）
     - **`num_mismatched == 0`、`num_unresolved == 0`、`coverage_errors == []`**
   - 校验逐 rank 明细列表 `ranks`（`len(ranks) == 96`）：
     - 每个元素 `r` 均满足 `r["checked"] == r["expected_checked"] == 435`、`r["num_mismatched"] == 0`、`r["unresolved"] == []`、`r["all_mismatched"] == []`；
     - 全部 96 个 rank 累计检查分片数 **`sum(r["checked"] for r in ranks) == 41,760`**（即 `96 × 435`）。
4. **Steps 1–3 三副本训练后分布式范数与梯度检查（区分代码自动判定范围与人工日志核验）**：
   - **代码自动判定边界说明**：[`raiden_checkpoint_engine.py` L806–L919](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py#L806-L919) 在 Step 1+ 对每个 replica 仅检查返回的 `sampler_workers` 列表非空（未核验 worker rank 的身份与唯一性，未显式断言 `len(sampler_workers) == 32`），且自动 pass 条件仅校验逐张量与全局的 **`numel`（`delta_numel == 0`）和 `L1` 范数（`delta_l1 <= 1e-3 * max(|L1|, 1.0)`）**；`Global L2 Norm` 仅打印在日志中、不参与代码自动判定。因此，**聚合 `numel` / `L1` 能检测造成聚合值不一致的异常，但不能单独证明 32 个不同 rank 集合的完整性与唯一性**。
   - **人工补充核验方法**：
     1. 核对 `head.log` 在 Steps 1、2、3 每一步均完整打印了 **3 个 replica（`[RAIDEN PARITY VERIFIED | Step k | Replica 0]`、`Replica 1`、`Replica 2`）** 的验证块；
     2. 核对每个 replica 的 `Total Parameters: 30532122624 across 579 tensors`（可检测导致聚合元素数或分片和偏离预期值的异常），并结合 `ray status` 辅助确认 96 个 rollout TPU 资源在线（注：二者作为运行期辅助核验，不替代 Step 0 `raiden_exact_parity_step_0.json` 对 `replica_rank_pairs` 唯一性的显式证明）；
     3. 人工核对每个 replica 日志中打印的 `Global L1 Norm` delta 与 **`Global L2 Norm` delta**（要求 `total_l2_delta <= 1e-3 * Trainer_L2`）；
     4. 确认 Steps 1–3 的 `actor/grad_norm` 均为有限正数，`0` 次非有限梯度跳过更新，`actor/lr == 2.0e-6`。
5. **Step 0 验证集（`validation/0.jsonl`，`n=30`）与 Steps 1–3 训练审计（`rollout_audit/{1..3}.jsonl`，24 个 session）人工早期离线对账**：
   - `validation/0.jsonl` 通过 `summarize_validation_dump(..., expected_n=30)` 完整性校验（`60 × 30 = 1,800` 个完整 session，`missing_num_turns == 0`）；对全部 1,800 个 session 离线调用 `retool_tpu.compute_score` 复算，`acc` 100% 一致，且 `score` 在浮点容差 **`math.isclose(recomputed_score, recorded_score, abs_tol=1e-6)`** 下 100% 一致（避免 float32 舍入造成误判）；
   - `rollout_audit/{1..3}.jsonl`（每步 8 个 session，共 24 个 session）离线重算 `acc`、`pred` 完全一致、`score` 满足 `math.isclose(..., abs_tol=1e-6)`，且 **`0` 条 `response_mask` 违规（`<tool_response>` 零泄漏）**；
   - W&B 新 Run 实时收到 Step 0 验证与 Steps 1–3 训练指标。

#### 6. 运行时长预估不确定性与 Checkpoint 恢复边界说明

1. **运行时长不作事前保证，以 64 卡实测为准**：
   - 32 卡 smoke 中 Step 1 的 `timing_s/gen = 2,116.86s` 是 3 个异步批次（2 个 warmup 批次 + Step 1 批次，共 192 道题）合并等待的总墙钟时间，日志并未分别记录这 3 个批次各自的完成时间戳与实际生成 token 总量；而 Step 2 的 `136s` 仅是单个 step 在 `ReplayBuffer` 已命中情况下的耗时，并非多步稳态均值。
   - 因此，按 `2,116.86s / 3 ≈ 705s` 线性外推三副本每步生成 `~240–270s` 仅是粗略数量级参考，不能作为运行时长保证。6 小时（`360` 分钟）预约窗口内能否跑完 25 步训练 + Step 0 / Step 25 两次 `n=30` 评估（每次 `1,800` 条 16K thinking 轨迹）及可能触发的动态 shape 编译，完全以 64 卡实测每步耗时与验证耗时为准。
2. **Checkpoint 配置边界（仅导出 `hf_model` 权重，不支持完整状态续训）**：
   - 当前 [`run_qwen3_30b_a3b_dapo_torchtitan.sh` L230-L239](run_qwen3_30b_a3b_dapo_torchtitan.sh#L230-L239) 配置为：
     ```yaml
     actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']
     trainer.resume_mode=disable
     ```
   - 因此设置 `SAVE_FREQ=5` 保存的是每 5 步导出的 HuggingFace 模型权重（可用于离线独立评估或作为后续推理/训练的初始模型权重），**并未保存 TorchTitan FSDP optimizer state、LR scheduler 状态或 dataloader 游标**，不能据此假定作业中断后可无损恢复 optimizer 和训练状态。

---

## 7. 已确认决策汇总

1. **跳过额外 32 卡 smoke，直接以 64 卡正式运行（`p3c-retool-25s-01`）的前 3 步（Steps 0–3）作为在线放行门槛**：在 `r-b0f80`（64 chips，`1 Trainer Slice + 3 Rollout Slices`，显式设置 `JOBENV_ROLLOUT_TP=32`、`JOBENV_ROLLOUT_EP=32`、`RUN_LIMIT_MIN=355`、`DRIVER_TIMEOUT_MIN=345`）从 Step 0 启动；若 Steps 0–3 通过三副本 96 ranks（`41,760` 分片）同步、范数校验、reward 对账与 mask 审计，则继续跑完至 Step 25；否则立即停止。
2. **`MAX_TURNS=16` 作为主配置**：明确设置 `max_user_turns=16`、`max_assistant_turns=16`，与 v5p 的 `ci0k54pl` / `rq1p980n` / `yrcsp4z6` 对齐。
3. **WandB Project 与新 PR 追踪**：使用 `google-trellis/verl-torchtpu-grpo-test`，PR #15 定格现有 32 卡早期部分验证结果与 64 卡启动/放行规范，64 卡正式运行使用新 W&B Run 与独立 PR 追踪。

---

## 8. 64 卡 TPU v7x 正式运行（`p3c-retool-25s-01`）Steps 0–3 在线放行检查全项通过与实时进展报告

### 8.1 运行标识、集群拓扑与节点池健康隔离记录

- **Run 名称**：`p3c-retool-25s-01`（命名空间 `r-b0f80`，队列 `multislice-queue`）
- **W&B Online 实时链接**：[`https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/smcnyyoh`](https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/smcnyyoh)
- **物理拓扑（64 × TPU v7x 物理芯片 = 128 逻辑 TensorCore，16 个 TPU Hosts + 1 个 CPU Head Pod）**：
  - **Trainer（1 × `2x2x4` Slice = 4 Hosts / 16 物理芯片 / 32 TensorCore）**：TorchTitan Actor (`FSDP2=8, CP=4, TP=1, EP=1`)；
  - **Rollout（3 × `2x2x4` Slices = 12 Hosts / 48 物理芯片 / 96 TensorCore）**：3 个独立 vLLM Async Rollout Replicas（每个副本 `TP=32, EP=32`，合计 96 个 sampler worker ranks）；
- **`EXCLUDE_NODEPOOLS` 排除坏盘与半占用节点池（已合入 [`run_ray_jobset_v7x_nap.sh`](../gke/run_ray_jobset_v7x_nap.sh)）**：
  - 启动前对 `bodaborg-tpu7x-nap` 集群上的 `2x2x4` 节点池进行排查：发现 `nap-tpu7x-stand-4t-13ryoomz` 节点 `gke-bodaborg-tpu7x-n-nap-tpu7x-stand--9fa8e677-w9c5` 的挂载卷 `/data` 存在 `mkdir: Input/output error` 宿主机 I/O 故障，另有 6 个池被其他用户 1/4 节点任务部分占用；
  - 通过 `EXCLUDE_NODEPOOLS="nap-tpu7x-stand-4t-13ryoomz,nap-tpu7x-stand-4t-1o71o48f,nap-tpu7x-stand-4t-1p4v1ri5,nap-tpu7x-stand-4t-9r5nozk4,nap-tpu7x-stand-4t-1w2840wy,nap-tpu7x-stand-4t-9mkyc6ug,nap-tpu7x-stand-4t-14vfng1m"` 注入 `cloud.google.com/gke-nodepool NotIn` 节点亲和性后，4 个 Slice 在 3 个健康空闲存量池（`1k4k5dbj`, `1df5e72s`, `a60bdzkc`）与 1 个新扩容池（`b1779xms`）上一次性完成调度（`17/17` Pods Running）。

---

### 8.2 Step 0 三副本（`3 × 32 = 96` ranks）逐元素 Raiden 权重同步核验（PASS）

原始产物已归档至 [`results/p3c-retool-25s-01/raiden_exact_parity_step_0.json`](results/p3c-retool-25s-01/raiden_exact_parity_step_0.json)，全部校验项 100% 满足 §6.5.5 第 3 条要求：

| 校验字段 | 期望值 | 实测值（`raiden_exact_parity_step_0.json`） | 结论 |
| --- | --- | --- | --- |
| `step` | `0` | `0` | **PASS** |
| `expected_ranks` / `num_results` | `96` / `96` | **`96` / `96`** | **PASS（完整覆盖 3 个 Rollout Slices）** |
| `expected_workers_per_replica` | `[32, 32, 32]` | **`[32, 32, 32]`** | **PASS（每个 Slice 精确 32 个 TPU 核心）** |
| `expected_checked_per_rank` / `checked_per_rank` | `435` / `[435]` | **`435` / `[435]`** | **PASS（全 96 ranks 无遗漏）** |
| `sum(r["checked"] for r in ranks)` | `41,760` (`96 × 435`) | **`41,760`** | **PASS（41,760 个分片逐元素 float32 比对）** |
| `replica_rank_pairs` 唯一性与完整性 | 精确覆盖 `{0,1,2} × {0..31}` | **96 个 `(replica, rank)` 组合完整无重** | **PASS** |
| `num_mismatched` / `num_unresolved` / `coverage_errors` | `0` / `0` / `[]` | **`0` / `0` / `[]`** | **PASS（零元素失配）** |
| Step 0 跨 3 副本端到端同步耗时 | - | **`38.2628 s`**（`P2P: 26.2617s`, `Parity Check: 0.1821s`） | **PASS** |

---

### 8.3 Step 0 `n=30`（`1,800` 条轨迹）AIME 2024 / 2025 基线评估与 `num_turns` 离线对账（PASS）

原始汇总已归档至 [`results/p3c-retool-25s-01/val0_gate3_summary.json`](results/p3c-retool-25s-01/val0_gate3_summary.json)：

| 评估指标（Step 0） | **TPU v7x 64-Chip `p3c-retool-25s-01` Step 0（`n=30`）** | **GB200 64-GPU `g64-retool-smoke-04` Step 0（`n=30`，PR #17）** | **Askhat v5p Regime 2 `yjowd03u` Step 0（`val_n = 8`）** |
| --- | --- | --- | --- |
| **AIME 2025 `acc/mean@N` (`pass@1`)** | **`31.78%`** (`286/900`, `0.317778`) | **`33.00%`** (`297/900`, `0.330000`) | `30.83%` (`0.308333`) |
| **AIME 2025 `acc/maj@N/mean`** | **`45.04%`** (`0.450433`) | `54.37%` (`0.543667`) | `39.96%`（`maj@8`，`0.399633`） |
| **AIME 2025 `acc/best@N/mean`** | **`71.74%`** (`0.717433`) | `72.45%` (`0.724467`) | - |
| **AIME 2025 `reward/mean@N`（Shaped）** | **`-0.2046`** (`-0.204556`) | `-0.1869` (`-0.186889`) | `-0.2213` (`-0.22125`) |
| **AIME 2024 `acc/mean@N` (`pass@1`)** | **`44.78%`** (`403/900`, `0.447778`) | **`45.56%`** (`410/900`, `0.455556`) | `48.75%` (`0.4875`) |
| **AIME 2024 `acc/maj@N/mean`** | **`62.49%`** (`0.624933`) | `64.81%` (`0.648067`) | `58.12%`（`maj@8`，`0.581233`） |
| **AIME 2024 `acc/best@N/mean`** | **`79.25%`** (`0.792533`) | `84.99%` (`0.849933`) | - |
| **AIME 2024 `reward/mean@N`（Shaped）** | **`+0.0343`** (`+0.034333`) | `+0.0504` (`+0.050444`) | `+0.1063` (`0.10625`) |
| **评估多轮轮数 `val-aux/num_turns/mean` (`min`–`max`)** | **`8.2733`** (`2`–`32`) | `8.1400` (`2`–`32`) | - |
| **评估轨迹含 `<tool_response>` 比例** | **`1,773 / 1,800` (`98.50%`, `5,646` 次调用)** | `1,773 / 1,800` (`98.50%`, `5,526` 次调用) | - |
| **`validation/0.jsonl` 离线 `num_turns` Shaped Reward 对账** | **`1,800 / 1,800` 100% 匹配（`0` mismatch）** | `1,800 / 1,800` 100% 匹配 | - |

- **参照来源说明**：Askhat 列取自 [`results/askhat-wandb-audit/askhat_runs_summary.json`](results/askhat-wandb-audit/askhat_runs_summary.json)（`yjowd03u`，Regime 2：16K + thinking + shaped，`val_n = 8`，与 `n=30` 的采样数不同，只能作量级参照）。TPU v7x 32 卡 `p3c-retool-smoke-01` 的 Step 0 每题仅 `n=2`（[`results/p3c-retool-smoke-01/val0_gate3_summary.json`](results/p3c-retool-smoke-01/val0_gate3_summary.json)），不可与 `n=30` 结果比较，因此不再列出；此前版本中 32 卡的 `n=30` 数值以及 `4z5c8z9k` / `h9c210da` 参照没有归档数据支撑，已删除。

---

### 8.4 Steps 1–9 全量核验结果与 Step 10 `/tmp/checkpoint` 临时存储驱逐修复（`p3c-retool-25s-02` 正在跑完 25 步）

原始汇总已归档至 [`results/p3c-retool-25s-01/steps_1_to_3_release_gate_summary.json`](results/p3c-retool-25s-01/steps_1_to_3_release_gate_summary.json) 与 [`results/p3c-retool-25s-01/steps_1_to_9_summary.json`](results/p3c-retool-25s-01/steps_1_to_9_summary.json)：

| Step | `grad_norm` | **`score/mean`** | `resp_len/mean` | **`clip_ratio`（门槛 `<5%`）** | `<tool_response>` 轨迹占比 (`调用次数`) | **`-2.0` `reward_CLIP`** | **`rollout_audit` 掩码违规 (`8` 会话/步)** | **三副本范数校验 (`Replica 0,1,2`)** | **`gen` / `update_actor` / `update_weights`** | **`timing_s/step`** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | `0.1705` | `-0.0611` | `1,578.9` | **`0.10%`** (`1/1024`) | `1,018 / 1,024` (`99.41%`, `2,639` 次) | **`0`** | **`0 / 8`** | **3/3 PASS** (`L1 Δ=0.415`, `L2 Δ=0.000792`) | `654.46s` / `171.17s` / **`22.89s`** | `850.08s`（首步含预热+XLA 编译） |
| **2** | `0.1355` | `-0.2478` | `2,462.7` | **`0.10%`** (`1/1024`) | `1,015 / 1,024` (`99.12%`, `3,718` 次) | **`0`** | **`0 / 8`** | **3/3 PASS** (`L1 Δ=0.391`, `L2 Δ=0.000794`) | **`0.04s`** / `139.08s` / **`16.55s`** | **`156.77s`** |
| **3** | `0.1229` | `-0.2891` | `2,747.5` | **`0.20%`** (`2/1024`) | `1,013 / 1,024` (`98.93%`, `3,809` 次) | **`0`** | **`0 / 8`** | **3/3 PASS** (`L1 Δ=0.146`, `L2 Δ=0.000793`) | **`0.04s`** / `128.22s` / **`15.06s`** | **`144.33s`** |
| **4** | `0.1522` | `-0.0245` | `1,859.2` | **`0.10%`** (`1/1024`) | `1,018 / 1,024` (`99.41%`, `2,894` 次) | **`0`** | **`0 / 8`** | **3/3 PASS** (`L1 Δ=0.025`, `L2 Δ=0.000792`) | `259.62s` / `84.15s` / **`14.72s`** | `359.45s` |
| **5** | `0.1358` | **`+0.0755`** | `2,086.2` | **`0.10%`** (`1/1024`) | `1,019 / 1,024` (`99.51%`, `3,621` 次) | **`0`** | **`0 / 8`** | **3/3 PASS** (`L1 Δ=0.197`, `L2 Δ=0.000793`) | **`0.62s`** / `71.33s` / **`14.70s`** | `323.40s`（含 `save_ckpt=235.84s`；纯训练 `87.56s`） |
| **6** | `0.1408` | **`+0.0363`** | `2,358.7` | **`0.10%`** (`1/1024`) | `1,014 / 1,024` (`99.02%`, `3,442` 次) | **`0`** | **`0 / 8`** | **3/3 PASS** (`L1 Δ=0.228`, `L2 Δ=0.000793`) | **`0.70s`** / `66.47s` / **`15.00s`** | **`83.15s`** |
| **7** | `0.1154` | **`+0.2266`** | `2,445.2` | **`0.10%`** (`1/1024`) | `1,015 / 1,024` (`99.12%`, `2,867` 次) | **`0`** | **`0 / 8`** | **3/3 PASS** (`L1 Δ=0.503`, `L2 Δ=0.000793`) | `281.83s` / `70.58s` / **`16.55s`** | `370.19s` |
| **8** | `0.1369` | **`+0.5112`** | `1,823.8` | **`0.10%`** (`1/1024`) | `1,016 / 1,024` (`99.22%`, `3,396` 次) | **`0`** | **`0 / 8`** | **3/3 PASS** (`L1 Δ=0.453`, `L2 Δ=0.000793`) | **`0.04s`** / `50.48s` / **`16.39s`** | **`67.98s`** |
| **9** | `0.1149` | **`+0.2284`** | `2,573.7` | **`0.10%`** (`1/1024`) | `1,016 / 1,024` (`99.22%`, `3,249` 次) | **`0`** | **`0 / 8`** | **3/3 PASS** (`L1 Δ=0.391`, `L2 Δ=0.000793`) | `400.46s` / `74.83s` / **`13.99s`** | `491.09s` |
| **Steps 1–9 合计** | `0.115–0.171` | **`-0.0611 -> +0.5112 / +0.2284`** | `1.58K–2.75K` | **`0.10%–0.20% << 5%`** | **`9,144 / 9,216` (`99.22%`, `29,635` 次调用)** | **`0 / 9,216`** | **`0 / 72` 会话（`0` 违规）** | **连续 9 步 × 3 副本 = 27 次全通过** | **`update_weights = 13.99s–22.89s`** | **Steps 2–9 去 `save_ckpt` 均值 `219.9s/step` (`3.67 min/step`)** |

- **Step 10 `/tmp/checkpoint` 宿主机 `ephemeral-storage` 驱逐根因与修复**：
  - `p3c-retool-25s-01` 在完成 Step 10 的训练更新后进入 `self._save_checkpoint()`，因 `SAVE_FREQ=5`、`MAX_CKPT_TO_KEEP=2` 且 [`run_qwen3_30b_a3b_dapo_torchtitan.sh`](run_qwen3_30b_a3b_dapo_torchtitan.sh) 将 `default_local_dir` 默认指向容器根盘 `/tmp/checkpoint/${experiment_name}`，[`TorchTitan._is_hf_checkpoint_writer`](../../../verl/workers/engine/torchtitan/transformer_impl.py#L717-L733) 对 `/tmp` 路径令每个 host 的 `local_rank == 0` 各自写出一份 `57 GiB` HF safetensors（Step 5 写第一份、Step 10 写第二份未轮转删除前达 `63.6–67.5 GiB`），触发 GKE kubelet `ephemeral-storage` 驱逐。
  - **已合入修复**：在 [`run_qwen3_30b_a3b_dapo_torchtitan.sh`](run_qwen3_30b_a3b_dapo_torchtitan.sh) 中，当存在 `VERL_NAP_RUN_DIR` 或 `RUN_DIR`（GCS Fuse `/data/...`）且未显式指定 `CKPTS_DIR` 时，将 `default_local_dir` 默认指向 `${VERL_NAP_RUN_DIR}/checkpoints`（仅全局 `rank == 0` 向 GCS 写出，完全不占用节点 `ephemeral-storage`）；同时保留 `p3c-retool-25s-01`（Steps 0–9）全部产物归档，并以 `JOBENV_SAVE_FREQ="-1"` + `JOBENV_CKPTS_DIR="@RUN_DIR@/checkpoints"` 启动 **`p3c-retool-25s-02`**（`17/17` Pods Running）跑完完整 25 步与 Step 25 `n=30` 评估。

---

### 8.5 `p3c-retool-25s-02`（64 卡 TPU v7x）25/25 步完整运行、Step 0 → Step 25 `n=30` 评估与跨平台对照

- **Run 名称**：`p3c-retool-25s-02`（`CODE_REF=348bbff9f4d303da43b27001a79d5734628d8b43`，GitHub tag [`run-code/p3c-retool-25s-01-02`](https://github.com/syangx39/verl-upstream/tree/run-code/p3c-retool-25s-01-02)，见 8.5.8；`JOBENV_SAVE_FREQ=-1`，`JOBENV_CKPTS_DIR=@RUN_DIR@/checkpoints`）
- **W&B 链接**：[`https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/wopup8dq`](https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/wopup8dq)
- **Checkpoint**：`SAVE_FREQ=-1`，整个 run 没有保存任何 checkpoint（包括 Step 25），Step 25 的权重没有保留；`CKPTS_DIR` 指向 GCS 运行目录，但这次 run 没有实际写入，所以 checkpoint 写 GCS 的耗时与 resume 都没有被实测（见 8.5.8）。
- **归档目录与离线校验**：[`results/p3c-retool-25s-02/`](results/p3c-retool-25s-02/)（运行 `python3 examples/tpu/retool/results/p3c-retool-25s-02/verify_evidence.py` 可离线复算全部 3,600 条验证样本的 `retool_tpu.compute_score`、配对 bootstrap 区间、原始审计会话与 TensorBoard 文件的 MD5 / SHA-256、从 200 个原始会话重算的整段掩码审计、每步墙钟拆分、Step 0 Raiden 逐元素归档与 25 步范数检查；运行 `python3 examples/tpu/retool/results/p3c-retool-25s-02/report_tables.py --check examples/tpu/retool/RETOOL_REWARD_CLIP_RERUN_PLAN_ZH.md` 校验下表与原始归档一致）。

#### 8.5.1 Step 0 → Step 25 AIME 2024 / 2025（`n=30`）评估结果

<!-- BEGIN AUTO-GENERATED p3c-retool-25s-02/eval (report_tables.py) -->
| 数据集 | Step 0 正确数 / 900（mean@30） | Step 25 正确数 / 900（mean@30） | 变化 | 配对 bootstrap 95% CI | 题目 升 / 降 / 平 | maj@30 Step 0 → 25 | best@30 Step 0 → 25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AIME 2024 | `399`（`44.33%`） | `492`（`54.67%`） | `+10.33` pp | `[+3.78, +17.22]` pp | 18 / 5 / 7 | `63.78%` → `71.59%` | `85.41%` → `85.19%` |
| AIME 2025 | `293`（`32.56%`） | `378`（`42.00%`） | `+9.44` pp | `[+3.67, +15.56]` pp | 19 / 3 / 8 | `48.77%` → `61.67%` | `69.77%` → `81.21%` |
<!-- END AUTO-GENERATED p3c-retool-25s-02/eval -->

#### 8.5.2 耗时、工具调用率、掩码审计与 Raiden 检查汇总

<!-- BEGIN AUTO-GENERATED p3c-retool-25s-02/wall (report_tables.py) -->
- Steps 2–25 的 `timing_s/step` 均值 `287.3` s（范围 `70.0`–`601.3` s；不含 Step 25 时 Steps 2–24 为 `291.5` s）。其中 `timing_s/gen`（在 `step` 计时内等 rollout 样本）Steps 2–24 均值 `194.7` s：奇数步均值 `0.29` s，偶数步均值 `372.9` s，即 trainer 每两步要等 rollout 一次。
- 端到端墙钟（相邻两步 TensorBoard 记录起点之差，与 W&B `_timestamp` 一致）Steps 2–24 均值 `341.7` s = `timing_s/step` `291.5` s + 上一步 TensorBoard 记录 `46.7` s（约 `114` 个 scalar，每个约 `0.41` s）+ 其余 `3.4` s（driver log 中同步结束到 `logger.log` 为 `2.9` s）。
- Step 25 墙钟 `1,905.6` s = `timing_s/step` `190.3` s + Step 25 `n=30` 验证 `1,663.8` s + 上一步 TensorBoard 记录 `48.3` s + 其余 `3.3` s。tqdm 的 `25/25` 总耗时 `2:59:32`（`10,772` s）在每步 TensorBoard 记录之后才更新，所以包含 Step 25 验证和全部 25 步的 TensorBoard 记录，不是纯训练时间。
- 训练轨迹 `25,600` 条，其中含 `<tool_response>` 的 `25,451` 条（`99.42%`），工具返回共 `84,914` 次，`score <= -1.99` 的轨迹 `0` 条。
- 整段掩码审计（[`rollout_mask_audit.py`](rollout_mask_audit.py)，从归档的原始 [`rollout_audit/`](results/p3c-retool-25s-02/rollout_audit/) 重算）：`200` 个会话违规 `0` 个；`721` 个 `<tool_response>` 块共 `24,014` 个 token，计入 loss 的 `0` 个；`721` 段 `mask=0` 全部恰好是注入的工具轮次（不符的 `0` 段）；`723` 个完整 `<tool_call>` 块不计入 loss 的 token `0` 个。
- Raiden：Step 0 三副本 `96` 个 rank、`41,760` 个张量分片逐元素比对，不一致 `0`；Steps 1–25 每步同步后的分布式范数检查 `25` / `25` 步三个副本全部通过（自动判定只看 `numel` 和 `L1`，是范数级检查，不等于逐元素一致）。
<!-- END AUTO-GENERATED p3c-retool-25s-02/wall -->

#### 8.5.3 前 5 步 vs 后 5 步训练窗口均值

<!-- BEGIN AUTO-GENERATED p3c-retool-25s-02/window (report_tables.py) -->
| 指标（5 步均值） | Steps 1–5 | Steps 21–25 |
| --- | --- | --- |
| 训练 reward `critic/score/mean` | `-0.108` | `+0.516` |
| `response_length/mean` | `2,109` | `2,939` |
| `num_turns/mean` | `8.50` | `8.74` |
| `response_length/clip_ratio` | `0.0018` | `0.0020` |
<!-- END AUTO-GENERATED p3c-retool-25s-02/window -->

#### 8.5.4 与我们 64× GB200 GPU（`g64-retool-25s-01`）及 Askhat TPU v5p 的 Step 25 对照

<!-- BEGIN AUTO-GENERATED p3c-retool-25s-02/compare (report_tables.py) -->
| Run | 硬件 | 每题采样 `n` | Step 0 AIME 2024 / 2025 | Step 25 AIME 2024 / 2025 | 训练 reward steps 1–5 / 21–25 |
| --- | --- | --- | --- | --- | --- |
| **我们 TPU `p3c-retool-25s-02`** | 64× TPU7x（trainer 16 + rollout 48） | 30 | `44.33%` / `32.56%` | `54.67%` / `42.00%` | `-0.108` / `+0.516` |
| 我们 GPU `g64-retool-25s-01` | 64× GB200（trainer 16 + rollout 48） | 30 | `44.22%` / `35.00%` | `54.00%` / `41.78%` | `-0.202` / `+0.432` |
| Askhat v5p `cp2`（`ci0k54pl`） | 48× v5p（trainer 16 + rollout 32；前 20 步 rollout 16） | 8 | –（未评估） | `58.33%` / `38.75%` | `-0.364` / `+0.479` |
| Askhat v5p `cp1-opt`（`rq1p980n`） | 48× v5p（trainer 16 + rollout 32） | 8 | –（未评估） | `56.67%` / `46.25%` | `-0.304` / `+0.483` |
| Askhat v5p `cp1-pr115`（`yrcsp4z6`） | 48× v5p（trainer 16 + rollout 32） | 8 | –（未评估） | `60.00%` / `46.67%` | `-0.306` / `+0.483` |
| Askhat v5p `cp1-pr115-async`（`zyrnkte8`） | 48× v5p（trainer 16 + rollout 32） | 8 | –（未评估） | `56.25%` / `46.25%` | `-0.275` / `+0.480` |
<!-- END AUTO-GENERATED p3c-retool-25s-02/compare -->

> **对比说明与证据边界**：
> 1. **我们 TPU v7x `p3c-retool-25s-02` vs 我们 GB200 `g64-retool-25s-01`（同 SFT 起点、同 `n=30` 评估）**：两条线使用同一个 SFT checkpoint（`p2-retool-sft-01/checkpoint/global_step_372/huggingface`）、相同的 16K + thinking + shaped reward + `MAX_TURNS=16` + `lr=2e-6` + `train_batch_size=64` + `rollout.n=16` 配置。Step 0 基线接近（AIME 2024 `44.33%` vs `44.22%`；AIME 2025 `32.56%` vs `35.00%`），Step 25 也接近（AIME 2024 **`54.67%` vs `54.00%`**，差 `+0.67` pp；AIME 2025 **`42.00%` vs `41.78%`**，差 `+0.22` pp），训练末段 Steps 21–25 的 `critic/score/mean` 分别为 `+0.516`（TPU）与 `+0.432`（GPU）。这是积极的初步 parity 信号，但每边只有一次正式训练 run，还不能证明两边的训练分布或收敛行为一致。
> 2. **与 Askhat TPU v5p 四条线（[`results/askhat-v5p/`](results/askhat-v5p/)，PR #19）的对照边界**：Askhat 在 Steps 21–25 的训练 reward 均值为 `+0.479`–`+0.483`，我们 TPU v7x 为 `+0.516`（略高 `+0.03`）；在 Step 25 的 AIME 评估上，我们 `n=30` 的 `54.67% / 42.00%` 接近 Askhat `n=8` 的范围（AIME 2024 `56.25%`–`60.00%`，AIME 2025 `38.75%`–`46.67%`；其中 AIME 2025 高于 Askhat `cp2` 的 `38.75%`）。注意两者存在两点不可直接归因的差异：(a) Askhat 评估每题仅采样 `n=8`（共 240 个样本，单次采样噪声比 `n=30` 的 900 个样本更大）；(b) Askhat 使用他自己训练的 `multiturn-sft-qwen3-30b-a3b-v5p`（W&B `myaf5pal`），且这四条 RL 线均设了 `val_before_train=False`，因此没有同 run 的 Step 0 评估。

#### 8.5.5 `p3c-retool-25s-02` 逐步训练与门禁明细（Steps 1–25）

<!-- BEGIN AUTO-GENERATED p3c-retool-25s-02/train (report_tables.py) -->
| Step | `score/mean` | `response_length/mean` | `clip_ratio` | `num_turns/mean` | `grad_norm` | `timing_s/step` | `gen` | `update_actor` | `update_weights` | 含 `<tool_response>` 轨迹 | 掩码审计违规 | Raiden 范数检查 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `-0.1673` | `1,508.0` | `0.10%` | `6.96` | `0.1706` | `833.6` | `628.1` | `183.7` | `20.6` | 1020 / 1024 | 0 / 8 | 3 / 3 |
| 2 | `-0.1983` | `2,648.2` | `0.10%` | `9.56` | `0.1289` | `181.8` | `0.0` | `165.9` | `14.7` | 1017 / 1024 | 0 / 8 | 3 / 3 |
| 3 | `-0.2363` | `2,389.9` | `0.10%` | `9.56` | `0.1344` | `109.2` | `0.0` | `94.0` | `13.4` | 1021 / 1024 | 0 / 8 | 3 / 3 |
| 4 | `-0.0409` | `1,863.1` | `0.49%` | `7.57` | `0.1502` | `377.1` | `261.6` | `100.0` | `14.5` | 1020 / 1024 | 0 / 8 | 3 / 3 |
| 5 | `+0.1043` | `2,137.2` | `0.10%` | `8.86` | `0.1346` | `76.5` | `0.0` | `60.6` | `15.0` | 1019 / 1024 | 0 / 8 | 3 / 3 |
| 6 | `+0.0308` | `2,138.3` | `0.10%` | `8.17` | `0.1317` | `410.4` | `326.3` | `68.0` | `15.1` | 1019 / 1024 | 0 / 8 | 3 / 3 |
| 7 | `+0.3736` | `2,092.2` | `0.10%` | `8.88` | `0.1386` | `82.4` | `0.3` | `65.7` | `15.3` | 1020 / 1024 | 0 / 8 | 3 / 3 |
| 8 | `+0.3770` | `2,159.7` | `0.10%` | `8.79` | `0.1281` | `466.5` | `386.4` | `64.0` | `15.2` | 1019 / 1024 | 0 / 8 | 3 / 3 |
| 9 | `+0.3601` | `2,404.7` | `0.10%` | `8.93` | `0.1323` | `99.8` | `0.2` | `83.2` | `15.4` | 1018 / 1024 | 0 / 8 | 3 / 3 |
| 10 | `+0.3349` | `2,705.7` | `0.10%` | `7.71` | `0.1196` | `480.6` | `375.2` | `90.2` | `14.3` | 1017 / 1024 | 0 / 8 | 3 / 3 |
| 11 | `+0.4759` | `2,108.9` | `0.10%` | `8.88` | `0.1189` | `70.0` | `0.2` | `55.2` | `13.6` | 1019 / 1024 | 0 / 8 | 3 / 3 |
| 12 | `+0.4584` | `2,622.4` | `0.10%` | `7.61` | `0.1102` | `444.8` | `350.1` | `78.8` | `15.0` | 1022 / 1024 | 0 / 8 | 3 / 3 |
| 13 | `+0.5146` | `2,185.9` | `0.10%` | `9.01` | `0.1286` | `80.5` | `0.2` | `64.2` | `15.1` | 1017 / 1024 | 0 / 8 | 3 / 3 |
| 14 | `+0.4475` | `2,588.0` | `0.10%` | `8.03` | `0.1079` | `524.7` | `438.7` | `69.2` | `15.9` | 1019 / 1024 | 0 / 8 | 3 / 3 |
| 15 | `+0.5261` | `2,338.2` | `0.10%` | `8.92` | `0.1032` | `78.6` | `0.2` | `62.0` | `15.4` | 1020 / 1024 | 0 / 8 | 3 / 3 |
| 16 | `+0.3547` | `2,747.0` | `0.10%` | `8.81` | `0.1014` | `562.1` | `465.0` | `80.7` | `15.4` | 1020 / 1024 | 0 / 8 | 3 / 3 |
| 17 | `+0.4312` | `2,740.3` | `0.10%` | `9.63` | `0.0986` | `95.7` | `0.7` | `78.2` | `15.8` | 1019 / 1024 | 0 / 8 | 3 / 3 |
| 18 | `+0.3593` | `3,197.4` | `0.10%` | `8.39` | `0.0959` | `555.8` | `454.8` | `85.0` | `14.1` | 1012 / 1024 | 0 / 8 | 3 / 3 |
| 19 | `+0.3975` | `2,614.8` | `0.10%` | `9.91` | `0.1075` | `90.9` | `0.2` | `74.1` | `15.6` | 1019 / 1024 | 0 / 8 | 3 / 3 |
| 20 | `+0.4518` | `3,162.4` | `0.10%` | `7.97` | `0.1338` | `563.0` | `461.4` | `85.2` | `15.4` | 1013 / 1024 | 0 / 8 | 3 / 3 |
| 21 | `+0.5557` | `2,795.2` | `0.20%` | `8.99` | `0.0937` | `91.2` | `0.2` | `74.5` | `15.5` | 1018 / 1024 | 0 / 8 | 3 / 3 |
| 22 | `+0.3871` | `3,259.2` | `0.20%` | `8.43` | `0.0886` | `568.0` | `452.8` | `93.2` | `21.0` | 1019 / 1024 | 0 / 8 | 3 / 3 |
| 23 | `+0.5986` | `2,707.4` | `0.20%` | `9.36` | `0.1033` | `94.5` | `0.7` | `77.2` | `15.6` | 1016 / 1024 | 0 / 8 | 3 / 3 |
| 24 | `+0.4881` | `3,111.0` | `0.10%` | `7.97` | `0.0877` | `601.3` | `503.0` | `81.8` | `15.5` | 1014 / 1024 | 0 / 8 | 3 / 3 |
| 25 | `+0.5525` | `2,822.3` | `0.29%` | `8.95` | `0.0926` | `190.3` | `56.6` | `117.8` | `14.9` | 1014 / 1024 | 0 / 8 | 3 / 3 |
<!-- END AUTO-GENERATED p3c-retool-25s-02/train -->

#### 8.5.6 每步墙钟拆分与 64× GB200 耗时对照

`timing_s/step` 只覆盖 `RayPPOTrainer.fit` 里 `with marked_timer("step")` 的部分：从 replay buffer 取样（`timing_s/gen`，即等 rollout）、`old_log_prob`、`adv`、`update_actor` 和 `on_step_end` 里的 `update_weights`。验证（`timing_s/testing`）、`_compute_metrics`、rollout / 审计 dump 的准备（TransferQueue 取数与 decode，写文件在后台线程）、`tq.kv_clear` 和 `logger.log` 都在计时之外（运行代码 [`trainer_base.py` L491–L556](https://github.com/syangx39/verl-upstream/blob/348bbff9f4d303da43b27001a79d5734628d8b43/verl/trainer/ppo/v1/trainer_base.py#L491-L556)）。`logger.log` 先写 W&B，再对每个指标调用一次 TensorBoard `add_scalar`，最后打印 console 行。TensorBoard 为每次 `add_scalar` 记下 `wall_time`，所以归档的事件文件（[`results/p3c-retool-25s-02/tensorboard/`](results/p3c-retool-25s-02/tensorboard/)）能独立给出每步 `logger.log` 的起点和 TensorBoard 写入时长；每步第一个 scalar 的时间与 W&B `_timestamp` 相差不超过 `0.02` s（W&B API 核对）。下表由 [`step_timing_breakdown.py`](step_timing_breakdown.py) 生成。

<!-- BEGIN AUTO-GENERATED p3c-retool-25s-02/timing (report_tables.py) -->
| Step | 墙钟（TensorBoard 记录起点间隔） | `timing_s/step` | 其中 `gen` | 其中 `update_actor` | 其中 `update_weights` | `timing_s/testing` | 上一步 TensorBoard 记录（scalar 数） | 计时外其余 | 同步结束 → `logger.log` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `957.4` | `833.6` | `628.1` | `183.7` | `20.6` | `0.0` | `120.9`（259） | `2.9` | `1.6` |
| 2 | `228.0` | `181.8` | `0.0` | `165.9` | `14.7` | `0.0` | `43.2`（111） | `3.0` | `2.8` |
| 3 | `155.3` | `109.2` | `0.0` | `94.0` | `13.4` | `0.0` | `42.6`（111） | `3.5` | `3.3` |
| 4 | `427.3` | `377.1` | `261.6` | `100.0` | `14.5` | `0.0` | `47.3`（111） | `3.0` | `2.7` |
| 5 | `122.5` | `76.5` | `0.0` | `60.6` | `15.0` | `0.0` | `43.8`（111） | `2.2` | `1.9` |
| 6 | `455.2` | `410.4` | `326.3` | `68.0` | `15.1` | `0.0` | `42.0`（111） | `2.8` | `2.5` |
| 7 | `128.8` | `82.4` | `0.3` | `65.7` | `15.3` | `0.0` | `43.7`（115） | `2.7` | `2.4` |
| 8 | `513.5` | `466.5` | `386.4` | `64.0` | `15.2` | `0.0` | `44.5`（115） | `2.5` | `2.2` |
| 9 | `153.2` | `99.8` | `0.2` | `83.2` | `15.4` | `0.0` | `50.6`（115） | `2.8` | `2.5` |
| 10 | `525.8` | `480.6` | `375.2` | `90.2` | `14.3` | `0.0` | `41.4`（115） | `3.7` | `3.5` |
| 11 | `117.2` | `70.0` | `0.2` | `55.2` | `13.6` | `0.0` | `43.8`（115） | `3.3` | `3.1` |
| 12 | `500.3` | `444.8` | `350.1` | `78.8` | `15.0` | `0.0` | `52.8`（115） | `2.7` | `2.4` |
| 13 | `134.6` | `80.5` | `0.2` | `64.2` | `15.1` | `0.0` | `51.4`（115） | `2.7` | `2.4` |
| 14 | `572.9` | `524.7` | `438.7` | `69.2` | `15.9` | `0.0` | `45.5`（115） | `2.7` | `2.5` |
| 15 | `126.2` | `78.6` | `0.2` | `62.0` | `15.4` | `0.0` | `44.8`（115） | `2.9` | `2.5` |
| 16 | `614.7` | `562.1` | `465.0` | `80.7` | `15.4` | `0.0` | `49.4`（115） | `3.2` | `2.8` |
| 17 | `150.7` | `95.7` | `0.7` | `78.2` | `15.8` | `0.0` | `51.7`（115） | `3.3` | `3.0` |
| 18 | `608.5` | `555.8` | `454.8` | `85.0` | `14.1` | `0.0` | `47.9`（115） | `4.8` | `3.5` |
| 19 | `137.0` | `90.9` | `0.2` | `74.1` | `15.6` | `0.0` | `43.0`（115） | `3.1` | `2.8` |
| 20 | `611.4` | `563.0` | `461.4` | `85.2` | `15.4` | `0.0` | `45.2`（115） | `3.2` | `2.9` |
| 21 | `151.1` | `91.2` | `0.2` | `74.5` | `15.5` | `0.0` | `55.0`（115） | `5.0` | `2.8` |
| 22 | `623.0` | `568.0` | `452.8` | `93.2` | `21.0` | `0.0` | `46.9`（115） | `8.1` | `6.9` |
| 23 | `149.5` | `94.5` | `0.7` | `77.2` | `15.6` | `0.0` | `51.5`（115） | `3.6` | `2.8` |
| 24 | `651.2` | `601.3` | `503.0` | `81.8` | `15.5` | `0.0` | `46.6`（115） | `3.3` | `3.0` |
| 25 | `1905.6` | `190.3` | `56.6` | `117.8` | `14.9` | `1663.8` | `48.3`（115） | `3.3` | `1666.8` |
| Steps 2–24 均值（23 步） | `341.7` | `291.5` | `194.7` | `80.5` | `15.3` | `0.0` | `46.7`（114） | `3.4` | `2.9` |
<!-- END AUTO-GENERATED p3c-retool-25s-02/timing -->

- “上一步 TensorBoard 记录”是第 N−1 步全部 scalar 的写入时长；“计时外其余” = 墙钟 − `timing_s/step` − `timing_s/testing` − 上一步 TensorBoard 记录。driver log 中 Raiden 同步结束（`[RAIDEN TELEMETRY | Orchestrator] Step N Completed`）到 `logger.log` 的时间与“计时外其余”吻合（Steps 2–24 均值 `2.9` s vs `3.4` s），所以墙钟与 `timing_s/step` 的差额几乎全部是 TensorBoard 写入。Step 1 含预热与 XLA 编译；Step 25 的“同步结束 → `logger.log`”包含 `n=30` 验证。
- 每个 scalar 约 `0.41` s，远高于写本地文件的开销；GPU 运行同样约 `0.41` s，与平台无关。推测原因（未单独实验）：镜像没有 TensorFlow 时，TensorBoard 用 `tensorboard.compat.tensorflow_stub` 的 `GFile`，对本地路径每写一条记录就 `open(..., "ab")`、写入、关闭一次；`TENSORBOARD_DIR` 在 GCS Fuse（`/data/...`）上，每次关闭都要把整个事件文件重新上传。验证办法：把 `TENSORBOARD_DIR` 改到本地盘（结束后再拷贝）或从 `LOGGER_LIST` 去掉 `tensorboard`，跑 3 步对比。
- rollout / 审计 JSONL 写 GCS 由后台线程完成，不在主循环的关键路径上。

<!-- BEGIN AUTO-GENERATED p3c-retool-25s-02/perf (report_tables.py) -->
| 指标（秒/步；除首行外为 Steps 2–24 均值） | GPU `g64-retool-25s-01` | TPU `p3c-retool-25s-02` | TPU / GPU |
| --- | --- | --- | --- |
| `timing_s/step`，Steps 2–25 均值 | `30.78` | `287.32` | `9.3×` |
| `timing_s/step` | `30.66` | `291.54` | `9.5×` |
| └ `gen`（`step` 计时内等 rollout） | `0.04` | `194.72` | – |
| └ `update_actor` | `24.45` | `80.46` | `3.3×` |
| └ `update_weights` | `5.47` | `15.29` | `2.8×` |
| 端到端墙钟 | `73.17` | `341.66` | `4.7×` |
| └ 计时外：上一步 TensorBoard 记录 | `39.96` | `46.72` | – |
| └ 计时外：其余（指标、dump 准备、`kv_clear`） | `2.56` | `3.40` | – |
| 每步 token 数（`perf/total_num_tokens`） | `3.01` M | `2.94` M | – |
| `timing_per_token_ms/update_actor` | `0.0081` | `0.0276` | `3.4×` |
<!-- END AUTO-GENERATED p3c-retool-25s-02/perf -->

> **读法**：
> 1. 两个 `timing_s/step` 均值本身准确：Steps 2–25 GPU `30.78` s、TPU `287.32` s，与各自的 `head.log` / metrics JSON 以及 W&B 历史三处一致。但含义不同：TPU 的 `timing_s/step` 里平均 `194.7` s 是等 rollout（偶数步约 `373` s，奇数步约 `0.3` s），GPU 每步只有 `0.04` s。所以 `9.3×` 主要反映 TPU 的 rollout 跟不上 trainer，不是 trainer 慢 9 倍。
> 2. 更可比的分项：`update_actor` `24.5` s vs `80.5` s（`3.3×`；两边每步 token 数相近，按 token 计 `3.4×`），`update_weights` `5.5` s vs `15.3` s（`2.8×`），端到端墙钟 `73.2` s vs `341.7` s（`4.7×`）。
> 3. 两边每步都有约 `40`–`47` s 花在计时外的 TensorBoard 写入，对 GPU 来说比整个 `step` 还长。GPU 的 rollout 因此多出约 40 s 余量：`gen ≈ 0` 只说明 rollout 至少跟得上 73 s 一步，不能说明去掉这部分开销后仍跟得上。
> 4. 每边只有一次运行，rollout 切分也不同（TPU 3 个 `TP=32, EP=32` 副本，GPU 12 个 `TP=4` 副本）。这些差距是值得继续调查的初步现象，不能单凭它们归因硬件性能。

#### 8.5.7 掩码审计方法与范围

- **方法**（[`rollout_mask_audit.py`](rollout_mask_audit.py)）：对每个审计会话，(1) 检查 `response_ids` 与 `response_mask` 等长，`segments` 恰好是 mask 的最大连续段；(2) 在整条 response 上（不按 segment）配对 `<tool_response>` … `</tool_response>`，块内含标签必须全部 `mask=0`，配对 `<tool_call>` … `</tool_call>`，必须全部 `mask=1`，`<|im_start|>` 不得计入 loss；(3) 每段 `mask=0` 必须恰好是注入的工具轮次 `\n<|im_start|>user\n<tool_response>…</tool_response><|im_end|>\n<|im_start|>assistant\n`，因此模型生成的 token 都在 loss 里；(4) 在拼接后的解码文本上按字符再做一遍同样的块检查，并核对每个标签在文本中的次数与其 token id 的次数一致（同时验证了 Qwen3 special token id 表）。
- **旧检查的缺口**：旧脚本只在每个 segment 的文本里找完整的 `<tool_response>`。如果一个工具返回块从 `mask=0` 段开始、内容和 `</tool_response>` 落进 `mask=1` 段，`mask=1` 段里没有开标签，旧检查不会报警。[`test_retool_evidence_tools_on_cpu.py`](../../../tests/special_tpu/test_retool_evidence_tools_on_cpu.py) 构造了这类会话：旧检查漏报，新检查报出。
- **结果**：从归档的 200 个原始会话重算，`0` 违规（见 8.5.2）。`p3c-retool-25s-01` Steps 1–9 的 72 个会话（GCS 原始文件，未归档）用新检查重跑，同样 `0` 违规。
- **范围**：每步只审计 8 个会话，25 步共 200 个，占 `25,600` 条训练轨迹的 `0.78%`；其余轨迹只有 `rollout/*.jsonl` 的文本级汇总（工具调用次数、`-2.0` 计数），没有 token 级 mask。

#### 8.5.8 可复现性：运行代码与 checkpoint

- **运行代码**：`run_record.json` 记录的 `348bbff9`（`p3c-retool-25s-01` 与 `-02` 相同）此前只在本地，现已推到 GitHub tag [`run-code/p3c-retool-25s-01-02`](https://github.com/syangx39/verl-upstream/tree/run-code/p3c-retool-25s-01-02)（[commit](https://github.com/syangx39/verl-upstream/commit/348bbff9f4d303da43b27001a79d5734628d8b43)，父提交 `c4726400` 是合入 PR #15 后的 `tpu-main`）。pods 实际运行的是 launcher 用 `git archive` 上传的 `gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3c-retool-25s-02/src-348bbff9f4d3.tar.gz`（sha256 `950dbf26…7701`）；[`verify_code_snapshot.py`](results/p3c-retool-25s-02/verify_code_snapshot.py) 逐个比对 tarball 内 `1,350` 个文件的 git blob hash 与 `git ls-tree -r 348bbff9`，全部一致，记录见 [`env/code_snapshot.json`](results/p3c-retool-25s-02/env/code_snapshot.json)。此后本分支在运行路径上的代码改动包括：`run_qwen3_30b_a3b_dapo_torchtitan.sh` 在未设 `CKPTS_DIR` 时默认写 GCS 运行目录（这次 run 显式设置了 `CKPTS_DIR`）；`v7x_nap/render.py` 在提交前拒绝重复的 slice nodepool 或与排除列表冲突的配置。这些均为该 run 完成后的更改；其余新增文件（审计、计时、校验脚本）只用于离线分析。
- **Checkpoint**：`-02` 用 `SAVE_FREQ=-1`、`resume_mode=disable`，整个 run 没有调用 `save_checkpoint`，Step 25 的权重没有保存。`p3c-retool-25s-01` 在 Step 10 写 `/tmp` checkpoint 时被 ephemeral-storage 驱逐（见 8.4）；之后默认改写 GCS 运行目录，但还没有任何 run 实际写过，所以写 GCS 的耗时、旧 checkpoint 轮转与 resume 都还没有实测（机制与体积估算见 PR #20）。GPU 正式 run `g64-retool-25s-01` 同样 `save_freq=-1`，也没有保存最终 checkpoint。
