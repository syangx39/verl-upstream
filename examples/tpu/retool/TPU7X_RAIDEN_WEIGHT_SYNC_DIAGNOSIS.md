# TPU v7x ReTool RL（Stage P1c）Raiden 初始权重同步 Tensor Mismatch 现象取证、待验证假设与受控验证计划

状态：**现象已取证 / 根因假设待受控实验验证（Working Hypothesis — 2026-11-03 修订）**
关联 PR：
- TPU v7x 实验进展记录：PR #7（[`REPRO_RESULTS_TPU7X.md`](https://github.com/syangx39/verl-upstream/pull/7)）
- NVIDIA GB200 实验结果记录：PR #9（[`REPRO_RESULTS_GB200_ZH.md`](https://github.com/syangx39/verl-upstream/pull/9)）
- 同物理加速器数量（Iso-Accelerator-Count）性能对照占位：PR #11（[`ISO_CHIP_PERF_BENCHMARK_PLAN.md`](https://github.com/syangx39/verl-upstream/pull/11)）

---

## 1. 问题概述与证据边界（Executive Summary & Scope of Evidence）

在 Google Cloud TPU v7x（Ironwood）双 Slice 拓扑上运行 **Qwen3-30B-A3B MoE 多轮 ReTool RL Smoke（Stage P1c）** 时，开启 `VERIFY_PARITY=all` 的诊断运行（`p1c-retool-rl-05`）在初始 `update_weights(0)`（Step-0 Raiden 权重同步）阶段触发了确定性的 `RaidenParityError`：

- **硬件拓扑更正与对账（与 PR #7 一致）**：
  - 单颗 TPU v7x 物理芯片包含 **2 个逻辑 TensorCore 设备**（每逻辑核 `96 GiB` HBM，单物理芯片共 `192 GiB` HBM）；每台 `ct7x-standard-4t` 主机搭载 **4 颗物理芯片 = 8 个逻辑 TensorCore 设备**。
  - `Stage P1c` 采用两个 `2×2×4` Slice（每个 Slice 为 **4 台主机 × 4 颗物理芯片 = 16 颗物理芯片 / 32 个逻辑 TensorCore 设备**）：
    - **Trainer Slice（`1 × 2×2×4`）**：**16 颗物理芯片 / 32 个逻辑 TensorCore 设备**（`NNODES_TRAINER=4, N_CHIPS_TRAINER=8` 逻辑设备/主机，`dp_shard=8, cp=4`，共 32 个 Trainer ranks）；
    - **Rollout Slice（`1 × 2×2×4`）**：**16 颗物理芯片 / 32 个逻辑 TensorCore 设备**（`NNODES_ROLLOUT=4, N_CHIPS_ROLLOUT=8` 逻辑设备/主机，vLLM `TP=32, EP=32`，共 32 个 Sampler ranks）；
    - **双 Slice 合计**：**32 颗物理 TPU v7x 芯片 / 64 个逻辑 TensorCore 设备**（此前版本误写为“每侧 32 颗物理芯片”，特此更正）。
- **已确认的故障现象（Confirmed Symptom）**：
  - 在 `p1c-retool-rl-05` 中，Step-0 初始权重同步（`RaidenController P2P = 2.2080 s`, `Sampler H2D DMA = 0.4275 s`）完成后，`RaidenParityCheck.report_exact` 报告：
    ```text
    verl.checkpoint_engine.raiden_checkpoint_engine.RaidenParityError: [RAIDEN PARITY EXACT | Step 0] 32 sampler ranks, [435] tensors checked per rank, 10414 mismatched, 0 unresolved
    ```
  - 在日志打印出的 `rank 0` 前 60 条 mismatch 明细中（受 `RaidenParityCheck.MAX_DETAIL_LINES = 60` 截断限制）：
    - `model.layers.3.mlp.experts.routed_experts.w13_weight` 的接收端平均绝对值约为参照值的一半（实测 `max|diff|=0.5195, ref_absmean=0.01824, recv_absmean=0.009274`，比值约 `0.508x`；**注：该比值与“拼接进 `w13_weight = [gate^T | up^T]` 的 `gate_proj` 或 `up_proj` 之一全零（或半截张量全零）”的方向一致，但仅凭标量均值本身尚不能证明具体哪一半全零，且该检查发生在 `_raiden_ws.h2d()` 与 `_raiden_fused_expert_weights` 融合/转置之后**）；
    - 随后打印的 `layers.30..47, 4..9` 的 MoE `w13_weight / w2_weight` 以及 `layers.0, 1, 10..18` 的 `self_attn.qkv_proj.weight` 全部表现为 **`recv_absmean = 0`**（全零）。
- **证据边界说明（为什么具体根因仍属于待验证假设 Working Hypothesis）**：
  - 在现有代码 [`verl/workers/rollout/vllm_rollout/tpu_utils.py#L604-L730`](../../../verl/workers/rollout/vllm_rollout/tpu_utils.py#L604-L730) 中，`parity.check_exact(...)` 发生在 **`self._raiden_ws.h2d()`（`L606`）之后**，以及 **MoE 专家权重拆包/融合（`_raiden_fused_expert_weights`，`L671–L684`）或 Dense 权重拼接/转置（`to_target_layout`，`L707–L728`）过程中**。
  - 因此，仅凭现有日志摘录，尚不能单独断定全零数据究竟产生于：
    1. **假设 H1（H2D 之前 Host Staging Buffer 已全零 / P2P 传输未完成即返回）**；
    2. **假设 H2（`_raiden_ws.h2d()` 主机到 TPU HBM DMA 拷贝阶段在中途截断或清零）**；或
    3. **假设 H3（H2D 之后的 TPU 张量视图拆包 / 转置 / 布局转换阶段产生全零）**。
  - 同样，“旧版 `tpu-sync` 命中 `direct_schedules` 导致 `.wait()` 提前返回”以及“切换新版运行时镜像即可修复该现象”属于**值得优先验证的工作假设（Working Hypotheses）**，尚未被调度器内部日志或新旧镜像对照实验最终证实。

---

## 2. 上游补丁关系与镜像包版本核实（Fact Corrections）

### 2.1 上游 PR 真实内容与跨仓库依赖核对（不盲目直接 Cherry-pick `#110` 和 `#116`）

此前草稿将 `jialei777/verl-upstream#110` 与 `verl-project/verl-hardware-plugin#89` 混为一谈，并建议直接同时 cherry-pick `#110` 与 `#116`。经逐一核对上游 PR 源码，真实对应关系如下：

| 上游 PR | 所属仓库与状态 | 真实修改内容 | 与本问题的关系及处理原则 |
| :--- | :--- | :--- | :--- |
| **[`jialei777/verl-upstream#110`](https://github.com/jialei777/verl-upstream/pull/110)** (`93058928`) | `jialei777/verl-upstream` (Open) | **Rollout Pause 与吞吐性能优化**（*`feat: 2.1x faster Qwen3-30B-A3B TPU GRPO step`*）：将 `_abort_replicas_for_sync` 推迟到 Trainer 端权重导出准备完成之后（使 rollout 暂停时间从 ~12s 降至 ~1.9s），将 `MAX_NUM_SEQS` 调至 128，并新增 `keep_prefix_cache_on_sync`。 | **不是 3-D MoE 专家导出补丁**。不能把它当作 `#89` 来 cherry-pick。 |
| **[`verl-project/verl-hardware-plugin#89`](https://github.com/verl-project/verl-hardware-plugin/pull/89)** (`cf1cb234`) | `verl-project/verl-hardware-plugin` (Merged) | **3-D MoE 专家分片直接导出与安装**（*`[TPU] perf: direct tpu-sync install of Qwen3 MoE experts (EP or TP-sliced) with local expert-shard export`*）：将 MoE 专家按 3-D `[E, I, H]` / `[E, H, I]` 导出并在 rollout 端走 `qwen3_moe direct` 安装。 | **位于独立的 `verl-hardware-plugin` 仓库**（修改路径为 `verl_hardware_plugin/accelerators/tpu/...`），依赖该仓库的插件架构实现，**不能直接 cherry-pick 到当前 `verl-upstream` 分支**；若需引入需单独评估跨仓库移植及依赖。 |
| **[`jialei777/verl-upstream#116`](https://github.com/jialei777/verl-upstream/pull/116)** (`32b5133e`) | `jialei777/verl-upstream` (Open, 标有 **`DO NOT MERGE`**) | **2-D Folded Expert Stack 的临时 Workaround**（*`DO NOT MERGE: Temporary workaaround Raiden weight sync for MoE models on torch-tpu >= 2026-09-24`*）：当使用 2-D 折叠专家栈（`[E_local * F, D]`）且 `torch_tpu`（`>= 2026-09-24` / `0.1.2.dev20261006`）中 TPU$\rightarrow$TPU `clone()` 共享源 3-D base storage buffer 时，通过 `adapt_for_raw_transfer` 探测并走 host round-trip 重新分配 2-D 存储。 | 属于针对 **2-D 折叠路径**的临时规避方案，且与 `#89`（3-D 原生导出）分属不同路径。**不应预先盲目合入**：应先保持当前 P1c 代码仅切换运行时镜像测试，只有当新版镜像确实报 `UnpackTorchTensor: ... leading (block) dimension ... does not match its base storage buffer` 时，再评估移植 `#116` 的必要部分。 |

### 2.2 容器镜像内 `torch-tpu` 与 `tpu-sync-torch` 版本核对（OCI Config 实测对账）

此前草稿中关于“新版镜像包含 `torch-tpu >= 0.2.0`”的表述与实际镜像审计存在冲突；同时 PR #7 `Stage P0.2` 表格曾将两个镜像的 `torch-tpu` 均记为 `0.1.1.dev20260912075415`。经直接提取两个镜像的 OCI Config 构建历史（`ARG TORCH_TPU_VERSION` / `ARG TPU_SYNC_VERSION` 及构建时 `importlib.metadata` 校验层），真实版本对照如下：

| 镜像标签与不可变 Digest | `torch` 版本 | `torch-tpu` 真实版本 | `tpu-sync-torch` 真实版本 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **`verl-tpu:v20261001-tsync990787257`** (`@sha256:dcece4bc3813359450f48d4523e5bd78e3139dc441a88b5c2e4d638caa054cbe`) | `2.13.0+cpu` | `0.1.1.dev20260912075415` | `0.0.1.dev20261001220631` | `p1a-moe-smoke-12`、`p1c-retool-rl-03` 与 `p1c-retool-rl-05` 实际运行所使用的旧版镜像 |
| **`verl-tpu:v20261006-tsync1006`** (`@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e`) | `2.13.0+cpu` | **`0.1.2.dev20261006000518`** | **`0.0.1.dev20261006025601`** | 仓库 `tests/special_tpu/gke/tpu-ci-image.env` 固定的新版镜像，**同时升级了 `torch-tpu` 与 `tpu-sync-torch`**（均为 `0.1.x.dev*` 开发构建，并非 `>= 0.2.0`） |

> **关键发现（已由 `p1c-retool-rl-06` 实测确认）**：`verl-tpu:v20261006-tsync1006`（`@sha256:d92a2b3d...`）同时包含 `torch-tpu==0.1.2.dev20261006000518` 与 `tpu-sync-torch==0.0.1.dev20261006025601`。在旧代码 `c2e17624` 中，Trainer 端在 `export_local_shards` 中对 3-D MoE 专家参数执行了 `local.reshape(local_shape).clone()`（2-D 折叠视图 `[E_local * F, D]`）。在 `torch-tpu==0.1.2.dev20261006000518` 上，该 TPU$\rightarrow$TPU `.reshape().clone()` 仍共享源 3-D `[E_local, F, D]` base storage buffer，导致 `p1c-retool-rl-06` 在 Step-0 首次创建 `WeightSynchronizer` 时明确命中**情形 B**：`ValueError: Tensor's leading (block) dimension does not match its base storage buffer; raw transfer requires a layout-preserving full view of the block-major storage`。因此阶段二按计划合入 `#116`（`32b5133e`, `adapt_for_raw_transfer`）并启动 **`p1c-retool-rl-07`**（commit `a7c6be29`）。

---

## 3. `p1c-retool-rl-05` 故障数据对账与 `VERIFY_PARITY=all` 机制核实

### 3.1 为什么日志汇总显示 `10,414` 个 Mismatch（而非 `325 × 32 = 10,400`）？

1. **原始日志精确输出（`p1c_05_driver.log:12283`）**：
   ```text
   verl.checkpoint_engine.raiden_checkpoint_engine.RaidenParityError: [RAIDEN PARITY EXACT | Step 0] 32 sampler ranks, [435] tensors checked per rank, 10414 mismatched, 0 unresolved
   ```
2. **与 `325 × 32 = 10,400` 差 14 个的原因**：
   - 在 [`verl/checkpoint_engine/raiden_checkpoint_engine.py#L457-L601`](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py#L457-L601) 中，`RaidenParityCheck.report_exact` 计算全集群总不匹配数 `num_mismatched = sum(r["num_mismatched"] for r in results)`（即 **`10,414`**，跨 `32 × 435 = 13,920` 个检查项，平均每 rank **`325.4375`** 个 mismatch，即 `10,414 = 32 × 325 + 14`）。
   - 然而，`report_exact` 在构造异常文本时执行了 `detail_text = "\n".join(details[: cls.MAX_DETAIL_LINES])`（其中 `MAX_DETAIL_LINES = 60`），且在 Sampler Worker 侧 `exact_result()` 中同样对 `self.mismatched` 做了 `[:60]` 截断。
   - 因为 `rank 0` 自身的 mismatch 数已超过 60 个，日志打印出的 60 行明细全部来自 `rank 0`，**并未打印出全部 32 个 rank 各自的 `r["num_mismatched"]` 数值**。
   - 此前草稿根据 `rank 0` 上首个失败张量 `model.layers.3.mlp.experts.routed_experts.w13_weight` 推算 `435 - 110 = 325` 并假定所有 32 个 rank 均为 `325`，会得出 `32 × 325 = 10,400`，比真实总数 `10,414` 少 **14 个**（说明 32 个 Sampler rank 之间的截断位置存在轻微差异，部分 rank 比 `rank 0` 多出 1 个或数个不匹配张量）。
   - **修正措施（文档与代码双侧补齐）**：
     1. 在报告中严格使用 `p1c-retool-rl-05` 日志实测总计 **`10,414 / 13,920`**；
     2. 已入队的 `p1c-retool-rl-06` 绑定的是提交前的基线代码 `CODE_REF=c2e17624`（保留“仅切换容器镜像”的严格单变量对照，因此**不包含**本 PR 新增的 `(replica, rank)` `0..31` 集合断言、`expected_checked_per_rank=435` 断言、JSON 持久化归档或 staging 诊断），故让 `-06` 继续排队完成，并在其结束后**人工核对 `head.log` 中的 Step-0 摘要行是否精确为 `32 sampler ranks, [435] tensors checked per rank, 0 mismatched, 0 unresolved`**；
     3. 同时，**本 PR 已在 [`verl/checkpoint_engine/raiden_checkpoint_engine.py`](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py)、[`verl/workers/rollout/vllm_rollout/tpu_utils.py`](../../../verl/workers/rollout/vllm_rollout/tpu_utils.py) 与 [`examples/tpu/retool/run_qwen3_30b_a3b_dapo_torchtitan.sh`](run_qwen3_30b_a3b_dapo_torchtitan.sh) 中补齐三项采集与校验能力**（供后续基于本 PR commit 启动的运行如 `p1c-retool-rl-07` 直接使用）：
        - **强制完整 `(replica, rank)` 集合与张量覆盖率断言**：在 `update_raiden_weights` 实际调用路径中向 `report_exact` 传入每个 replica 的预期 worker 数 `expected_workers_per_replica` 与 `expected_checked_per_rank=435`（同时在 Sampler Worker 侧通过 `compute_expected_checked_params(vllm_model, received_lm_head)` 按 `id(p)` 去重记录该 worker 实际应经 `check_exact` 检查的参数总数 `expected_checked`：仅当 `!received_lm_head` 且 `lm_head.weight` 与 `embed_tokens.weight` 为不同 `nn.Parameter` 实例时剔除 `id(lm_head.weight)`，避免共享同一 `nn.Parameter` 的 tied-weight 模型重复扣减），强制校验每个 replica 返回的 rank ID 集合精确为 `0..W_r-1`（如 `0..31`，拒绝 `100..131` 或重复/缺失 rank）且每个 rank 检查的张量数精确等于预期参数数（如 `435`，拒绝 `checked=1` 等部分检查）；
        - **未截断逐 Rank 全量明细持久化归档与写失败 Fail-Closed**：`run_qwen3_30b_a3b_dapo_torchtitan.sh` 将 `RAIDEN_PARITY_DUMP_DIR` 指向共享运行目录（`${VERL_NAP_RUN_DIR}/env` 或 `${default_local_dir}/parity`）并通过 `+ray_kwargs.ray_init.runtime_env.env_vars.RAIDEN_PARITY_DUMP_DIR` 与 `+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.parity_dump_dir` 传入 Ray 进程；`report_exact` 将全部 32 个 rank 的完整结果写入 `${RAIDEN_PARITY_DUMP_DIR}/raiden_exact_parity_step_<step>.json`，若目录创建或文件写入失败则直接记入 `coverage_errors` 并抛出 `RaidenParityError`（Fail-Closed，绝不在归档失败时静默返回 `PASS`）；
        - **仅在 Mismatch 时惰性触发的 Staging Buffer 诊断及明确归因边界**：`install_raiden_weights` 将融合前 `self._raiden_staging` 中的 `gate / up / down` 引用传给 `check_exact(..., staging_tensors=...)`，**仅当 `diff != 0.0`（且 `VERL_RAIDEN_DIAG_STAGING != "0"`）时才计算 `absmean` 与 `zero_frac` 并调用 `.item()`**，在无 mismatch 的正常同步中**零额外归约与零同步开销**。注意归因边界：由于 `self._raiden_staging` 是在 `self._raiden_ws.h2d()` **之后**读取的，若 `self._raiden_staging` 非零而融合后张量为零可定位 **H3**（`install_raiden_weights` 内部融合/转置异常）；若 `self._raiden_staging` 已为零则可排除 **H3**（证明损坏发生在 `install_raiden_weights` 拆包之前），**但仅凭该指标本身尚不能单独区分 H1（Trainer D2H / Follower 发送竞态）与 H2（Host gRPC 传输或 Sampler `h2d()` DMA 拷贝截断）**。

### 3.2 现有 `VERIFY_PARITY=all` 代码的实际检查能力边界

检视 [`verl/checkpoint_engine/raiden_checkpoint_engine.py#L332-L675`](../../../verl/checkpoint_engine/raiden_checkpoint_engine.py#L332-L675) 可确认，`VERIFY_PARITY=all` 由两个阶段组成，其覆盖范围和判定条件如下：

| 阶段 / 步数 | 触发函数 | 实际比对内容与通过条件 | 局限性与注意事项 |
| :--- | :--- | :--- | :--- |
| **Step 0（初始同步）** | `exact_install_kwargs` (`not global_steps`) $\rightarrow$ `check_exact` + `report_exact` | 在 `install_raiden_weights` 中（`_raiden_ws.h2d()` 及拆包/拼接/转置之后、覆盖 vLLM 参数之前），将每个 Sampler rank 收到的张量与 vLLM 从初始 HF checkpoint 加载的对应参数均转为 `float32`（`target.detach().float()`, `received.detach().float()`）后做逐元素数值比对 `max\|ref - new\| == 0.0`。在当前 2-D 折叠 MoE 路径下，每 rank 检查 `435` 个 vLLM 参数张量（`48 × 2 = 96` 个 MoE `w13/w2` + `48 × 7 = 336` 个每层稠密/路由/Norm 参数 + `3` 个全局 `embed_tokens / norm / lm_head`）。 | **仅在 Step 0 运行一次，且为 `float32` 逐元素数值相等（而非原始二进制位比较，例如不区分 `+0.0` 与 `-0.0`）**。注意：在 `p1c-retool-rl-06` 所用的基线代码 `c2e17624` 中，`report_exact` 仅检查 `len(results) > 0` 且 `num_mismatched == 0 and num_unresolved == 0`，未在代码内自动断言 `32` 个不同 rank 与 `[435]` 项/rank，因此审阅 `-06` 日志时须核对完整摘要行；本 PR 代码已新增 `(replica, rank)` `0..W_r-1` 集合与 `expected_checked` 强制断言。 |
| **Step 1 及后续步（`global_steps > 0`）** | `verify_norms` $\rightarrow$ `_compare_replica_norms` | 汇总 Trainer 所有 rank 导出的逐张量 `(numel, l1, l2_sq)` 与每个 Sampler Replica 聚合后的逐张量统计。**实际判定通过的条件仅检查 `numel`（`delta_numel == 0`）与 `L1` 范数（`delta_l1 <= 1e-3 * max(\|t_agg_l1\|, 1.0)`）**；全局 `L2` 范数虽然被计算并打印在日志中，但**未参与 `mismatches` 或最终 `if` 通过判定**。 | **Norm 通过 $\neq$ 逐元素数值相等**：一方面 Step 1+ 未做 Trainer 与 Sampler 的逐元素比对；另一方面范数检查对张量内部元素顺序错位/切片置换不敏感。若需要证明 Step 1+ 逐元素一致，必须额外实现 Trainer 当前权重与 Sampler 权重的逐元素对照。 |

---

## 4. 待验证的工作假设与源码级佐证（Working Hypotheses & Code-Level Evidence）

基于 `p1c-retool-rl-03`（Step 1 使用同步前预热轨迹时 `rollout_ppl=1.1322, ppl_ratio=0.9923, grad_norm=0.7330` 正常，Step 2 使用 Step-0 同步后权重时 `rollout_ppl=249.33` 乱码；Step 3 因 `timing_s/gen=0.02 s` 消费的是异步队列中已缓存的 Step-0 批次，不能独立证明 Step-1 同步后新生成的 rollout 仍然损坏）、`p1c-retool-rl-05`（Step 0 `10,414` 个张量 mismatch 且后续大量张量 `recv_absmean=0`）以及对 `p1a-moe-smoke-12` 和 `tpu_sync` C++ 源码的交叉核对，我们梳理出以下三个待受控运行最终确认的假设（按优先级排序）：

1. **工作假设 H1（主假设：旧版 `tpu-sync-torch==0.0.1.dev20261001220631` 中 `PushWeightsReshardedLocal` 并发合并 D2H 竞态导致 Follower 线程推送未初始化的全零 Host Staging Buffer）**：
   - **C++ 源码级机制佐证（`WeightSynchronizerBase::PushWeightsReshardedLocal`）**：
     - 在 `Qwen3-30B-A3B`（Sampler `TP=32, EP=32`）下，不同参数具有不同的目标等价类（32 路 TP/EP 切分的专家与线性层、`num_kv_heads=4` 在 32 个 TP rank 上 8 路复制的 `k_proj/v_proj`、以及 32 路全复制的 Norm 与路由门控 `mlp.gate`）。因此 `BroadcastEngine` 会对同一个传输 `uuid` 向源端 Trainer Worker 并发发送多个 `StartTransferRequest` RPC。
     - 在旧版 `tpu-sync-torch==0.0.1.dev20261001220631` 的 `WeightSynchronizerBase::PushWeightsReshardedLocal` 中，首个到达的 **Leader RPC 线程**在调用 `D2hLayer(l, uuid)` **之前**即把 `uuid` 插入 `completed_d2h_uuids_` 并释放锁；随后微秒级到达的 **Follower RPC 线程**检查 `completed_d2h_uuids_.insert(uuid).second` 得到 `already_completed = true`（日志分支 `Coalescing D2H copy (already completed for uuid ...)`），从而**完全跳过了 `d2h_layer_futures[l].Await()`**，直接将 Host Staging Buffer（`shard_info.host_ptr`）推送给远端 Sampler。
     - 在 **Step 0**（`WeightSynchronizer` 首次创建、Host Staging Buffer 由 `memset(0)` 初始化为全零）时，Trainer 与 Sampler 均按 `(-global_numel, name)` 顺序注册 `579` 个传输张量（前 `2` 个为词表嵌入，`idx=2..145` 按字典序排列 `layers.0, 1, 10..19, 2, 20..29, 3, 30..39, 4, 40..47, 5..9` 的 3 个 MoE 专家栈，其中 `idx=71, 72, 73` 恰好对应第 25 层 `layers.3` 的 `down_proj, gate_proj, up_proj`；`idx=146..578` 为 433 个稠密与 Norm 张量）。若 Follower 线程在 `idx=72/73`（对应拼接进 `layers.3.mlp.experts.routed_experts.w13_weight` 的 `gate_proj` / `up_proj`）附近追上并超越 Leader 线程尚未完成的 `D2hLayer` + `DetileBuffer`，就会导致其中一个专家栈（或半数切片）仍为初始全零，从而表现为 `layers.3...w13_weight` 的 `recv_absmean = 0.009274` 约为 `ref_absmean = 0.01824` 的一半（注：“其中一个栈或半个张量全零”是基于该均值比值与字典序位置的推断，而非由标量均值直接证明），而排在其后的 `layers.30..39, 4, 40..47, 5..9` 及 `qkv_proj` 因 Leader 尚未完成 D2H，被 Follower 线程直接推送了初始全零的 Host Staging Buffer（`recv_absmean = 0`）。
   - **来自 `p1a-moe-smoke-12` 的历史日志交叉印证**：
     - 回查同样运行在旧镜像 `@sha256:dcece4bc...` 上但未开启 `VERIFY_PARITY` 的 `p1a-moe-smoke-12`（`p1a_12_driver.log`）可见：**Step 0 初始同步后生成的预热批次（供 `step:1` 和 `step:2` 使用）同样发生了权重损坏**（`training_ppl = 6443.22` / `5379.75`，`rollout_corr/kl = 3.14`，GSM8K `critic/score/mean = 0.0`，生成长度 100% 撞满 `1024`）；因所有回答得分为 `0`，`step:1` 梯度范数 `actor/grad_norm = 0.0`（Trainer 权重未变）。
     - 而在 **`step:1` 结束触发的第二次 Raiden 权重同步（Step 1 Sync）** 中，由于复用了同一个 `self._trainer_raiden_ws`（`bind_weights`），如果该持久化 Host Staging Buffer 随后在 Step 0 结束前被 Leader 线程完整填入初始权重，这与 `P1a` 在第二次同步后生成的 `step:3` 轨迹恢复正常（**`training_ppl = 1.0964`, `rollout_ppl = 1.0943`, `rollout_corr/kl = 0.0017`, `critic/score/mean = 0.7578`, GSM8K 验证集准确率 `58.59%`**）的现象一致。
   - **新版 `tpu-sync-torch==0.0.1.dev20261006025601` 中的对应修复**：
     - 在 `verl-tpu:v20261006-tsync1006`（`@sha256:d92a2b3d...`）包含的 `tpu-sync-torch==0.0.1.dev20261006025601` 中，`PushWeightsReshardedLocal` 新增了逐层 D2H 完成状态同步结构 `UuidD2hState`（Follower 线程在推送每个 layer group 前必须通过 `d2h_state->mu.Await(...)` 等待 Leader 线程将该层的 `d2h_state->layer_ready[target_layer]` 或 `all_ready` 置为 `true`）。
   - **如何最终验证**：
     - 保持代码 commit `c2e176240dcff0a25911b0576880e46624f7249b` 与全部超参不变，仅将镜像切换为 `verl-tpu:v20261006-tsync1006`（`@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e`）并开启 `VERIFY_PARITY=all` 运行 **`p1c-retool-rl-06`**。
2. **工作假设 H2（`_raiden_ws.h2d()` 主机到设备 DMA 拷贝异常）**：
   - **机制推测**：Host Staging Buffer 已完整收到非零权重，但 `_raiden_ws.h2d()` 在批量拷贝 579 个张量到 TPU HBM 时在 `idx=73` 附近发生截断或清零。
   - **如何验证与归因边界**：由於 `tpu_utils.py` 中的 `self._raiden_staging` 位于 TPU HBM 且在 `self._raiden_ws.h2d()` **之后**读取，观察到 `self._raiden_staging[k]` 为零只能证明全零发生在 `install_raiden_weights` 拆包/融合之前（即排除 H3），**不能单独区分 H1 与 H2**。若新版镜像仍复现 mismatch 且 `self._raiden_staging[k]` 为零，需进一步对比 `self._raiden_ws.h2d()` 执行前的 CPU Host Buffer 或 C++ `tpu-sync` D2H/传输日志以区分 H1 与 H2。
3. **工作假设 H3（H2D 之后的 TPU 端张量视图拆包 / 转置异常）**：
   - **机制推测**：`self._raiden_staging[k]` 在 TPU HBM 中非零，但在 `_raiden_fused_expert_weights()` 或 `to_target_layout()` 执行 `view / cat / transpose / contiguous` 时因 XLA/PjRt 视图或延迟执行问题读到全零。
   - **如何验证**：若新版镜像仍复现 mismatch，通过本 PR 在 `check_exact`（仅 `diff != 0.0` 时触发）中记录的 `staging_gate_absmean / staging_up_absmean / staging_down_absmean` 与转换后 `new13 / new2` 的 `recv_absmean` 对比：若 `staging_*_absmean > 0` 而 `recv_absmean == 0`（或半零），即可直接定位为 H3。

---

## 5. 下一步分阶段受控验证计划与当前执行状态（Staged Execution Plan & Live Status）

遵循“**先固定代码只验证新版运行时镜像，按需再评估补丁，严格定义 Parity Gate**”的原则，分两步推进：

### 阶段一：环境审计与受控 Runtime 镜像验证（`p1c-retool-rl-06` 已执行完毕，实测命中情形 B）

1. **受控运行配置与执行结果（`p1c-retool-rl-06`，`09:27:52Z – 09:41:32Z`）**：
   - **Run Name**: `p1c-retool-rl-06`（与 `p1c-retool-rl-05` 保持完全相同的代码 commit `c2e176240dcff0a25911b0576880e46624f7249b` 与超参，**唯一变量为容器镜像** `verl-tpu:v20261006-tsync1006` `@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e`）；
   - **实测结果（触发情形 B）**：
     - `p1c-retool-rl-06` 在两个 `2×2×4` TPU v7x Slice 上正常完成模型加载与 XLA 编译，但在 Step-0 首次执行 `send_weights -> _create_and_register -> create_torch_weight_synchronizer` 时（尚未进入 D2H/P2P 传输与 `check_exact`）即抛出：
       ```text
       ValueError: Tensor's leading (block) dimension does not match its base storage buffer; raw transfer requires a layout-preserving full view of the block-major storage.
       ```
     - **原因确认**：在 `verl-tpu:v20261006-tsync1006` 所含的 `torch-tpu==0.1.2.dev20261006000518` 中，Trainer 端 `export_local_shards` 对 3-D MoE 专家权重（`[E_local=128, F, D]`）执行的 `local.reshape(local_shape).clone()`（折叠为 2-D `[128 * F, D]`）仍共享底层 3-D TPU storage buffer（其块主序首维仍为 `128` 而非 `128 * F`），被 `tpu-sync` 的 `UnpackTorchTensor` 拒绝注册。
2. **受控实验结果判定分支回顾与推进**：
   - **情形 A**：不适用（`-06` 在创建 `WeightSynchronizer` 时即触发情形 B）；
   - **情形 B（已触发）**：`p1c-retool-rl-06` 在 Step-0 绑定 2-D 折叠专家栈时明确抛出 `Tensor's leading (block) dimension does not match its base storage buffer`，按计划进入**阶段二**，合入 [`jialei777/verl-upstream#116`](https://github.com/jialei777/verl-upstream/pull/116)（`32b5133e`, `adapt_for_raw_transfer`）并结合本 PR 完整诊断代码启动 **`p1c-retool-rl-07`**；
   - **情形 C**：若后续 `p1c-retool-rl-07`（包含 `#116` 与本 PR 诊断代码，commit `a7c6be29`）仍复现 Step-0 mismatch，通过持久化到 `${RAIDEN_PARITY_DUMP_DIR}/raiden_exact_parity_step_0.json` 的未截断逐 rank 明细以及仅在 mismatch 时触发的 `self._raiden_staging`（H2D 后、融合前 `gate / up / down`）统计，先区分故障发生在 **`install_raiden_weights` 之前（H1 或 H2，`staging_*_absmean == 0`）** 还是 **H2D 后的融合/转置阶段（H3，`staging_*_absmean > 0` 但 `recv_absmean == 0`）**；若确认 `staging_*_absmean == 0`（排除 H3），再结合 Host Staging Buffer / C++ `tpu-sync` 日志进一步区分 H1 与 H2。

### 阶段二：合入 `#116`（`adapt_for_raw_transfer`）并执行 `p1c-retool-rl-07` 验证（当前运行中）

1. **已合入 `#116` 并提交受控验证运行（`p1c-retool-rl-07`）**：
   - 鉴于阶段一 `p1c-retool-rl-06` 实测触发 2-D folded expert stack 的 `UnpackTorchTensor` 错误（情形 B），本 PR 已 cherry-pick [`jialei777/verl-upstream#116`](https://github.com/jialei777/verl-upstream/pull/116)（`32b5133ee6414979cb5947d24dc0750e7556e497` -> 本分支 `a7c6be29965cc567c666ba01613dfcddb5578989`）：在首次同步前通过 `probe_raw_transfer_rejected_classes` 探测被 `UnpackTorchTensor` 拒绝的张量等价类，仅对该类张量（即 2-D 折叠 MoE 专家栈）在 `adapt_for_raw_transfer` 中执行 CPU host round-trip 重新分配真 2-D block-major TPU buffer，稠密参数保持零开销直传；
   - **当前运行 `p1c-retool-rl-07`**：已基于 commit **`a7c6be29965cc567c666ba01613dfcddb5578989`**（同时包含 `#116` `adapt_for_raw_transfer` 与本 PR 新增的 `(replica, rank)` `0..31` 集合断言、`compute_expected_checked_params` `435` 项/rank 覆盖断言、`raiden_exact_parity_step_0.json` 持久化归档及 mismatch-only staging 诊断）搭配新版镜像 `verl-tpu:v20261006-tsync1006`（`@sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e`）与 `VERIFY_PARITY=all` 在 `bodaborg-tpu7x-nap` 提交运行（`gs://ubench-logs/xiaotongyang/retool-v7x/runs/p1c-retool-rl-07/`）；
   - [`verl-project/verl-hardware-plugin#89`](https://github.com/verl-project/verl-hardware-plugin/pull/89) 的 3-D direct install 方案因涉及跨仓库依赖，单独评估，不与 `#116` 混用。
2. **明确定义的 `Stage P1c` 通过门槛（Gate Criteria）**：
   - **Step 0（`exact` 逐元素数值相等校验门槛）**：在 `VERIFY_PARITY=all` 下，全部 **32 个 Sampler ranks** 的全部预期检查项（当前 2-D 折叠路径下为 `435 / 435` 项/rank；若未来采用 3-D direct install 则按对应 expert region 检查项计数）必须 **`exact` 100% 通过（转为 `float32` 后 `max|ref - new| == 0.0`，`0 mismatched, 0 unresolved`）**；
   - **Step 1–2（`norm` 范数校验 + 训练/推理闭环健康度门槛）**：
     - 通过现有 `verify_norms` 校验（全模型 `numel` 与逐张量/全局 `L1` 范数在 `1e-3` 相对容差内一致；**注：报告中须明确注明 Step 1–2 的 `norm` 通过仅验证 `numel` 与 `L1` 范数一致，不等于逐元素数值相等**）；
     - 同步后生成的 Rollout 困惑度比值 `actor/rollout_corr/ppl_ratio` 稳定在 `~1.0` 正常区间（无乱码或 PPL 暴涨）；
     - 连续完成真实梯度更新（`actor/grad_norm > 0`），触发多轮 SandboxFusion 代码工具调用，且 `rollout_audit` 的 Token 级 `response_mask` 违规数为 **`0`**。
3. **证据归档与推进 `Stage P3`**：
   - 完整保存 `Stage P1c` 的原始控制台日志、Hydra resolved config、容器包版本清单与逐 rank parity 明细；
   - 在 `Stage P1c` 满足上述全部门槛后，再启动 **`Stage P3`（TPU v7x 25-step 正式 RL 训练 + Step 0/25 AIME 2024/2025 `n=30` 评估）**。
