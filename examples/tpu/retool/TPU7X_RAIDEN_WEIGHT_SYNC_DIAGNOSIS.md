# TPU v7x ReTool RL（P1c）Raiden 初始权重同步 Tensor Mismatch 诊断报告

## 1. 问题概述（Executive Summary）

在 TPU v7x 双 Slice（`2 × 4x4 = 32` 芯片/Slice，共 64 芯片：32 芯片 TorchTitan Trainer + 32 芯片 vLLM Rollout Sampler）上运行 **Qwen3-30B-A3B MoE 多轮 ReTool RL Smoke（Stage P1c）** 时，初始 `update_weights(0)`（Step-0 Raiden P2P 权重同步）在 `2.65 s` 内返回完成，但开启 `VERIFY_PARITY=all`（`RaidenParityCheck` 逐张量精确比对与全局范数校验）后捕获到确定性的 **部分传输截断（Partial Transfer Truncation / Zero-Filled Staging Buffers）**：

- **前 `73 / 579` 个传输张量（对应 vLLM 侧 `110 / 435` 个参数张量，约 `29.4 GiB`）100% 逐位一致（`max|diff| = 0.0`）**：包括 `lm_head.weight`、`model.embed_tokens.weight`、`layers.0, 1, 10..19, 2, 20..29` 共 23 层的全部 MoE 专家权重（`w13_weight` 与 `w2_weight`），以及 `layers.3.mlp.experts.routed_experts.w2_weight`（wire index `71`）和 `layers.3.mlp.experts.routed_experts.w13_weight` 的前半部分 `gate_proj.weight`（wire index `72`）。
- **从第 `74` 个传输张量（wire index `73`：`model.layers.3.mlp.experts.up_proj.weight`）开始，后续 `506 / 579` 个传输张量（对应 vLLM 侧 `325 / 435` 个参数张量）在 Sampler 端主机暂存缓冲区（Host Staging Buffer）中全部为 `0.0`**：
  - `layers.3.mlp.experts.routed_experts.w13_weight` 因前半 `gate_proj`（`idx=72`）正常接收、后半 `up_proj`（`idx=73`）全零，导致 `recv_absmean = 0.009274 ≈ 0.5 × ref_absmean (0.01824)`；
  - 其余 24 层 MoE 专家权重（`layers.30..39, 4, 40..47, 5..9`，`idx=74..145`）以及全部 `433` 个 Dense 权重（`idx=146..578`，包括所有层的 `q_proj / k_proj / v_proj / o_proj`、MoE router `gate.weight` 与 RMSNorm）在 `_raiden_ws.h2d()` 执行时均为 `0.0`。

### 核心结论：这是 Runtime Bug 还是漏合了 Upstream Patch？

**两者兼有（Both）：**
1. **底层 `tpu-sync`（Raiden）运行时 Bug（主因）**：P1c 运行所使用的基础镜像搭载的是 10 月 1 日构建的 `tpu-sync-torch==0.0.1.dev20261001220631`。该版本在多主机 Direct-Push 路径（`can_fast_path_direct=True`）及接收端完成计数上存在已知缺陷，`transfer_future.wait()` 在仅完成约前 `73` 个大张量（~29.4 GiB）的数据搬运时即提前返回，导致 Sampler 在后续 `506` 个张量尚未写入主机暂存缓冲区时便执行了 `install_raiden_weights`（`_raiden_ws.h2d()`）。这些缺陷已在后续 `tpu-sync-torch==0.0.1.dev20261006025601`（对应 `verl-tpu:v20261006-tsync1006` 镜像）中修复（移除 broken `direct_schedules` 快路径并统一走 `BroadcastEngine`、修复多层 Host Buffer 容量计算及接收端 per-layer chunk 完成计数）。
2. **缺少两个上游补丁（Missing Upstream Patches）**：我们的分支基于 `a27db71c`（2026-10-03），尚未合入此后提交的两个关键补丁：
   - **`verl-project/verl-hardware-plugin#89` / `jialei777/verl-upstream#110`**（*Export MoE expert stacks as 3-D `[E, I, H]` for Raiden weight sync*）：将 MoE 专家栈从 2-D 折叠导出（`[E * rows, cols]` + `.reshape(local_shape).clone()`）改为原生 3-D `[E, I, H]` 导出，消除每张芯片约 `3.8 GiB` 的额外 2-D clone 缓冲并对齐 `tpu-sync` 原生 3-D 分片语义。
   - **`jialei777/verl-upstream#116`**（*Clone non-matching-block-dim tensors via CPU in `adapt_for_raw_transfer`*）：在升级到新版 `torch-tpu`（`>= 0.2.0`）后，`tensor.reshape(...).clone()` 在 TPU 设备上通过 `CopyTpuToTpu` 复用原 3-D 底层存储布局，会触发 `tpu-sync` 的 `UnpackTorchTensor: Tensor 'layer_0_shard_0' leading (block) dimension (6144) does not match its base storage buffer (4)` 报错；`#116`（或 `#110/#89` 的原生 3-D 导出）是切换到新版镜像的必要补丁。

---

## 2. 复现实验现象与 P1a vs P1c 对比

### 2.1 运行配置对比

| 维度 | Stage P1a（`p1a-moe-smoke-12`） | Stage P1c（`p1c-retool-rl-03`） | Stage P1c + Parity（`p1c-retool-rl-05`） |
| :--- | :--- | :--- | :--- |
| **任务与序列长度** | 单轮 GSM8K Smoke（`max_prompt=512, max_resp=512`） | 多轮 ReTool Math RL（`max_prompt=2048, max_resp=16384`） | 多轮 ReTool Math RL（`max_prompt=2048, max_resp=16384`） |
| **模型与起点** | `Qwen/Qwen3-30B-A3B` | P2 Step-372 SFT Checkpoint（30B-A3B） | P2 Step-372 SFT Checkpoint（30B-A3B） |
| **Trainer 并行拓扑** | 1-D FSDP：`dp_shard=32, cp=1` | 2-D FSDP + Ring Attention：`dp_shard=8, cp=4` | 2-D FSDP + Ring Attention：`dp_shard=8, cp=4` |
| **Trainer Eager 模式** | `TPU_EAGER_MODE=DEFER_AND_FUSE` | `TPU_EAGER_MODE=DEFER_NEVER` | `TPU_EAGER_MODE=DEFER_NEVER` |
| **Sampler 并行拓扑** | 1 Replica × 32 Chips（`TP=32, EP=32`） | 1 Replica × 32 Chips（`TP=32, EP=32`） | 1 Replica × 32 Chips（`TP=32, EP=32`） |
| **`VERIFY_PARITY` 设置** | `off`（默认关闭） | `off`（默认关闭） | `all`（开启 `exact + norm` 校验） |
| **Step-0 初始权重同步耗时** | `2.60 s` | `2.73 s` | `2.65 s`（P2P 传输 `2.2080 s` + H2D `0.4275 s`） |
| **Step-0 同步后行为** | Step 1–2 Warmup Rollout 同样乱码（`ppl_ratio=31.53 / 32.18, score=0.0`），Step 1 训练后 Step 3 恢复（`ppl_ratio=1.0020, score=0.7578`） | Step 1（使用预热在 Step-0 同步前生成的 Rollout）完全正常（`ppl_ratio=0.9923`），Step 2–3（使用同步后权重）乱码（`rollout_ppl=249.3, score=-1.0`） | 在 Step-0 `update_weights(0)` 阶段直接拦截并抛出 `RaidenParityError` |

### 2.2 为什么 P1a 看起来“通过”了，而 P1c 暴露了问题？

回溯 `p1a-moe-smoke-12` 与 `p1c-retool-rl-03` 的逐步指标可以发现，**Step-0 初始 Raiden 权重同步实际上在 P1a 和 P1c 中都发生了不完整传输**：

1. **在 `p1a-moe-smoke-12` 中**：
   - 当时 `VERIFY_PARITY` 默认关闭（`off`），且 `NUM_WARMUP_BATCHES=2` 在 `update_weights(0)` **之后**启动异步生成。
   - 因此 `p1a-moe-smoke-12` 的 Step 1 和 Step 2 实际上都使用了被 Step-0 Raiden 同步破坏的 Sampler 权重：`rollout_corr/ppl_ratio` 高达 `31.530` 和 `32.180`，`rollout_corr/kl` 为 `3.1396` 和 `3.1676`，奖励 `critic/score/mean = 0.0000`，导致 `actor/grad_norm = 0.0000`。
   - 到了 Step 1 训练结束后的第二次同步 `update_weights(1)`（此时双端 `WeightSynchronizer` 已完成首次内存页初始化与连接建立，且 P1a 为 1-D `dp_shard=32, cp=1`），同步成功完成，Step 3 恢复正常（`rollout_corr/ppl_ratio = 1.0020`，`critic/score/mean = 0.7578`）。
2. **在 `p1c-retool-rl-03` 中**：
   - 在 Attempt 2（`NUM_WARMUP_BATCHES=1`）中，预热 Batch 在 Step-0 `update_weights(0)` 覆盖 vLLM 权重之前就已经由 HF 初始权重生成完毕：
     - **Step 1 完全正常**：`rollout_ppl = 1.1322`、`training_ppl = 1.1235`、`ppl_ratio = 0.9923`、`kl = -0.0080`，触发 `18` 次沙箱代码工具调用，Multi-turn response/tool mask 0 违规，`actor/grad_norm = 0.7330`。这证明 **vLLM 多轮采样、SandboxFusion 工具调用、TorchTitan `SimpleFSDP=8, CP=4` 前反向训练本身全部工作正常**。
     - **Step 2–3 权重被破坏**：一旦使用经 Raiden 同步后的 Sampler 权重生成新轨迹，vLLM 立即输出乱码（`rollout_ppl = 249.33`，`ppl_ratio = 0.0156`，奖励全部为 `-1.0`）。

---

## 3. `p1c-retool-rl-05`（`VERIFY_PARITY=all`）逐张量精确取证

在 `p1c-retool-rl-05` 中开启 `VERIFY_PARITY=all`（来自 `verl-project/verl-hardware-plugin#85`）后，`update_raiden_weights(global_steps=0)` 输出了完整的遥测与逐张量对比日志：

### 3.1 Raiden 编排器阶段耗时

```text
Trainer Rank 0: registering 578 sharded and 1 full tensors
[RAIDEN CONTROLLER] All 32 Trainer and 32 Sampler units verified and registered.
[RAIDEN TELEMETRY | Orchestrator] Step 0 Completed in 21.84s:
  * Sampler Quiesce/Pause  : 0.0000s
  * Trainer Raiden Init    : 13.7112s
  * Sampler Raiden Init    : 5.4735s
  * Raiden Barrier Check   : 0.0005s
  * RaidenController P2P   : 2.2080s
  * Sampler H2D DMA        : 0.4275s
```

### 3.2 传输顺序与精确截断点（Wire Index `72 -> 73`）

`RaidenCheckpointEngine.send_weights()` 与 `vLLMRaidenWorkerExtension.init_raiden_sync_on_worker()` 均按 `(-global_numel, name)` 对 `579` 个传输张量（Wire Tensors）排序注册：

| Wire Index 范围 | 张量类别与名称（按 `(-global_numel, name)` 排序） | 单张量全局形状（2-D 折叠模式） | 单张量全局大小 | Sampler 接收结果（全 32 Ranks） |
| :--- | :--- | :--- | :--- | :--- |
| `idx = 0..1` | `lm_head.weight`, `model.embed_tokens.weight` | `[151936, 2048]` | `622.3 MB` × 2 | **PASS**（`max\|diff\| = 0.0`） |
| `idx = 2..70` | 23 个 MoE 层（字典序 `layers.0, 1, 10..19, 2, 20..29`）的 `down_proj`、`gate_proj`、`up_proj` | `[98304, 2048]` 或 `[24576, 768]` | `402.7 MB` × 69 | **PASS**（对应 46 个 vLLM `w13/w2` 张量，`max\|diff\| = 0.0`） |
| `idx = 71` | `model.layers.3.mlp.experts.down_proj.weight` | `[98304, 2048]` | `402.7 MB` | **PASS**（`layers.3...w2_weight` `max\|diff\| = 0.0`） |
| `idx = 72` | `model.layers.3.mlp.experts.gate_proj.weight` | `[98304, 2048]` | `402.7 MB` | **PASS**（写入 `layers.3...w13_weight` 前半部分正常） |
| **`idx = 73`（截断点）** | **`model.layers.3.mlp.experts.up_proj.weight`** | **`[98304, 2048]`** | **`402.7 MB`** | **FAIL（全 0）**：导致 `layers.3...w13_weight` 后半全 0，`recv_absmean = 0.009274` vs `ref_absmean = 0.01824` |
| `idx = 74..145` | 其余 24 个 MoE 层（字典序 `layers.30..39, 4, 40..47, 5..9`）的 `down_proj`、`gate_proj`、`up_proj` | `[98304, 2048]` | `402.7 MB` × 72 | **FAIL（全 0）**：`recv_absmean = 0.000000` |
| `idx = 146..578` | 全部 48 层的 Dense 权重（`q_proj`, `o_proj`, `k_proj`, `v_proj`, `mlp.gate.weight`, `RMSNorm`） | `[4096, 2048]` 等 | 共 `433` 个张量 | **FAIL（全 0）**：`recv_absmean = 0.000000` |

全局范数汇总（`verify_norms`）同样印证了这一截断比例：
- **Trainer 全局参数量与范数**：`numel = 30,532,122,624`，`L1 = 278,864,458.65`，`L2 = 2,711.61`
- **Sampler 接收后范数**：`numel = 30,532,122,624`，`L1 = 135,517,090.30`（正好约为前 `73 / 146` 个大张量的 L1 范数总和），`L2 = 1,890.02`

---

## 4. 根因剖析（Root Cause Analysis）

### 4.1 根因一：`tpu-sync-torch==0.0.1.dev20261001220631` 的 Direct-Push 与完成计数缺陷

在 8 台 Trainer 主机（32 芯片）→ 8 台 Sampler 主机（1 个 Replica × 32 芯片）的拓扑下：
- Sampler 的 32 个 worker 分别持有不同的 `(TP, EP)` 分片，因此在 `ReshardPlanner` 中被划分为 `32` 个等价类（每个等价类长度 `max_eq_class_len = 1 <= n_seed`）。
- 在 10 月 1 日构建的 `tpu-sync-torch==0.0.1.dev20261001220631` 中，这会命中 `can_fast_path_direct = True` 的旧版 `direct_schedules` 路径，且在多主机 chunk 折叠/聚合（`coalesced_transfers`）与接收端 `expected_block_count` / `expected_layer_chunk_counts` 跟踪之间存在不一致，同时 `RaidenController.start_transfer` 在 Direct-Push 模式下未等待接收端所有层的 `OnDataReceived` 确认即返回。
- 结果是：在 `2.2080 s` 的网络传输窗口内，仅前 `73` 个排在最前面的大张量（约 `29.4 GiB`，对应约 `13.3 GB/s` 聚合传输速率）完成了写入，随后编排器立即触发 `install_raiden_weights`（`_raiden_ws.h2d()`）并调用 `release_raiden_sync_buffers()` 释放了 Trainer 侧绑定张量，导致后续 `506` 个张量在 Sampler 端保持为全零初始态。
- **修复版本**：在 `tpu-sync-torch==0.0.1.dev20261006025601`（打包在 `verl-tpu:v20261006-tsync1006` 镜像中）中：
  1. 移除了有缺陷的 `can_fast_path_direct` / `direct_schedules` 分支，将所有权重同步统一收敛到 `BroadcastEngine` 流水线调度；
  2. 修复了 PyTorch `WeightSynchronizer` 多层 Host Buffer 按层容量查询（此前误用第 0 层 `slice_byte_size()`）以及非 `(8, 128)` 对齐分片的 `skip_tiling` 判定；
  3. 补全了接收端 `expected_layer_chunk_counts` 的注册与完成等待。

### 4.2 根因二：P1c（`CP=4, FSDP=8`）下的分片导出与上游缺失补丁

在 P1c 中，为了容纳 16K 多轮工具交互序列，Trainer 启用了 `context_parallel_size=4, data_parallel_shard_size=8`。在 `verl/workers/engine/torchtitan/transformer_impl.py` 中，TPU 默认使用 `spmd_backend="default"`，TorchTitan 的 `apply_simple_fsdp` 将权重分片在扁平化的 1-D `"fsdp"` mesh（大小 `dp_shard * cp = 32`）上，唯独启用了 `_split_qkv_on_save` 的 `q_proj / k_proj / v_proj` 在导出 hook 中返回 2-D `(dp_shard=8, cp=4)` mesh，从而在 `export_local_shards()` 中触发跨芯片 all-gather 并作为 replicated 张量注册（这也是日志中显示 `registering 578 sharded and 1 full tensors` 或部分张量走 full-tensor 路径的原因）。

此外，我们的分支（基于 `a27db71c`）缺少此后在上游合入的两个关键 PR：

1. **MoE 专家栈 3-D `[E, I, H]` 原生导出（`verl-project/verl-hardware-plugin#89` / `jialei777/verl-upstream#110`）**：
   - 当前分支的 `expert_stack_wire()` 将 3-D MoE 专家栈 `[128, 768, 2048]` 折叠为 2-D `[98304, 2048]`，并在每次同步时执行 `local.reshape(local_shape).clone()`，在每张芯片上额外分配约 `3.8 GiB` 的 2-D 拷贝缓冲。
   - 合入 `#89` / `#110` 后，Trainer 与 Sampler 直接以 3-D `[128, 768, 2048]` / `[128, 2048, 768]` 注册给 Raiden，无需在 Trainer 端 `.reshape().clone()`，也无需在 Sampler 端做二次 2-D 拆包。
2. **新版 `torch-tpu` 下折叠张量视图的底层存储块维度检查（`jialei777/verl-upstream#116`）**：
   - 当切换到包含新版 `tpu-sync` 的镜像（同时包含 `torch-tpu >= 0.2.0`）时，TPU 上的 `tensor.reshape(2D).clone()` 会通过 `CopyTpuToTpu` 继承源 3-D 张量的底层 PjRt 存储描述符（leading dim 仍为 `4` 个本地专家，而非折叠后的 `4 × 1536 = 6144`）。
   - 这会导致 `tpu-sync` 的 `UnpackTorchTensor` 抛出：
     ```text
     RuntimeError: UnpackTorchTensor: Tensor 'layer_0_shard_0' leading (block) dimension (6144) does not match its base storage buffer (4)
     ```
   - `jialei777/verl-upstream#116` 在 `adapt_for_raw_transfer()` 中通过 `.cpu().to(t.device)` 强制分配匹配 2-D 形状的新设备存储；而同时合入 `#110`（`verl-hardware-plugin#89`）直接导出 3-D 专家栈则从源头避免了 2-D 折叠视图。

---

## 5. 修复与验证计划（Remediation Plan）

1. **Cherry-pick 上游补丁**：
   - Cherry-pick `jialei777/verl-upstream#110`（对应 `verl-project/verl-hardware-plugin#89`）：将 MoE 专家权重改为 3-D `[E, I, H]` 原生 Raiden 导出；
   - Cherry-pick `jialei777/verl-upstream#116`：在 `adapt_for_raw_transfer` 中对底层存储 leading dim 不匹配的视图走 CPU 中转克隆，确保在 `torch-tpu 0.2.0` 下兼容所有 reshape 视图。
2. **升级 TPU 运行时镜像**：
   - 将 P1c JobSet 镜像从 `verl-tpu:v20261001-tsync990787257`（`tpu-sync-torch==0.0.1.dev20261001220631`）切换至包含修复后 `tpu-sync` 的 `verl-tpu:v20261006-tsync1006`（`tpu-sync-torch==0.0.1.dev20261006025601`）。
3. **验证标准（Gate Criteria）**：
   - 开启 `VERIFY_PARITY=all` 运行 `P1c`，确认 Step 0、Step 1、Step 2 的 `RaidenParityCheck`（`435 / 435` 个 vLLM 张量 exact check + distributed L1/L2 norm check）100% 通过；
   - 确认 Step 1–2 的 `rollout_corr/ppl_ratio` 保持在 `~1.0` 附近，随后推进 `P3`（25-step 正式训练与 AIME 2024/2025 评估）。
