# Reproduction results: ReTool (Qwen3-30B-A3B) on TPU 7x

Status: **In progress** (started 2026-10-29). P0 preparation is complete; hardware stages (`P1a`, `P1b`, `P1c`, `P2`, `P3`) are executing and recorded below as each stage finishes.
Plan: [`REPRO_PLAN_TPU7X.md`](REPRO_PLAN_TPU7X.md) (rev 3).
Owner: @syangx39.

---

## 1. P0 preparation verification summary

All P0 preparation deliverables (`P0.1`–`P0.4`, `P0.6`, `P0.8`–`P0.10`) are merged into `tpu-main` at commit `cfabaada` and verified before hardware submission:

| Item | PR / Commit | Verification | Status |
|---|---|---|---|
| `P0.1` JobSet launcher (`examples/tpu/gke/run_ray_jobset_v7x_nap.sh`) | #4 (`cfabaada`) | Server-side `kubectl create --dry-run=server` passed on the TPU 7x GKE cluster for both `ray-jobset-v7x-nap.yaml` and `ray-jobset-v7x-nap-prewarm.yaml`; 23 CPU tests in `tests/special_tpu/test_v7x_nap_launcher_on_cpu.py` passed (including UID-precondition delete, separated stdout/stderr `owned_create` with `launch-id` adoption, and fail-closed atomic GCS attempt-slot claiming) | PASS |
| `P0.2` Container image & package audit | #4 (`cfabaada`) | Inspected OCI layers of reference image `verl-tpu:v20261001-tsync990787257` (`sha256:dcece4bc3813359450f48d4523e5bd78e3139dc441a88b5c2e4d638caa054cbe`) and CI image `verl-tpu:v20261006-tsync1006` (`sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e`); findings documented in section 1.1 | PASS |
| `P0.3` Staged model & datasets | - | Staged under `<INPUTS_DIR>` with verified SHA-256 digests and row counts (section 1.2) | PASS |
| `P0.4` & `P0.6` Output persistence & allowlisted env dump | #4 (`cfabaada`) | `examples/tpu/gke/v7x_nap/pod_env_snapshot.py` writes allowlisted, secret-redacted pod and `/proc` Ray worker env snapshots (`env/<pod>-env.json`, `env/<pod>-procs.jsonl`) to `<RUN_DIR>/env/` on GCS | PASS |
| `P0.5` Host ordering preparation | #4 (`cfabaada`), #5 (`ed0170cf`) | `tpu_node_rank.py` computes topological/job-index rank and labels each Ray node (`verl-tpu-node-rank`, `verl-tpu-slice`); `RayResourcePool.get_placement_groups()` sorts by `(slice_key, rank, ip)` on every call and logs `slice=`, `rank=`, and `tpu-worker-id=`; 102 CPU tests in `tests/single_controller/test_placement_group_host_order_on_cpu.py` passed | PASS (hardware validation in `P1a` / `R2`) |
| `P0.8`, `P0.9`, `P0.10` Observability & rollout audit | #6 (`32c4efd1`) | Runtime `training.max_norm` vs `optim.clip_grad` logged in `TorchTitanEngine`; opt-in `+trainer.rollout_audit_dir` dumps token IDs, `response_mask`, decoded segments, reward I/O, `num_turns`, and `tq_layouts`; `summarize_validation_dump` enforces completeness by `question_id` and session ID (`mean_acc=None` if incomplete); 27 CPU tests passed in `tests/trainer/ppo/v1/test_rollout_audit_on_cpu.py` | PASS |

### 1.1 Container image package audit (`P0.2`)

| Component | Reference image (`v20261001-tsync990787257`, `sha256:dcece4bc...`) | Newer CI image (`v20261006-tsync1006`, `sha256:d92a2b3d...`) | Impact on TPU 7x runs |
|---|---|---|---|
| `python` / `ray` | Python 3.12, `ray==2.56.0` | Python 3.12, `ray==2.56.0` | Compatible |
| `torch` / `torch-tpu` | `torch==2.13.0+cpu`, `torch-tpu==0.1.1.dev20260912075415` | `torch==2.13.0+cpu`, **`torch-tpu==0.1.2.dev20261006000518`** | `multihost_wrapper.py` sets `NODE_RANK = JOB_COMPLETION_INDEX`; launcher defaults to `NODE_RANK_SOURCE=job-index`. Note: `v20261006-tsync1006` upgrades **both** `torch-tpu` and `tpu-sync-torch` |
| `libtpu` / `jax` | `libtpu==0.0.47`, `jax==0.10.2` | `libtpu==0.0.47`, `jax==0.10.2` | Compatible |
| `tpu-sync-torch` | `0.0.1.dev20261001220631` (no `unbind_weights()`) | `0.0.1.dev20261006025601` (has `unbind_weights()` and per-layer `UuidD2hState` wait in `PushWeightsReshardedLocal`) | On `v20261001-tsync990787257`, runs must pass `RELEASE_SYNC_BUFFERS=False` and `VERL_RAIDEN_RELEASE_BUFFERS=0` so `RaidenCheckpointEngine` does not call missing `unbind_weights()` |
| `vllm` / `vllm-torchtpu` | `vllm==0.29.0`, `vllm-torchtpu` @ `9faafb17` | `vllm==0.29.0`, `vllm-torchtpu` @ `9faafb17` | Compatible |
| `torchtitan` | `0.1.0` @ `f880cc73` (`TrainingConfig.max_norm` default `1.0`) | `0.1.0` @ `f880cc73` | Confirms `D7` static inspection (verified at runtime via `P0.9` log) |
| `TransferQueue` | `0.1.8` (no `save_checkpoint` / `load_checkpoint`) | `0.1.8` (no `save_checkpoint` / `load_checkpoint`) | Neither image has `TransferQueue >= 0.1.9`; `R13` can only pass via a verified quiescent checkpoint boundary, otherwise multi-window RL segments are reported separately |

### 1.2 Staged input artifacts (`P0.3`)

| Artifact | Relative path under `<INPUTS_DIR>` | Rows / Shards | Bytes | SHA-256 / Commit |
|---|---|---|---|---|
| Base model (`Qwen/Qwen3-30B-A3B`) | `assets/hf/Qwen3-30B-A3B/` | 16 safetensors shards (`model-00001-of-00016` .. `00016`) + `model.safetensors.index.json` + config/tokenizer | 61,084,187,391 | HF commit `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| ReTool SFT dataset | `dataset/ReTool-SFT/data/train-00000-of-00001.parquet` | 2,000 rows (`max_turns` up to 17) | 8,901,717 | `92a29cea03f72dc79066e1697f4d7423d91bd6a02bd1aac50f5c9dfd89b27faa` |
| DAPO-Math-17k | `dataset/BytedTsinghua-SIA/DAPO-Math-17k/data/train-00000-of-00001.parquet` | 1,791,700 rows (17,917 unique prompts x 100) | 715,244,619 | `22e266f54502ce271ee08713feb8b30caf97a4ea948aca0e0506f58c4cdd0545` |
| AIME 2024 | `dataset/Maxwell-Jia/AIME_2024/data/train-00000-of-00001.parquet` | 30 rows | 29,259 | `1332803ea03085cce9222b6c0b295b4115507ab5984aee9d3e9cb3db3c2ee948` |
| AIME 2025 | `dataset/yentinglin/aime_2025/data/train-00000-of-00001.parquet` | 30 rows | 18,454 | `4f4432a5252cf673ddd696a3dba24f75bfd662c0ef1d54b6e0f442c24119f8ed` |
| GSM8K train (`P1a`) | `data/gsm8k/train.parquet` | 7,473 rows | 2,095,889 | `89cd3cb8d28e5274e7f0bf71ff541ea5654ac9e30589ac4b5d19c3f783a3858c` |
| GSM8K test (`P1a`) | `data/gsm8k/test.parquet` | 1,319 rows | 377,550 | `e801bce6b15925630ea9976dc14419e49735a90fd54d267911e6701ebdc0d489` |

---

## 2. Run log

| Stage | Run name | Code SHA | Image digest | Slices (`2x2x4`) | Steps (start->end) | Submit / Admit / Ready / End (UTC) | Median step time | Status | Output prefix |
|---|---|---|---|---|---|---|---|---|---|
| `P1b` | `p1b-retool-sft-01` | `cfabaada` | `sha256:dcece4bc...` | 1 (1 trainer) | 1->3 | 11:06:20 / 11:15:33 / 11:16:31 / 11:32:31 | 28.4 s (steps 1–3: 55.6 s, 28.4 s, 27.1 s) | **SUCCEEDED** (exit 0) | `<RUN_DIR>/p1b-retool-sft-01` |
| `P2` | `p2-retool-sft-01` | `cfabaada` | `sha256:dcece4bc...` | 1 (1 trainer) | 1->372 | 11:46:17 / 12:03:14 / 12:04:12 / 13:36:05 | 9.57 s mean / 10.0 s median warm (Epochs 2–6; Epoch 1 mean 15.73 s) | **SUCCEEDED** (exit 0) | `<RUN_DIR>/p2-retool-sft-01` |
| `P1a` | `p1a-moe-smoke-12` | `6fa65e49` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 1->3 | 19:07:34 / 19:11:31 / 19:13:03 / 19:36:40 | 11.45 s (`398.5 s` step 1 compile, `11.45 s` step 2, `7.94 s` step 3; Raiden sync `2.25–2.35 s` P2P / `3.04–3.21 s` total) | **SUCCEEDED WITH CAVEAT** (exit 0; topology/placement & Step-3 recovery `score=0.7578, ppl_ratio=1.0020` after Step-1 sync verified, but Steps 1–2 warmup batches after Step-0 initial sync had corrupted weights `ppl_ratio=31.5/32.2, grad_norm=0.0`, so job exit 0 alone does not prove Step-0 sync correctness) | `<RUN_DIR>/p1a-moe-smoke-12` |
| `P1c` | `p1c-retool-rl-03` (att. 1) | `6162456a` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 1->2 (`num_warmup=2`) | 20:54:25 / 20:54:25 / 20:55:47 / 21:17:41 | `413.83 s` step 1, `11.71 s` step 2 (Raiden sync `3.92 s` / `4.29 s`) | **FAILED GATE** (exit 0, both warmup steps used corrupted step-0 Raiden sync: `rollout_ppl=249.3/144.1`, `score=-1.0`, `grad_norm=0.0`) | `<RUN_DIR>/p1c-retool-rl-03` |
| `P1c` | `p1c-retool-rl-03` (att. 2) | `c2e17624` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 1->3 (`num_warmup=1`) | 00:26:52 / 00:26:52 / 00:28:14 / 00:50:19 | `317.85 s` step 1, `438.92 s` step 2, `11.77 s` step 3 | **PARTIAL / FAILED GATE** (exit 0; Step 1 pre-sync batch had valid `rollout_ppl=1.1322`, `ppl_ratio=0.9923`, `grad_norm=0.7330`, `18` tool calls, `0` mask violations; Step 2 post-step-0-sync and Step 3 buffered post-step-0-sync batch had corrupted `rollout_ppl=249.33`, `ppl_ratio=0.0156`) | `<RUN_DIR>/p1c-retool-rl-03` |
| `P1c` | `p1c-retool-rl-05` (`VERIFY_PARITY=all`) | `c2e17624` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 0 (init sync parity check) | 01:41:28 / 01:51:55 / 01:53:18 / 02:06:21 | failed fast at step-0 `RaidenParityCheck.report_exact` | **DIAGNOSTIC FAIL-FAST** (`RaidenParityError` at Step 0 across `32` sampler ranks x `435` tensors/rank = `13,920` checks: **`10,414` total mismatched**, `0` unresolved; `rank 0` first 60 truncated detail lines show `layers.3...w13_weight` at `recv_absmean=0.009274` vs `ref_absmean=0.01824` and subsequent printed tensors at `recv_absmean=0`) | `<RUN_DIR>/p1c-retool-rl-05` |
| `P1c` | `p1c-retool-rl-06` (`VERIFY_PARITY=all`, newer image) | `c2e17624` | `sha256:d92a2b3d...` | 2 (1 trainer + 1 rollout) | 0 (init sync registration) | 05:22:13 / 09:26:03 / 09:27:52 / 09:41:32 | failed at step-0 `WeightSynchronizer` registration | **CASE B TRIGGERED** (on `torch-tpu==0.1.2.dev20261006000518`, 2-D folded MoE `.reshape().clone()` shares 3-D base storage buffer and hit `ValueError: Tensor's leading (block) dimension does not match its base storage buffer`; triggered Stage Two port of `#116` `adapt_for_raw_transfer`) | `<RUN_DIR>/p1c-retool-rl-06` |
| `P1c` | `p1c-retool-rl-07` (`VERIFY_PARITY=all`, newer image + `#116` + PR #10) | `a7c6be29` | `sha256:d92a2b3d...` | 2 (1 trainer + 1 rollout) | 1->3 (`num_warmup=1`) | 09:47:40 / queued in `priority-dev` | - | **QUEUED (Stage Two test with `adapt_for_raw_transfer` + full PR #10 diagnostics)** | `<RUN_DIR>/p1c-retool-rl-07` |
| `P3` | `p3-retool-rl-01` | `6162456a` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 1->25 | cancelled pending `P1c` step-0 sync verification | - | **BLOCKED on `P1c`** | `<RUN_DIR>/p3-retool-rl-01` |

---

## 3. Checks (`R1`–`R13`)

| Check | Kind | Phase | Target / Reference | Measured on TPU 7x | Verdict |
|---|---|---|---|---|---|
| `R1` | gate | every run | `LIBTPU_INIT_ARGS` contains `--xla_tpu_bf16_emission_mode=PROMOTE_F32` on all active roles | `P1b`, `P2` (trainer) and `P1a` (both trainer `ray::WorkerDict.*` and rollout `ray::RayWorkerWrapper.execute_method`, `ray::CheckpointEngineWorker`, `ray::AgentLoopWorkerTQ`): verified in `<RUN_DIR>/{p1b-retool-sft-01,p2-retool-sft-01,p1a-moe-smoke-12}/env/*-procs.jsonl` across all 4 trainer hosts and all 4 rollout hosts (`LIBTPU_INIT_ARGS` includes `--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32`, `TPU_ACCELERATOR_TYPE="tpu7x"`) | **PASS (`P1a`, `P1b`, `P2`)** |
| `R2` | gate | `P1a`, `P1c` | 32 trainer devices (4D mesh) + 32 rollout devices (TP=32, EP=32) on distinct `2x2x4` slices; trainer rank order matches `verl-tpu-node-rank`; Raiden sync completes | `P1a` (`p1a-moe-smoke-12`): 32 trainer devices (`tpu-group-0`, `FSDP=8, CP=4`) and 32 rollout devices (`tpu-group-1`, `TP=32, EP=32`) ran on separate `2x2x4` slices. On `tpu-group-0`, host IPs were in strictly reverse lexicographic order (`10.224.53.11, 10.224.43.11, 10.224.37.11, 10.224.33.11` for workers `0..3`) and were placed `0, 1, 2, 3` by `verl-tpu-node-rank` (`tpu-group-1`: `10.224.75.7, 10.224.30.7, 10.224.84.7, 10.224.81.7` -> `0, 1, 2, 3`). Initial Raiden P2P sync completed in `17.79 s` (`19.59 s` total), and warm per-step Raiden P2P syncs completed in **`2.35 s`, `2.25 s`, `2.33 s`** (`3.04–3.21 s` total). Note: while Step 3 recovered cleanly after Step-1 sync (`ppl_ratio=1.0020`, `score=0.7578`), Steps 1–2 warmup batches after Step-0 sync had corrupted weights (`ppl_ratio=31.5/32.2`) | **PASS (`P1b`, `P2`, `P1a` topology & Step-1+ sync) / FAIL (`P1a`/`P1c` Step-0 sync on old image)** |
| `R3` | observation | `P1b` | SFT 3-step smoke `train/loss` (`0.7950 -> 0.5787 -> 0.5177`, step 3 `+-0.05`) and step-3 `val/loss` (`0.4050`, `+-0.05`) | `train/loss`: `0.7932` (step 1) -> `0.5881` (step 2) -> `0.5224` (step 3, `+0.0047` vs ref); step-3 `val/loss`: `0.4095` (`+0.0045` vs ref). Moreover, step-1 `train/loss` in `P2` (`0.7931681871414185`) was identical to `P1b` (`0.7931681871414185`) and within `-0.0009` of #108's v6e smoke (`0.7941`) | **PASS** |
| `R4` | observation | `P2` | SFT steps 363–372 mean `train/loss` (`0.039429` in TPU v5p 6-epoch ref `myaf5pal`; `0.038` in the author's 6-epoch doc) and step-372 in-sample `val/loss` (`0.026605` in `myaf5pal`) | `P2` (372 steps, 6 epochs): `train/loss` decreased smoothly from `0.7932` (step 1) to `0.3832` (step 62, Epoch 1), `0.2818` (step 124, Epoch 2), `0.1612` (step 186, Epoch 3), `0.1017` (step 248, Epoch 4), `0.0612` (step 310, Epoch 5), and `0.0359` (step 372, Epoch 6; **steps 363–372 mean `0.039265`**, `-0.000164` vs `myaf5pal`'s `0.039429`). In-sample `val/loss` across epochs 1–6: `0.2713 -> 0.1793 -> 0.1078 -> 0.0618 -> 0.0395 -> 0.0263` (within `+-0.0005` of `myaf5pal`'s `0.2710 -> 0.1796 -> 0.1076 -> 0.0622 -> 0.0392 -> 0.0266` at every single epoch; total tokens identical at `23,798,553`). Both `global_step_186/huggingface` and `global_step_372/huggingface` were exported for downstream evaluation | **PASS** (see section 4.2 & 4.3) |
| `R5a` | gate | `P1b`, `P2` | `global_step_N/huggingface` export complete (16 bf16 shards, index, config, tokenizer) and reloadable | `P1b` (`global_step_3/huggingface`) and `P2` (`global_step_186/huggingface`, `global_step_372/huggingface`): each exported all 16 bf16 `.safetensors` shards + `model.safetensors.index.json` (`18,867` tensors, `61,064,245,248` param bytes, `lm_head.weight` & `model.embed_tokens.weight` present, `tie_word_embeddings=false`) + config/tokenizer (`27` objects, `61,084,185,820` bytes total) | **PASS (`P1b`, `P2`)** |
| `R5b` | gate | `P1c`, `P3` | Staged SFT export matches source hashes across all hosts; TorchTitan + multi-host vLLM load cleanly; first Raiden sync completes without corrupting weights | `p1c-retool-rl-03` & `p1c-retool-rl-05` staged `global_step_372/huggingface` cleanly across all 8 hosts (`45.8–67.7 s`); before Raiden sync, vLLM generated valid multi-turn tool trajectories (`rollout_ppl=1.1322`, `ppl_ratio=0.9923`, `18` tool calls). However, `p1c-retool-rl-05` (`VERIFY_PARITY=all`) failed at Step-0 `RaidenParityCheck.report_exact` with **`10,414 / 13,920` mismatched tensor checks** across `32` sampler ranks (`435` tensors/rank, `0` unresolved; on `rank 0`, `layers.3...w13_weight` had `recv_absmean=0.009274` vs `ref_absmean=0.01824` and subsequent printed tensors had `recv_absmean=0`; see section 4.0.1 and PR #10) | **FAIL (`p1c-retool-rl-05` caught `10,414 / 13,920` Step-0 mismatches on old image; `p1c-retool-rl-06` queued with newer image)** |
| `R6` | observation | `P3` | Step-0 AIME 2024 (`22.44%`) and AIME 2025 (`14.17%`) mean@30 (+-4.5 pp), maj@30, best@30 | Blocked on `P1c` gate (`P3` will not run until `P1c` passes 2 real updates with tool calls and valid `response_mask`) | BLOCKED on `P1c` |
| `R7` | observation | `P3` | Early RL numerics vs TPU v5p reference `fcze3xlx` (steps 3–10 `actor/ppo_kl` `0.0101–0.0123` mean `0.0110`; `actor/rollout_corr/k3_kl` `0.0099–0.0124` mean `0.0109`; `actor/grad_norm` `0.1222–0.1548` mean `0.1345`; `critic/rewards/mean` rising from `-0.5233` at step 1 to `+0.3531` at step 10 and `+0.6562` at step 23) | Blocked on `P1c` gate (full 23-step `fcze3xlx` baseline recorded in section 4.3) | BLOCKED on `P1c` |
| `R8` | observation | `P3` | Step-25 AIME 2024 (`30.00%`) and AIME 2025 (`18.33%`) mean@30 (+-5 pp) | Blocked on `P1c` gate | BLOCKED on `P1c` |
| `R9` | observation | `P2`, `P3` | Median step time and stage split (`gen`, `update_actor`, `update_weights`) vs v6e (`18.6 s` SFT, `108 s` RL) | `P2` SFT: **warm mean `9.57 s/step`**, **median `10.0 s/step`**, **p90 `10.0 s/step`** (`min=9 s, max=11 s` across Epochs 2–6, `n=300` steps; **1.94x faster** than v6e's `18.6 s/step`). Epoch 1 mean was `15.73 s/step` while XLA bucket shapes compiled; cached 32-sample validation took `12 s`. RL step time blocked on `P1c` | **RECORDED (`P2`)** / blocked on `P1c` |
| `R10` | gate | `P3` | Complete step-0 and step-25 eval dumps (30 questions x 30 sessions by trajectory ID for both AIME 2024 and 2025) | Blocked on `P1c` gate | BLOCKED on `P1c` |
| `R11` | gate | `P1c`, `P3` | Token-level rollout audit dump (`response_mask == 0` on tool results, `1` on assistant turns, multi-turn propagation, `compute_score` behavior) | In `p1c-retool-rl-03` (Attempt 2, `NUM_WARMUP_BATCHES=1`), Step 1 (`rollout/1.jsonl`, `rollout_audit/1.jsonl`) executed **`18` SandboxFusion tool calls** across 32 trajectories (`num_turns` up to `8`, `num_non_loss_tokens > 0`), and all audited multi-turn trajectories had **`0` mask violations** (`response_mask == 0` on every `<tool_response>` segment) with `actor/grad_norm = 0.7330`. However, Steps 2–3 suffered from Step-0 Raiden sync corruption (`rollout_ppl=249.33`) | **PARTIAL PASS (Step 1 verified tool calls & mask; Steps 2–3 blocked on Step-0 Raiden sync fix)** |
| `R12` | gate | `P1c`, `P3` | Post-Raiden-sync policy version & staleness within `max_off_policy_threshold=4` | `P1a` (`p1a-moe-smoke-12`): `trajectory_staleness/max = 1 <= max_off_policy_threshold=4` across all 3 steps, and after `update_weights(1)` synced at the end of Step 1, Step 3 achieved `rollout_corr/ppl_ratio = 1.0020`, `rollout_corr/kl = 0.0017`, `critic/score/mean = 0.7578` (whereas Steps 1–2 warmup batches after `update_weights(0)` had `ppl_ratio = 31.530 / 32.180`). In `P1c` (`p1c-retool-rl-05`), `VERIFY_PARITY=all` caught `10,414 / 13,920` tensor mismatches at Step 0 across the 32 sampler ranks | **PASS (`P1a` post-step-1)** / **FAIL (`P1a` & `P1c` step-0 init sync on old image)** |
| `R13` | gate | `P4a` | Resume equivalence (`TransferQueue==0.1.8` in current images lacks `save_checkpoint`/`load_checkpoint`; requires quiescent boundary verification or `TransferQueue >= 0.1.9`) | Blocked unless quiescent boundary verified in `P4a` | BLOCKED (pending `P4a`) |

---

## 4. Stage details

### 4.0 Stage `P1a`: 3-step Qwen3-30B-A3B dual-slice platform smoke (`p1a-moe-smoke-12`) — **Topology/Placement & Step-1+ Recovery PASS (Step-0 Sync Corrupted Steps 1–2 Warmup)**

- **Configuration**: `examples/tpu/grpo/run_qwen3_30b_a3b_torchtitan.sh` on `2 x 2x2x4` TPU 7x slices (32 physical chips = 64 TensorCore devices: `tpu-group-0` 16 physical chips / 32 trainer TensorCore devices with `SimpleFSDP=8, CP=4, EP=1, TP=1`, `TPU_EAGER_MODE=DEFER_NEVER`; `tpu-group-1` 16 physical chips / 32 rollout TensorCore devices with 1 multi-host vLLM server `TP=32, EP=32`, `RELEASE_SYNC_BUFFERS=False`, `VERL_RAIDEN_RELEASE_BUFFERS=0`, `TOTAL_TRAINING_STEPS=3`, `LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32"`).
- **Host ordering (`P0.5` / `R2`)**:
  - Trainer slice (`tpu-group-0`):
    - `verl-tpu-node-rank=0` (`tpu-worker-id=0`): `10.224.53.11` (`p1a-moe-smoke-12-trainer-0-0-0`)
    - `verl-tpu-node-rank=1` (`tpu-worker-id=1`): `10.224.43.11` (`p1a-moe-smoke-12-trainer-0-0-1`)
    - `verl-tpu-node-rank=2` (`tpu-worker-id=2`): `10.224.37.11` (`p1a-moe-smoke-12-trainer-0-0-2`)
    - `verl-tpu-node-rank=3` (`tpu-worker-id=3`): `10.224.33.11` (`p1a-moe-smoke-12-trainer-0-0-3`)
  - Rollout slice (`tpu-group-1`):
    - `verl-tpu-node-rank=0` (`tpu-worker-id=0`): `10.224.75.7` (`p1a-moe-smoke-12-rollout-0-0-0`)
    - `verl-tpu-node-rank=1` (`tpu-worker-id=1`): `10.224.30.7` (`p1a-moe-smoke-12-rollout-0-0-1`)
    - `verl-tpu-node-rank=2` (`tpu-worker-id=2`): `10.224.84.7` (`p1a-moe-smoke-12-rollout-0-0-2`)
    - `verl-tpu-node-rank=3` (`tpu-worker-id=3`): `10.224.81.7` (`p1a-moe-smoke-12-rollout-0-0-3`)
  - Notice that on `tpu-group-0` the 4 trainer host IPs were in strictly reverse lexicographic order (`53.11 > 43.11 > 37.11 > 33.11`), and on `tpu-group-1` the rollout host IPs were also non-monotonic (`75.7, 30.7, 84.7, 81.7`). `RayResourcePool.get_placement_groups()` sorted both slices into exact topological order `0, 1, 2, 3` via `verl-tpu-node-rank`.
- **Per-step metrics, Raiden P2P weight sync, and policy propagation (`R2`, `R12`)**:

| Step | `critic/score/mean` | `actor/grad_norm` | `actor/ppo_kl` | `rollout_corr/ppl_ratio` | `rollout_corr/kl` | `trajectory_staleness` (mean / max) | Raiden P2P / Total sync | `timing_s/update_actor` | `timing_s/step` |
|---|---|---|---|---|---|---|---|---|---|
| 0 (init sync) | - | - | - | - | - | - | `17.791 s` / `19.588 s` | - | - |
| 1 (warmup batch 0) | `0.0000` | `0.0000` | `1.22e-4` | `31.530` | `3.1396` | `0.000` / `0` | `2.345 s` / `3.207 s` | `107.258 s` (compile) | `398.518 s` |
| 2 (warmup batch 1) | `0.0000` | `0.0000` | `-2.28e-4` | `32.180` | `3.1676` | `1.000` / `1` | `2.254 s` / `3.044 s` | `6.034 s` | `11.451 s` |
| 3 (post-sync policy) | **`0.7578`** | `0.4500` | `1.42e-4` | **`1.0020`** | **`0.0017`** | `0.1016` / `1` | `2.335 s` / `3.121 s` | `3.524 s` | **`7.944 s`** |
| 3 (GSM8K val) | `val-core/openai/gsm8k/acc/mean@1 = 0.5859375` | - | - | - | - | - | - | - | `79.769 s` (val) |

- **Artifacts & interpretation boundary**:
  - Dumped `<RUN_DIR>/p1a-moe-smoke-12/rollout/{1,2,3}.jsonl`, `<RUN_DIR>/p1a-moe-smoke-12/rollout_audit/{1,2,3}.jsonl`, `<RUN_DIR>/p1a-moe-smoke-12/validation/3.jsonl`, and `<RUN_DIR>/p1a-moe-smoke-12/env/*-{env.json,procs.jsonl}`.
  - `verdict.json`: `status="SUCCEEDED"`, `exit_code=0`, `duration_s=1413`.
  - **Important caveat**: Although the 3-step job exited 0 and recovered cleanly at Step 3 (`ppl_ratio=1.0020, score=0.7578, GSM8K acc=58.59%`) after the Step-1 Raiden sync (`update_weights(1)`), the two warmup batches generated immediately after the Step-0 initial sync (`update_weights(0)`) exhibited severe weight corruption (`ppl_ratio=31.530 / 32.180`, `grad_norm=0.0000`, `score=0.0000`). Thus "job finished with exit 0" cannot be treated as proof of Step-0 sync correctness.

### 4.0.1 Stage `P1c` (`p1c-retool-rl-03`, `p1c-retool-rl-05`, & queued `p1c-retool-rl-06`): ReTool multi-turn RL smoke & Step-0 Raiden sync diagnosis (Working Hypotheses)

- **Attempt 1 (`p1c-retool-rl-03-edd33f27`, `NUM_WARMUP_BATCHES=2, TOTAL_TRAINING_STEPS=2`, `code_sha: 6162456a`)**:
  - Both Step 1 and Step 2 consumed warmup batches generated after the Step-0 initial Raiden sync (`update_weights(0)`), which produced corrupted rollouts (`rollout_ppl=249.327 / 144.104`, `ppl_ratio=40.668 / 43.711`, `actor/grad_norm=0.0000`, `0` tool calls).
- **Attempt 2 (`p1c-retool-rl-03-040c806c`, `NUM_WARMUP_BATCHES=1, TOTAL_TRAINING_STEPS=3`, `code_sha: c2e17624`)**:
  - Reduced `NUM_WARMUP_BATCHES=1` and ran 3 steps (`00:26:52 -> 00:50:19 UTC`):

| Step | `critic/score/mean` | `actor/grad_norm` | `rollout_corr/rollout_ppl` | `rollout_corr/training_ppl` | `rollout_corr/ppl_ratio` | `response_length/mean` (`clip_ratio`) | `num_turns` (`tool_calls`) | `timing_s/gen` | `timing_s/update_actor` | `timing_s/step` |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (pre-sync batch) | **`-0.5562`** | **`0.7330`** | **`1.1322`** | **`1.1235`** | **`0.9923`** | `5,803.2` (`0.5625`) | up to `8` (**`18` tool calls**, `0` mask violations) | `151.31 s` | `159.89 s` | `317.85 s` |
| 2 (post-step-0-sync) | `-1.0000` | `0.0000` | `249.327` | `4.192` | `0.0156` | `8,192.0` (`1.0000`) | `2` (`0` tool calls) | `424.91 s` | `9.69 s` | `438.92 s` |
| 3 (buffered post-step-0-sync batch) | `-1.0000` | `0.0000` | `249.327` | `4.192` | `0.0156` | `8,192.0` (`1.0000`) | `2` (`0` tool calls) | `0.02 s` (buffered) | `7.31 s` | `11.77 s` |

  - **Key takeaway from Attempt 2**: Before Raiden overwrites vLLM's weights, vLLM + SandboxFusion + TorchTitan on TPU 7x work end-to-end on Step 1 (`rollout_ppl=1.1322`, `training_ppl=1.1235`, `ppl_ratio=0.9923`, `18` tool calls, `0` mask violations in `rollout_audit/1.jsonl`, `actor/grad_norm=0.7330`). The corruption on Step 2 is strictly introduced by Step-0 Raiden `update_weights(0)`. Note also that Step 3 had `timing_s/gen = 0.02 s (buffered)` because it consumed trajectories already buffered in the async queue under the corrupted Step-0 weights (`policy_version` from Step 0), so Step 3 in `p1c-retool-rl-03` does **not** independently prove that new rollouts generated after the Step-1 sync (`update_weights(1)`) remain corrupted.
- **Attempt 3 (`p1c-retool-rl-05`, `VERIFY_PARITY=all`, `code_sha: c2e17624`) — Step-0 `float32` Elementwise Parity Diagnosis**:
  - Enabled `+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.verify_parity=all` (`RaidenParityCheck.check_exact` at Step 0, casting `target` and `received` to `float32` and checking `max|ref - new| == 0.0` on all 32 sampler ranks before overwriting vLLM parameters; note that `check_exact` runs **after** `_raiden_ws.h2d()` and `_raiden_fused_expert_weights` / `to_target_layout` unfolding/transpose).
  - `p1c-retool-rl-05` failed fast at Step-0 `update_weights(0)` (`02:06:21 UTC`) with:
    ```text
    verl.checkpoint_engine.raiden_checkpoint_engine.RaidenParityError: [RAIDEN PARITY EXACT | Step 0] 32 sampler ranks, [435] tensors checked per rank, 10414 mismatched, 0 unresolved
      * rank 0: model.layers.3.mlp.experts.routed_experts.w13_weight: max|diff|=0.5195 ref_absmean=0.01824 recv_absmean=0.009274
      * rank 0: model.layers.30.mlp.experts.routed_experts.w13_weight: max|diff|=0.543 ref_absmean=0.01898 recv_absmean=0
      * rank 0: model.layers.30.mlp.experts.routed_experts.w2_weight: max|diff|=0.3633 ref_absmean=0.01939 recv_absmean=0
      ...
      * rank 0: model.layers.47.mlp.experts.routed_experts.w2_weight: max|diff|=1.078 ref_absmean=0.0205 recv_absmean=0
      * rank 0: model.layers.5.mlp.experts.routed_experts.w13_weight: max|diff|=0.5781 ref_absmean=0.01844 recv_absmean=0
      ...
      * rank 0: model.layers.0.self_attn.qkv_proj.weight: max|diff|=0.2207 ref_absmean=0.01046 recv_absmean=0
      * rank 0: model.layers.18.self_attn.qkv_proj.weight: max|diff|=0.2295 ref_absmean=0.02161 recv_absmean=0
    ```
  - **Empirical facts, evidence boundaries, and working hypotheses (full details in PR #10 `TPU7X_RAIDEN_WEIGHT_SYNC_DIAGNOSIS.md`)**:
    1. **Exact mismatch count (`10,414 / 13,920`)**: Across `32` sampler ranks x `435` tensors/rank = `13,920` checks, `report_exact` recorded **`10,414` total mismatches** (`325.4375` per rank on average, i.e. `32 x 325 + 14`) and `0` unresolved. Because `report_exact` truncates printed details to `MAX_DETAIL_LINES = 60`, only the first 60 mismatch lines from `rank 0` were printed in the driver log (where `layers.3.mlp.experts.routed_experts.w13_weight` had `recv_absmean = 0.009274` vs `ref_absmean = 0.01824`, and subsequent printed tensors had `recv_absmean = 0`).
    2. **Evidence boundaries & working hypotheses (H1 / H2 / H3)**:
       - **Working Hypothesis H1 (Primary — `PushWeightsReshardedLocal` D2H coalescing race in `tpu-sync-torch==0.0.1.dev20261001220631`)**: In `0.0.1.dev20261001220631`, the leader RPC thread inserts `uuid` into `completed_d2h_uuids_` *before* calling `D2hLayer(l, uuid)`. Concurrent follower RPC threads (triggered by distinct target equivalence classes across TP=32 / GQA-replicated / fully-replicated tensors) see `already_completed = true`, skip `d2h_layer_futures[l].Await()`, and push the zero-initialized host staging buffer before the leader finishes D2H. In `verl-tpu:v20261006-tsync1006` (`@sha256:d92a2b3d...`), `tpu-sync-torch==0.0.1.dev20261006025601` adds per-layer `UuidD2hState` synchronization (`d2h_state->mu.Await(...)`). If the persistent host staging buffer is subsequently populated by the leader thread before Step 0 finishes, this is consistent with `P1a`'s recovery at Step 3 after the Step-1 sync.
       - **Working Hypotheses H2 (Host-to-TPU `_raiden_ws.h2d()` DMA truncation) & H3 (Post-H2D tensor unfolding/transpose issue)**: Because `check_exact` runs after `_raiden_ws.h2d()` and layout conversion, existing logs alone cannot rule out H2 or H3 without controlled experiments and post-H2D staging diagnostics (added in PR #10).
    3. **Attempt 4 (`p1c-retool-rl-06`, `code_sha: c2e17624`, newer image `@sha256:d92a2b3d...`) & Attempt 5 (`p1c-retool-rl-07`, `code_sha: a7c6be29`, newer image + `#116` `adapt_for_raw_transfer` + PR #10 diagnostics)**:
       - `p1c-retool-rl-06` ran from `09:27:52 -> 09:41:32 UTC` at baseline code `c2e17624` on `verl-tpu:v20261006-tsync1006` (`@sha256:d92a2b3d...`) and hit **Case B** at Step-0 `WeightSynchronizer` registration (`_create_and_register`): `ValueError: Tensor's leading (block) dimension does not match its base storage buffer; raw transfer requires a layout-preserving full view of the block-major storage` (because on `torch-tpu==0.1.2.dev20261006000518`, `local.reshape(local_shape).clone()` of a 2-D folded MoE expert view `[E_local * F, D]` shares the 3-D `[E_local, F, D]` base storage buffer).
       - Per the staged plan in PR #10, we cherry-picked `jialei777/verl-upstream#116` (`32b5133e`, `adapt_for_raw_transfer` host-roundtrip rebind for rejected tensor classes) onto PR #10 (`a7c6be29`) and submitted **`p1c-retool-rl-07`** (`09:47:40 UTC`, queued in `priority-dev`) with `VERIFY_PARITY=all`. `P3` (`p3-retool-rl-01`) remains blocked until `P1c` passes all Step-0 `exact` and Step 1–2 `norm` / training gates.

### 4.1 Stage `P1b`: 3-step ReTool SFT smoke (`p1b-retool-sft-01`) — **PASS**

- **Configuration**: `examples/tpu/retool/run_qwen3_30b_a3b_sft_torchtitan.sh` on `1 x 2x2x4` TPU 7x slice (4 hosts, 16 chips = 32 TensorCore devices; `SimpleFSDP=8, CP=4, EP=1, TP=1`, `TPU_EAGER_MODE=DEFER_NEVER`, `MAX_LENGTH=16384`, `TRAIN_BATCH_SIZE=32`, `TOTAL_TRAINING_STEPS=3`, `SAVE_FREQ=3`, `TEST_FREQ=3`, `LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32"`).
- **Host ordering (`P0.5`)**:
  - Physical / JobSet completion order vs IP address on `tpu-group-0`:
    - `verl-tpu-node-rank=0` (`tpu-worker-id=0`): `10.224.130.8` (`p1b-retool-sft-01-trainer-0-0-0`)
    - `verl-tpu-node-rank=1` (`tpu-worker-id=1`): `10.224.106.8` (`p1b-retool-sft-01-trainer-0-0-1`)
    - `verl-tpu-node-rank=2` (`tpu-worker-id=2`): `10.224.114.8` (`p1b-retool-sft-01-trainer-0-0-2`)
    - `verl-tpu-node-rank=3` (`tpu-worker-id=3`): `10.224.113.8` (`p1b-retool-sft-01-trainer-0-0-3`)
  - Without `verl-tpu-node-rank`, lexicographic IP sorting would have placed worker 1 first (`10.224.106.8`), worker 3 second (`10.224.113.8`), worker 2 third (`10.224.114.8`), and worker 0 last (`10.224.130.8`). With #5 (`ed0170cf`), all 4 hosts were placed in topological order `0, 1, 2, 3` and completed FSDP + Ulysses CP collectives cleanly.
- **Effective gradient clipping (`P0.9` / `D7`)**:
  - Runtime log confirmed: `WARNING:2026-10-29 04:19:03,748:TorchTitan grad clipping uses training.max_norm=1.0; optim.clip_grad=1.0 is not passed to TorchTitan`.
- **Per-step metrics (`R3`)**:

| Step | TPU 7x `train/loss` | TPU v6e ref `train/loss` | Diff (7x - v6e) | `train/grad_norm` | `train/lr` | `train/global_tokens` | Step time |
|---|---|---|---|---|---|---|---|
| 1 | `0.793168` | `0.7950` (`0.7941` in #108 body) | `-0.0018` (`-0.0009`) | `13.9740` | `7.75e-06` | `59,418` | `55.57 s` (first compile) |
| 2 | `0.588094` | `0.5787` (`0.5874` in #108 body) | `+0.0094` (`+0.0007`) | `6.0704` | `3.25e-06` | `64,348` | `28.43 s` |
| 3 | `0.522368` | `0.5177` (`0.5215` in #108 body) | `+0.0047` (`+0.0009`) | `2.8164` | `1.00e-06` | `55,789` | `27.10 s` |
| 3 (val) | `val/loss = 0.409530` | `val/loss = 0.4050` (`0.4093` in #108 body) | `+0.0045` (`+0.0002`) | - | - | 32 val samples | `46.0 s` |

- **Exported HF checkpoint (`R5a`)**:
  - Path: `<RUN_DIR>/p1b-retool-sft-01/checkpoint/global_step_3/huggingface/`
  - Contents: 16 bf16 shards (`model-00001-of-00016.safetensors` through `model-00016-of-00016.safetensors`), `model.safetensors.index.json` (`1,699,757` bytes, `18,867` tensors, `61,064,245,248` param bytes), `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` (27 objects, `61,084,185,820` bytes total; export write time `9m 59s`).
  - `verdict.json`: `status="SUCCEEDED"`, `node_rank_ok=true`, `duration_s=961`, `timed_out=false`.

### 4.2 Stage `P2`: Full 6-epoch (372-step) ReTool cold-start SFT (`p2-retool-sft-01`) — **PASS**

- **Configuration**: `examples/tpu/retool/run_qwen3_30b_a3b_sft_torchtitan.sh` with script defaults (`TOTAL_EPOCHS=6`, `62` steps/epoch = `372` steps, `TRAIN_BATCH_SIZE=32`, `MAX_LENGTH=16384`, `SAVE_FREQ=186`, `TEST_FREQ=62`, `optim.lr=1e-5`, `lr_warmup_steps_ratio=0.1` = 37 warmup steps, cosine decay to `1e-6`, `LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32"`).
- **Host ordering (`P0.5`)**:
  - Physical / JobSet completion order vs IP address on `tpu-group-0`:
    - `verl-tpu-node-rank=0` (`tpu-worker-id=0`): `10.224.180.7` (`p2-retool-sft-01-trainer-0-0-0`)
    - `verl-tpu-node-rank=1` (`tpu-worker-id=1`): `10.224.242.7` (`p2-retool-sft-01-trainer-0-0-1`)
    - `verl-tpu-node-rank=2` (`tpu-worker-id=2`): `10.224.243.7` (`p2-retool-sft-01-trainer-0-0-2`)
    - `verl-tpu-node-rank=3` (`tpu-worker-id=3`): `10.224.181.7` (`p2-retool-sft-01-trainer-0-0-3`)
  - Without `verl-tpu-node-rank`, lexicographic IP order would have placed worker 3 (`10.224.181.7`) second (`[0, 3, 1, 2]`). With `verl-tpu-node-rank`, all 4 hosts ran in topological order `0, 1, 2, 3`.
- **Per-epoch convergence and throughput summary (`R4`, `R9`)**:

| Epoch | Steps | First `train/loss` | Last `train/loss` | Epoch mean `train/loss` | Last-10-step mean `train/loss` | Max `grad_norm` | End `grad_norm` | End-of-epoch `val/loss` (in-sample) | Pure training time (61 steps) | Mean step time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1–62 | `0.793168` | `0.383225` | `0.426880` | `0.347975` | `14.3615` (step 3) | `0.4312` | `0.271294` (step 62) | `1,000 s` (+ `54 s` step 62 + val) | `15.73 s` (compiling buckets) |
| 2 | 63–124 | `0.267472` | `0.281752` | `0.273512` | `0.266374` | `0.6228` | `0.4842` | `0.179302` (step 124) | `583 s` (+ `12 s` val) | `9.57 s` |
| 3 | 125–186 | `0.195968` | `0.161228` | `0.176805` | `0.175233` | `1.7405` | `0.5058` | `0.107801` (step 186) | `583 s` (+ `12 s` val + `669 s` HF export) | `9.55 s` |
| 4 | 187–248 | `0.127889` | `0.101719` | `0.107684` | `0.111980` | `0.9924` | `0.5146` | `0.061826` (step 248) | `584 s` (+ `12 s` val) | `9.58 s` |
| 5 | 249–310 | `0.049699` | `0.061234` | `0.063903` | `0.065669` | `0.8272` | `0.4382` | `0.039511` (step 310) | `583 s` (+ `13 s` val) | `9.57 s` |
| 6 | 311–372 | `0.080677` | `0.035875` | `0.040940` | **`0.039265`** | `0.9555` | `0.4085` | **`0.026332`** (step 372) | `584 s` (+ `12 s` val + `598 s` HF export) | `9.58 s` |

- **Final 10 steps (`steps 363–372`, `R4`)**:
  - `train/loss` values: `[0.040568, 0.036545, 0.045109, 0.040919, 0.035504, 0.040524, 0.038072, 0.046814, 0.032716, 0.035875]`
  - Mean `train/loss` (`steps 363–372`): **`0.039265`** (median `0.039298`), `step 372 val/loss`: **`0.026332`**, total tokens processed: `23,798,553` (`0.0238 B`).
  - Comparison with reference notes:
    - Step 1 `train/loss` (`0.793168`) is identical to `P1b` and matches #108's v6e smoke (`0.7941`) within `-0.0009`.
    - Over 6 epochs on the 2,000-row dataset, `train/loss` reaches `0.0393` (mean of steps 363–372) and `0.0359` (step 372), which matches the `train/loss = 0.038` reported in the recipe author's 6-epoch TPU v6e report. Meanwhile, Epoch 1 (`val/loss = 0.2713`) and Epoch 2 (`mean train/loss = 0.2735`, `last-10 mean = 0.2664`, `val/loss = 0.1793`) bracket the `train/loss ~ 0.25, val/loss = 0.214` numbers quoted in #108's PR summary. Because `SAVE_FREQ=186` exported both `global_step_186/huggingface` and `global_step_372/huggingface`, `P3` starts from `global_step_372/huggingface` per `D9` while preserving `global_step_186/huggingface` for comparison if needed.
- **Exported HF checkpoints (`R5a`)**:
  - Mid-run export: `<RUN_DIR>/p2-retool-sft-01/checkpoint/global_step_186/huggingface/` (27 objects, `61,084,185,820` bytes, 16 bf16 `.safetensors` shards, `18,867` tensors).
  - Final export: `<RUN_DIR>/p2-retool-sft-01/checkpoint/global_step_372/huggingface/` (27 objects, `61,084,185,820` bytes, 16 bf16 `.safetensors` shards, `18,867` tensors, `61,064,245,248` param bytes, `lm_head.weight` & `model.embed_tokens.weight` present, `tie_word_embeddings=false`).
  - `verdict.json`: `status="SUCCEEDED"`, `node_rank_ok=true`, `duration_s=5548` (`92m 28s` head runtime, including model staging, 372 training steps, 6 validations, and two 57 GiB HF checkpoint exports), `timed_out=false`.

### 4.3 Reference baseline comparison: TPU v5p runs (`myaf5pal` and `fcze3xlx`)

To establish exact step-by-step parity targets across SFT (`R4`) and DAPO RL (`R6`–`R9`), both reference runs linked in PR #2 (`myaf5pal` for 6-epoch SFT and `fcze3xlx` for DAPO RL) are recorded below alongside our completed TPU 7x Stage `P2` run (`p2-retool-sft-01`).

#### 4.3.1 Cold-start SFT parity: TPU v5p `myaf5pal` vs. TPU 7x `p2-retool-sft-01`

- **Reference setup (`myaf5pal` / `multiturn-sft-qwen3-30b-a3b-v5p`)**: `finished` in `3,295.6 s` (`54m 56s`), `4 nodes x 4 chips = 16` TPU v5p devices (`data_parallel_shard_size=16`, `context_parallel_size=1`, `use_splash_attention=true`, `attn_type="varlen"`), `train_batch_size=32`, `micro_batch_size_per_gpu=1`, `max_length=16384`, `lr=1e-5` (cosine decay to `1e-6`, `lr_warmup_steps_ratio=0.1`), `clip_grad=1.0`, `seed=42`, `372` steps (`6` epochs), `train/total_tokens = 23,798,553` (`0.023798553 B` — identical to `p2-retool-sft-01`).
- **Epoch-by-epoch comparison (`R4`)**:

| Checkpoint / Metric | TPU v5p ref (`myaf5pal`) | TPU 7x `P2` (`p2-retool-sft-01`) | Delta (`7x - v5p`) |
|---|---|---|---|
| Step 1 `train/loss` | `0.791439` | `0.793168` | `+0.001729` |
| Epoch 1 (steps 1–62) mean / last-10 `train/loss` | `0.426701` / `0.347942` | `0.426880` / `0.347975` | `+0.000179` / `+0.000033` |
| Epoch 1 (step 62) `val/loss` | `0.271026` | `0.271294` | `+0.000268` |
| Epoch 2 (steps 63–124) mean / last-10 `train/loss` | `0.273237` / `0.266112` | `0.273512` / `0.266374` | `+0.000275` / `+0.000262` |
| Epoch 2 (step 124) `val/loss` | `0.179572` | `0.179302` | `-0.000270` |
| Epoch 3 (steps 125–186) mean / last-10 `train/loss` | `0.176942` / `0.175286` | `0.176805` / `0.175233` | `-0.000137` / `-0.000053` |
| Epoch 3 (step 186) `val/loss` | `0.107577` | `0.107801` | `+0.000224` |
| Epoch 4 (steps 187–248) mean / last-10 `train/loss` | `0.107932` / `0.111990` | `0.107684` / `0.111980` | `-0.000248` / `-0.000010` |
| Epoch 4 (step 248) `val/loss` | `0.062240` | `0.061826` | `-0.000414` |
| Epoch 5 (steps 249–310) mean / last-10 `train/loss` | `0.064055` / `0.066167` | `0.063903` / `0.065669` | `-0.000152` / `-0.000498` |
| Epoch 5 (step 310) `val/loss` | `0.039194` | `0.039511` | `+0.000317` |
| Epoch 6 (steps 311–372) mean / last-10 `train/loss` | `0.041088` / **`0.039429`** | `0.040940` / **`0.039265`** | `-0.000148` / **`-0.000164`** |
| Epoch 6 (step 372) `val/loss` | **`0.026605`** | **`0.026332`** | **`-0.000273`** |
| Total tokens (`train/total_tokens`) | `23,798,553` | `23,798,553` | `0` |

The TPU v7x SFT loss curve closely reproduces the TPU v5p SFT reference (`myaf5pal`) across all six epochs (within `+-0.0005` on every epoch mean, last-10-step mean, and validation loss, despite different sharding: `SimpleFSDP=8, CP=4` on TPU 7x vs `SimpleFSDP=16, CP=1` on TPU v5p). Note that identical `train/total_tokens = 23,798,553` supports workload consistency without proving identical per-step sample order or token IDs, and `val/loss` remains in-sample.

#### 4.3.2 DAPO RL reference trajectory: TPU v5p `fcze3xlx` (steps 1–23)

- **Reference setup (`fcze3xlx` / `retool-dapo-qwen3-30b-a3b-v5p-cp2`)**:
  - **Scope & config comparison vs default `P3`**: `fcze3xlx` ran the 16K / 16-turn / `n=16` override (`max_response_length=16384`, `max_turns=16`, `n=16`, `lr=2e-6`, `clip_grad=0.0625`, trainer `CP=2, FSDP=8`, rollout `TP=16, EP=16`, `enable_thinking=true`) rather than the default 8K `P3` target (`max_response_length=8192`, `max_turns=8`, `n=8`, `lr=1.4e-6`, `clip_grad=0.088`, trainer `CP=4, FSDP=8`, rollout `TP=32, EP=32`, `enable_thinking=false`). It therefore serves as a behavioral and systems reference rather than a same-config step-by-step parity gate for 8K `P3`.
  - **Runtime & termination**: `state="crashed"` after completing `training/global_step = 23 / 150` (`_runtime = 30,573.5 s` = `8h 30m`, including two synchronous checkpoint saves of `1,430.7 s` at step 10 and `1,349.8 s` at step 20). Because `trainer.val_before_train=false` and `trainer.test_freq=25`, no `val-core/*` (AIME 2024 / AIME 2025) evaluations were logged before the run terminated at step 23.
  - **Topology & sharding**: Trainer `4 nodes x 4 chips = 16` devices (`strategy="torchtitan"`, `use_simple_fsdp=true`, `context_parallel_size=2`, `data_parallel_shard_size=8`, `use_splash_attention=true`, `attn_type="varlen"`, `max_seq_len=18432`, `ppo_max_token_len_per_gpu=9216`, `use_dynamic_bsz=true`, actor `activation_checkpoint="full"`, ref `activation_checkpoint="selective"`) + Rollout `4 nodes x 4 chips = 16` devices (`tensor_model_parallel_size=16`, `expert_parallel_size=16`, `USE_MOE_SPARSE_CORE=0`, `max_model_len=18432`, `max_num_batched_tokens=4096`, `max_num_seqs=512`, `block_size=256`, `gpu_memory_utilization=0.85`, `checkpoint_engine.backend="raiden"`).
  - **Hyperparameters**: `train_batch_size=64`, `ppo_mini_batch_size=64`, `n=16`, `prompt_length=2048`, `response_length=16384`, `lr=2e-6`, `betas=[0.9, 0.999]`, `clip_grad=0.0625`, `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `loss_agg_mode="token-mean"`, `policy_loss.loss_mode="bypass_mode"`, `algorithm.rollout_correction={bypass_mode: true, loss_type: "ppo_clip", rollout_is: "token", rollout_is_threshold: 2}`, `trainer_mode="separate_async"` (`num_warmup_batches=2`, `parameter_sync_step=1`, `max_off_policy_threshold=4`), `multi_turn={enable: true, format: "hermes", max_assistant_turns: 16, max_user_turns: 16, max_tool_response_length: 256}`.
  - **Window summaries (`R7`, `R9`)**:
    - Steps 3–10: `actor/ppo_kl` `0.010070–0.012328` (mean `0.010971`), `actor/rollout_corr/k3_kl` `0.009899–0.012421` (mean `0.010882`), `actor/grad_norm` `0.122238–0.154805` (mean `0.134469`), `critic/rewards/mean` step 3 `-0.445996` -> step 10 `+0.353125` (window min `-0.445996` at step 3, max `+0.381152` at step 8, mean `+0.026599`).
    - All 23 steps: `critic/rewards/mean` increases from `-0.5233` (step 1) to `+0.1005` (step 6), `+0.3531` (step 10), `+0.5825` (step 13), and **`+0.6562` (step 23)** (note: `examples/tpu/retool/retool_tpu.py` shapes incorrect multi-turn tool trajectories into `[-1.0, -0.6]`, so exact solve rate must be read from per-sample `acc` rather than linearly inverted from `critic/rewards/mean`); `timing_s/gen` `0.2–2,724.7 s` (mean `1,001.6 s`; `0.22 s` at steps 11 & 21 when the async buffer was already full), `timing_s/update_actor` `99.8–190.8 s` (mean `134.9 s`), `timing_s/update_weights` `9.8–16.4 s` (mean `14.6 s`).
- **Complete 23-step trajectory (`fcze3xlx`)**:

| Step | `critic/rewards/mean` | `response_length/mean` | `response_length/clip_ratio` | `num_turns/mean` | `actor/grad_norm` | `actor/ppo_kl` | `timing_s/step` | `timing_s/gen` | `timing_s/update_actor` | `timing_s/update_weights` |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `-0.5233` | `1,612.9` | `0.000977` | `6.48` | `0.1649` | `0.01259` | `2,869.6 s` | `2,724.7 s` | `125.4 s` | `15.0 s` |
| 2 | `-0.4161` | `2,039.9` | `0.000977` | `9.59` | `0.1453` | `0.01132` | `576.3 s` | `440.7 s` | `119.4 s` | `15.4 s` |
| 3 | `-0.4460` | `2,791.2` | `0.001953` | `9.37` | `0.1222` | `0.01066` | `1,303.4 s` | `1,096.7 s` | `190.8 s` | `15.2 s` |
| 4 | `-0.1520` | `2,053.2` | `0.000977` | `8.44` | `0.1548` | `0.01158` | `197.6 s` | `60.3 s` | `120.8 s` | `15.3 s` |
| 5 | `-0.2827` | `2,433.3` | `0.000977` | `8.39` | `0.1269` | `0.01098` | `1,896.9 s` | `1,750.8 s` | `132.5 s` | `12.7 s` |
| 6 | `+0.1005` | `1,925.3` | `0.000977` | `8.60` | `0.1498` | `0.01233` | `583.1 s` | `469.1 s` | `101.6 s` | `11.6 s` |
| 7 | `-0.0002` | `2,491.2` | `0.000977` | `8.67` | `0.1264` | `0.01078` | `1,378.1 s` | `1,234.1 s` | `131.3 s` | `11.9 s` |
| 8 | `+0.3812` | `1,971.0` | `0.000977` | `8.78` | `0.1398` | `0.01098` | `856.5 s` | `736.4 s` | `107.7 s` | `11.6 s` |
| 9 | `+0.2589` | `2,567.5` | `0.000977` | `8.76` | `0.1245` | `0.01007` | `1,271.0 s` | `1,121.7 s` | `138.1 s` | `9.8 s` |
| 10 | `+0.3531` | `2,167.9` | `0.001953` | `8.11` | `0.1313` | `0.01039` | `2,471.9 s` (ckpt `1,430.7 s`) | `904.2 s` | `121.1 s` | `15.1 s` |
| 11 | `+0.1180` | `3,214.9` | `0.001953` | `9.71` | `0.1017` | `0.00843` | `203.0 s` | `0.2 s` | `186.0 s` | `15.9 s` |
| 12 | `+0.2935` | `2,447.8` | `0.001953` | `8.66` | `0.1263` | `0.00972` | `1,494.6 s` | `1,349.8 s` | `127.3 s` | `16.3 s` |
| 13 | `+0.5825` | `1,808.3` | `0.000977` | `7.20` | `0.1471` | `0.01091` | `1,267.1 s` | `1,150.2 s` | `99.8 s` | `16.4 s` |
| 14 | `+0.4082` | `2,355.2` | `0.003906` | `8.88` | `0.1196` | `0.00870` | `862.2 s` | `724.3 s` | `121.4 s` | `15.7 s` |
| 15 | `+0.2918` | `2,603.1` | `0.001953` | `8.25` | `0.1158` | `0.00855` | `1,444.1 s` | `1,292.8 s` | `135.2 s` | `15.3 s` |
| 16 | `+0.4662` | `2,419.2` | `0.001953` | `8.97` | `0.1170` | `0.00850` | `917.9 s` | `768.3 s` | `132.6 s` | `16.1 s` |
| 17 | `+0.3906` | `2,586.9` | `0.000977` | `8.95` | `0.1080` | `0.00902` | `1,474.2 s` | `1,324.4 s` | `133.5 s` | `15.5 s` |
| 18 | `+0.4184` | `2,298.6` | `0.000977` | `9.19` | `0.1341` | `0.00909` | `1,137.0 s` | `945.4 s` | `175.1 s` | `15.2 s` |
| 19 | `+0.3231` | `2,827.7` | `0.000977` | `9.80` | `0.1079` | `0.00833` | `1,246.9 s` | `1,083.9 s` | `147.5 s` | `14.6 s` |
| 20 | `+0.3169` | `2,501.2` | `0.000977` | `9.41` | `0.1103` | `0.00834` | `2,415.2 s` (ckpt `1,349.8 s`) | `919.2 s` | `131.4 s` | `14.0 s` |
| 21 | `+0.2702` | `3,251.5` | `0.006836` | `9.64` | `0.1004` | `0.00796` | `187.5 s` | `0.2 s` | `170.1 s` | `15.8 s` |
| 22 | `+0.4424` | `2,635.4` | `0.000977` | `9.12` | `0.1004` | `0.00750` | `1,795.3 s` | `1,639.2 s` | `140.4 s` | `14.9 s` |
| 23 | `+0.6562` | `2,200.7` | `0.000977` | `7.52` | `0.1060` | `0.00833` | `1,430.7 s` | `1,299.6 s` | `114.7 s` | `15.6 s` |

