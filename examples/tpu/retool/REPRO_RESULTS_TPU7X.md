# Reproduction results: ReTool (Qwen3-30B-A3B) on TPU 7x

Status: **In progress** (started 2026-10-29). P0 preparation is complete; hardware stages (`P1a`, `P1b`, `P1c`, `P2`, `P3`) are executing and recorded below as each stage finishes.
Plan: [`REPRO_PLAN_TPU7X.md`](REPRO_PLAN_TPU7X.md) (rev 3).
Owner: @syangx39.

---

## 1. P0 preparation verification summary

All P0 preparation deliverables (`P0.1`–`P0.4`, `P0.6`, `P0.8`–`P0.10`) are merged into `tpu-main` at commit `cfabaada` and verified before hardware submission:

| Item | PR / Commit | Verification | Status |
|---|---|---|---|
| `P0.1` JobSet launcher (`examples/tpu/gke/run_ray_jobset_v7x_nap.sh`) | #4 (`cfabaada`) | Server-side `kubectl create --dry-run=server` passed on `bodaborg-tpu7x-nap` for both `ray-jobset-v7x-nap.yaml` and `ray-jobset-v7x-nap-prewarm.yaml`; 23 CPU tests in `tests/special_tpu/test_v7x_nap_launcher_on_cpu.py` passed (including UID-precondition delete, separated stdout/stderr `owned_create` with `launch-id` adoption, and fail-closed atomic GCS attempt-slot claiming) | PASS |
| `P0.2` Container image & package audit | #4 (`cfabaada`) | Inspected OCI layers of reference image `verl-tpu:v20261001-tsync990787257` (`sha256:dcece4bc3813359450f48d4523e5bd78e3139dc441a88b5c2e4d638caa054cbe`) and CI image `verl-tpu:v20261006-tsync1006` (`sha256:d92a2b3d6667584d67ea51874a721228a96c799263ab40f4aee168680476f64e`); findings documented in section 1.1 | PASS |
| `P0.3` Staged model & datasets | - | Staged under `<INPUTS_DIR>` with verified SHA-256 digests and row counts (section 1.2) | PASS |
| `P0.4` & `P0.6` Output persistence & allowlisted env dump | #4 (`cfabaada`) | `examples/tpu/gke/v7x_nap/pod_env_snapshot.py` writes allowlisted, secret-redacted pod and `/proc` Ray worker env snapshots (`env/<pod>-env.json`, `env/<pod>-procs.jsonl`) to `<RUN_DIR>/env/` on GCS | PASS |
| `P0.5` Host ordering preparation | #4 (`cfabaada`), #5 (`ed0170cf`) | `tpu_node_rank.py` computes topological/job-index rank and labels each Ray node (`verl-tpu-node-rank`, `verl-tpu-slice`); `RayResourcePool.get_placement_groups()` sorts by `(slice_key, rank, ip)` on every call and logs `slice=`, `rank=`, and `tpu-worker-id=`; 102 CPU tests in `tests/single_controller/test_placement_group_host_order_on_cpu.py` passed | PASS (hardware validation in `P1a` / `R2`) |
| `P0.8`, `P0.9`, `P0.10` Observability & rollout audit | #6 (`32c4efd1`) | Runtime `training.max_norm` vs `optim.clip_grad` logged in `TorchTitanEngine`; opt-in `+trainer.rollout_audit_dir` dumps token IDs, `response_mask`, decoded segments, reward I/O, `num_turns`, and `tq_layouts`; `summarize_validation_dump` enforces completeness by `question_id` and session ID (`mean_acc=None` if incomplete); 27 CPU tests passed in `tests/trainer/ppo/v1/test_rollout_audit_on_cpu.py` | PASS |

### 1.1 Container image package audit (`P0.2`)

| Component | Reference image (`v20261001-tsync990787257`, `sha256:dcece4bc...`) | Newer CI image (`v20261006-tsync1006`, `sha256:d92a2b3d...`) | Impact on TPU 7x runs |
|---|---|---|---|
| `python` / `ray` | Python 3.12, `ray==2.56.0` | Python 3.12, `ray==2.56.0` | Compatible |
| `torch` / `torch-tpu` | `torch==2.13.0+cpu`, `torch-tpu==0.1.1.dev20260912075415` | `torch==2.13.0+cpu`, `torch-tpu==0.1.1.dev20260912075415` | `multihost_wrapper.py` sets `NODE_RANK = JOB_COMPLETION_INDEX`; launcher defaults to `NODE_RANK_SOURCE=job-index` |
| `libtpu` / `jax` | `libtpu==0.0.47`, `jax==0.10.2` | `libtpu==0.0.47`, `jax==0.10.2` | Compatible |
| `tpu-sync-torch` | `0.0.1.dev20261001220631` (no `unbind_weights()`) | `0.0.1.dev20261006025601` (has `unbind_weights()`) | On `v20261001-tsync990787257`, runs must pass `RELEASE_SYNC_BUFFERS=False` and `VERL_RAIDEN_RELEASE_BUFFERS=0` so `RaidenCheckpointEngine` does not call missing `unbind_weights()` |
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
| `P1a` | `p1a-moe-smoke-01` | `cfabaada` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 1->3 | 10:36 / 10:36 (prewarm) / - / - | - | Running (prewarm 4/8 -> 8/8) | `<RUN_DIR>/p1a-moe-smoke-01` |
| `P1b` | pending | `cfabaada` | `sha256:dcece4bc...` | 1 (1 trainer) | 1->3 | - | - | Pending | `<RUN_DIR>/p1b-*` |
| `P1c` | pending | `cfabaada` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 1->2 | - | - | Pending (`P1b` export) | `<RUN_DIR>/p1c-*` |
| `P2` | pending | `cfabaada` | `sha256:dcece4bc...` | 1 (1 trainer) | 1->372 | - | - | Pending | `<RUN_DIR>/p2-*` |
| `P3` | pending | `cfabaada` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 1->25 | - | - | Pending (`P2` export) | `<RUN_DIR>/p3-*` |

---

## 3. Checks (`R1`–`R13`)

| Check | Kind | Phase | Target / Reference | Measured on TPU 7x | Verdict |
|---|---|---|---|---|---|
| `R1` | gate | every run | `LIBTPU_INIT_ARGS` contains `--xla_tpu_bf16_emission_mode=PROMOTE_F32` on all active roles | Pending first hardware run | PENDING |
| `R2` | gate | `P1a`, `P1c` | 32 trainer devices (4D mesh) + 32 rollout devices (TP=32, EP=32) on distinct `2x2x4` slices; trainer rank order matches `verl-tpu-node-rank`; Raiden sync completes | Pending `P1a` | PENDING |
| `R3` | observation | `P1b` | SFT step-1 `train/loss` vs `0.7941` (+-0.02) | Pending `P1b` | PENDING |
| `R4` | observation | `P2` | SFT steps 363–372 mean `train/loss` (~0.25) and step-372 in-sample `val/loss` (`0.214`, +-10%) | Pending `P2` | PENDING |
| `R5a` | gate | `P1b`, `P2` | `global_step_N/huggingface` export complete (16 bf16 shards, index, config, tokenizer) and reloadable | Pending `P1b` | PENDING |
| `R5b` | gate | `P1c`, `P3` | Staged SFT export matches source hashes across all hosts; TorchTitan + multi-host vLLM load cleanly; first Raiden sync completes | Pending `P1c` | PENDING |
| `R6` | observation | `P3` | Step-0 AIME 2024 (`22.44%`) and AIME 2025 (`14.17%`) mean@30 (+-4.5 pp), maj@30, best@30 | Pending `P3` | PENDING |
| `R7` | observation | `P3` | Early RL numerics (`actor/ppo_kl`, `actor/rollout_corr/k3_kl`, steps 3–10) | Pending `P3` | PENDING |
| `R8` | observation | `P3` | Step-25 AIME 2024 (`30.00%`) and AIME 2025 (`18.33%`) mean@30 (+-5 pp) | Pending `P3` | PENDING |
| `R9` | observation | `P2`, `P3` | Median step time and stage split (`gen`, `update_actor`, `update_weights`) vs v6e (`18.6 s` SFT, `108 s` RL) | Pending `P2`, `P3` | PENDING |
| `R10` | gate | `P3` | Complete step-0 and step-25 eval dumps (30 questions x 30 sessions by trajectory ID for both AIME 2024 and 2025) | Pending `P3` | PENDING |
| `R11` | gate | `P1c`, `P3` | Token-level rollout audit dump (`response_mask == 0` on tool results, `1` on assistant turns, multi-turn propagation, `compute_score` behavior) | Pending `P1c` | PENDING |
| `R12` | gate | `P1c`, `P3` | Post-Raiden-sync policy version & staleness within `max_off_policy_threshold=4` | Pending `P1c` | PENDING |
| `R13` | gate | `P4a` | Resume equivalence (`TransferQueue==0.1.8` in current images lacks `save_checkpoint`/`load_checkpoint`; requires quiescent boundary verification or `TransferQueue >= 0.1.9`) | Blocked unless quiescent boundary verified in `P4a` | BLOCKED (pending `P4a`) |
