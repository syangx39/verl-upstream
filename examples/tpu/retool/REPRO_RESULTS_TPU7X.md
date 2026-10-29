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
| `P1b` | `p1b-retool-sft-01` | `cfabaada` | `sha256:dcece4bc...` | 1 (1 trainer) | 1->3 | 11:06:20 / 11:15:33 / 11:16:31 / 11:32:31 | 28.4 s (steps 1–3: 55.6 s, 28.4 s, 27.1 s) | **SUCCEEDED** (exit 0) | `<RUN_DIR>/p1b-retool-sft-01` |
| `P2` | `p2-retool-sft-01` | `cfabaada` | `sha256:dcece4bc...` | 1 (1 trainer) | 1->372 | 11:46:17 / 12:03:14 / 12:04:12 / 13:36:05 | 9.57 s mean / 10.0 s median warm (Epochs 2–6; Epoch 1 mean 15.73 s) | **SUCCEEDED** (exit 0) | `<RUN_DIR>/p2-retool-sft-01` |
| `P1a` | `p1a-moe-smoke-05` | `cfabaada` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 1->3 | 12:18:52 (`w0` running on 1st slice; `w1` waiting for 2nd slice) | - | Pre-warming (4/8 pods running) | `<RUN_DIR>/p1a-moe-smoke-05` |
| `P1c` | `p1c-retool-rl-01` | `cfabaada` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 1->2 | queued (`P1b` export ready) | - | Queued | `<RUN_DIR>/p1c-retool-rl-01` |
| `P3` | `p3-retool-rl-01` | `cfabaada` | `sha256:dcece4bc...` | 2 (1 trainer + 1 rollout) | 1->25 | queued (`P2` step-372 export ready) | - | Queued | `<RUN_DIR>/p3-retool-rl-01` |

---

## 3. Checks (`R1`–`R13`)

| Check | Kind | Phase | Target / Reference | Measured on TPU 7x | Verdict |
|---|---|---|---|---|---|
| `R1` | gate | every run | `LIBTPU_INIT_ARGS` contains `--xla_tpu_bf16_emission_mode=PROMOTE_F32` on all active roles | `P1b`, `P2`: verified in `<RUN_DIR>/{p1b-retool-sft-01,p2-retool-sft-01}/env/*-procs.jsonl` across all 4 trainer hosts (`LIBTPU_INIT_ARGS` includes `--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32`, `TPU_ACCELERATOR_TYPE="tpu7x"`). Rollout role pending `P1a`/`P1c`/`P3` | **PASS (trainer in `P1b`, `P2`)** / pending rollout |
| `R2` | gate | `P1a`, `P1c` | 32 trainer devices (4D mesh) + 32 rollout devices (TP=32, EP=32) on distinct `2x2x4` slices; trainer rank order matches `verl-tpu-node-rank`; Raiden sync completes | `P1b` & `P2` (1-slice trainer): in both runs the 4 hosts had non-monotonic IP order (`P1b`: `10.224.130.8, 10.224.106.8, 10.224.114.8, 10.224.113.8`; `P2`: `10.224.180.7, 10.224.242.7, 10.224.243.7, 10.224.181.7` for workers `0..3`) and were ordered `0, 1, 2, 3` by `verl-tpu-node-rank`. Dual-slice trainer+rollout + Raiden sync pending `P1a`/`P1c` | Partial (`P1b`/`P2` trainer host order PASS; dual-slice pending `P1a`/`P1c`) |
| `R3` | observation | `P1b` | SFT 3-step smoke `train/loss` (`0.7950 -> 0.5787 -> 0.5177`, step 3 `+-0.05`) and step-3 `val/loss` (`0.4050`, `+-0.05`) | `train/loss`: `0.7932` (step 1) -> `0.5881` (step 2) -> `0.5224` (step 3, `+0.0047` vs ref); step-3 `val/loss`: `0.4095` (`+0.0045` vs ref). Moreover, step-1 `train/loss` in `P2` (`0.7931681871414185`) was bit-identical to `P1b` (`0.7931681871414185`) and within `-0.0009` of #108's v6e smoke (`0.7941`) | **PASS** |
| `R4` | observation | `P2` | SFT steps 363–372 mean `train/loss` (~0.25 in #108; `0.038` in the author's 6-epoch doc) and step-372 in-sample `val/loss` (`0.214` in #108) | `P2` (372 steps, 6 epochs): `train/loss` decreased smoothly from `0.7932` (step 1) to `0.3832` (step 62, Epoch 1), `0.2818` (step 124, Epoch 2), `0.1612` (step 186, Epoch 3), `0.1017` (step 248, Epoch 4), `0.0612` (step 310, Epoch 5), and `0.0359` (step 372, Epoch 6; **steps 363–372 mean `0.0393`**, median `0.0393`, matching the author's 6-epoch doc's `0.038`). In-sample `val/loss` across epochs 1–6: `0.2713 -> 0.1793 -> 0.1078 -> 0.0618 -> 0.0395 -> 0.0263`. Both `global_step_186/huggingface` and `global_step_372/huggingface` were exported for downstream evaluation | **RECORDED** (see section 4.2 analysis) |
| `R5a` | gate | `P1b`, `P2` | `global_step_N/huggingface` export complete (16 bf16 shards, index, config, tokenizer) and reloadable | `P1b` (`global_step_3/huggingface`) and `P2` (`global_step_186/huggingface`, `global_step_372/huggingface`): each exported all 16 bf16 `.safetensors` shards + `model.safetensors.index.json` (`18,867` tensors, `61,064,245,248` param bytes, `lm_head.weight` & `model.embed_tokens.weight` present, `tie_word_embeddings=false`) + config/tokenizer (`27` objects, `61,084,185,820` bytes total) | **PASS (`P1b`, `P2`)** |
| `R5b` | gate | `P1c`, `P3` | Staged SFT export matches source hashes across all hosts; TorchTitan + multi-host vLLM load cleanly; first Raiden sync completes | Pending `P1c`, `P3` | PENDING |
| `R6` | observation | `P3` | Step-0 AIME 2024 (`22.44%`) and AIME 2025 (`14.17%`) mean@30 (+-4.5 pp), maj@30, best@30 | Pending `P3` | PENDING |
| `R7` | observation | `P3` | Early RL numerics (`actor/ppo_kl`, `actor/rollout_corr/k3_kl`, steps 3–10) | Pending `P3` | PENDING |
| `R8` | observation | `P3` | Step-25 AIME 2024 (`30.00%`) and AIME 2025 (`18.33%`) mean@30 (+-5 pp) | Pending `P3` | PENDING |
| `R9` | observation | `P2`, `P3` | Median step time and stage split (`gen`, `update_actor`, `update_weights`) vs v6e (`18.6 s` SFT, `108 s` RL) | `P2` SFT: **warm mean `9.57 s/step`**, **median `10.0 s/step`**, **p90 `10.0 s/step`** (`min=9 s, max=11 s` across Epochs 2–6, `n=300` steps; **1.94x faster** than v6e's `18.6 s/step`). Epoch 1 mean was `15.73 s/step` while XLA bucket shapes compiled; cached 32-sample validation took `12 s`. RL step time pending `P3` | **RECORDED (`P2`)** / pending `P3` |
| `R10` | gate | `P3` | Complete step-0 and step-25 eval dumps (30 questions x 30 sessions by trajectory ID for both AIME 2024 and 2025) | Pending `P3` | PENDING |
| `R11` | gate | `P1c`, `P3` | Token-level rollout audit dump (`response_mask == 0` on tool results, `1` on assistant turns, multi-turn propagation, `compute_score` behavior) | Pending `P1c` | PENDING |
| `R12` | gate | `P1c`, `P3` | Post-Raiden-sync policy version & staleness within `max_off_policy_threshold=4` | Pending `P1c` | PENDING |
| `R13` | gate | `P4a` | Resume equivalence (`TransferQueue==0.1.8` in current images lacks `save_checkpoint`/`load_checkpoint`; requires quiescent boundary verification or `TransferQueue >= 0.1.9`) | Blocked unless quiescent boundary verified in `P4a` | BLOCKED (pending `P4a`) |

---

## 4. Stage details

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
    - Step 1 `train/loss` (`0.793168`) matches `P1b` bit-for-bit and matches #108's v6e smoke (`0.7941`) within `-0.0009`.
    - Over 6 epochs on the 2,000-row dataset, `train/loss` reaches `0.0393` (mean of steps 363–372) and `0.0359` (step 372), which matches the `train/loss = 0.038` reported in the recipe author's 6-epoch TPU v6e report. Meanwhile, Epoch 1 (`val/loss = 0.2713`) and Epoch 2 (`mean train/loss = 0.2735`, `last-10 mean = 0.2664`, `val/loss = 0.1793`) bracket the `train/loss ~ 0.25, val/loss = 0.214` numbers quoted in #108's PR summary. Because `SAVE_FREQ=186` exported both `global_step_186/huggingface` and `global_step_372/huggingface`, `P3` starts from `global_step_372/huggingface` per `D9` while preserving `global_step_186/huggingface` for comparison if needed.
- **Exported HF checkpoints (`R5a`)**:
  - Mid-run export: `<RUN_DIR>/p2-retool-sft-01/checkpoint/global_step_186/huggingface/` (27 objects, `61,084,185,820` bytes, 16 bf16 `.safetensors` shards, `18,867` tensors).
  - Final export: `<RUN_DIR>/p2-retool-sft-01/checkpoint/global_step_372/huggingface/` (27 objects, `61,084,185,820` bytes, 16 bf16 `.safetensors` shards, `18,867` tensors, `61,064,245,248` param bytes, `lm_head.weight` & `model.embed_tokens.weight` present, `tie_word_embeddings=false`).
  - `verdict.json`: `status="SUCCEEDED"`, `node_rank_ok=true`, `duration_s=5548` (`92m 28s` head runtime, including model staging, 372 training steps, 6 validations, and two 57 GiB HF checkpoint exports), `timed_out=false`.
