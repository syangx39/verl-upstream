# Resolved-config diff: TPU `p3-retool-rl-11` vs GPU `G2`

`config_diff.tsv` lists every resolved verl config key that differs between the two runs (or that only one side's verl version has), produced by `diff_resolved_configs.py` from the config block each run logs at start-up.

| Input | Location | SHA-256 |
|---|---|---|
| TPU `p3-retool-rl-11` driver log (verl `aa10dfed`) | `gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3-retool-rl-11/attempts/bef2992e/logs/driver.log` | `abc80864365fb3da4bc0c10f3e3e26d116ae2a83be12cf8a44fb23cbd98098cb` |
| GPU `G2` head log (verl `64ef634b`) | [`gb200/logs/g2-retool-rl-25step-01/head.log`](../../gb200/logs/g2-retool-rl-25step-01/head.log) | `c943e7ab8631454ecf7785acae645321a5690566e49ae705ae2cf53f3bba8012` |

```bash
gcloud storage cp gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3-retool-rl-11/attempts/bef2992e/logs/driver.log /tmp/
python examples/tpu/retool/results/config-diff-p3-retool-rl-11-vs-g2/diff_resolved_configs.py \
  /tmp/driver.log examples/tpu/retool/gb200/logs/g2-retool-rl-25step-01/head.log \
  --left-name tpu_p3_retool_rl_11 --right-name gpu_g2 > config_diff.tsv
```

## Result: 108 differing keys

| Category | Keys | What differs |
|---|---|---|
| `platform` | 67 | device counts; trainer layout (TPU `CP=4`, `DP shard=8`; GPU `FSDP2=4`); rollout `TP=32/EP=32` vs `TP=4`; attention (`varlen` vs `flex`), activation checkpointing, torch compile, SPMD backend; vLLM memory and batching; weight sync (`raiden` vs `nccl`); TransferQueue backend |
| `paths/naming` | 16 | run directories, model staging paths, experiment and project names, `PYTHONPATH` |
| `token-budget` | 5 | dynamic-batch token budget per device (`4608` vs `16384`, actor / ref / rollout / data) and `torchtitan.max_seq_len` (`18432` vs `16384`) |
| `schema` | 20 | keys that only one verl version has: rollout and reward-model quantization config objects (both off), `rollout.topk_log_probs` (GPU `0`), `algorithm.rollout_correction.score_centering` (GPU `False`), `distillation.distillation_loss.tail_mass_eps` (distillation not used) |
| `review` | 0 | - |

Every other key is equal, including the learning rate, `optim.clip_grad` (`0.088` on both sides, though the effective TorchTitan threshold is `1.0` on both), batch sizes, `n`, `max_turns`, response length, KL settings, clip ratios, rollout correction, validation `n`, sampling parameters and the reward config (`reward.custom_reward_function.path: None` on both sides). The training and validation files are the same datasets; only their path prefix differs (listed under `paths/naming`).

The `schema` rows only show that the two verl versions have different config fields; they do not show that the algorithm code is the same. That still has to be checked in the source of the two commits (see the re-run plan, section 6).
