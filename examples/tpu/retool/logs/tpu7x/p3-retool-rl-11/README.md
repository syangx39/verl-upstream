# Stage P3 evidence: `p3-retool-rl-11`

Derived evidence for section 4.0.2 and checks `R6`–`R12` of [`REPRO_RESULTS_TPU7X.md`](../../../exp/tpu7x/REPRO_RESULTS_TPU7X.md). The raw run files stay in the run directory on GCS; the JSON files here were produced from them by `verify_p3_results.py`, using the code of the run's commit `aa10dfed`.

## Raw files

Run directory: `gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3-retool-rl-11/`

| File (relative to the run directory) | Contents | SHA-256 |
|---|---|---|
| `attempts/bef2992e/logs/driver.log` | Ray job driver log: resolved config, per-step metrics, Raiden parity lines (32,015,184 bytes) | `abc80864365fb3da4bc0c10f3e3e26d116ae2a83be12cf8a44fb23cbd98098cb` |
| `validation/0.jsonl` | Step-0 AIME 2024 + 2025 validation dump (8,555,870 bytes) | `3d621a94f86e1ef323b4106ddc9a564f6f3a86b68e2535f6b8e85bf8ec066c92` |
| `validation/25.jsonl` | Step-25 AIME 2024 + 2025 validation dump (8,514,582 bytes) | `4f7026d3cfc4275d6a7626b16a7ce60e8ae54ef326279144850162bc9b95329c` |
| `rollout_audit/{1..25}.jsonl` | Token-level audit of 8 training sessions per step (200 records) | per-file digests of the 25 files, sorted by step and hashed again: `f0462be79f2bbfb9f0f1ae3ab60f3d4afd65002b35fd514115885336f76910b1` |
| `env/raiden_exact_parity_step_0.json` | Per-rank Step-0 exact parity report | - |

AIME sources (staged inputs, section 1.2): `gs://ubench-logs/xiaotongyang/retool-v7x/inputs/dataset/Maxwell-Jia/AIME_2024/data/train-00000-of-00001.parquet` (`1332803e...`) and `.../yentinglin/aime_2025/data/train-00000-of-00001.parquet` (`4f4432a5...`).

## Files here

- `validation_summaries.json`: `summarize_validation_dump` on both dumps, with `expected_n=30` and the expected question ids computed from the AIME source rows through the run's prompt mapping (`map_fn`) and `validation_question_id`. Includes per-question means.
- `scorer_check.json`: every validation row and every audited training session re-scored with `math_dapo.compute_score`, with `strict_box_verify=False` and `True`, and compared with the logged values; plus what ReTool's `compute_score` would have returned.
- `train_metrics.json`: per-step metrics parsed from `driver.log` (the section 4.0.2 table) and the Step-0 / Step-25 `val-core` metrics.

## Reproduce

```bash
RUN=gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3-retool-rl-11
IN=gs://ubench-logs/xiaotongyang/retool-v7x/inputs/dataset
mkdir -p data/src
gcloud storage cp "$RUN/attempts/bef2992e/logs/driver.log" data/
gcloud storage cp -r "$RUN/validation" "$RUN/rollout_audit" data/
gcloud storage cp "$IN/Maxwell-Jia/AIME_2024/data/train-00000-of-00001.parquet" data/src/AIME_2024.parquet
gcloud storage cp "$IN/yentinglin/aime_2025/data/train-00000-of-00001.parquet" data/src/aime_2025.parquet
# --repo: any checkout that contains commit aa10dfed (the PR #10 branch); needs pyarrow
python examples/tpu/retool/logs/tpu7x/p3-retool-rl-11/verify_p3_results.py --data data --repo . --out /tmp/p3-check
```

## Summary

- **Eval dumps (`R10`)**: both dumps are complete. For each of AIME 2024 and AIME 2025: 30 of 30 expected questions, 0 missing or unexpected, exactly 30 sessions per question, 1 output per session, every session with `acc`, every `gts` equal to the source answer. Means of the session final outputs: Step 0 `25.56%` / `17.33%`, Step 25 `26.78%` / `17.78%`, equal to the logged mean@30.
- **Scorer actually used**: verl's default `math_dapo.compute_score` with `strict_box_verify=False` and no tool-use reward shaping, for both training and validation. It reproduces the logged acc, score and pred of all 3,600 validation rows and all 200 audited training sessions. ReTool's `compute_score` (`strict_box_verify=True`, plus shaping of incorrect multi-turn rollouts) would have scored 147 of the 200 audited sessions differently. Scored with `strict_box_verify=True`, the AIME mean@30 would be Step 0 `25.78%` / `17.44%` and Step 25 `27.78%` / `17.78%`.
