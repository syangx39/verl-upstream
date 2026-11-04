# GB200 run evidence

- [G2 25-step run: logs, metrics, environment, samples and recomputation](./g2-retool-rl-25step-01/)
- [G64-D 64-GPU 25-step run: logs, metrics, 16-node environment, Step 0/25 samples, rollout and mask-audit summary](./g64-retool-25s-01/)
- [G64-D 64-GPU 5-step smoke validation run: logs, metrics, 16-node environment, Step 0/5 samples, rollout and mask-audit summary](./g64-retool-smoke-04/)

Shared G64-D scripts:

- [`build_g64_run_evidence.py`](./build_g64_run_evidence.py): builds a run directory above from the downloaded GCS run directory.
- [`gen_g64_report_tables.py`](./gen_g64_report_tables.py): generates the report tables (`--run <run> --print-markdown`).
- [`recompute_g64_metrics.py`](./recompute_g64_metrics.py): rescores a run's validation samples with the runtime scorer and checks the 16-node reports (`--run <run>`).
- [`repro_rollout_is_level.py`](./repro_rollout_is_level.py): token- vs. sequence-level `rollout_is_*` metrics on synthetic data (CPU `torch`).

Original run artifacts are retained unchanged.
