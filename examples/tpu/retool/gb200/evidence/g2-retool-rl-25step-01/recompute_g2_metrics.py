#!/usr/bin/env python3
"""Independently recompute and verify Stage G2 (g2-retool-rl-25step-01) training, timing, and AIME evaluation metrics."""

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_validation_samples(path: Path, expected_step: int, metrics_step: dict) -> dict:
    rows = load_jsonl(path)
    assert len(rows) == 1800, f"Expected 1800 validation samples in {path.name}, got {len(rows)}"
    by_ds_q: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with_tool = 0
    total_tools = 0
    for r in rows:
        assert r["step"] == expected_step
        assert "output_tail_300" in r and len(r["output_tail_300"]) <= 300
        by_ds_q[r["data_source"]][r["question_id"]].append(r)
        if r["tool_response_count"] > 0:
            with_tool += 1
        total_tools += r["tool_response_count"]

    # Verify pred/acc extraction from output_tail_300 using verl/utils/reward_score/math_dapo.py (stdlib-only module).
    # Note on runtime scorer resolution in G2 (verified in head.log lines 765-766 and 899-901):
    # - The launch script passed legacy CLI overrides `custom_reward_function.path=pkg://examples.tpu.retool.retool_tpu`
    #   and `custom_reward_function.name=compute_score` (inherited from examples/tpu/retool/run_qwen3_32b_dapo_torchtitan.sh).
    # - Under `trainer.use_v1=True` (`TaskRunnerV1` in `verl/trainer/main_ppo.py`), `migrate_legacy_reward_impl` is not
    #   called, so `config.reward.custom_reward_function.path` remained `None` (`head.log#L899-L901`), and
    #   `verl/trainer/ppo/reward.py:get_custom_reward_fn` fell back to `default_compute_score`
    #   (`verl/utils/reward_score/__init__.py:59-62`), which invokes `math_dapo.compute_score(..., strict_box_verify=False)`.
    # Below we verify 100% agreement with the runtime scorer (`strict_box_verify=False`) AND also recompute metrics
    # under `strict_box_verify=True` (`retool_tpu.compute_score`) to quantify the exact difference (5 samples at Step 0,
    # 9 samples at Step 25 where Markdown around `Answer:` caused Minerva regex extraction to return `**45`/`**`/`\(204\)`).
    import importlib.util

    math_dapo_path = REPO_ROOT / "verl" / "utils" / "reward_score" / "math_dapo.py"
    spec = importlib.util.spec_from_file_location("math_dapo", math_dapo_path)
    assert spec is not None and spec.loader is not None
    math_dapo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_dapo)
    strict_By_ds_q: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    strict_vs_runtime_diffs: list[dict] = []
    for r in rows:
        res_runtime = math_dapo.compute_score(r["output_tail_300"], str(r["gts"]), strict_box_verify=False)
        pred_runtime = res_runtime["pred"] if res_runtime["pred"] is not None else ""
        assert pred_runtime == r["pred"] and bool(res_runtime["acc"]) == bool(r["acc"]), (
            f"Mismatch on {r['uid']}: {res_runtime} vs {r}"
        )
        expected_score = 1.0 if r["acc"] else -1.0
        assert abs(float(r["score"]) - expected_score) < 1e-6, (
            f"Score mismatch on {r['uid']}: {r['score']} vs {expected_score}"
        )

        res_strict = math_dapo.compute_score(r["output_tail_300"], str(r["gts"]), strict_box_verify=True)
        pred_strict = res_strict["pred"] if res_strict["pred"] is not None else ""
        strict_By_ds_q[r["data_source"]][r["question_id"]].append(bool(res_strict["acc"]))
        if bool(res_strict["acc"]) != bool(res_runtime["acc"]):
            strict_vs_runtime_diffs.append(
                {
                    "uid": r["uid"],
                    "data_source": r["data_source"],
                    "question_id": r["question_id"],
                    "gts": str(r["gts"]),
                    "runtime_strict_false_pred": pred_runtime,
                    "runtime_strict_false_acc": bool(res_runtime["acc"]),
                    "strict_true_pred": pred_strict,
                    "strict_true_acc": bool(res_strict["acc"]),
                }
            )
    raw_tail_verified = True

    out: dict[str, dict] = {}
    for ds in ("aime_2024", "aime_2025"):
        q_map = by_ds_q[ds]
        assert len(q_map) == 30, f"{ds}: expected 30 questions, got {len(q_map)}"
        per_q_means = []
        per_q_stds = []
        total_correct = 0
        for qid, samples in sorted(q_map.items()):
            assert len(samples) == 30, f"{ds} {qid}: expected n=30 samples, got {len(samples)}"
            accs = [1.0 if s["acc"] else 0.0 for s in samples]
            total_correct += sum(1 for s in samples if s["acc"])
            m = sum(accs) / 30.0
            var = sum((x - m) ** 2 for x in accs) / 30.0
            per_q_means.append(m)
            per_q_stds.append(math.sqrt(var))

        mean_at_30 = sum(per_q_means) / len(per_q_means)
        std_at_30 = sum(per_q_stds) / len(per_q_stds)
        logged_mean = metrics_step[f"val-core/{ds}/acc/mean@30"]
        logged_std = metrics_step[f"val-aux/{ds}/acc/std@30"]
        logged_reward_mean = metrics_step[f"val-aux/{ds}/reward/mean@30"]
        assert abs(mean_at_30 - logged_mean) < 1e-6, f"{ds} mean@30 mismatch: {mean_at_30} vs {logged_mean}"
        assert abs(std_at_30 - logged_std) < 1e-6, f"{ds} std@30 mismatch: {std_at_30} vs {logged_std}"
        assert abs((2.0 * mean_at_30 - 1.0) - logged_reward_mean) < 1e-6, (
            f"{ds} reward/mean@30 mismatch: {2.0 * mean_at_30 - 1.0} vs {logged_reward_mean}"
        )

        # Also compute strict_box_verify=True metrics for side-by-side comparison
        strict_q_map = strict_By_ds_q[ds]
        strict_per_q_means = []
        strict_per_q_stds = []
        strict_total_correct = 0
        for qid, strict_accs_bool in sorted(strict_q_map.items()):
            strict_accs = [1.0 if a else 0.0 for a in strict_accs_bool]
            strict_total_correct += sum(1 for a in strict_accs_bool if a)
            sm = sum(strict_accs) / 30.0
            svar = sum((x - sm) ** 2 for x in strict_accs) / 30.0
            strict_per_q_means.append(sm)
            strict_per_q_stds.append(math.sqrt(svar))
        strict_mean_at_30 = sum(strict_per_q_means) / len(strict_per_q_means)
        strict_std_at_30 = sum(strict_per_q_stds) / len(strict_per_q_stds)

        out[ds] = {
            "num_questions": len(q_map),
            "samples_per_question": 30,
            "runtime_scorer": "default_compute_score -> math_dapo.compute_score(strict_box_verify=False)",
            "total_correct_samples": f"{total_correct} / 900",
            "raw_tail_300_extraction_verified": raw_tail_verified,
            "recomputed_acc_mean_at_30": round(mean_at_30, 6),
            "logged_acc_mean_at_30": round(logged_mean, 6),
            "recomputed_acc_std_at_30": round(std_at_30, 6),
            "logged_acc_std_at_30": round(logged_std, 6),
            "strict_box_verify_true_total_correct_samples": f"{strict_total_correct} / 900",
            "strict_box_verify_true_acc_mean_at_30": round(strict_mean_at_30, 6),
            "strict_box_verify_true_acc_std_at_30": round(strict_std_at_30, 6),
            "logged_acc_maj_at_30": round(metrics_step[f"val-core/{ds}/acc/maj@30/mean"], 6),
            "logged_acc_best_at_30": round(metrics_step[f"val-core/{ds}/acc/best@30/mean"], 6),
            "logged_acc_maj_at_8": round(metrics_step[f"val-aux/{ds}/acc/maj@8/mean"], 6),
            "logged_acc_best_at_8": round(metrics_step[f"val-aux/{ds}/acc/best@8/mean"], 6),
            "maj_best_verification_note": (
                "mean@30 and std@30 are exact deterministic reductions over the 900 samples; "
                "best@k and maj@k in verl use 1000-iteration numpy bootstrap (seed=42) where maj@k tie-breaking "
                "depends on pre-sort async rollout completion order at runtime."
            ),
        }
    out["strict_box_verify_vs_runtime_diff_count"] = len(strict_vs_runtime_diffs)
    out["strict_box_verify_vs_runtime_diffs"] = strict_vs_runtime_diffs
    out["tool_usage"] = {
        "trajectories_with_tool_response": with_tool,
        "total_trajectories": len(rows),
        "ratio": round(with_tool / len(rows), 6),
        "total_tool_responses": total_tools,
        "num_turns_mean": round(metrics_step["val-aux/num_turns/mean"], 4),
    }
    return out


def main() -> None:
    ev_dir = Path(__file__).resolve().parent
    metrics = json.loads((ev_dir / "g2_25step_metrics.json").read_text(encoding="utf-8"))
    rollout_sum = json.loads((ev_dir / "rollout_and_audit_summary.json").read_text(encoding="utf-8"))

    steps_1_25 = [metrics[str(i)] for i in range(1, 26)]
    steps_2_25 = [metrics[str(i)] for i in range(2, 26)]
    steps_3_25 = [metrics[str(i)] for i in range(3, 26)]

    val0 = verify_validation_samples(ev_dir / "validation_step0_samples.jsonl", 0, metrics["0"])
    val25 = verify_validation_samples(ev_dir / "validation_step25_samples.jsonl", 25, metrics["25"])

    summary = {
        "run_name": "g2-retool-rl-25step-01",
        "code_sha": "64ef634b173c69ea31eff598caeee81d9a9caea1",
        "image_digest": "verl-torchtitan-cu130-arm64@sha256:a139e2b491ac23f06f790fe4277afb9bd543a94bc740c92173e5fdb13ddaba40",
        "hardware": "2 nodes x 4 NVIDIA GB200 GPUs = 8 GPUs (4 trainer FSDP2=4 + 4 rollout vLLM TP=4)",
        "step_25_checkpoint_saved": False,
        "step_25_checkpoint_note": "SAVE_FREQ=-1 (trainer.save_freq=-1); Step 25 evaluation ran in-memory before exit.",
        "wall_clock_breakdown_from_head_log": {
            "main_ppo_start_utc": "2026-11-02 23:47:07.289",
            "step0_weight_sync_done_utc": "2026-11-02 23:50:05.530",
            "step0_eval_start_utc": "2026-11-02 23:50:05.950",
            "step0_eval_end_tqdm_start_utc": "2026-11-02 23:53:06",
            "step25_weight_sync_done_utc": "2026-11-03 00:11:30.170",
            "step25_eval_start_utc": "2026-11-03 00:11:30.538",
            "step25_eval_and_audit_done_utc": "2026-11-03 00:14:42.324",
            "step0_eval_wall_s": 180.1,
            "training_25_steps_only_wall_s": 1104.5,
            "step25_eval_wall_s": round(metrics["25"]["timing_s/testing"], 2),
            "tqdm_25_steps_plus_step25_eval_wall_s": 1296.0,
            "tqdm_25_steps_plus_step25_eval_formatted": "21m 36s (21m 35s at step:25 log line, includes 25 training steps + Step 25 eval; excludes Step 0 eval)",
            "both_evals_plus_25_steps_wall_s": 1476.4,
            "both_evals_plus_25_steps_formatted": "24m 36s (23:50:05.950 -> 00:14:42.324, includes Step 0 eval + 25 training steps + Step 25 eval)",
            "full_main_ppo_wall_s": 1657.1,
            "full_main_ppo_formatted": "27m 37s (23:47:07.289 -> 00:14:44.392, includes Ray init + 30B load + Step 0 sync + both evals + 25 steps)",
        },
        "training_25_steps": {
            "pg_loss_mean_1_25": round(statistics.mean(s["actor/pg_loss"] for s in steps_1_25), 4),
            "grad_norm_mean_1_25": round(statistics.mean(s["actor/grad_norm"] for s in steps_1_25), 4),
            "grad_norm_min_max_1_25": [
                round(min(s["actor/grad_norm"] for s in steps_1_25), 4),
                round(max(s["actor/grad_norm"] for s in steps_1_25), 4),
            ],
            "training_ppl_mean_1_25": round(
                statistics.mean(s["actor/rollout_corr/training_ppl"] for s in steps_1_25), 4
            ),
            "rollout_ppl_mean_1_25": round(
                statistics.mean(s["actor/rollout_corr/rollout_ppl"] for s in steps_1_25), 4
            ),
            "ppl_ratio_mean_1_25": round(
                statistics.mean(s["actor/rollout_corr/ppl_ratio"] for s in steps_1_25), 4
            ),
            "ppl_ratio_min_max_1_25": [
                round(min(s["actor/rollout_corr/ppl_ratio"] for s in steps_1_25), 4),
                round(max(s["actor/rollout_corr/ppl_ratio"] for s in steps_1_25), 4),
            ],
            "response_length_mean_1_25": round(
                statistics.mean(s["response_length/mean"] for s in steps_1_25), 2
            ),
            "num_turns_mean_1_25": round(statistics.mean(s["num_turns/mean"] for s in steps_1_25), 2),
            "update_actor_s_mean_3_25": round(
                statistics.mean(s["timing_s/update_actor"] for s in steps_3_25), 2
            ),
            "update_weights_s_mean_1_25": round(
                statistics.mean(s["timing_s/update_weights"] for s in steps_1_25), 2
            ),
            "update_weights_s_mean_3_25": round(
                statistics.mean(s["timing_s/update_weights"] for s in steps_3_25), 2
            ),
            "step_time_s_mean_2_25": round(statistics.mean(s["timing_s/step"] for s in steps_2_25), 2),
            "step_time_s_mean_3_25": round(statistics.mean(s["timing_s/step"] for s in steps_3_25), 2),
            "step_time_s_mean_1_25": round(statistics.mean(s["timing_s/step"] for s in steps_1_25), 2),
            "total_tokens_sum_1_25": int(sum(s["perf/total_num_tokens"] for s in steps_1_25)),
        },
        "validation_step0": val0,
        "validation_step25": val25,
        "rollout_and_audit_totals": rollout_sum["totals"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
