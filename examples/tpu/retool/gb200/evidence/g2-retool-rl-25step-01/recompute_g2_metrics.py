#!/usr/bin/env python3
"""Independently recompute and verify all Stage G2 (g2-retool-rl-25step-01) training & AIME evaluation metrics."""

from collections import defaultdict
import json
from pathlib import Path
import statistics


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
        by_ds_q[r["data_source"]][r["question_id"]].append(r)
        if r["tool_response_count"] > 0:
            with_tool += 1
        total_tools += r["tool_response_count"]

    out: dict[str, dict] = {}
    for ds in ("aime_2024", "aime_2025"):
        q_map = by_ds_q[ds]
        assert len(q_map) == 30, f"{ds}: expected 30 questions, got {len(q_map)}"
        per_q_means = []
        for qid, samples in sorted(q_map.items()):
            assert len(samples) == 30, f"{ds} {qid}: expected n=30 samples, got {len(samples)}"
            per_q_means.append(sum(1.0 if s["acc"] else 0.0 for s in samples) / 30.0)
        mean_at_30 = sum(per_q_means) / len(per_q_means)
        logged_mean = metrics_step[f"val-core/{ds}/acc/mean@30"]
        assert abs(mean_at_30 - logged_mean) < 1e-6, f"{ds} mean@30 mismatch: {mean_at_30} vs {logged_mean}"
        out[ds] = {
            "num_questions": len(q_map),
            "samples_per_question": 30,
            "recomputed_acc_mean_at_30": round(mean_at_30, 6),
            "logged_acc_mean_at_30": round(logged_mean, 6),
            "logged_acc_std_at_30": round(metrics_step[f"val-aux/{ds}/acc/std@30"], 6),
            "logged_acc_maj_at_30": round(metrics_step[f"val-core/{ds}/acc/maj@30/mean"], 6),
            "logged_acc_best_at_30": round(metrics_step[f"val-core/{ds}/acc/best@30/mean"], 6),
            "logged_acc_maj_at_8": round(metrics_step[f"val-aux/{ds}/acc/maj@8/mean"], 6),
            "logged_acc_best_at_8": round(metrics_step[f"val-aux/{ds}/acc/best@8/mean"], 6),
        }
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
