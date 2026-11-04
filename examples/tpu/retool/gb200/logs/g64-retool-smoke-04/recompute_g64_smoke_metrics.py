#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Independently recompute Stage G64-D (64-GPU GB200) smoke validation metrics and shaped-reward parity."""

import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_expected_shaped_score(acc: bool, num_turns: int) -> float:
    if acc:
        return 1.0
    num_tool_calls = max(int(num_turns) - 2, 0) // 2
    return -1.0 + 0.1 * min(num_tool_calls, 4)


def verify_validation_samples(path: Path, expected_step: int, metrics_step: dict) -> dict:
    rows = load_jsonl(path)
    assert len(rows) == 1800, f"Expected 1800 validation samples in {path.name}, got {len(rows)}"
    by_ds_q: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with_tool = 0
    total_tools = 0

    math_dapo_path = REPO_ROOT / "verl" / "utils" / "reward_score" / "math_dapo.py"
    spec = importlib.util.spec_from_file_location("math_dapo", math_dapo_path)
    assert spec is not None and spec.loader is not None
    math_dapo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_dapo)

    for r in rows:
        assert r["step"] == expected_step
        assert "output_tail_300" in r and len(r["output_tail_300"]) <= 300
        assert "num_turns" in r and r["num_turns"] is not None
        by_ds_q[r["data_source"]][r["question_id"]].append(r)
        if r["tool_response_count"] > 0:
            with_tool += 1
        total_tools += r["tool_response_count"]

        res_strict = math_dapo.compute_score(r["output_tail_300"], str(r["gts"]), strict_box_verify=True)
        pred_strict = res_strict["pred"] if res_strict["pred"] is not None else ""
        assert pred_strict == r["pred"] and bool(res_strict["acc"]) == bool(r["acc"]), (
            f"Strict box verify mismatch on {r['uid']}: {res_strict} vs {r}"
        )
        expected_score = compute_expected_shaped_score(bool(r["acc"]), int(r["num_turns"]))
        assert abs(float(r["score"]) - expected_score) < 1e-6, (
            f"Shaped score mismatch on {r['uid']}: {r['score']} vs {expected_score} (num_turns={r['num_turns']})"
        )

    out: dict[str, dict] = {}
    for ds in ("aime_2024", "aime_2025"):
        q_map = by_ds_q[ds]
        assert len(q_map) == 30, f"{ds}: expected 30 questions, got {len(q_map)}"
        per_q_means = []
        per_q_stds = []
        per_q_reward_means = []
        total_correct = 0
        for qid, samples in sorted(q_map.items()):
            assert len(samples) == 30, f"{ds} {qid}: expected n=30 samples, got {len(samples)}"
            accs = [1.0 if s["acc"] else 0.0 for s in samples]
            scores = [float(s["score"]) for s in samples]
            total_correct += sum(1 for s in samples if s["acc"])
            m = sum(accs) / 30.0
            var = sum((x - m) ** 2 for x in accs) / 30.0
            per_q_means.append(m)
            per_q_stds.append(math.sqrt(var))
            per_q_reward_means.append(sum(scores) / 30.0)

        mean_at_30 = sum(per_q_means) / len(per_q_means)
        std_at_30 = sum(per_q_stds) / len(per_q_stds)
        reward_mean_at_30 = sum(per_q_reward_means) / len(per_q_reward_means)
        logged_mean = metrics_step[f"val-core/{ds}/acc/mean@30"]
        logged_std = metrics_step[f"val-aux/{ds}/acc/std@30"]
        logged_reward_mean = metrics_step[f"val-aux/{ds}/reward/mean@30"]
        assert abs(mean_at_30 - logged_mean) < 1e-6, f"{ds} mean@30 mismatch: {mean_at_30} vs {logged_mean}"
        assert abs(std_at_30 - logged_std) < 1e-6, f"{ds} std@30 mismatch: {std_at_30} vs {logged_std}"
        assert abs(reward_mean_at_30 - logged_reward_mean) < 1e-6, (
            f"{ds} shaped reward/mean@30 mismatch: {reward_mean_at_30} vs {logged_reward_mean}"
        )

        out[ds] = {
            "num_questions": len(q_map),
            "samples_per_question": 30,
            "runtime_scorer": "retool_tpu.compute_score (strict_box_verify=True, USE_SHAPED_REWARD=True)",
            "total_correct_samples": f"{total_correct} / 900",
            "shaped_score_reconciliation": "1800 / 1800 exact match",
            "recomputed_acc_mean_at_30": round(mean_at_30, 6),
            "logged_acc_mean_at_30": round(logged_mean, 6),
            "recomputed_acc_std_at_30": round(std_at_30, 6),
            "logged_acc_std_at_30": round(logged_std, 6),
            "recomputed_shaped_reward_mean_at_30": round(reward_mean_at_30, 6),
            "logged_shaped_reward_mean_at_30": round(logged_reward_mean, 6),
            "logged_acc_maj_at_30": round(metrics_step[f"val-core/{ds}/acc/maj@30/mean"], 6),
            "logged_acc_best_at_30": round(metrics_step[f"val-core/{ds}/acc/best@30/mean"], 6),
        }

    out["tool_usage"] = {
        "trajectories_with_tool_response": with_tool,
        "total_trajectories": len(rows),
        "ratio": round(with_tool / len(rows), 6),
        "total_tool_responses": total_tools,
        "num_turns_mean": round(metrics_step["val-aux/num_turns/mean"], 4),
    }
    return out


def verify_env_artifacts(env_dir: Path) -> dict:
    prechecks = sorted(env_dir.glob("*-precheck.json"))
    g0s = sorted(env_dir.glob("*-g0-verify.json"))
    smis = sorted(env_dir.glob("*-nvidia-smi-L.txt"))
    assert len(prechecks) == 16, f"Expected 16 precheck JSONs, got {len(prechecks)}"
    assert len(g0s) == 16, f"Expected 16 g0-verify JSONs, got {len(g0s)}"
    assert len(smis) == 16, f"Expected 16 nvidia-smi-L files, got {len(smis)}"
    git_shas = set()
    lock_shas = set()
    ckpt_shas = set()
    total_gpus = 0
    for p in prechecks:
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d.get("status") == "PASS", f"Precheck failed in {p.name}: {d}"
        git_shas.add(d.get("git_sha"))
        lock_shas.add(d.get("uv_lock_sha256"))
    for g in g0s:
        d = json.loads(g.read_text(encoding="utf-8"))
        assert d.get("status") == "PASS", f"G0 check failed in {g.name}: {d}"
        ckpt = d.get("g0_checkpoint_manifest", {})
        assert ckpt.get("composite_sha256_match") is True, f"Checkpoint SHA mismatch in {g.name}"
        ckpt_shas.add(ckpt.get("composite_sha256"))
    for smi in smis:
        lines = [line for line in smi.read_text(encoding="utf-8").splitlines() if line.startswith("GPU ")]
        assert len(lines) == 4, f"Expected 4 GPUs in {smi.name}, got {len(lines)}"
        total_gpus += len(lines)
    assert len(git_shas) == 1, f"Mismatched git_sha across nodes: {git_shas}"
    assert len(lock_shas) == 1, f"Mismatched uv_lock_sha256 across nodes: {lock_shas}"
    assert len(ckpt_shas) == 1, f"Mismatched checkpoint composite_sha256 across nodes: {ckpt_shas}"
    assert total_gpus == 64, f"Expected 64 GPUs across 16 nodes, got {total_gpus}"
    return {
        "num_nodes_verified": 16,
        "total_gpus_verified": total_gpus,
        "git_sha": next(iter(git_shas)),
        "uv_lock_sha256": next(iter(lock_shas)),
        "sft_checkpoint_composite_sha256": next(iter(ckpt_shas)),
    }


def main() -> None:
    ev_dir = Path(__file__).resolve().parent
    metrics = json.loads((ev_dir / "g64_smoke_04_metrics.json").read_text(encoding="utf-8"))
    rollout_sum = json.loads((ev_dir / "rollout_and_audit_summary.json").read_text(encoding="utf-8"))
    env_info = verify_env_artifacts(ev_dir / "env")

    val0 = verify_validation_samples(ev_dir / "validation_step0_samples.jsonl", 0, metrics["0"])
    val5 = verify_validation_samples(ev_dir / "validation_step5_samples.jsonl", 5, metrics["5"])

    steps_2_5 = [metrics[str(i)] for i in range(2, 6)]

    summary = {
        "run_name": "g64-retool-smoke-04",
        "wandb_run_url": "https://wandb.ai/google-trellis/verl-torchtpu-grpo-test/runs/hv1k9rwo",
        "env_verification": env_info,
        "hardware": (
            "16 nodes x 4 NVIDIA GB200 GPUs = 64 GPUs on a4x-highgpu-4g-a4x-pool-5 "
            "(4 trainer nodes = 16 GPUs FSDP2=16 + 12 rollout nodes = 48 GPUs = 12 x TP=4 vLLM V1 Async Replicas)"
        ),
        "timing_summary": {
            "step_1_s": round(metrics["1"]["timing_s/step"], 2),
            "steady_state_steps_2_to_5_mean_s": round(statistics.mean(s["timing_s/step"] for s in steps_2_5), 2),
            "steady_state_steps_2_to_5_min_s": round(min(s["timing_s/step"] for s in steps_2_5), 2),
            "steady_state_steps_2_to_5_max_s": round(max(s["timing_s/step"] for s in steps_2_5), 2),
            "steady_state_tput_tok_per_s_per_gpu_mean": round(
                statistics.mean(s["perf/throughput"] for s in steps_2_5), 1
            ),
            "steady_state_gen_s_mean": round(statistics.mean(s["timing_s/gen"] for s in steps_2_5), 3),
            "steady_state_update_actor_s_mean": round(
                statistics.mean(s["timing_s/update_actor"] for s in steps_2_5), 2
            ),
        },
        "validation_step_0": val0,
        "validation_step_5": val5,
        "rollout_and_audit_totals": rollout_sum["totals"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
