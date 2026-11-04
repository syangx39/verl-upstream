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
"""Recompute Stage G64-D (64-GPU GB200) validation metrics of a run from its archived samples.

Usage: ``python3 recompute_g64_metrics.py --run g64-retool-25s-01`` (or ``--run g64-retool-smoke-04``).

Scope (what this script does and does not establish):

* Validation (Step 0 and the final step, 1,800 samples each): every sample is rescored by the runtime scorer
  ``examples/tpu/retool/retool_tpu.py::compute_score``. Its source is taken verbatim from that file (via
  ``ast``) and executed with ``verl/utils/reward_score/math_dapo.py``, so no scoring formula is re-implemented
  here; this avoids importing the module's training-time dependencies (datasets, sandbox tool, RLHFDataset).
  The scorer input is the archived ``output_tail_300`` plus the dumped ``num_turns``. This is equivalent to
  scoring the full output only because ``math_dapo.compute_score`` itself truncates ``solution_str`` to its
  last 300 characters; the script checks ``len(output_tail_300) == min(300, output_chars)`` but cannot check
  the tail content against the full text, which is not archived (only ``output_sha256`` is).
* Training (Steps 1..N, 1,024 trajectories per step): only the aggregated ``rollout_and_audit_summary.json`` is read.
  The training trajectories were NOT rescored per sample from full outputs here, and the V1 training
  rollout dump has no ``num_turns`` field, so exact per-sample shaped-score recomputation is not possible
  from that dump alone.
* Checkpoint: checks that all 16 nodes report the same staged composite SHA-256 equal to the pinned value.
  The verifier version used in this run only size-checked ``--ref-dir`` after the pinned match, so its
  ``ref_composite_sha256`` field echoes the pinned value rather than an independent hash of the reference.
"""

import argparse
import ast
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOGS_DIR.parents[4]
RETOOL_TPU_PATH = REPO_ROOT / "examples" / "tpu" / "retool" / "retool_tpu.py"
MATH_DAPO_PATH = REPO_ROOT / "verl" / "utils" / "reward_score" / "math_dapo.py"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_runtime_compute_score():
    """Return ``retool_tpu.compute_score`` executed from its verbatim source with the repo's ``math_dapo``."""
    spec = importlib.util.spec_from_file_location("math_dapo", MATH_DAPO_PATH)
    assert spec is not None and spec.loader is not None
    math_dapo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_dapo)

    src = RETOOL_TPU_PATH.read_text(encoding="utf-8")
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "compute_score"]
    assert len(nodes) == 1, f"expected exactly one compute_score in {RETOOL_TPU_PATH}"
    namespace = {"math_dapo": math_dapo, "Any": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RETOOL_TPU_PATH), "exec"), namespace)
    return namespace["compute_score"]


def verify_validation_samples(path: Path, expected_step: int, metrics_step: dict, compute_score) -> dict:
    rows = load_jsonl(path)
    assert len(rows) == 1800, f"Expected 1800 validation samples in {path.name}, got {len(rows)}"
    by_ds_q: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with_tool = 0
    total_tools = 0

    for r in rows:
        assert r["step"] == expected_step
        tail = r["output_tail_300"]
        assert len(tail) == min(300, int(r["output_chars"])), f"{r['uid']}: tail is not the last-300-char window"
        assert r.get("num_turns") is not None, f"{r['uid']}: missing num_turns"
        by_ds_q[r["data_source"]][r["question_id"]].append(r)
        if r["tool_response_count"] > 0:
            with_tool += 1
        total_tools += r["tool_response_count"]

        res = compute_score(r["data_source"], tail, str(r["gts"]), {"num_turns": int(r["num_turns"])})
        assert res["pred"] == r["pred"] and bool(res["acc"]) == bool(r["acc"]), f"{r['uid']}: {res} vs {r}"
        assert abs(float(res["score"]) - float(r["score"])) < 1e-6, (
            f"{r['uid']}: runtime scorer {res['score']} vs logged {r['score']} (num_turns={r['num_turns']})"
        )

    out: dict[str, dict] = {}
    for ds in ("aime_2024", "aime_2025"):
        q_map = by_ds_q[ds]
        assert len(q_map) == 30, f"{ds}: expected 30 questions, got {len(q_map)}"
        per_q_means, per_q_stds, per_q_reward_means = [], [], []
        total_correct = 0
        for qid, samples in sorted(q_map.items()):
            assert len(samples) == 30, f"{ds} {qid}: expected n=30 samples, got {len(samples)}"
            accs = [1.0 if s["acc"] else 0.0 for s in samples]
            total_correct += int(sum(accs))
            m = sum(accs) / 30.0
            per_q_means.append(m)
            per_q_stds.append(math.sqrt(sum((x - m) ** 2 for x in accs) / 30.0))
            per_q_reward_means.append(sum(float(s["score"]) for s in samples) / 30.0)

        mean_at_30 = sum(per_q_means) / len(per_q_means)
        std_at_30 = sum(per_q_stds) / len(per_q_stds)
        reward_mean_at_30 = sum(per_q_reward_means) / len(per_q_reward_means)
        for name, ours, key in (
            ("mean@30", mean_at_30, f"val-core/{ds}/acc/mean@30"),
            ("std@30", std_at_30, f"val-aux/{ds}/acc/std@30"),
            ("shaped reward/mean@30", reward_mean_at_30, f"val-aux/{ds}/reward/mean@30"),
        ):
            assert abs(ours - metrics_step[key]) < 1e-6, f"{ds} {name}: recomputed {ours} vs logged {metrics_step[key]}"

        out[ds] = {
            "num_questions": len(q_map),
            "samples_per_question": 30,
            "total_correct_samples": f"{total_correct} / 900",
            "rescored_samples_matching_logged_pred_acc_score": "900 / 900",
            "recomputed_acc_mean_at_30": round(mean_at_30, 6),
            "logged_acc_mean_at_30": round(metrics_step[f"val-core/{ds}/acc/mean@30"], 6),
            "recomputed_acc_std_at_30": round(std_at_30, 6),
            "recomputed_shaped_reward_mean_at_30": round(reward_mean_at_30, 6),
            "logged_shaped_reward_mean_at_30": round(metrics_step[f"val-aux/{ds}/reward/mean@30"], 6),
            "logged_acc_maj_at_30": round(metrics_step[f"val-core/{ds}/acc/maj@30/mean"], 6),
            "logged_acc_best_at_30": round(metrics_step[f"val-core/{ds}/acc/best@30/mean"], 6),
        }

    out["tool_usage"] = {
        "trajectories_with_tool_response": with_tool,
        "total_trajectories": len(rows),
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
    git_shas, lock_shas, ckpt_shas, ref_modes = set(), set(), set(), set()
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
        assert ckpt.get("composite_sha256") == ckpt.get("expected_composite_sha256"), g.name
        ckpt_shas.add(ckpt.get("composite_sha256"))
        # Reports without ref_check_mode come from the pre-fix verifier, whose pinned fast path only
        # size-checked --ref-dir and echoed the pinned value into ref_composite_sha256.
        ref_modes.add(ckpt.get("ref_check_mode", "size_only (pre-fix verifier; ref_composite_sha256 echoes pin)"))
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
        "staged_checkpoint_composite_sha256_all_nodes": next(iter(ckpt_shas)),
        "reference_dir_check_mode": sorted(ref_modes),
    }


def load_runs() -> dict:
    spec = importlib.util.spec_from_file_location("gen_g64_report_tables", LOGS_DIR / "gen_g64_report_tables.py")
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen.RUNS


def main() -> None:
    runs = load_runs()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", choices=sorted(runs), default="g64-retool-smoke-04")
    args = parser.parse_args()
    final = runs[args.run]["final_step"]
    ev_dir = LOGS_DIR / args.run
    metrics = json.loads((ev_dir / runs[args.run]["metrics"]).read_text(encoding="utf-8"))
    rollout_sum = json.loads((ev_dir / "rollout_and_audit_summary.json").read_text(encoding="utf-8"))
    env_info = verify_env_artifacts(ev_dir / "env")
    compute_score = load_runtime_compute_score()

    val0 = verify_validation_samples(ev_dir / "validation_step0_samples.jsonl", 0, metrics["0"], compute_score)
    valn = verify_validation_samples(
        ev_dir / f"validation_step{final}_samples.jsonl", final, metrics[str(final)], compute_score
    )
    window = [metrics[str(i)] for i in range(2, final + 1)]

    summary = {
        "run_name": args.run,
        "env_verification": env_info,
        "validation_scorer": {
            "function": "examples/tpu/retool/retool_tpu.py::compute_score (verbatim source, executed via ast)",
            "input": "archived output_tail_300 (math_dapo truncates to the last 300 chars) + dumped num_turns",
            "limitation": "full outputs are not archived; tail content is not checked against the full text",
        },
        f"timing_steps_2_to_{final}": {
            "note": f"timing_s/step only (post-step dump/logging work is outside it); {len(window)} steps",
            "step_1_s": round(metrics["1"]["timing_s/step"], 2),
            "mean_s": round(statistics.mean(s["timing_s/step"] for s in window), 2),
            "min_s": round(min(s["timing_s/step"] for s in window), 2),
            "max_s": round(max(s["timing_s/step"] for s in window), 2),
            "throughput_tok_per_s_per_gpu_mean": round(statistics.mean(s["perf/throughput"] for s in window), 1),
            "gen_s_mean": round(statistics.mean(s["timing_s/gen"] for s in window), 3),
            "update_actor_s_mean": round(statistics.mean(s["timing_s/update_actor"] for s in window), 2),
        },
        "validation_step_0": val0,
        f"validation_step_{final}": valn,
        "training_rollouts": {
            "source": "rollout_and_audit_summary.json (aggregates only)",
            "per_sample_rescoring": "not performed (no num_turns in the V1 training rollout dump)",
            "totals": rollout_sum["totals"],
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
