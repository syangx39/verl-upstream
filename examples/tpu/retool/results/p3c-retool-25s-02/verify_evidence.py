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
"""Verify the archived evidence of a TPU v7x NAP ReTool run (no TPU, no network).

Usage: ``python3 verify_evidence.py [--final-step 25]`` from anywhere; reads the files next to this script.

Checks (any failure raises ``AssertionError``):

* ``metrics.json`` equals a fresh parse of ``head_taskrunner.log``;
* every validation sample (Step 0 and the final step, ``30 x 30`` per dataset) is rescored by the runtime scorer
  ``examples/tpu/retool/retool_tpu.py::compute_score``, taken verbatim from that file (via ``ast``) and executed with
  ``verl/utils/reward_score/math_dapo.py``; ``pred``/``acc``/``score`` must match the dump. The scorer input is the
  archived last-300-character window, which is equivalent to the full output only because ``math_dapo`` truncates
  to the last 300 characters itself (full outputs are not archived);
* recomputed ``acc/mean@30`` and shaped ``reward/mean@30`` equal the logged ``val-core`` / ``val-aux`` values;
* ``eval_summary.json`` equals a fresh recomputation (counts and paired bootstrap interval);
* the Raiden distributed norm check passed for all three rollout replicas at every training step;
* the token-mask audit found no violation.
"""

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
RETOOL_TPU_PATH = REPO_ROOT / "examples" / "tpu" / "retool" / "retool_tpu.py"
MATH_DAPO_PATH = REPO_ROOT / "verl" / "utils" / "reward_score" / "math_dapo.py"

spec = importlib.util.spec_from_file_location("build_evidence", HERE / "build_evidence.py")
assert spec is not None and spec.loader is not None
build_evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_evidence)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_runtime_compute_score():
    """Return ``retool_tpu.compute_score`` executed from its verbatim source with the repo's ``math_dapo``."""
    mspec = importlib.util.spec_from_file_location("math_dapo", MATH_DAPO_PATH)
    assert mspec is not None and mspec.loader is not None
    math_dapo = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(math_dapo)
    src = RETOOL_TPU_PATH.read_text(encoding="utf-8")
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "compute_score"]
    assert len(nodes) == 1, f"expected exactly one compute_score in {RETOOL_TPU_PATH}"
    namespace = {"math_dapo": math_dapo, "Any": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RETOOL_TPU_PATH), "exec"), namespace)
    return namespace["compute_score"]


def verify_validation(rows: list[dict], step: int, logged: dict, compute_score) -> dict:
    assert len(rows) == 1800, f"step {step}: expected 1800 samples, got {len(rows)}"
    for r in rows:
        assert r["step"] == step
        tail = r["output_tail_300"]
        assert len(tail) == min(300, int(r["output_chars"])), f"{r['uid']}: tail is not the last-300-char window"
        res = compute_score(r["data_source"], tail, r["gts"], {"num_turns": int(r["num_turns"])})
        assert res["pred"] == r["pred"] and bool(res["acc"]) == bool(r["acc"]), f"{r['uid']}: {res} vs {r}"
        assert abs(float(res["score"]) - float(r["score"])) < 1e-6, f"{r['uid']}: {res['score']} vs {r['score']}"
    out = {}
    for ds, q_map in build_evidence.group_by_question(rows).items():
        assert len(q_map) == 30 and all(len(v) == 30 for v in q_map.values()), f"{ds}: expected 30 x 30 samples"
        mean = sum(sum(s["acc"] for s in v) / 30 for v in q_map.values()) / 30
        reward = sum(sum(s["score"] for s in v) / 30 for v in q_map.values()) / 30
        for name, ours, key in (
            ("acc/mean@30", mean, f"val-core/{ds}/acc/mean@30"),
            ("reward/mean@30", reward, f"val-aux/{ds}/reward/mean@30"),
        ):
            assert abs(ours - logged[key]) < 1e-6, f"step {step} {ds} {name}: recomputed {ours} vs logged {logged[key]}"
        out[ds] = {"correct": sum(s["acc"] for v in q_map.values() for s in v), "acc_mean_at_30": round(mean, 6)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final-step", type=int, default=25)
    args = parser.parse_args()
    final = args.final_step

    lines = (HERE / "head_taskrunner.log").read_text(encoding="utf-8").splitlines()
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    assert build_evidence.parse_step_metrics(lines) == metrics, "metrics.json differs from head_taskrunner.log"
    assert all(str(s) in metrics for s in range(0, final + 1)), "missing step metrics"

    compute_score = load_runtime_compute_score()
    v0 = load_jsonl(HERE / "validation_step0_samples.jsonl")
    vn = load_jsonl(HERE / f"validation_step{final}_samples.jsonl")
    val = {0: verify_validation(v0, 0, metrics["0"], compute_score)}
    val[final] = verify_validation(vn, final, metrics[str(final)], compute_score)

    ev = json.loads((HERE / "eval_summary.json").read_text(encoding="utf-8"))
    fresh = build_evidence.eval_summary(v0, vn, metrics["0"], metrics[str(final)], final)
    assert json.loads(json.dumps(fresh)) == ev, "eval_summary.json differs from a fresh recomputation"

    norms = json.loads((HERE / "raiden_norm_checks.json").read_text(encoding="utf-8"))
    assert all(norms.get(str(s)) == [0, 1, 2] for s in range(1, final + 1)), f"norm checks incomplete: {norms}"
    rollout = json.loads((HERE / "rollout_and_audit_summary.json").read_text(encoding="utf-8"))
    assert rollout["totals"]["audit_mask_violations"] == 0, "token-mask audit violations found"

    print(
        json.dumps(
            {
                "metrics_steps": len(metrics),
                "validation": val,
                "eval_summary": "matches fresh recomputation",
                "raiden_norm_checks": f"steps 1-{final} x replicas 0,1,2 passed",
                "rollout_totals": rollout["totals"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
