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
* every raw file listed in ``raw_manifest.json`` (the audit sessions, gunzipped, and the TensorBoard scalars) has the
  recorded size, MD5 (the GCS object ``md5Hash``) and SHA-256;
* the whole-session token-mask audit (``examples/tpu/retool/rollout_mask_audit.py``) recomputed from the raw
  ``rollout_audit/<step>.jsonl.gz`` sessions equals ``rollout_and_audit_summary.json`` and finds no violation;
* ``step_timing_breakdown.json`` equals a fresh recomputation from ``metrics.json``, the TensorBoard scalars and
  ``head_taskrunner.log``;
* the Step 0 Raiden element-wise check (``env/raiden_exact_parity_step_0.json``) covered 3 x 32 ranks with no
  mismatch, and the Raiden distributed norm check passed for all three rollout replicas at every training step
  (a norm-level check, not an element-wise comparison);
* ``env/code_snapshot.json`` names the commit of ``env/run_record.json``.
"""

import argparse
import ast
import base64
import gzip
import hashlib
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


def verify_raw_manifest() -> int:
    """Every raw file has the size / MD5 / SHA-256 recorded when it was archived from GCS."""
    manifest = json.loads((HERE / "raw_manifest.json").read_text(encoding="utf-8"))
    for f in manifest["files"]:
        data = (HERE / f["path"]).read_bytes()
        if f["path"].endswith(".gz"):
            data = gzip.decompress(data)
        assert len(data) == f["bytes"], f"{f['path']}: {len(data)} bytes, manifest {f['bytes']}"
        assert base64.b64encode(hashlib.md5(data).digest()).decode() == f["md5_base64"], f"{f['path']}: md5 differs"
        assert hashlib.sha256(data).hexdigest() == f["sha256"], f"{f['path']}: sha256 differs"
    return len(manifest["files"])


def verify_mask_audit(final: int) -> dict:
    """Recompute the whole-session audit from the raw sessions and compare with the archived summary."""
    rollout = json.loads((HERE / "rollout_and_audit_summary.json").read_text(encoding="utf-8"))
    fresh = build_evidence.audit_steps(HERE / "rollout_audit", final, ".jsonl.gz")
    for s, audit in fresh.items():
        archived = {k: rollout["steps"][s][k] for k in audit}
        assert archived == audit, f"step {s}: archived audit {archived} != recomputed {audit}"
    t = rollout["totals"]
    for key in build_evidence.mask_audit.COUNT_KEYS:
        assert t[key] == sum(a[key] for a in fresh.values()), f"totals[{key}] differs from the steps"
    assert t["audit_sessions"] == 8 * final, f"expected {8 * final} audited sessions, got {t['audit_sessions']}"
    for key in (
        "audit_mask_violations",
        "audit_tool_response_tokens_in_loss",
        "audit_tool_call_tokens_not_in_loss",
        "audit_mask0_runs_not_tool_turn",
        "audit_im_start_in_loss",
    ):
        assert t[key] == 0, f"token-mask audit: {key} = {t[key]} ({t['violation_reasons']})"
    assert t["audit_mask_length_matches"] == t["audit_sessions"] and t["violation_reasons"] == {}
    return t


def verify_raiden(final: int) -> dict:
    exact = json.loads((HERE / "env" / "raiden_exact_parity_step_0.json").read_text(encoding="utf-8"))
    ranks = exact["ranks"]
    assert exact["num_results"] == exact["expected_ranks"] == len(ranks) == 96, "Step 0 exact parity: rank count"
    assert sorted((r["replica"], r["rank"]) for r in ranks) == [(p, k) for p in range(3) for k in range(32)]
    assert exact["num_mismatched"] == 0 and exact["num_unresolved"] == 0 and not exact["coverage_errors"]
    assert all(r["checked"] == r["expected_checked"] == 435 and r["num_mismatched"] == 0 for r in ranks)
    norms = json.loads((HERE / "raiden_norm_checks.json").read_text(encoding="utf-8"))
    assert all(norms.get(str(s)) == [0, 1, 2] for s in range(1, final + 1)), f"norm checks incomplete: {norms}"
    return {"step0_exact_ranks": len(ranks), "step0_exact_tensors": sum(r["checked"] for r in ranks)}


def verify_code_snapshot() -> str:
    snap = json.loads((HERE / "env" / "code_snapshot.json").read_text(encoding="utf-8"))
    record = json.loads((HERE / "env" / "run_record.json").read_text(encoding="utf-8"))
    sha = record["code_sha"]
    assert snap["code_sha"] == sha == snap["gcs_tarball"]["pax_comment"], "code snapshot names another commit"
    assert snap["github"]["commit_url"].endswith(sha) and sha[:12] in snap["gcs_tarball"]["uri"]
    return sha


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

    raw_files = verify_raw_manifest()
    audit = verify_mask_audit(final)
    timing = json.loads((HERE / "step_timing_breakdown.json").read_text(encoding="utf-8"))
    fresh_timing = build_evidence.timing_breakdown(HERE, metrics, final)
    assert json.loads(json.dumps(fresh_timing)) == timing, "step_timing_breakdown.json differs from a recomputation"
    raiden = verify_raiden(final)
    code_sha = verify_code_snapshot()

    print(
        json.dumps(
            {
                "metrics_steps": len(metrics),
                "validation": val,
                "eval_summary": "matches fresh recomputation",
                "raw_files_matching_manifest": raw_files,
                "mask_audit_from_raw_sessions": audit,
                "timing_summary": timing["summary"],
                "raiden": {**raiden, "norm_checks": f"steps 1-{final} x replicas 0,1,2 passed"},
                "code_sha": code_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
