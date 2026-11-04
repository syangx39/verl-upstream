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
"""Build a G64-D evidence directory from a downloaded GB200 run directory.

Input: a local copy of ``gs://<bucket>/<RUN_ROOT>/<RUN_NAME>/`` (``head.log``, ``exit_code.txt``, ``.launch_lock.json``,
``env/``, ``validation/``, ``rollout/``, ``rollout_audit/``). Output (in ``--out-dir``):

* ``head.log``, ``exit_code.txt``, ``env/`` (per-node ``*-precheck.json``, ``*-g0-verify.json``, ``*-nvidia-smi-L.txt``,
  ``*-topo.txt``, sandbox report; launch-id-prefixed duplicates are skipped) and ``env/launch_lock.json``;
* ``<metrics-name>``: every ``TaskRunnerV1`` ``step:N`` line of ``head.log`` parsed into ``{step: {metric: value}}``;
* ``validation_step{0,N}_samples.jsonl``: compact validation samples (hashed question id, ``pred``/``acc``/``score``,
  ``num_turns``, ``output_sha256`` prefix and the last 300 output characters, the window ``math_dapo`` scores);
* ``rollout_and_audit_summary.json``: per-step training-rollout aggregates and token-mask audit counts. The audit
  is computed from each audited session's ``segments`` (runs of equal ``response_mask``): a session is a violation
  if its mask length mismatches, the mask array disagrees with a segment, a ``mask=1`` segment contains
  ``<tool_response>`` or a ``mask=0`` segment contains ``<tool_call>``.

Usage::

    python3 build_g64_run_evidence.py --src <downloaded run dir> --out-dir g64-retool-25s-01 \
        --metrics-name g64_25s_01_metrics.json --final-step 25
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

STEP_LINE_RE = re.compile(r"\bstep:(\d+) - (.*)")
NP_SCALAR_RE = re.compile(r"^np\.\w+\((.*)\)$")


def parse_step_metrics(head_log: Path) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for line in head_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_LINE_RE.search(line)
        if not m:
            continue
        kv: dict = {}
        for part in m.group(2).split(" - "):
            key, sep, value = part.partition(":")
            if not sep:
                continue
            value = NP_SCALAR_RE.sub(r"\1", value.strip())
            try:
                kv[key.strip()] = float(value)
            except ValueError:
                kv[key.strip()] = value
        metrics[m.group(1)] = kv
    return metrics


def compact_validation(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prompt, text = str(r.get("input", "")), str(r.get("output", ""))
        out.append(
            json.dumps(
                {
                    "step": int(r["step"]),
                    "data_source": r["data_source"],
                    "question_id": "sha1:" + hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16],
                    "uid": r["uid"],
                    "gts": str(r["gts"]),
                    "pred": str(r.get("pred", "")),
                    "acc": bool(r["acc"]),
                    "num_turns": int(r["num_turns"]),
                    "score": float(r["score"]),
                    "reward": float(r.get("reward", r["score"])),
                    "tool_response_count": text.count("<tool_response>"),
                    "output_chars": len(text),
                    "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                    "output_tail_300": text[-300:],
                },
                ensure_ascii=False,
            )
        )
    return out


def audit_sessions(path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    res = dict(
        audit_sessions=len(rows),
        audit_mask_length_matches=0,
        audit_sessions_with_tool_response=0,
        audit_tool_call_segments=0,
        audit_tool_response_segments=0,
        audit_mask_violations=0,
    )
    for a in rows:
        mask = a.get("response_mask", [])
        length_ok = bool(a.get("mask_length_matches")) and len(a.get("response_ids", [])) == len(mask)
        res["audit_mask_length_matches"] += length_ok
        bad = not length_ok or bool(a.get("error")) or bool(a.get("missing_fields"))
        tool_responses = 0
        for seg in a.get("segments", []):
            text = seg.get("text", "")
            if any(m != seg["mask"] for m in mask[seg["start"] : seg["end"]]):
                bad = True
            if seg["mask"] == 1:
                res["audit_tool_call_segments"] += text.count("<tool_call>")
                bad = bad or "<tool_response>" in text
            else:
                tool_responses += text.count("<tool_response>")
                bad = bad or "<tool_call>" in text
        res["audit_tool_response_segments"] += tool_responses
        res["audit_sessions_with_tool_response"] += tool_responses > 0
        res["audit_mask_violations"] += bad
    return res


def rollout_step(path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = [str(r.get("output", "")) for r in rows]
    return dict(
        num_trajectories=len(rows),
        trajectories_with_tool_response=sum("<tool_response>" in o for o in outputs),
        total_tool_responses=sum(o.count("<tool_response>") for o in outputs),
        clip_neg2_count=sum(float(r.get("score", 0.0)) <= -1.99 for r in rows),
    )


def build(src: Path, out_dir: Path, metrics_name: str, final_step: int) -> dict:
    env_out = out_dir / "env"
    env_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "exit_code.txt", out_dir / "exit_code.txt")
    shutil.copy2(src / "head.log", out_dir / "head.log")
    shutil.copy2(src / ".launch_lock.json", env_out / "launch_lock.json")
    launch_prefix = json.loads((src / ".launch_lock.json").read_text(encoding="utf-8"))["launch_id"] + "-"
    for p in sorted((src / "env").iterdir()):
        if p.suffix in (".json", ".txt") and not p.name.startswith(launch_prefix):
            shutil.copy2(p, env_out / p.name)

    metrics = parse_step_metrics(src / "head.log")
    (out_dir / metrics_name).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for step in (0, final_step):
        lines = compact_validation(src / "validation" / f"{step}.jsonl")
        (out_dir / f"validation_step{step}_samples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    steps: dict[str, dict] = {}
    totals: dict[str, int] = {}
    total_keys = {
        "num_trajectories": "total_trajectories",
        "trajectories_with_tool_response": "trajectories_with_tool_response",
        "total_tool_responses": "total_tool_responses",
        "clip_neg2_count": "total_clip_neg2_count",
        "audit_sessions": "total_audit_sessions",
    }
    for s in range(1, final_step + 1):
        row = {
            "step": s,
            **rollout_step(src / "rollout" / f"{s}.jsonl"),
            **audit_sessions(src / "rollout_audit" / f"{s}.jsonl"),
        }
        steps[str(s)] = row
        for key, value in row.items():
            if key != "step":
                tk = total_keys.get(key, key)
                totals[tk] = totals.get(tk, 0) + value
    totals["trajectories_with_tool_response_ratio"] = round(
        totals["trajectories_with_tool_response"] / totals["total_trajectories"], 6
    )
    summary = {"steps": steps, "totals": totals}
    (out_dir / "rollout_and_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="downloaded GCS run directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="evidence directory to write")
    parser.add_argument("--metrics-name", required=True, help="file name for the parsed step metrics JSON")
    parser.add_argument("--final-step", type=int, required=True, help="last training step (validated at the end)")
    args = parser.parse_args()
    summary = build(args.src, args.out_dir, args.metrics_name, args.final_step)
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
