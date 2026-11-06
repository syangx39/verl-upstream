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
"""Fetch the data behind ``REPRO_GUIDE.md`` into ``data/`` (W&B + GCS).

Needs ``WANDB_API_KEY`` with read access to ``google-trellis/verl-torchtpu-grpo-test`` (the key is never written
anywhere) and ``gcloud`` with read access to ``gs://ubench-logs``. Writes:

* ``data/wandb_runs.json``: config subset and per-step metrics (steps 0–25) of every run in ``make_report.RUNS``,
  plus the evaluation of the base model without SFT;
* ``data/eval_per_question.json``: correct / total responses per problem at Step 0 and Step 25 for the TPU v7x and
  GB200 runs, from their validation dumps (Askhat's runs only logged aggregates to W&B);
* ``data/train_prompts.json``: the training prompts used at each step (first 16 hex digits of the SHA-1 of the
  rendered prompt) for the TPU v7x and GB200 runs, from their rollout dumps;
* ``data/sft_loss.json``: SFT loss / LR / tokens per step (TPU v7x: driver log on GCS; TPU v5p: W&B);
* ``data/tb_log_windows.json``: per step, the wall time of the first TensorBoard scalar and the time spent writing
  all scalars of the step (``../step_timing_breakdown.py``) for the TPU v7x and GB200 runs (Askhat's runs had no
  TensorBoard logger).

Usage::

    WANDB_API_KEY=... python3 fetch_runs.py
"""

import argparse
import base64
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import step_timing_breakdown  # noqa: E402
from make_report import (  # noqa: E402
    BASE_EVAL_RUN,
    CONFIG_KEYS,
    DATA,
    MAX_STEP,
    METRICS,
    RUNS,
    SFT_V5P_RUN,
    VAL_PREFIX,
    WANDB_PROJECT,
)

GCS_RUN_DIRS = {
    "v7x": "gs://ubench-logs/xiaotongyang/retool-v7x/runs/p3c-retool-25s-02",
    "gb200": "gs://ubench-logs/xiaotongyang/retool-gb200/runs/g64-retool-25s-01",
}
SFT_V7X_DRIVER_LOG = "gs://ubench-logs/xiaotongyang/retool-v7x/runs/p2-retool-sft-01/attempts/b9ebe40d/logs/driver.log"
STEP_LINE = re.compile(r"\bstep:(\d+) - (.*)$")


def gql(key: str, query: str, variables: dict) -> dict:
    auth = base64.b64encode(f"api:{key}".encode()).decode()
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read().decode())
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


def wandb_run(key: str, run_id: str) -> dict:
    entity, project = WANDB_PROJECT.split("/")
    query = """query($e: String!, $p: String!, $r: String!) {
      project(entityName: $e, name: $p) {
        run(name: $r) { name displayName state createdAt config history(samples: 5000) }
      }
    }"""
    run = gql(key, query, {"e": entity, "p": project, "r": run_id})["project"]["run"]
    return {
        "display_name": run["displayName"],
        "state": run["state"],
        "created_at": run["createdAt"],
        "config": json.loads(run["config"]),
        "history": [json.loads(r) for r in run["history"] or []],
    }


def flatten(node: object, prefix: str = "") -> dict:
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = node
    return out


def flat_config(raw: dict) -> dict:
    cfg = {}
    for top, entry in raw.items():
        if top.startswith("_"):
            continue
        cfg.update(flatten(entry.get("value") if isinstance(entry, dict) and "value" in entry else entry, f"{top}."))
    return cfg


def compact_row(row: dict) -> dict:
    out = {"step": int(row["_step"]), "_timestamp": row["_timestamp"]}
    for k, v in row.items():
        if (k in METRICS or (k.startswith(VAL_PREFIX) and "/acc/mean@" in k)) and isinstance(v, int | float):
            out[k] = v
    return out


def gcs_lines(uri: str) -> Iterator[str]:
    proc = subprocess.Popen(["gcloud", "storage", "cat", uri], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    yield from proc.stdout
    if proc.wait() != 0:
        raise RuntimeError(f"gcloud storage cat {uri} failed")


def fetch_wandb(key: str) -> dict:
    out = {"wandb_project": WANDB_PROJECT, "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "runs": {}}
    run_ids = [r for spec in RUNS for r, _, _ in spec["segments"]] + [BASE_EVAL_RUN]
    for run_id in run_ids:
        run = wandb_run(key, run_id)
        cfg = flat_config(run["config"])
        rows = [compact_row(r) for r in run["history"] if "_step" in r]
        out["runs"][run_id] = {
            "display_name": run["display_name"],
            "state": run["state"],
            "created_at": run["created_at"],
            "config": {k: cfg[k] for k in CONFIG_KEYS if k in cfg},
            "history": sorted((r for r in rows if r["step"] <= MAX_STEP and len(r) > 2), key=lambda r: r["step"]),
        }
        print(f"{run_id} {run['display_name']} {run['state']}: {len(out['runs'][run_id]['history'])} rows")
    return out


def fetch_eval_per_question() -> dict:
    out = {}
    for key, run_dir in GCS_RUN_DIRS.items():
        steps = {}
        for step in (0, MAX_STEP):
            counts = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
            for line in gcs_lines(f"{run_dir}/validation/{step}.jsonl"):
                r = json.loads(line)
                c = counts[r["data_source"]][r["question_id"]]
                c[0] += int(bool(r["acc"]))
                c[1] += 1
            steps[str(step)] = {ds: dict(sorted(q.items())) for ds, q in sorted(counts.items())}
        out[key] = {"source": f"{run_dir}/validation/{{0,{MAX_STEP}}}.jsonl", "steps": steps}
        print(f"{key}: per-question validation counts for steps 0 and {MAX_STEP}")
    return out


def fetch_train_prompts() -> dict:
    out = {}
    for key, run_dir in GCS_RUN_DIRS.items():
        steps = {}
        for step in range(1, MAX_STEP + 1):
            ids = {
                hashlib.sha1(json.loads(line)["input"].encode()).hexdigest()[:16]
                for line in gcs_lines(f"{run_dir}/rollout/{step}.jsonl")
            }
            steps[str(step)] = sorted(ids)
        out[key] = {"source": f"{run_dir}/rollout/<step>.jsonl", "id": "sha1(input)[:16]", "steps": steps}
        print(f"{key}: training prompt ids for steps 1-{MAX_STEP}")
    return out


def parse_step_line(rest: str) -> dict:
    kv = {}
    for part in rest.split(" - "):
        k, sep, v = part.rpartition(":")
        if sep:
            try:
                kv[k.strip()] = float(v)
            except ValueError:
                pass
    return kv


def fetch_sft(key: str) -> dict:
    v7x = collections.defaultdict(dict)
    for line in gcs_lines(SFT_V7X_DRIVER_LOG):
        m = STEP_LINE.search(line)
        if m:
            v7x[int(m.group(1))].update(parse_step_line(m.group(2)))
    run = wandb_run(key, SFT_V5P_RUN)
    v5p = collections.defaultdict(dict)
    for r in run["history"]:
        v5p[int(r["_step"])].update({k: v for k, v in r.items() if isinstance(v, int | float)})

    def pack(steps: dict, source: str) -> dict:
        train = [
            [s, d["train/loss"], d["train/lr"], d["train/global_tokens"]]
            for s, d in sorted(steps.items())
            if "train/loss" in d
        ]
        val = [[s, d["val/loss"]] for s, d in sorted(steps.items()) if "val/loss" in d]
        return {
            "source": source,
            "columns": ["step", "train/loss", "train/lr", "train/global_tokens"],
            "train": train,
            "val": val,
        }

    print("sft: v7x driver log and v5p W&B history")
    return {"v7x": pack(v7x, SFT_V7X_DRIVER_LOG), "v5p": pack(v5p, f"wandb:{WANDB_PROJECT}/{SFT_V5P_RUN}")}


def fetch_tb_windows() -> dict:
    out = {}
    for key, run_dir in GCS_RUN_DIRS.items():
        listing = subprocess.run(
            ["gcloud", "storage", "ls", f"{run_dir}/tensorboard/"], capture_output=True, text=True, check=True
        ).stdout.split()
        best = None
        with tempfile.TemporaryDirectory() as tmp:
            for uri in (u for u in listing if "/events.out.tfevents." in u):
                local = Path(tmp) / uri.rsplit("/", 1)[1]
                subprocess.run(["gcloud", "storage", "cp", uri, str(local)], capture_output=True, check=True)
                events = step_timing_breakdown.read_scalar_events(local)
                if best is None or len(events) > len(best[1]):
                    best = (uri, events)
        assert best is not None, f"no TensorBoard event file under {run_dir}/tensorboard/"
        windows = step_timing_breakdown.tb_log_windows(best[1])
        out[key] = {"source": best[0], "steps": {str(s): w for s, w in sorted(windows.items())}}
        print(f"{key}: TensorBoard logging windows from {best[0]}")
    return out


def write(name: str, obj: dict) -> None:
    DATA.mkdir(exist_ok=True)
    (DATA / name).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["wandb", "eval", "prompts", "sft", "tb"], action="append")
    args = parser.parse_args()
    only = set(args.only or ["wandb", "eval", "prompts", "sft", "tb"])
    key = os.environ.get("WANDB_API_KEY", "")
    if only & {"wandb", "sft"} and not key:
        sys.exit("WANDB_API_KEY is not set")
    if "wandb" in only:
        write("wandb_runs.json", fetch_wandb(key))
    if "eval" in only:
        write("eval_per_question.json", fetch_eval_per_question())
    if "prompts" in only:
        write("train_prompts.json", fetch_train_prompts())
    if "sft" in only:
        write("sft_loss.json", fetch_sft(key))
    if "tb" in only:
        write("tb_log_windows.json", fetch_tb_windows())


if __name__ == "__main__":
    main()
