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
"""Fetch Askhat's TPU v5p ReTool runs from W&B into a compact JSON (``askhat_v5p_runs.json``).

Selects every run in ``google-trellis/verl-torchtpu-grpo-test`` whose display name contains ``v5p`` and keeps only
the config fields and per-step metrics used by ``plot_askhat_v5p.py``. Needs ``WANDB_API_KEY`` in the environment;
the key is never written anywhere.

Usage::

    WANDB_API_KEY=... python3 fetch_askhat_v5p.py --out askhat_v5p_runs.json
"""

import argparse
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

ENTITY, PROJECT = "google-trellis", "verl-torchtpu-grpo-test"
CONFIG_PATHS = {
    "enable_thinking": ("data", "apply_chat_template_kwargs", "enable_thinking"),
    "max_prompt_length": ("data", "max_prompt_length"),
    "max_response_length": ("data", "max_response_length"),
    "train_batch_size": ("data", "train_batch_size"),
    "rollout_n": ("actor_rollout_ref", "rollout", "n"),
    "val_n": ("actor_rollout_ref", "rollout", "val_kwargs", "n"),
    "max_user_turns": ("actor_rollout_ref", "rollout", "multi_turn", "max_user_turns"),
    "max_assistant_turns": ("actor_rollout_ref", "rollout", "multi_turn", "max_assistant_turns"),
    "rollout_tp": ("actor_rollout_ref", "rollout", "tensor_model_parallel_size"),
    "rollout_ep": ("actor_rollout_ref", "rollout", "expert_parallel_size"),
    "actor_lr": ("actor_rollout_ref", "actor", "optim", "lr"),
    "clip_grad": ("actor_rollout_ref", "actor", "optim", "clip_grad"),
    "ppo_mini_batch_size": ("actor_rollout_ref", "actor", "ppo_mini_batch_size"),
    "loss_mode": ("actor_rollout_ref", "actor", "policy_loss", "loss_mode"),
    "bypass_mode": ("algorithm", "rollout_correction", "bypass_mode"),
    "trainer_nnodes": ("trainer", "nnodes"),
    "trainer_n_gpus_per_node": ("trainer", "n_gpus_per_node"),
    "total_training_steps": ("trainer", "total_training_steps"),
    "test_freq": ("trainer", "test_freq"),
    "custom_reward_path": ("reward", "custom_reward_function", "path"),
    "model_path": ("actor_rollout_ref", "model", "path"),
    "trainer_cp": ("actor_rollout_ref", "actor", "torchtitan", "context_parallel_size"),
    "trainer_dp_shard": ("actor_rollout_ref", "actor", "torchtitan", "data_parallel_shard_size"),
    "ppo_max_token_len_per_gpu": ("actor_rollout_ref", "actor", "ppo_max_token_len_per_gpu"),
    "rollout_dp": ("actor_rollout_ref", "rollout", "data_parallel_size"),
    "rollout_nnodes": ("actor_rollout_ref", "rollout", "nnodes"),
    "rollout_n_gpus_per_node": ("actor_rollout_ref", "rollout", "n_gpus_per_node"),
    "rollout_max_num_seqs": ("actor_rollout_ref", "rollout", "max_num_seqs"),
    "rollout_temperature": ("actor_rollout_ref", "rollout", "temperature"),
    "val_temperature": ("actor_rollout_ref", "rollout", "val_kwargs", "temperature"),
    "val_top_p": ("actor_rollout_ref", "rollout", "val_kwargs", "top_p"),
    "train_files": ("data", "train_files"),
    "val_files": ("data", "val_files"),
    "val_before_train": ("trainer", "val_before_train"),
    "resume_mode": ("trainer", "resume_mode"),
    "resume_from_path": ("trainer", "resume_from_path"),
    "default_local_dir": ("trainer", "default_local_dir"),
    # The Ray working_dir package hash in PYTHONPATH changes whenever the uploaded code changes.
    "ray_pythonpath": ("ray_kwargs", "ray_init", "runtime_env", "env_vars", "PYTHONPATH"),
    "ckpt_async_save": ("actor_rollout_ref", "actor", "checkpoint", "async_save"),
    "ckpt_save_contents": ("actor_rollout_ref", "actor", "checkpoint", "save_contents"),
    "ckpt_load_contents": ("actor_rollout_ref", "actor", "checkpoint", "load_contents"),
    "max_actor_ckpt_to_keep": ("trainer", "max_actor_ckpt_to_keep"),
    "use_v1": ("trainer", "use_v1"),
    "num_warmup_batches": ("trainer", "v1", "separate_async", "num_warmup_batches"),
    "max_off_policy_threshold": ("trainer", "v1", "sampler", "max_off_policy_threshold"),
    "rollout_is": ("algorithm", "rollout_correction", "rollout_is"),
    "save_freq": ("trainer", "save_freq"),
    "val_only": ("trainer", "val_only"),
    "sft_lr": ("optim", "lr"),
    "sft_warmup_ratio": ("optim", "lr_warmup_steps_ratio"),
    "sft_epochs": ("trainer", "total_epochs"),
    "sft_train_batch_size": ("data", "train_batch_size"),
    "sft_max_length": ("data", "max_length"),
    "sft_model": ("model", "path"),
}
METRIC_PREFIXES = ("val-core/", "val-aux/aime")
METRICS = (
    "critic/score/mean",
    "response_length/mean",
    "response_length/clip_ratio",
    "num_turns/mean",
    "actor/grad_norm",
    "actor/lr",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/testing",
    "timing_s/save_checkpoint",
    "perf/throughput",
    "train/loss",
    "train/lr",
    "val/loss",
)


def gql(key: str, query: str, variables: dict) -> dict:
    auth = base64.b64encode(f"api:{key}".encode()).decode()
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode())
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


def list_runs(key: str) -> list[dict]:
    query = """query($entity: String!, $project: String!, $cursor: String) {
      project(entityName: $entity, name: $project) {
        runs(first: 100, after: $cursor) {
          pageInfo { hasNextPage endCursor }
          edges { node { name displayName state createdAt heartbeatAt config } }
        }
      }
    }"""
    runs, cursor = [], None
    while True:
        page = gql(key, query, {"entity": ENTITY, "project": PROJECT, "cursor": cursor})["project"]["runs"]
        runs += [e["node"] for e in page["edges"]]
        if not page["pageInfo"]["hasNextPage"]:
            return runs
        cursor = page["pageInfo"]["endCursor"]


def history(key: str, name: str) -> list[dict]:
    query = """query($entity: String!, $project: String!, $name: String!) {
      project(entityName: $entity, name: $project) { run(name: $name) { history(samples: 2000) } }
    }"""
    rows = gql(key, query, {"entity": ENTITY, "project": PROJECT, "name": name})["project"]["run"]["history"] or []
    return [json.loads(r) for r in rows]


def config_value(cfg: dict, path: tuple[str, ...]):
    node = cfg
    for i, part in enumerate(path):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
        if i == 0 and isinstance(node, dict) and "value" in node and len(path) > 1:
            node = node["value"]
    return node.get("value") if isinstance(node, dict) and set(node) == {"value"} else node


def compact(row: dict) -> dict:
    out = {k: row[k] for k in ("_step", "training/global_step", "_timestamp") if k in row}
    for k, v in row.items():
        if (k in METRICS or k.startswith(METRIC_PREFIXES)) and isinstance(v, int | float):
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ["WANDB_API_KEY"]
    runs = [r for r in list_runs(key) if "v5p" in (r.get("displayName") or "")]
    result = {
        "wandb_project": f"{ENTITY}/{PROJECT}",
        "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [],
    }
    for r in sorted(runs, key=lambda x: x["createdAt"]):
        cfg = json.loads(r["config"]) if isinstance(r["config"], str) else (r["config"] or {})
        rows = [compact(x) for x in history(key, r["name"])]
        rows = [x for x in rows if len(x) > 3]
        result["runs"].append(
            {
                "run_id": r["name"],
                "display_name": r["displayName"],
                "state": r["state"],
                "created_at": r["createdAt"],
                "heartbeat_at": r["heartbeatAt"],
                "config": {k: config_value(cfg, p) for k, p in CONFIG_PATHS.items()},
                "history": rows,
            }
        )
        print(f"{r['name']} {r['displayName']} {r['state']} rows={len(rows)}")
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
