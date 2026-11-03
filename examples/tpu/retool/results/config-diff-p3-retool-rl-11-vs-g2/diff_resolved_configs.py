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
"""Diff the resolved verl configs printed in two run logs and classify each differing key.

The resolved config is the ``pprint`` dict that ``main_ppo`` logs at start-up (the block that begins with
``{'actor_rollout_ref'``). Both configs are flattened to dotted keys; every key whose value differs, or that only one
side has, is written as one TSV row with a category:

  paths/naming     run directories, model staging paths, experiment and project names
  platform         device counts, parallel layout, compile and attention settings, rollout engine and sync backend
  token-budget     dynamic-batch token budgets and the TorchTitan sequence length
  schema           a key that only one side's verl version has
  review           anything else (to be checked by hand)
"""

import argparse
import ast
import csv
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PREFIX = re.compile(r"^\(\w+ pid=\d+(?:, ip=[\d.]+)?\)\s?")
ABSENT = "<absent>"
RULES = (
    (
        "paths/naming",
        r"(_dir|\.path|experiment_name|project_name|served_model_name|PYTHONPATH|train_files|val_files|"
        r"RAIDEN_PARITY_DUMP_DIR)$",
    ),
    ("token-budget", r"(max_token_len_per_gpu|torchtitan\.max_seq_len)$"),
    (
        "platform",
        r"(nnodes|n_gpus_per_node|trainer\.device|torchtitan\.(attn_type|activation_checkpoint|"
        r"context_parallel_size|data_parallel_shard_size|spmd_backend|use_torch_compile|use_simple_fsdp|"
        r"use_splash_attention|tpu_eager_mode|fp32_logits|mixed_precision_param|pad_to_length\w*)|\.use_torch_compile|"
        r"rollout\.(tensor_model_parallel_size|expert_parallel_size|gpu_memory_utilization|max_num_batched_tokens|"
        r"layered_summon|engine_kwargs\.\S+|checkpoint_engine\.\S+|tpu_slices|ray_actor_max_concurrency|"
        r"standalone_gpu_memory_utilization)|^transfer_queue\.|VLLM_USE_V1|RAIDEN_PARITY_EXPECTED_CHECKED_PER_RANK)",
    ),
)


def resolved_config(path: str) -> dict:
    lines = [ANSI.sub("", line.rstrip("\n")) for line in open(path, errors="replace")]
    for start, line in enumerate(lines):
        if PREFIX.sub("", line).startswith("{'actor_rollout_ref'"):
            block, depth = [], 0
            for raw in lines[start:]:
                if not PREFIX.match(raw):
                    continue  # interleaved output of other processes
                text = PREFIX.sub("", raw)
                block.append(text)
                depth += text.count("{") - text.count("}")
                if depth == 0:
                    return ast.literal_eval("\n".join(block))
    raise SystemExit(f"no resolved config found in {path}")


def flatten(value, prefix: str = "") -> dict:
    if not isinstance(value, dict):
        return {prefix[:-1]: value}
    out = {}
    for key, item in value.items():
        out.update(flatten(item, f"{prefix}{key}."))
    return out


def category(key: str, left, right) -> str:
    for name, pattern in RULES:
        if re.search(pattern, key):
            return name
    return "schema" if ABSENT in (left, right) else "review"


def _fmt(value) -> str:
    return value if isinstance(value, str) else repr(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left_log")
    parser.add_argument("right_log")
    parser.add_argument("--left-name", default="tpu")
    parser.add_argument("--right-name", default="gpu")
    args = parser.parse_args()
    left, right = flatten(resolved_config(args.left_log)), flatten(resolved_config(args.right_log))
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(["key", args.left_name, args.right_name, "category"])
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key, ABSENT), right.get(key, ABSENT)
        if a != b:
            writer.writerow([key, _fmt(a), _fmt(b), category(key, a, b)])


if __name__ == "__main__":
    main()
