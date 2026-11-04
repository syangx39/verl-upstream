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
"""Show that the logged ``actor/rollout_corr/rollout_is_*`` values are sequence-level statistics.

Runs the repository's own ``compute_rollout_correction_and_rejection_mask`` (source taken verbatim from
``verl/trainer/ppo/rollout_corr_helper.py``; ``masked_sum``/``masked_mean`` verbatim from
``verl/utils/torch_functional.py``) on synthetic 16K-thinking-like data with a token-level trainer/rollout
KL of about 0.01 nats/token, once with ``rollout_is="token"`` (the trainer-side config in head.log) and once
with ``rollout_is="sequence"`` (the ``RolloutCorrectionConfig`` dataclass default that the V1 actor's bypass
loss falls back to because the V1 trainer never copies ``algorithm.rollout_correction`` into
``actor.policy_loss.rollout_correction``).

Requires ``torch`` (CPU is enough): ``python3 repro_rollout_is_level.py``
"""

import ast
import math
import types
import typing
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[6]
HELPER_FUNCS = {
    "_parse_rollout_is_threshold",
    "compute_rollout_correction_weights",
    "compute_is_metrics",
    "compute_offpolicy_metrics",
    "compute_rollout_correction_and_rejection_mask",
}


def _extract(path: Path, names: set[str]) -> str:
    src = path.read_text(encoding="utf-8")
    parts = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            parts.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SAFETY_BOUND" for t in node.targets):
            parts.append(ast.get_source_segment(src, node))
    return "\n\n".join(parts)


class _DataProto:
    def __init__(self, tensors):
        self.batch = tensors

    @classmethod
    def from_dict(cls, tensors):
        return cls(tensors)


def load_helper():
    verl_f = types.ModuleType("verl_F")
    exec(
        "import torch\n" + _extract(REPO_ROOT / "verl/utils/torch_functional.py", {"masked_sum", "masked_mean"}),
        verl_f.__dict__,
    )
    ns = {
        "torch": torch,
        "verl_F": verl_f,
        "DataProto": _DataProto,
        "Optional": typing.Optional,
        "Any": typing.Any,
        "math": math,
    }
    exec(_extract(REPO_ROOT / "verl/trainer/ppo/rollout_corr_helper.py", HELPER_FUNCS), ns)
    return ns["compute_rollout_correction_and_rejection_mask"]


def main() -> None:
    fn = load_helper()
    torch.manual_seed(0)
    batch, max_len = 128, 16384
    lengths = torch.distributions.LogNormal(math.log(2000.0), 0.8).sample((batch,)).clamp(80, max_len).long()
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    rollout_lp = -torch.distributions.Exponential(torch.tensor(8.0)).sample((batch, max_len))
    sigma = 0.14  # symmetric per-token log-prob noise -> KL ~ sigma^2 / 2 ~ 0.01 nats/token
    train_lp = rollout_lp + sigma * torch.randn(batch, max_len) - 0.5 * sigma**2
    rollout_lp = torch.where(mask, rollout_lp, torch.zeros_like(rollout_lp))
    train_lp = torch.where(mask, train_lp, torch.zeros_like(train_lp))

    keys = (
        "kl",
        "k3_kl",
        "rollout_is_mean",
        "rollout_is_ratio_fraction_low",
        "rollout_is_seq_fraction_low",
        "rollout_is_ratio_fraction_high",
        "rollout_is_max",
        "rollout_is_min",
    )
    print(f"synthetic batch={batch}, mean response length={lengths.float().mean():.0f} tokens")
    for level in ("token", "sequence"):
        _, _, metrics = fn(
            old_log_prob=train_lp,
            rollout_log_prob=rollout_lp,
            response_mask=mask,
            rollout_is=level,
            rollout_is_threshold=2.0,
        )
        print(f"rollout_is={level!r}: " + ", ".join(f"{k}={metrics['rollout_corr/' + k]:.4g}" for k in keys))


if __name__ == "__main__":
    main()
