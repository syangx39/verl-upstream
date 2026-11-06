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
"""Executable reference for Qwen3-30B-A3B TorchTitan checkpoint size, SAVE_FREQ overhead, and directory behavior.

This script accompanies ``examples/tpu/retool/CHECKPOINT_MECHANISM_ZH.md`` so that every calculation and code-path
behavior discussed in the document can be inspected and run directly on CPU:

1. ``qwen3_30b_a3b_param_breakdown()`` computes the exact parameter count (``30,532,122,624``) from the Qwen3-30B-A3B
   architecture and derives the byte / GiB sizes of:
   - ``hf_model`` (``bfloat16`` weights only, 2 B/param = ``56.87 GiB`` raw tensor bytes, ``56.89 GiB`` on disk with
     safetensors headers + tokenizer);
   - TorchTitan DCP full training state (``float32`` master weights 4 B + AdamW ``exp_avg`` 4 B + AdamW ``exp_avg_sq``
     4 B = 12 B/param = ``341.22 GiB`` across 4 trainer hosts / 32 ranks).
2. ``simulate_torchtitan_ckpt_layout()`` reproduces the exact directory layout and rotation behavior of
   ``TorchTitanEngine.save_checkpoint`` / ``load_checkpoint`` (``verl/workers/engine/torchtitan/transformer_impl.py``)
   and V1 ``PPOTrainer._save_checkpoint`` / ``_load_checkpoint`` (``verl/trainer/ppo/v1/trainer_base.py``) on a temp
   directory, demonstrating why:
   - ``save_contents=['hf_model']`` + ``resume_mode='auto'`` raises ``FileNotFoundError`` (missing ``step-N/``);
   - ``keep_latest_k`` rotates ``actor/huggingface/`` via ``BaseCheckpointManager`` but never deletes older
     ``global_step_M/step-M/`` DCP directories because ``self.checkpointer.folder`` is overwritten to ``global_step_N``
     on every save step;
   - ``async_save=True`` in our fork skips writing ``latest_checkpointed_iteration.txt`` while still running a
     synchronous DCP save, causing ``resume_mode='auto'`` to silently restart from step 0.
3. ``save_freq_overhead_table()`` computes the checkpoint overhead table across ``SAVE_FREQ in (5, 10, 20, 25)``.

Usage::

    python3 examples/tpu/retool/ckpt_size_and_overhead.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

GIB = 1024**3
GB = 1000**3

# Exact total_size field in gs://.../p2-retool-sft-01/checkpoint/global_step_372/huggingface/model.safetensors.index.json
SFT_SAFETENSORS_INDEX_TOTAL_BYTES = 61_064_245_248


def qwen3_30b_a3b_param_breakdown() -> dict[str, int | float]:
    """Compute exact parameter count and checkpoint sizes of Qwen3-30B-A3B."""
    hidden = 2048
    layers = 48
    vocab = 151_936
    num_heads = 32
    num_kv_heads = 4
    head_dim = 128
    num_experts = 128
    moe_inter = 768

    # Embedding + final RMSNorm + untied lm_head
    embed_params = vocab * hidden + hidden + vocab * hidden

    # Per transformer layer:
    # - input_layernorm (hidden) + post_attention_layernorm (hidden)
    # - q_proj (hidden -> num_heads * head_dim) + k_proj + v_proj + o_proj + q_norm (head_dim) + k_norm (head_dim)
    # - MoE router gate (hidden -> num_experts) + 128 experts * (gate_proj, up_proj, down_proj: 3 * hidden * moe_inter)
    attn_params = (
        hidden * (num_heads * head_dim)
        + hidden * (num_kv_heads * head_dim)
        + hidden * (num_kv_heads * head_dim)
        + (num_heads * head_dim) * hidden
        + 2 * head_dim
    )
    norms_params = 2 * hidden
    moe_params = hidden * num_experts + num_experts * (3 * hidden * moe_inter)
    per_layer_params = attn_params + norms_params + moe_params
    total_params = embed_params + layers * per_layer_params

    hf_bf16_bytes = total_params * 2
    assert hf_bf16_bytes == SFT_SAFETENSORS_INDEX_TOTAL_BYTES, (
        f"Expected {SFT_SAFETENSORS_INDEX_TOTAL_BYTES}, got {hf_bf16_bytes}"
    )

    # TorchTitan DCP full training state: fp32 master weights (4 B) + AdamW exp_avg (4 B) + AdamW exp_avg_sq (4 B)
    dcp_fp32_model_bytes = total_params * 4
    dcp_adam_m_bytes = total_params * 4
    dcp_adam_v_bytes = total_params * 4
    dcp_total_bytes = dcp_fp32_model_bytes + dcp_adam_m_bytes + dcp_adam_v_bytes

    return {
        "total_params": total_params,
        "hf_bf16_bytes": hf_bf16_bytes,
        "hf_bf16_gib": round(hf_bf16_bytes / GIB, 2),
        "hf_bf16_gb": round(hf_bf16_bytes / GB, 2),
        "dcp_total_bytes": dcp_total_bytes,
        "dcp_total_gib": round(dcp_total_bytes / GIB, 2),
        "dcp_total_gb": round(dcp_total_bytes / GB, 2),
        "dcp_per_trainer_host_gib_4hosts": round(dcp_total_bytes / 4 / GIB, 2),
        "dcp_per_rank_gib_32ranks": round(dcp_total_bytes / 32 / GIB, 2),
        "dcp_to_hf_size_ratio": dcp_total_bytes // hf_bf16_bytes,
    }


def simulate_torchtitan_ckpt_layout(
    root_dir: Path,
    save_steps: tuple[int, ...] = (10, 20, 30),
    save_contents: tuple[str, ...] = ("hf_model",),
    max_ckpt_to_keep: int = 2,
    async_save: bool = False,
) -> dict[str, object]:
    """Mirror the exact filesystem actions of V1 PPOTrainer + TorchTitanEngine.save_checkpoint / load_checkpoint."""
    hf_registered: list[Path] = []

    for step in save_steps:
        local_global_step_folder = root_dir / f"global_step_{step}"
        actor_local_path = local_global_step_folder / "actor"
        actor_local_path.mkdir(parents=True, exist_ok=True)

        # 1. HF rotation bookkeeping before writing (verl/workers/engine/torchtitan/transformer_impl.py L689-L694)
        if "hf_model" in save_contents:
            while max_ckpt_to_keep > 0 and len(hf_registered) >= max_ckpt_to_keep:
                old_hf = hf_registered.pop(0)
                if old_hf.exists():
                    shutil.rmtree(old_hf)

        # 2. TorchTitan DCP save (transformer_impl.py L696-L704):
        #    parent_dir = os.path.dirname(local_path)  # -> .../global_step_<step>
        #    self.checkpointer.folder = parent_dir
        #    self.checkpointer.save(curr_step=step)    # writes <folder>/step-<step>/
        if any(k in save_contents for k in ("model", "optimizer")):
            parent_dir = actor_local_path.parent
            step_dir = parent_dir / f"step-{step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / ".metadata").write_text("dcp-metadata\n", encoding="utf-8")
            # TorchTitan _purge_stale_checkpoints only inspects os.listdir(self.checkpointer.folder),
            # which is global_step_<step>/ and only ever contains a single step-<step>/ directory!
            dcp_children = [p for p in parent_dir.iterdir() if p.is_dir() and p.name.startswith("step-")]
            while max_ckpt_to_keep > 0 and len(dcp_children) > max_ckpt_to_keep:
                shutil.rmtree(dcp_children.pop(0))

        # 3. HF export (transformer_impl.py L706-L711)
        if "hf_model" in save_contents:
            hf_dir = actor_local_path / "huggingface"
            hf_dir.mkdir(parents=True, exist_ok=True)
            (hf_dir / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
            hf_registered.append(hf_dir)

        # 4. Driver writes dataloader state + tracker (verl/trainer/ppo/v1/trainer_base.py L1029-L1054)
        (local_global_step_folder / "data.pt").write_text("dataloader_state\n", encoding="utf-8")
        if not async_save:
            (root_dir / "latest_checkpointed_iteration.txt").write_text(f"{step}\n", encoding="utf-8")

    # Simulate V1 _load_checkpoint(resume_mode="auto") + TorchTitanEngine.load_checkpoint
    tracker = root_dir / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        resume_outcome = "trains_from_scratch_no_tracker"
    else:
        latest_step = int(tracker.read_text(encoding="utf-8").strip())
        dcp_step_dir = root_dir / f"global_step_{latest_step}" / f"step-{latest_step}"
        if not (dcp_step_dir / ".metadata").exists():
            resume_outcome = f"FileNotFoundError: missing {dcp_step_dir.relative_to(root_dir)}"
        else:
            resume_outcome = f"resumed_step_{latest_step}"

    existing_dcp = sorted(str(p.relative_to(root_dir)) for p in root_dir.glob("global_step_*/step-*"))
    existing_hf = sorted(str(p.relative_to(root_dir)) for p in root_dir.glob("global_step_*/actor/huggingface"))
    return {
        "existing_dcp_dirs": existing_dcp,
        "existing_hf_dirs": existing_hf,
        "tracker_exists": tracker.exists(),
        "resume_auto_outcome": resume_outcome,
    }


def save_freq_overhead_table(
    step_time_s: float = 342.0,
    dcp_sync_save_s: float = 1412.0,
    dcp_net_save_s: float = 510.0,
    hf_only_save_s: float = 235.84,
) -> list[dict[str, float | int]]:
    """Compute checkpoint overhead ratios for SAVE_FREQ in (5, 10, 20, 25)."""
    rows: list[dict[str, float | int]] = []
    for freq in (5, 10, 20, 25):
        interval_s = freq * step_time_s
        rows.append(
            {
                "save_freq": freq,
                "interval_s": round(interval_s, 1),
                "dcp_raw_overhead_pct": round(100.0 * dcp_sync_save_s / interval_s, 1),
                "dcp_net_overhead_pct": round(100.0 * dcp_net_save_s / interval_s, 1),
                "hf_only_overhead_pct": round(100.0 * hf_only_save_s / interval_s, 1),
                "max_lost_train_min": round(interval_s / 60.0, 1),
            }
        )
    return rows


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        hf_only = simulate_torchtitan_ckpt_layout(tmp_root / "hf_only", save_contents=("hf_model",), max_ckpt_to_keep=2)
        full_dcp = simulate_torchtitan_ckpt_layout(
            tmp_root / "full_dcp", save_contents=("model", "optimizer", "extra", "hf_model"), max_ckpt_to_keep=2
        )
        async_trap = simulate_torchtitan_ckpt_layout(
            tmp_root / "async_trap", save_contents=("model", "optimizer", "extra"), async_save=True
        )
    summary = {
        "param_and_size_breakdown": qwen3_30b_a3b_param_breakdown(),
        "simulations": {
            "hf_only_with_auto_resume": hf_only,
            "full_dcp_plus_hf_keep_2": full_dcp,
            "async_save_true_trap": async_trap,
        },
        "save_freq_overhead": save_freq_overhead_table(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
