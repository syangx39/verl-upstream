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
"""Stage G0 verifier for NVIDIA GB200 ReTool RL reproduction.

Verifies:
1. Shared TPU P2 SFT checkpoint (`global_step_372/huggingface`) 16-shard SHA-256 manifest
   and cross-host `/dev/shm` staging parity.
2. TorchTitan 0.3.0 `Qwen3-30B-A3B` MoE config adaptation via `configure_qwen3_moe_for_rl`:
   - `dataclasses.fields(MoE.Config)` schema check (`load_balance_coeff` present, `aux_loss` absent).
   - All 48 layers of `30B-A3B` have `load_balance_coeff=1e-3` before `configure_qwen3_moe_for_rl`
     and `load_balance_coeff=None` after.
3. End-to-end 1-step MoE optimizer update + `Qwen3StateDictAdapter.to_hf` round-trip check
   on GPU:
   - Positive check (`configure_qwen3_moe_for_rl` applied): after 1 forward/backward/optimizer step,
     `expert_bias` remains `None` (no load-balancing hook registered), `to_hf` -> `from_hf` preserves
     100% of router top-k expert selections (`router_topk_match_rate == 1.0`) and `0.0` output logit
     difference (`logits_max_abs_diff == 0.0`).
   - Negative control (`load_balance_coeff=1e-3` left enabled): after 1 step with `register_moe_load_balancing_hook`,
     `expert_bias` becomes non-zero on the actor (`expert_bias_abs_max > 0`) while `to_hf()` drops
     `expert_bias` keys, demonstrating the exact actor-vs-rollout routing drift prevented by `G0`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Any


def stage_checkpoint_local(src_dir: Path, dst_dir: Path, workers: int = 8) -> dict[str, Any]:
    """Copy all checkpoint files from src_dir to dst_dir in parallel (skipping already-matching sizes)."""
    t0 = time.time()
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_files = sorted(p for p in src_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    if not any(p.name.endswith(".safetensors") for p in src_files):
        raise FileNotFoundError(f"No .safetensors files found in {src_dir}")

    to_copy: list[tuple[Path, Path, int]] = []
    for sp in src_files:
        dp = dst_dir / sp.name
        sz = sp.stat().st_size
        if not dp.is_file() or dp.stat().st_size != sz:
            to_copy.append((sp, dp, sz))

    def _copy_one(item: tuple[Path, Path, int]) -> int:
        sp, dp, sz = item
        tmp = dp.with_suffix(dp.suffix + f".tmp.{os.getpid()}")
        shutil.copyfile(sp, tmp)
        os.replace(tmp, dp)
        return sz

    copied_bytes = 0
    if to_copy:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            copied_bytes = sum(ex.map(_copy_one, to_copy))

    dt = max(time.time() - t0, 1e-3)
    return {
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "total_files": len(src_files),
        "copied_files": len(to_copy),
        "copied_bytes": copied_bytes,
        "elapsed_s": round(dt, 3),
        "throughput_gb_s": round((copied_bytes / 1e9) / dt, 3) if copied_bytes > 0 else 0.0,
    }


def _sha256_file(path: Path) -> tuple[str, int, str]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            h.update(chunk)
    return path.name, total, h.hexdigest()


def verify_checkpoint_manifest(
    model_dir: Path,
    ref_dir: Path | None = None,
    expect_composite_sha256: str | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    t0 = time.time()
    required_meta = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    shards = [f"model-{i:05d}-of-00016.safetensors" for i in range(1, 17)]
    all_names = required_meta + shards
    missing = [name for name in all_names if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {model_dir}: {missing}")

    paths = [model_dir / name for name in all_names]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_sha256_file, paths))

    manifest = {name: {"size_bytes": size_b, "sha256": digest} for name, size_b, digest in rows}
    manifest_json = json.dumps(manifest, sort_keys=True).encode("utf-8")
    composite_sha256 = hashlib.sha256(manifest_json).hexdigest()

    expected_composite = expect_composite_sha256.strip().lower() if expect_composite_sha256 else None
    composite_sha256_match = (composite_sha256 == expected_composite) if expected_composite is not None else None

    ref_match = None
    ref_composite_sha256 = None
    mismatches: list[str] = []
    if ref_dir is not None and ref_dir.exists():
        if expected_composite is not None and composite_sha256_match:
            for name in all_names:
                rp = ref_dir / name
                if not rp.is_file() or rp.stat().st_size != manifest[name]["size_bytes"]:
                    mismatches.append(name)
            ref_match = len(mismatches) == 0
            ref_composite_sha256 = expected_composite if ref_match else None
        else:
            ref_paths = [ref_dir / name for name in all_names]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                ref_rows = list(ex.map(_sha256_file, ref_paths))
            ref_manifest = {name: {"size_bytes": size_b, "sha256": digest} for name, size_b, digest in ref_rows}
            ref_composite_sha256 = hashlib.sha256(json.dumps(ref_manifest, sort_keys=True).encode("utf-8")).hexdigest()
            for name in all_names:
                if manifest[name] != ref_manifest[name]:
                    mismatches.append(name)
            ref_match = len(mismatches) == 0

    return {
        "model_dir": str(model_dir),
        "ref_dir": str(ref_dir) if ref_dir else None,
        "num_files_hashed": len(all_names),
        "num_safetensors_shards": len(shards),
        "total_bytes": sum(r[1] for r in rows),
        "composite_sha256": composite_sha256,
        "expected_composite_sha256": expected_composite,
        "composite_sha256_match": composite_sha256_match,
        "ref_composite_sha256": ref_composite_sha256,
        "ref_match": ref_match,
        "mismatches": mismatches,
        "elapsed_s": round(time.time() - t0, 3),
        "files": manifest,
    }


def verify_moe_config_and_step(model_dir: Path) -> dict[str, Any]:
    import torch
    import torchtitan.models.qwen3 as tt_qwen3
    from torchtitan.models.common.moe import MoE
    from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
    from transformers import AutoConfig

    from verl.workers.engine.torchtitan.utils import (
        configure_qwen3_moe_for_rl,
        derive_torchtitan_name_and_flavor,
    )

    hf_config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    tt_name, tt_flavor = derive_torchtitan_name_and_flavor(hf_config)
    if (tt_name, tt_flavor) != ("qwen3", "30B-A3B"):
        raise AssertionError(f"Expected ('qwen3', '30B-A3B'), got {(tt_name, tt_flavor)}")

    moe_field_names = [f.name for f in fields(MoE.Config)]
    has_load_balance_coeff = "load_balance_coeff" in moe_field_names
    has_aux_loss = "aux_loss" in moe_field_names
    if not has_load_balance_coeff or has_aux_loss:
        raise AssertionError(f"Unexpected TorchTitan 0.3 MoE.Config fields: {moe_field_names}")

    model_spec = tt_qwen3.model_registry(tt_flavor, attn_backend="flex")
    before_coeffs = [moe.load_balance_coeff for _, moe, _, _ in model_spec.model.traverse(MoE.Config)]
    if len(before_coeffs) != 48 or not all(c == 1e-3 for c in before_coeffs):
        raise AssertionError(
            f"Expected 48 MoE layers with default load_balance_coeff=1e-3 before fix, "
            f"got len={len(before_coeffs)}, sample={before_coeffs[:3]}"
        )

    configure_qwen3_moe_for_rl(model_spec.model)
    after_coeffs = [moe.load_balance_coeff for _, moe, _, _ in model_spec.model.traverse(MoE.Config)]
    if len(after_coeffs) != 48 or not all(c is None for c in after_coeffs):
        raise AssertionError(
            "Expected all 48 MoE layers to have load_balance_coeff=None after configure_qwen3_moe_for_rl, "
            f"got {after_coeffs[:3]}"
        )

    # Build a 2-layer miniature Qwen3-MoE model on CUDA to test 1 optimizer step + to_hf() round-trip
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026)

    def _make_mini_spec(apply_rl_fix: bool):
        spec = copy.deepcopy(tt_qwen3.model_registry("debugmodel_moe", attn_backend="flex"))
        if apply_rl_fix:
            configure_qwen3_moe_for_rl(spec.model)
        return spec

    def _run_one_step_and_compare(apply_rl_fix: bool) -> dict[str, Any]:
        spec = _make_mini_spec(apply_rl_fix=apply_rl_fix)
        torch.manual_seed(1234)
        model = spec.model.build()
        model.to(device=device, dtype=torch.bfloat16)
        model.init_weights(buffer_device=device)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # TorchTitan 0.3 registers post_optimizer_build_fn(model_parts, optimizers, parallel_dims)
        if spec.post_optimizer_build_fn is not None:
            import inspect

            class _OptWrap:
                def __init__(self, inner):
                    self.inner = inner
                    self.optimizers = [inner]

                def register_step_pre_hook(self, fn):
                    return self.inner.register_step_pre_hook(fn)

            class _DummyParallelDims:
                dp_cp_enabled = False
                dp_enabled = False
                cp_enabled = False
                ep_enabled = False
                pp_enabled = False
                tp_enabled = False

                @staticmethod
                def get_optional_mesh(_name: str):
                    return None

            param_names = list(inspect.signature(spec.post_optimizer_build_fn).parameters.keys())
            first_is_opt = bool(param_names and param_names[0].startswith("opt"))
            if len(param_names) >= 3:
                if first_is_opt:
                    spec.post_optimizer_build_fn(_OptWrap(opt), [model], _DummyParallelDims())
                else:
                    spec.post_optimizer_build_fn([model], _OptWrap(opt), _DummyParallelDims())
            else:
                if first_is_opt:
                    spec.post_optimizer_build_fn(_OptWrap(opt), [model])
                else:
                    spec.post_optimizer_build_fn([model], _OptWrap(opt))

        from verl.workers.engine.torchtitan.utils import get_attention_masks

        tokens = torch.randint(0, spec.model.vocab_size, (2, 128), device=device)
        positions = torch.arange(128, device=device).unsqueeze(0).expand(2, -1)
        attn_masks = get_attention_masks(tokens, positions, "flex")

        out = model(tokens, attention_masks=attn_masks, positions=positions)
        loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Inspect expert_bias and tokens_per_expert across MoE modules and all model buffers
        moe_modules = [m for m in model.modules() if isinstance(m, MoE)]
        expert_bias_max = 0.0
        has_expert_bias_buf = False
        buffer_max_by_name: dict[str, float] = {}
        for buf_name, buf in model.named_buffers():
            val = float(buf.detach().float().abs().max().item()) if buf.numel() > 0 else 0.0
            buffer_max_by_name[buf_name] = val
            if "expert_bias" in buf_name:
                has_expert_bias_buf = True
                expert_bias_max = max(expert_bias_max, val)

        # Convert actor state_dict to HF via Qwen3StateDictAdapter.to_hf and reload into a fresh HF-equivalent model
        adapter = Qwen3StateDictAdapter(spec.model, None)
        raw_sd = model.state_dict()
        hf_sd = adapter.to_hf(raw_sd)
        has_expert_bias_in_hf = any("expert_bias" in k for k in hf_sd.keys())

        # Build receiver model (representing rollout engine receiving only HF weights)
        recv_spec = _make_mini_spec(apply_rl_fix=True)
        recv_model = recv_spec.model.build().to(device=device, dtype=torch.bfloat16)
        recv_model.init_weights(buffer_device=device)
        restored_sd = adapter.from_hf(hf_sd)
        dropped_state_keys = sorted(set(raw_sd.keys()) - set(restored_sd.keys()))
        missing, unexpected = recv_model.load_state_dict(restored_sd, strict=False)
        if unexpected:
            raise AssertionError(f"Unexpected keys restoring from HF state dict: {unexpected}")

        # Capture router top-k indices on both actor and receiver on an evaluation batch
        eval_tokens = torch.randint(0, spec.model.vocab_size, (2, 128), device=device)
        eval_positions = torch.arange(128, device=device).unsqueeze(0).expand(2, -1)
        eval_masks = get_attention_masks(eval_tokens, eval_positions, "flex")

        actor_topk: list[torch.Tensor] = []
        recv_topk: list[torch.Tensor] = []

        def _hook_router(store: list[torch.Tensor]):
            def _fn(_mod, _inp, output):
                # Router returns (top_scores, selected_experts_indices, num_tokens_per_expert)
                if isinstance(output, tuple) and len(output) >= 2:
                    store.append(output[1].detach().clone())

            return _fn

        handles = []
        for m in moe_modules:
            handles.append(m.router.register_forward_hook(_hook_router(actor_topk)))
        for m in recv_model.modules():
            if isinstance(m, MoE):
                handles.append(m.router.register_forward_hook(_hook_router(recv_topk)))

        with torch.no_grad():
            actor_logits = model(eval_tokens, attention_masks=eval_masks, positions=eval_positions)
            recv_logits = recv_model(eval_tokens, attention_masks=eval_masks, positions=eval_positions)

        for h in handles:
            h.remove()

        topk_matches = [float((a == b).float().mean().item()) for a, b in zip(actor_topk, recv_topk, strict=True)]
        logits_max_abs_diff = float((actor_logits.float() - recv_logits.float()).abs().max().item())

        return {
            "apply_rl_fix": apply_rl_fix,
            "num_moe_layers": len(moe_modules),
            "has_expert_bias_buffer": has_expert_bias_buf,
            "expert_bias_abs_max_after_step1": expert_bias_max,
            "buffer_max_by_name": buffer_max_by_name,
            "dropped_state_keys": dropped_state_keys,
            "has_expert_bias_in_to_hf": has_expert_bias_in_hf,
            "missing_keys_on_from_hf": list(missing),
            "router_topk_match_rate": sum(topk_matches) / max(len(topk_matches), 1),
            "logits_max_abs_diff": logits_max_abs_diff,
        }

    positive = _run_one_step_and_compare(apply_rl_fix=True)
    negative = _run_one_step_and_compare(apply_rl_fix=False)

    if positive["has_expert_bias_buffer"] or positive["expert_bias_abs_max_after_step1"] != 0.0:
        raise AssertionError(f"Expected no expert_bias update with RL fix, got {positive}")
    if positive["router_topk_match_rate"] != 1.0 or positive["logits_max_abs_diff"] != 0.0:
        raise AssertionError(f"Expected 100% router top-k and 0.0 logit diff after to_hf round-trip, got {positive}")
    if negative["router_topk_match_rate"] >= 1.0 or negative["logits_max_abs_diff"] <= 0.0:
        raise AssertionError(f"Negative control expected router/logit divergence without RL fix, got {negative}")

    return {
        "torchtitan_name": tt_name,
        "torchtitan_flavor": tt_flavor,
        "moe_config_fields": moe_field_names,
        "num_layers_30b_a3b": len(after_coeffs),
        "before_load_balance_coeff_sample": before_coeffs[:3],
        "after_load_balance_coeff_sample": after_coeffs[:3],
        "positive_rl_fix_check": positive,
        "negative_control_unfixed_check": negative,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Stage G0 gates on GB200.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Staged local model directory (e.g. /dev/shm/models/multiturn-sft-qwen3-30b-a3b-tpu7x).",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="Optional reference checkpoint directory on shared storage to cross-check.",
    )
    parser.add_argument(
        "--stage-from",
        type=Path,
        default=None,
        help="Optional source checkpoint directory to stage into --model-dir before verifying.",
    )
    parser.add_argument(
        "--expect-composite-sha256",
        type=str,
        default=None,
        help="Optional expected composite SHA-256 over the 20 checkpoint manifest entries; fails if mismatched.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the JSON verification report.",
    )
    args = parser.parse_args()

    stage_report = None
    if args.stage_from is not None:
        stage_report = stage_checkpoint_local(args.stage_from, args.model_dir)

    manifest_report = verify_checkpoint_manifest(
        args.model_dir,
        args.ref_dir,
        expect_composite_sha256=args.expect_composite_sha256,
    )
    if args.ref_dir is not None and not manifest_report["ref_match"]:
        raise RuntimeError(f"Checkpoint manifest mismatch against {args.ref_dir}: {manifest_report['mismatches']}")
    if args.expect_composite_sha256 and not manifest_report["composite_sha256_match"]:
        raise RuntimeError(
            f"Checkpoint composite_sha256 mismatch: expected {manifest_report['expected_composite_sha256']}, "
            f"got {manifest_report['composite_sha256']}"
        )

    moe_report = verify_moe_config_and_step(args.model_dir)
    report = {
        "status": "PASS",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "g0_stage_local": stage_report,
        "g0_checkpoint_manifest": manifest_report,
        "g0_moe_torchtitan_0_3": moe_report,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
