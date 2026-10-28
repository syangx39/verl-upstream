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
"""Regression guard: ``max_ckpt_to_keep`` must bound the TorchTitan engine's HuggingFace exports.

``TorchTitanEngine.save_checkpoint`` used to forward ``max_ckpt_to_keep`` only to TorchTitan's DCP
``keep_latest_k``, so with ``checkpoint.save_contents=['hf_model']`` every ``<local_path>/huggingface`` export was
kept forever (for Qwen3-30B-A3B under host-local ``/tmp``: ~57 GiB per trainer host and save).

The engine module cannot be imported on a CPU-only runner (torchtitan is not installed there), so, as in
tests/workers/test_engine_detach_tree_on_cpu.py, the rotation is tested on the shared ``BaseCheckpointManager``
helper the engine delegates to, and the call sites are checked on the source.
"""

import ast
from pathlib import Path

import pytest

import verl

ENGINE_SRC = Path(verl.__file__).parent / "workers" / "engine" / "torchtitan" / "transformer_impl.py"
STEPS = (10, 20, 30, 40, 50)


def _engine_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(ENGINE_SRC.read_text())
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TorchTitanEngine")
    return next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _method_calls(node: ast.AST, attr: str) -> list[ast.Call]:
    """Every ``<expr>.<attr>(...)`` call inside ``node``, in source order."""
    calls = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == attr
    ]
    return sorted(calls, key=lambda call: (call.lineno, call.col_offset))


def _single_call(fn: ast.FunctionDef, attr: str) -> ast.Call:
    calls = _method_calls(fn, attr)
    assert len(calls) == 1, f"expected exactly one .{attr}(...) call in {fn.name}, found {len(calls)}"
    return calls[0]


def test_save_checkpoint_rotates_hf_exports_around_the_export():
    fn = _engine_method("save_checkpoint")
    export = _single_call(fn, "_save_hf_checkpoint")
    ensure = _single_call(fn, "ensure_checkpoint_capacity")
    register = _single_call(fn, "register_checkpoint")
    barrier = _single_call(fn, "barrier")

    # Free room before writing; register (and prune past the limit) only once every rank finished writing.
    assert ensure.lineno < export.lineno < barrier.lineno < register.lineno
    assert ast.unparse(ensure.func.value) == ast.unparse(register.func.value)
    assert [ast.unparse(arg) for arg in ensure.args] == ["max_ckpt_to_keep"]
    # Rotate the export itself, never the whole step folder (it also holds the trainer's data.pt).
    assert [ast.unparse(arg) for arg in register.args] == [
        "os.path.join(local_path, 'huggingface')",
        "max_ckpt_to_keep",
    ]


def test_hf_exports_are_rotated_by_the_ranks_that_write_them():
    # Under host-local /tmp every host writes its own copy, so pruning on global rank 0 alone would leak the rest.
    for name in ("save_checkpoint", "_save_hf_checkpoint"):
        calls = _method_calls(_engine_method(name), "_is_hf_checkpoint_writer")
        assert [ast.unparse(call) for call in calls] == ["self._is_hf_checkpoint_writer(local_path)"], name

    fn = _engine_method("save_checkpoint")
    ensure = _single_call(fn, "ensure_checkpoint_capacity")
    guards = [
        node.test
        for node in ast.walk(fn)
        if isinstance(node, ast.If) and any(child is ensure for stmt in node.body for child in ast.walk(stmt))
    ]
    assert any(_method_calls(test, "_is_hf_checkpoint_writer") for test in guards), [ast.unparse(t) for t in guards]


@pytest.mark.parametrize(
    "max_ckpt_to_keep, max_old_exports_during_save, kept_steps",
    [(2, 1, [40, 50]), (1, 1, [50]), (None, len(STEPS) - 1, list(STEPS))],
)
def test_rotation_keeps_only_the_latest_hf_exports(
    tmp_path, monkeypatch, max_ckpt_to_keep, max_old_exports_during_save, kept_steps
):
    import torch.distributed

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

    def exports() -> list[Path]:
        return sorted(tmp_path.glob("global_step_*/actor/huggingface"), key=lambda p: int(p.parts[-3].split("_")[-1]))

    # Same object and call sequence as TorchTitanEngine.save_checkpoint on an HF-writer rank (PPO trainer layout).
    rotation = BaseCheckpointManager(model=None, optimizer=None)
    for step in STEPS:
        step_dir = tmp_path / f"global_step_{step}"
        rotation.ensure_checkpoint_capacity(max_ckpt_to_keep)
        assert len(exports()) <= max_old_exports_during_save

        hf_dir = step_dir / "actor" / "huggingface"
        hf_dir.mkdir(parents=True)
        (hf_dir / "model.safetensors").write_bytes(b"weights")
        (step_dir / "data.pt").write_bytes(b"dataloader")  # written by the trainer, not by the engine
        rotation.register_checkpoint(str(hf_dir), max_ckpt_to_keep)

    assert [int(p.parts[-3].split("_")[-1]) for p in exports()] == kept_steps
    assert len(list(tmp_path.glob("global_step_*/data.pt"))) == len(STEPS)
