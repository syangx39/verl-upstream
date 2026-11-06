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
"""CPU unit tests for examples/tpu/retool/ckpt_size_and_overhead.py."""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples" / "tpu" / "retool" / "ckpt_size_and_overhead.py"

spec = importlib.util.spec_from_file_location("ckpt_size_and_overhead", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
ckpt_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ckpt_mod
spec.loader.exec_module(ckpt_mod)


def test_qwen3_30b_a3b_param_and_dcp_size_breakdown():
    info = ckpt_mod.qwen3_30b_a3b_param_breakdown()
    assert info["total_params"] == 30_532_122_624
    assert info["hf_bf16_bytes"] == 61_064_245_248
    assert info["dcp_total_bytes"] == 366_385_471_488
    assert info["dcp_to_hf_size_ratio"] == 6
    assert info["hf_bf16_gib"] == 56.87
    assert info["dcp_total_gib"] == 341.22


def test_torchtitan_checkpoint_layout_and_resume_simulations(tmp_path):
    hf_only = ckpt_mod.simulate_torchtitan_ckpt_layout(
        tmp_path / "hf_only", save_contents=("hf_model",), max_ckpt_to_keep=2
    )
    assert hf_only["existing_dcp_dirs"] == []
    assert hf_only["existing_hf_dirs"] == [
        "global_step_20/actor/huggingface",
        "global_step_30/actor/huggingface",
    ]
    assert str(hf_only["resume_auto_outcome"]).startswith("FileNotFoundError:")

    full_dcp = ckpt_mod.simulate_torchtitan_ckpt_layout(
        tmp_path / "full_dcp", save_contents=("model", "optimizer", "extra", "hf_model"), max_ckpt_to_keep=2
    )
    assert full_dcp["existing_dcp_dirs"] == [
        "global_step_10/step-10",
        "global_step_20/step-20",
        "global_step_30/step-30",
    ]
    assert full_dcp["existing_hf_dirs"] == [
        "global_step_20/actor/huggingface",
        "global_step_30/actor/huggingface",
    ]
    assert full_dcp["resume_auto_outcome"] == "resumed_step_30"

    async_trap = ckpt_mod.simulate_torchtitan_ckpt_layout(
        tmp_path / "async_trap", save_contents=("model", "optimizer", "extra"), async_save=True
    )
    assert async_trap["tracker_exists"] is False
    assert async_trap["resume_auto_outcome"] == "trains_from_scratch_no_tracker"


def test_save_freq_overhead_monotonic():
    rows = ckpt_mod.save_freq_overhead_table()
    assert [r["save_freq"] for r in rows] == [5, 10, 20, 25]
    assert rows[0]["dcp_raw_overhead_pct"] > rows[1]["dcp_raw_overhead_pct"] > rows[2]["dcp_raw_overhead_pct"]
