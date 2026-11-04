# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for the Raiden checkpoint engine: pure helpers, parity checks, and the update_raiden_weights orchestration.

tpu_sync / torch_tpu are not installed on CPU runners. The engine imports them lazily, so the orchestration test
stubs the two tpu_sync modules it touches and drives update_raiden_weights with fake manager / replicas / controller.
"""

import itertools
import sys
import threading
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import verl.checkpoint_engine.raiden_checkpoint_engine as raiden
from verl.checkpoint_engine.raiden_checkpoint_engine import (
    RaidenCheckpointEngine,
    RaidenParityCheck,
    RaidenParityError,
    _dim0_shard_info,
    _replica_num_workers,
    _sampler_job_name,
    compute_expected_checked_params,
    filter_tied_embeddings,
    merge_rank_stats,
)

# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------


def test_filter_tied_embeddings():
    items = [("model.embed_tokens.weight", 1), ("lm_head.weight", 2), ("model.norm.weight", 3)]
    assert [k for k, _ in filter_tied_embeddings(items, tie_word_embeddings=True)] == [
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]
    # Untied models (Qwen3-8B+) train a separate lm_head: it must be sent.
    assert filter_tied_embeddings(items, tie_word_embeddings=False) == items
    # Without an embedding in the set there is nothing lm_head could be copied from: keep it.
    no_embed = [("lm_head.weight", 2), ("model.norm.weight", 3)]
    assert filter_tied_embeddings(no_embed, tie_word_embeddings=True) == no_embed


def test_compute_expected_checked_params_untied_and_tied():
    class _ToyModel(torch.nn.Module):
        def __init__(self, tie_mode: str):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(8, 4)
            self.model.norm = torch.nn.LayerNorm(4, elementwise_affine=True)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            if tie_mode == "shared_param":
                # Shared nn.Parameter instance: named_parameters() already deduplicates to 3 params
                # (embed_tokens.weight, norm.weight, norm.bias), and all 3 are checked by check_exact.
                self.lm_head.weight = self.model.embed_tokens.weight

    # 1. Untied model: trainer sends lm_head.weight (received_lm_head=True), all 4 distinct params are checked.
    untied = _ToyModel("untied")
    assert len(list(untied.named_parameters())) == 4
    assert compute_expected_checked_params(untied, received_lm_head=True) == 4

    # 2. Tied model with shared nn.Parameter (lm_head.weight is model.embed_tokens.weight):
    #    named_parameters() yields 3 unique parameters, and all 3 are checked by check_exact.
    #    Must NOT double-subtract 1 (must return 3, not 2).
    tied_shared = _ToyModel("shared_param")
    assert tied_shared.lm_head.weight is tied_shared.model.embed_tokens.weight
    assert len(list(tied_shared.named_parameters())) == 3
    assert compute_expected_checked_params(tied_shared, received_lm_head=False) == 3

    # 3. Tied model where lm_head.weight is a separate nn.Parameter buffer copied from embed_tokens.weight
    #    in step 3 (received_lm_head=False, lm_head.weight is not embed_tokens.weight):
    #    named_parameters() yields 4 parameters, but lm_head.weight is copied after check_exact -> expected 3.
    tied_separate = _ToyModel("separate_param")
    assert tied_separate.lm_head.weight is not tied_separate.model.embed_tokens.weight
    assert len(list(tied_separate.named_parameters())) == 4
    assert compute_expected_checked_params(tied_separate, received_lm_head=False) == 3


class _FakeMesh:
    def __init__(self, sizes):
        self._sizes = sizes
        self.ndim = len(sizes)

    def size(self, dim):
        return self._sizes[dim]


def _patch_placement(monkeypatch, local_shape, global_offset, full_shape):
    from verl.workers.engine import spec as spec_mod

    place = spec_mod.BlockPlacement(tuple(local_shape), tuple(global_offset), tuple(full_shape))
    monkeypatch.setattr(spec_mod, "derive_dtensor_placement", lambda spec: (place, True, None))


def test_dim0_shard_info_even_shard0(monkeypatch):
    # FSDP2 Shard(0) over 4 ranks, this rank holds rows 6:9 of a [12, 16] tensor -> block 2 of 4.
    _patch_placement(monkeypatch, (3, 16), (6, 0), (12, 16))
    spec = SimpleNamespace(mesh=_FakeMesh([4]), full_shape=(12, 16))
    assert _dim0_shard_info(spec) == ((3, 16), 4, 2)


def test_dim0_shard_info_rejects_unsupported_layouts(monkeypatch):
    # Uneven dim-0 cut: torch and Raiden disagree on who gets the remainder.
    _patch_placement(monkeypatch, (4, 16), (0, 0), (10, 16))
    assert _dim0_shard_info(SimpleNamespace(mesh=_FakeMesh([3]), full_shape=(10, 16))) is None
    # Not a pure dim-0 cut (dim 1 is split): not flat-contiguous.
    _patch_placement(monkeypatch, (12, 8), (0, 8), (12, 16))
    assert _dim0_shard_info(SimpleNamespace(mesh=_FakeMesh([2]), full_shape=(12, 16))) is None
    # 2-D mesh, single-rank mesh, and no mesh at all.
    assert _dim0_shard_info(SimpleNamespace(mesh=_FakeMesh([2, 2]), full_shape=(12, 16))) is None
    _patch_placement(monkeypatch, (12, 16), (0, 0), (12, 16))
    assert _dim0_shard_info(SimpleNamespace(mesh=_FakeMesh([1]), full_shape=(12, 16))) is None
    assert _dim0_shard_info(SimpleNamespace(mesh=None, full_shape=(12, 16))) is None


def test_merge_rank_stats_sums_partial_stats():
    rank_stats = {
        0: {"per_tensor": {"w": {"l1": 1.0, "l2_sq": 4.0, "numel": 2}, "norm": {"l1": 3.0, "l2_sq": 9.0, "numel": 1}}},
        1: {"per_tensor": {"w": {"l1": 2.0, "l2_sq": 5.0, "numel": 2}}},
    }
    merged = merge_rank_stats(rank_stats)
    assert merged["per_tensor"]["w"] == {"l1": 3.0, "l2_sq": 9.0, "numel": 4, "l2": 3.0}
    assert merged["total_numel"] == 5
    assert merged["num_tensors"] == 2
    assert merged["l1_norm"] == 6.0
    assert merged["l2_sq"] == 18.0
    assert merged["l2_norm"] == pytest.approx(18.0**0.5)


def test_sampler_job_name_and_worker_count():
    assert _sampler_job_name(0, 1) == "sampler"
    assert [_sampler_job_name(i, 3) for i in range(3)] == ["sampler0", "sampler1", "sampler2"]
    assert _replica_num_workers(SimpleNamespace(world_size=4)) == 4
    assert _replica_num_workers(SimpleNamespace(world_size=None, workers=[1, 2])) == 2
    assert _replica_num_workers(SimpleNamespace()) == 1


@pytest.mark.parametrize(
    "kwarg, env, expected",
    [(True, "", True), (False, "", False), (True, "0", False), (False, "1", True), ("false", "", False)],
)
def test_release_buffers_flag(monkeypatch, kwarg, env, expected):
    from verl.checkpoint_engine import tpu_weight_registry

    monkeypatch.setattr(tpu_weight_registry, "get_tpu_weight_registry", lambda: None)
    monkeypatch.setenv("VERL_RAIDEN_RELEASE_BUFFERS", env)
    engine = RaidenCheckpointEngine(release_buffers_after_sync=kwarg)
    assert engine.release_buffers_after_sync is expected


# ---------------------------------------------------------------------------------------------
# RaidenParityCheck
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, mode",
    [
        (None, "off"),
        (False, "off"),
        (True, "norm"),
        ("1", "norm"),
        ("false", "off"),
        ("EXACT", "exact"),
        ("all", "all"),
    ],
)
def test_parse_mode(value, mode):
    assert RaidenParityCheck.parse_mode(value) == mode


def test_parse_mode_rejects_unknown():
    with pytest.raises(ValueError):
        RaidenParityCheck.parse_mode("sometimes")


def test_modes_and_exact_only_on_step0():
    assert RaidenParityCheck("all").norm and RaidenParityCheck("all").exact
    assert RaidenParityCheck("norm").norm and not RaidenParityCheck("norm").exact
    exact = RaidenParityCheck("exact")
    assert exact.exact_install_kwargs(None) == {"exact_parity": True}
    assert exact.exact_install_kwargs(0) == {"exact_parity": True}
    # Trained weights no longer equal what vLLM loaded, so later steps cannot run the exact check.
    assert exact.exact_install_kwargs(3) == {}
    assert RaidenParityCheck("norm").exact_install_kwargs(0) == {}
    assert RaidenParityCheck.from_config({"raiden": {"verify_parity": "all"}}).mode == "all"
    assert RaidenParityCheck.from_config({}).mode == "off"


def test_check_exact_records_value_and_shape_mismatches():
    parity = RaidenParityCheck("exact")
    target = torch.arange(6, dtype=torch.bfloat16).view(2, 3)
    parity.check_exact("same", target, target.clone())
    parity.check_exact("value", target, target + 1)
    parity.check_exact("shape", target, target.view(3, 2))
    result = parity.exact_result(rank=5, unresolved=["x"])
    assert result["rank"] == 5
    assert result["checked"] == 2
    assert result["num_mismatched"] == 2
    assert result["mismatched"][0].startswith("value:")
    assert result["mismatched"][1].startswith("shape:")
    assert result["unresolved"] == ["x"]


def _exact(rank, mismatched=(), unresolved=(), checked=10, expected_checked=None):
    payload = {
        "rank": rank,
        "checked": checked,
        "num_mismatched": len(mismatched),
        "mismatched": list(mismatched),
        "unresolved": list(unresolved),
    }
    if expected_checked is not None:
        payload["expected_checked"] = expected_checked
    return {"exact_parity": payload}


def test_report_exact_passes_when_every_rank_matches(tmp_path):
    # One list of worker results per replica, as collective_rpc returns them.
    assert (
        RaidenParityCheck.report_exact(
            0,
            [[_exact(0), _exact(1)], [_exact(0), _exact(1)]],
            expected_ranks=4,
            expected_checked_per_rank=10,
            dump_dir=str(tmp_path),
        )
        is True
    )
    archive = tmp_path / "raiden_exact_parity_step_0.json"
    assert archive.is_file()


def test_report_exact_archives_untruncated_mismatches_and_enforces_coverage(tmp_path):
    parity = RaidenParityCheck("exact")
    ref = torch.ones((2, 4, 6), dtype=torch.bfloat16)
    recv_half_zero = ref.clone()
    recv_half_zero[..., 3:] = 0.0
    parity.check_exact(
        "model.layers.3.mlp.experts.routed_experts.w13_weight",
        ref,
        recv_half_zero,
        staging_tensors={"gate": torch.ones((4, 4)), "up": torch.zeros((4, 4))},
    )
    for idx in range(70):
        parity.check_exact(f"layer.{idx}.weight", torch.ones(2), torch.zeros(2))
    res0 = parity.exact_result(rank=0, unresolved=[], expected_checked=71)
    assert len(res0["mismatched"]) == RaidenParityCheck.MAX_DETAIL_LINES
    assert len(res0["all_mismatched"]) == 71
    assert "w13_h0(gate)_absmean=" in res0["all_mismatched"][0]
    assert "w13_h1(up)_absmean=0(zero=1)" in res0["all_mismatched"][0]
    assert "staging_gate_absmean=1(zero=0)" in res0["all_mismatched"][0]
    assert "staging_up_absmean=0(zero=1)" in res0["all_mismatched"][0]

    with pytest.raises(RaidenParityError, match="71 mismatched"):
        RaidenParityCheck.report_exact(0, [[{"exact_parity": res0}]], dump_dir=str(tmp_path))

    import json

    archived = json.loads((tmp_path / "raiden_exact_parity_step_0.json").read_text())
    assert len(archived["ranks"][0]["all_mismatched"]) == 71

    # Duplicate rank within a replica, missing expected ranks, or non-0..W-1 rank IDs must fail closed.
    with pytest.raises(RaidenParityError, match="duplicate sampler"):
        RaidenParityCheck.report_exact(0, [[_exact(0), _exact(0)]], dump_dir=str(tmp_path))
    with pytest.raises(RaidenParityError, match="expected 4 sampler worker results"):
        RaidenParityCheck.report_exact(0, [[_exact(0), _exact(1)]], expected_ranks=4, dump_dir=str(tmp_path))
    with pytest.raises(RaidenParityError, match=r"replica 0 expected rank IDs 0\.\.31"):
        RaidenParityCheck.report_exact(
            0,
            [[_exact(r, checked=435) for r in range(100, 132)]],
            expected_ranks=32,
            expected_checked_per_rank=435,
            dump_dir=str(tmp_path),
        )
    with pytest.raises(RaidenParityError, match="non-uniform or zero tensors checked"):
        RaidenParityCheck.report_exact(0, [[_exact(0, checked=10), _exact(1, checked=9)]], dump_dir=str(tmp_path))


def test_report_exact_fails_closed_when_json_archive_fails(tmp_path, monkeypatch):
    # Point dump_dir at an existing regular file so os.makedirs(target_dir, exist_ok=True) fails.
    tmp_path.mkdir(parents=True, exist_ok=True)
    bad_dir = tmp_path / "not_a_dir"
    bad_dir.write_text("file")
    with pytest.raises(RaidenParityError, match=r"UNARCHIVED: failed to archive exact_parity JSON report"):
        RaidenParityCheck.report_exact(
            0,
            [[_exact(0), _exact(1)]],
            expected_ranks=2,
            expected_checked_per_rank=10,
            dump_dir=str(bad_dir),
        )
    # Also verify _resolve_dump_dir picks up VERL_NAP_RUN_DIR when dump_dir and RAIDEN_PARITY_DUMP_DIR are unset.
    monkeypatch.delenv("RAIDEN_PARITY_DUMP_DIR", raising=False)
    monkeypatch.setenv("VERL_NAP_RUN_DIR", str(tmp_path / "nap_run"))
    assert RaidenParityCheck._resolve_dump_dir(None) == str(tmp_path / "nap_run" / "env")


def test_check_exact_computes_staging_stats_only_on_mismatch(monkeypatch):
    class _SpyStaging:
        def __init__(self, t):
            self._t = t
            self.detach_calls = 0

        def detach(self):
            self.detach_calls += 1
            return self._t.detach()

    parity = RaidenParityCheck("exact")
    ref = torch.ones((2, 4, 6), dtype=torch.bfloat16)
    spy_gate = _SpyStaging(torch.ones((4, 4)))
    spy_up = _SpyStaging(torch.zeros((4, 4)))

    # 1. Exact match (diff == 0.0): staging_tensors must NOT be reduced or touched at all.
    parity.check_exact(
        "model.layers.0.mlp.experts.routed_experts.w13_weight",
        ref,
        ref.clone(),
        staging_tensors={"gate": spy_gate, "up": spy_up},
    )
    assert spy_gate.detach_calls == 0
    assert spy_up.detach_calls == 0
    assert parity.mismatched == []

    # 2. Mismatch (diff != 0.0): staging_tensors ARE reduced lazily.
    bad = ref.clone()
    bad[..., 3:] = 0.0
    parity.check_exact(
        "model.layers.0.mlp.experts.routed_experts.w13_weight",
        ref,
        bad,
        staging_tensors={"gate": spy_gate, "up": spy_up},
    )
    assert spy_gate.detach_calls == 1
    assert spy_up.detach_calls == 1
    assert "staging_gate_absmean=1(zero=0)" in parity.mismatched[-1]
    assert "staging_up_absmean=0(zero=1)" in parity.mismatched[-1]

    # 3. Opt-out via VERL_RAIDEN_DIAG_STAGING=0 skips staging reduction even on mismatch.
    monkeypatch.setenv("VERL_RAIDEN_DIAG_STAGING", "0")
    parity.check_exact(
        "model.layers.1.mlp.experts.routed_experts.w13_weight",
        ref,
        bad,
        staging_tensors={"gate": spy_gate, "up": spy_up},
    )
    assert spy_gate.detach_calls == 1
    assert spy_up.detach_calls == 1
    assert "staging_gate_absmean" not in parity.mismatched[-1]


@pytest.mark.parametrize(
    "install_results, match",
    [
        ([[_exact(0), _exact(1, mismatched=["w: max|diff|=1"])]], "1 mismatched"),
        ([[_exact(0, unresolved=["model.foo.weight"])]], "1 unresolved"),
        ([[{"h2d": 0.1}], []], "no sampler returned a result"),
        ([], "no sampler returned a result"),
    ],
)
def test_report_exact_raises(install_results, match):
    with pytest.raises(RaidenParityError, match=match):
        RaidenParityCheck.report_exact(0, install_results)


def _t(l1, numel, l2_sq=None, replicas=None):
    d = {"l1": l1, "numel": numel, "l2_sq": l1 * l1 if l2_sq is None else l2_sq}
    if replicas is not None:
        d["replicas"] = replicas
    return d


def _trainer(per_tensor):
    return {
        "per_tensor": per_tensor,
        "total_numel": sum(t["numel"] for t in per_tensor.values()),
        "l1_norm": sum(t["l1"] for t in per_tensor.values()),
        "l2_norm": sum(t["l2_sq"] for t in per_tensor.values()) ** 0.5,
    }


# Trainer view of three tensors: a TP-sharded weight, a GQA k_proj whose single head is replicated on both TP ranks,
# and an unsharded norm.
_TRAINER = _trainer(
    {
        "model.layers.0.mlp.down_proj.weight": _t(10.0, 8, l2_sq=50.0),
        "model.layers.0.self_attn.k_proj.weight": _t(3.0, 4),
        "model.norm.weight": _t(2.0, 4),
    }
)


def _sampler_workers(down_l1=(4.0, 6.0)):
    return [
        {
            "per_tensor": {
                "model.layers.0.mlp.down_proj.weight": _t(down_l1[i], 4, l2_sq=25.0, replicas=1),
                "model.layers.0.self_attn.k_proj.weight": _t(3.0, 4, replicas=2),
                "model.norm.weight": _t(2.0, 4, replicas=2),
            }
        }
        for i in range(2)
    ]


def test_compare_replica_norms_passes_for_sharded_gqa_and_replicated():
    assert RaidenParityCheck._compare_replica_norms("Step 1", _TRAINER, _sampler_workers()) is True


def test_compare_replica_norms_raises_on_mismatch():
    with pytest.raises(RaidenParityError, match=r"(?s)RAIDEN PARITY MISMATCH \| Step 1.*down_proj"):
        RaidenParityCheck._compare_replica_norms("Step 1", _TRAINER, _sampler_workers(down_l1=(4.0, 7.0)))


def test_compare_replica_norms_raises_on_missing_tensor():
    workers = _sampler_workers()
    for w in workers:
        del w["per_tensor"]["model.norm.weight"]
    with pytest.raises(RaidenParityError, match="numel"):
        RaidenParityCheck._compare_replica_norms("Step 1", _TRAINER, workers)


# ---------------------------------------------------------------------------------------------
# Fakes for the orchestrator (update_raiden_weights) and verify_norms
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _RaidenId:
    job_name: str
    job_replica_id: str
    data_name: str


class _Remote:
    """Mimics an actor method: ``.remote(...)`` returns an awaitable."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        async def _call():
            return self._fn(*args, **kwargs)

        return _call()


class _FakeServer:
    def __init__(self, replica_idx, num_workers, results):
        self.replica_idx = replica_idx
        self.num_workers = num_workers
        self.results = results
        self.calls = []
        self.global_steps = None
        self.collective_rpc = _Remote(self._collective_rpc)
        self.set_global_steps = _Remote(self._set_global_steps)

    def _collective_rpc(self, method, kwargs=None):
        self.calls.append((method, kwargs))
        return self.results.get(method, lambda: [None] * self.num_workers)()

    def _set_global_steps(self, step):
        self.global_steps = step


class _FakeController:
    def __init__(self, registered=()):
        self._lock = threading.Lock()
        self._registered_shards = {u: object() for u in registered}
        self.transfers = []

    def start_transfer(self, **kwargs):
        self.transfers.append(kwargs)

        class _Future:
            async def wait(self):
                return None

        return _Future()


class _FakeActorWG:
    def __init__(self, world_size, events):
        self.world_size = world_size
        self.events = events

    def update_weights(self, global_steps=None, mode=None):
        self.events.append(("update_weights", global_steps, mode))
        return None

    def release_raiden_sync_buffers(self):
        self.events.append(("release",))
        return None


class _FakeManager:
    def __init__(
        self,
        num_trainer,
        num_replicas,
        tp,
        verify_parity="off",
        register=True,
        results=None,
        expected_checked_per_rank=None,
        parity_dump_dir=None,
    ):
        self.events = []
        raiden_cfg = {"verify_parity": verify_parity, "parallelism": 4}
        if expected_checked_per_rank is not None:
            raiden_cfg["expected_checked_per_rank"] = expected_checked_per_rank
        if parity_dump_dir is not None:
            raiden_cfg["parity_dump_dir"] = parity_dump_dir
        self.config = SimpleNamespace(engine_kwargs={"raiden": raiden_cfg})
        self.actor_wg = _FakeActorWG(num_trainer, self.events)
        self.replicas = [
            SimpleNamespace(world_size=tp, server_handle=_FakeServer(i, tp, results or {})) for i in range(num_replicas)
        ]
        units = [_RaidenId("trainer", str(r), "weights") for r in range(num_trainer)]
        units += [
            _RaidenId(_sampler_job_name(i, num_replicas), str(k), "weights")
            for i in range(num_replicas)
            for k in range(tp)
        ]
        self.raiden_controller = _FakeController(units if register else units[:-1])

    async def abort_replicas(self):
        self.events.append(("abort",))

    async def resume_generation_replicas(self):
        self.events.append(("resume",))


@pytest.fixture
def stub_tpu_sync(monkeypatch):
    """Install the two tpu_sync modules update_raiden_weights imports, without the real (TPU-only) package."""
    pkg = types.ModuleType("tpu_sync")
    api = types.ModuleType("tpu_sync.api")
    common = types.ModuleType("tpu_sync.api.common")
    rpc = types.ModuleType("tpu_sync.rpc")
    controller = types.ModuleType("tpu_sync.rpc.raiden_controller")
    common.RaidenId = _RaidenId
    controller.RaidenMemoryType = SimpleNamespace(DRAM="DRAM")
    for name, mod in [
        ("tpu_sync", pkg),
        ("tpu_sync.api", api),
        ("tpu_sync.api.common", common),
        ("tpu_sync.rpc", rpc),
        ("tpu_sync.rpc.raiden_controller", controller),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


async def _no_sleep(*_args, **_kwargs):
    """Stands in for asyncio.sleep in the engine's polling loops so timeout paths run instantly."""
    return None


# ---------------------------------------------------------------------------------------------
# update_raiden_weights orchestration
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("num_replicas", [1, 3])
async def test_update_raiden_weights_single_transfer_over_all_replicas(stub_tpu_sync, num_replicas):
    install = {"install_raiden_weights": lambda: [{"total": 0.2, "h2d": 0.1, "sync": 0.0}] * 2}
    manager = _FakeManager(num_trainer=4, num_replicas=num_replicas, tp=2, results=install)

    metrics = await raiden.update_raiden_weights(manager, global_steps=5)

    # Exactly one transfer whose destination holds every TP worker of every replica, each under a unique job name.
    (transfer,) = manager.raiden_controller.transfers
    assert transfer["src_units"] == [_RaidenId("trainer", str(r), "weights") for r in range(4)]
    expected_jobs = ["sampler"] if num_replicas == 1 else [f"sampler{i}" for i in range(num_replicas)]
    assert transfer["dst_units"] == [_RaidenId(j, str(k), "weights") for j in expected_jobs for k in range(2)]
    assert len(set(transfer["dst_units"])) == 2 * num_replicas
    assert transfer["parallelism"] == 4
    assert transfer["req_id"] == "verl_step_5"

    # Every replica was initialized under its own job name, installed, and tagged with the new weight version.
    for i, replica in enumerate(manager.replicas):
        methods = [m for m, _ in replica.server_handle.calls]
        assert methods == ["init_raiden_sync_on_worker", "install_raiden_weights"]
        assert replica.server_handle.calls[0][1]["job_name"] == expected_jobs[i]
        assert replica.server_handle.global_steps == 5

    names = [e[0] for e in manager.events]
    # The samplers are quiesced concurrently with the trainer-side export, so abort and update_weights may
    # come in either order; both precede the transfer (release is requested right after it) and the resume.
    assert sorted(names[:2]) == ["abort", "update_weights"]
    assert names[2:] == ["release", "resume"]
    assert {
        "timing_s/tpu-sync/quiesce",
        "timing_s/tpu-sync/trainer_init",
        "timing_s/tpu-sync/sampler_init",
        "timing_s/tpu-sync/barrier",
        "timing_s/tpu-sync/p2p_transfer",
        "timing_s/tpu-sync/sampler_h2d",
        "timing_s/tpu-sync/total_sync",
    } <= set(metrics)
    assert metrics["timing_s/tpu-sync/sampler_h2d_pure"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_update_raiden_weights_skips_abort_on_first_sync(stub_tpu_sync):
    manager = _FakeManager(num_trainer=2, num_replicas=1, tp=2)
    await raiden.update_raiden_weights(manager, global_steps=0)
    assert ("abort",) not in manager.events
    assert manager.events[-1] == ("resume",)


@pytest.mark.asyncio
async def test_update_raiden_weights_barrier_timeout_lists_missing_units(stub_tpu_sync, monkeypatch):
    manager = _FakeManager(num_trainer=2, num_replicas=1, tp=2, register=False)
    clock = itertools.count(0, 100)  # every perf_counter() call advances 100 s: the 60 s barrier times out at once
    monkeypatch.setattr(raiden.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(raiden.asyncio, "sleep", _no_sleep)
    with pytest.raises(RuntimeError, match=r"Missing Sampler: \[.*job_replica_id='1'"):
        await raiden.update_raiden_weights(manager, global_steps=1)
    assert manager.raiden_controller.transfers == []
    assert ("resume",) not in manager.events


@pytest.mark.asyncio
async def test_update_raiden_weights_exact_parity_failure_does_not_resume(stub_tpu_sync):
    install = {"install_raiden_weights": lambda: [_exact(0), _exact(1, mismatched=["qkv: max|diff|=0.5"])]}
    manager = _FakeManager(num_trainer=2, num_replicas=1, tp=2, verify_parity="exact", results=install)

    with pytest.raises(RaidenParityError, match="1 mismatched"):
        await raiden.update_raiden_weights(manager, global_steps=0)

    # The exact check was requested from the samplers on the step-0 sync, and generation stays paused.
    assert manager.replicas[0].server_handle.calls[1] == ("install_raiden_weights", {"exact_parity": True})
    assert ("resume",) not in manager.events


@pytest.mark.asyncio
async def test_update_raiden_weights_exact_parity_pass_resumes(stub_tpu_sync, tmp_path):
    install = {"install_raiden_weights": lambda: [_exact(0, expected_checked=10), _exact(1, expected_checked=10)]}
    manager = _FakeManager(
        num_trainer=2,
        num_replicas=1,
        tp=2,
        verify_parity="exact",
        results=install,
        expected_checked_per_rank=10,
        parity_dump_dir=str(tmp_path),
    )
    await raiden.update_raiden_weights(manager, global_steps=0)
    assert manager.events[-1] == ("resume",)
    assert (tmp_path / "raiden_exact_parity_step_0.json").is_file()


@pytest.mark.asyncio
async def test_update_raiden_weights_exact_parity_rejects_checked_1_and_bad_rank_ids(stub_tpu_sync, tmp_path):
    # Negative test from the update_raiden_weights entry point: 32 ranks all returning checked=1 must fail
    # both via engine_kwargs.raiden.expected_checked_per_rank=435 and via worker expected_checked=435.
    install_checked_1 = {"install_raiden_weights": lambda: [_exact(r, checked=1) for r in range(32)]}
    manager_cfg = _FakeManager(
        num_trainer=32,
        num_replicas=1,
        tp=32,
        verify_parity="all",
        results=install_checked_1,
        expected_checked_per_rank=435,
        parity_dump_dir=str(tmp_path),
    )
    with pytest.raises(RaidenParityError, match=r"expected \[435\] tensors checked per rank, got \[1\]"):
        await raiden.update_raiden_weights(manager_cfg, global_steps=0)
    assert ("resume",) not in manager_cfg.events

    # Even without engine_kwargs.raiden.expected_checked_per_rank, worker expected_checked=435 rejects checked=1.
    install_worker_mismatch = {
        "install_raiden_weights": lambda: [_exact(r, checked=1, expected_checked=435) for r in range(32)]
    }
    manager_worker = _FakeManager(
        num_trainer=32,
        num_replicas=1,
        tp=32,
        verify_parity="all",
        results=install_worker_mismatch,
        parity_dump_dir=str(tmp_path),
    )
    with pytest.raises(RaidenParityError, match="checked 1 tensors, expected 435 installed model parameters"):
        await raiden.update_raiden_weights(manager_worker, global_steps=0)
    assert ("resume",) not in manager_worker.events

    # Negative test from the update_raiden_weights entry point: 32 ranks returning rank IDs 100..131 must fail.
    install_bad_ranks = {"install_raiden_weights": lambda: [_exact(r, checked=435) for r in range(100, 132)]}
    manager_bad_ranks = _FakeManager(
        num_trainer=32,
        num_replicas=1,
        tp=32,
        verify_parity="all",
        results=install_bad_ranks,
        expected_checked_per_rank=435,
        parity_dump_dir=str(tmp_path),
    )
    with pytest.raises(RaidenParityError, match=r"replica 0 expected rank IDs 0\.\.31"):
        await raiden.update_raiden_weights(manager_bad_ranks, global_steps=0)
    assert ("resume",) not in manager_bad_ranks.events


class _FakeRegistry:
    def __init__(self, stats):
        self.stats = stats
        self.get_stats = _Remote(lambda step: self.stats.get(step))


def _norm_manager(sampler_results):
    results = {"get_model_weights_stats": lambda: sampler_results}
    return _FakeManager(num_trainer=2, num_replicas=1, tp=2, verify_parity="norm", results=results)


@pytest.mark.asyncio
async def test_update_raiden_weights_norm_parity_failure_does_not_resume(stub_tpu_sync, monkeypatch):
    registry = _FakeRegistry({3: {"master": _TRAINER}})
    monkeypatch.setattr(raiden.ray, "get_actor", lambda *a, **k: registry)
    manager = _norm_manager(_sampler_workers(down_l1=(4.0, 7.0)))

    with pytest.raises(RaidenParityError, match="RAIDEN PARITY MISMATCH"):
        await raiden.update_raiden_weights(manager, global_steps=3)
    assert ("resume",) not in manager.events


@pytest.mark.asyncio
async def test_verify_norms_passes_with_sharded_trainer_stats(monkeypatch):
    # Sharded export: each trainer rank posts partial stats; the orchestrator sums them.
    half = {
        name: {"l1": t["l1"] / 2, "l2_sq": t["l2_sq"] / 2, "numel": t["numel"] // 2}
        for name, t in _TRAINER["per_tensor"].items()
    }
    registry = _FakeRegistry({3: {"ranks": {0: {"per_tensor": half}, 1: {"per_tensor": half}}}})
    monkeypatch.setattr(raiden.ray, "get_actor", lambda *a, **k: registry)
    await RaidenParityCheck("norm").verify_norms(_norm_manager(_sampler_workers()), global_steps=3)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stats, sampler_results, match",
    [
        ({}, None, "no complete trainer stats"),
        # Only one of the two trainer ranks posted its partial stats.
        ({3: {"ranks": {0: {"per_tensor": {}}}}}, None, "no complete trainer stats"),
        ({3: {"master": _TRAINER}}, [None, None], "no sampler stats from replica 0"),
    ],
)
async def test_verify_norms_raises_when_stats_are_missing(monkeypatch, stats, sampler_results, match):
    monkeypatch.setattr(raiden.ray, "get_actor", lambda *a, **k: _FakeRegistry(stats))
    monkeypatch.setattr(raiden.asyncio, "sleep", _no_sleep)
    manager = _norm_manager(sampler_results if sampler_results is not None else _sampler_workers())
    with pytest.raises(RaidenParityError, match=match):
        await RaidenParityCheck("norm").verify_norms(manager, global_steps=3)


@pytest.mark.asyncio
async def test_verify_norms_skips_step0():
    # The step-0 sync carries the checkpoint vLLM already loaded; the exact check covers it.
    await RaidenParityCheck("norm").verify_norms(manager=None, global_steps=0)
