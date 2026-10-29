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
"""CPU tests for the host order of placement groups in ``verl.single_controller.ray.base``.

``sort_placement_group_by_node_ip`` orders placement groups by node IP, or by the ``verl-tpu-node-rank`` Ray node
label when the node of every placement group carries an integer value. ``ray.nodes()`` and the placement group
table are faked, no Ray cluster is started.
"""

import logging
import random

import pytest
import ray
import ray._private.state
from ray._raylet import PlacementGroupID
from ray.util.placement_group import PlacementGroup

from verl.single_controller.ray import base as ray_base
from verl.single_controller.ray.base import (
    TPU_NODE_RANK_LABEL,
    RayResourcePool,
    RayWorkerGroup,
    sort_placement_group_by_node_ip,
)

# Topological ranks of the hosts of an 8-host TPU 7x slice listed in IP order: the worker order that the IP sort
# produced in the port where the mesh formed but the first FSDP collective halted.
IP_ORDER_RANKS = [1, 6, 5, 0, 2, 4, 7, 3]


class PlacedGroup(PlacementGroup):
    """A placement group that is already placed, so ``ready()`` needs no Ray cluster."""

    def ready(self):
        return None


class FakeCluster:
    """``ray.nodes()`` entries and placement group table of a fake Ray cluster."""

    def __init__(self):
        self.nodes: list[dict] = []
        self.table: dict = {}

    def add_node(self, ip: str, rank=None) -> str:
        """Add a node; unless ``rank`` is None, ``str(rank)`` becomes its ``verl-tpu-node-rank`` label."""
        node_id = f"{len(self.nodes) + 1:056x}"
        labels = {"ray.io/node-id": node_id}
        if rank is not None:
            labels[TPU_NODE_RANK_LABEL] = str(rank)
        self.nodes.append(
            {"NodeID": node_id, "NodeManagerAddress": ip, "Alive": True, "Labels": labels, "Resources": {"CPU": 8.0}}
        )
        return node_id

    def add_pg(self, bundles_to_node_id: dict, bundle_count: int = 1) -> PlacedGroup:
        """Add a placement group whose bundles sit on the given node ids ({} = unknown to the table)."""
        pg = PlacedGroup(PlacementGroupID.from_random(), bundle_cache=[{"CPU": 1.0}] * bundle_count)
        self.table[pg.id] = (
            {"state": "CREATED", "bundles_to_node_id": dict(bundles_to_node_id)} if bundles_to_node_id else {}
        )
        return pg

    def add_host(self, ip: str, rank=None, bundles: int = 1) -> PlacedGroup:
        """Add a node and a placement group with all of its bundles on that node (one per TPU host)."""
        node_id = self.add_node(ip, rank)
        return self.add_pg(dict.fromkeys(range(bundles), node_id), bundle_count=bundles)

    def placement_group_table(self, placement_group_id=None) -> dict:
        return self.table.get(placement_group_id, {})

    def ip_of(self, pg: PlacementGroup) -> str:
        bundles = self.table[pg.id]["bundles_to_node_id"]
        node_id = bundles[min(bundles)]
        return next(node["NodeManagerAddress"] for node in self.nodes if node["NodeID"] == node_id)

    def ips(self, pgs: list[PlacementGroup]) -> list[str]:
        return [self.ip_of(pg) for pg in pgs]


@pytest.fixture
def cluster(monkeypatch):
    fake = FakeCluster()
    monkeypatch.setattr(ray, "nodes", lambda: fake.nodes)
    monkeypatch.setattr(
        ray._private.state.GlobalState,
        "placement_group_table",
        lambda self, placement_group_id=None: fake.placement_group_table(placement_group_id),
    )
    return fake


def legacy_sort_by_node_ip(pgs: list[PlacementGroup]) -> list[PlacementGroup]:
    """``sort_placement_group_by_node_ip`` as it was before the topology label (tpu-main @ 5f2cbd4a)."""
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    pg_ip = {}
    for pg in pgs:
        specs = ray._private.state.state.placement_group_table(pg.id)
        bundles_map = specs.get("bundles_to_node_id", {})
        if bundles_map:
            min_b_idx = min(bundles_map.keys())
            node_id = bundles_map[min_b_idx]
            pg_ip[pg.id] = node_ip.get(node_id, "")
        else:
            pg_ip[pg.id] = ""
    return sorted(pgs, key=lambda pg: pg_ip[pg.id])


def sort_and_capture_log(pgs: list[PlacementGroup], caplog) -> tuple[list[PlacementGroup], logging.LogRecord]:
    """Sort ``pgs`` and return the result with the single order record logged by the call."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=ray_base.logger.name):
        ordered = sort_placement_group_by_node_ip(pgs)
    records = [r for r in caplog.records if r.getMessage().startswith("Placement group order by ")]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    return ordered, records[0]


def test_labels_on_all_nodes_give_rank_order(cluster, caplog):
    ips = [f"10.8.{i}.{20 + i}" for i in range(8)]  # IP string order == list order
    rank_of = dict(zip(ips, IP_ORDER_RANKS, strict=True))
    pgs = [cluster.add_host(ip, rank_of[ip], bundles=4) for ip in random.Random(0).sample(ips, len(ips))]
    assert [rank_of[ip] for ip in cluster.ips(legacy_sort_by_node_ip(pgs))] == IP_ORDER_RANKS

    ordered, record = sort_and_capture_log(pgs, caplog)

    assert [rank_of[ip] for ip in cluster.ips(ordered)] == list(range(8))
    assert record.levelno == logging.INFO
    message = record.getMessage()
    assert f"by Ray node label {TPU_NODE_RANK_LABEL}" in message
    rank0_ip = ips[IP_ORDER_RANKS.index(0)]
    rank0_node = next(node["NodeID"] for node in cluster.nodes if node["NodeManagerAddress"] == rank0_ip)
    assert f"[0] {rank0_ip} node={rank0_node[:8]} rank=0" in message
    assert "rank=n/a" not in message


def test_no_labels_keep_legacy_ip_order(cluster, caplog):
    a = cluster.add_node("10.0.0.25")
    b = cluster.add_node("10.0.0.3")
    c = cluster.add_node("10.0.0.100")
    d = cluster.add_node("10.0.1.7")
    pgs = [
        cluster.add_pg({0: b, 1: b}),
        cluster.add_pg({3: d, 2: c}),  # the lowest bundle index decides: c
        cluster.add_pg({0: a}),
        cluster.add_pg({0: ""}),  # bundle not placed yet
        cluster.add_pg({}),  # unknown to the placement group table
        cluster.add_pg({0: "f" * 56}),  # node missing from ray.nodes()
        cluster.add_pg({0: c}),  # same node as pgs[1]: stays after it
        cluster.add_pg({0: d}),
    ]

    ordered, record = sort_and_capture_log(pgs, caplog)

    # Node IPs compare as strings: "" < "10.0.0.100" < "10.0.0.25" < "10.0.0.3" < "10.0.1.7".
    assert ordered == [pgs[3], pgs[4], pgs[5], pgs[1], pgs[6], pgs[2], pgs[0], pgs[7]]
    assert ordered == legacy_sort_by_node_ip(pgs)
    assert record.levelno == logging.INFO
    assert record.getMessage().startswith("Placement group order by node ip: [0] ? node=n/a rank=n/a")


@pytest.mark.parametrize("seed", range(25))
def test_no_labels_match_legacy_order_randomized(cluster, seed):
    rng = random.Random(seed)
    node_ids = [
        cluster.add_node(f"10.{rng.randint(0, 2)}.{rng.randint(0, 12)}.{rng.randint(0, 255)}")
        for _ in range(rng.randint(1, 10))
    ]
    pgs = []
    for _ in range(rng.randint(0, 16)):
        bundle_indices = rng.sample(range(8), rng.randint(0, 4))
        pgs.append(cluster.add_pg({idx: rng.choice(node_ids + [""]) for idx in bundle_indices}))
    if pgs and rng.random() < 0.5:
        pgs.append(rng.choice(pgs))  # the same placement group twice

    assert sort_placement_group_by_node_ip(pgs) == legacy_sort_by_node_ip(pgs)


@pytest.mark.parametrize("unlabeled", ["node_without_label", "placement_group_not_placed"])
def test_partial_labels_fall_back_to_ip_order_with_warning(cluster, caplog, unlabeled):
    pgs = [cluster.add_host("10.0.0.4", 0), cluster.add_host("10.0.0.3", 2), cluster.add_host("10.0.0.1", 3)]
    if unlabeled == "node_without_label":
        pgs.append(cluster.add_host("10.0.0.2"))
    else:
        pgs.append(cluster.add_pg({0: ""}))

    ordered, record = sort_and_capture_log(pgs, caplog)

    assert ordered == legacy_sort_by_node_ip(pgs)
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert (
        f"by node ip (Ray node label {TPU_NODE_RANK_LABEL} ignored, missing on 1 of 4 placement group nodes" in message
    )
    assert "not a non-negative integer" not in message
    # The log still shows the ranks that are set, so the unlabeled host stands out.
    assert message.count("rank=n/a") == 1 and "rank=0" in message


@pytest.mark.parametrize("bad_value", ["abc", "1.5", "-1", "", "0x2", "\u00b3"])
def test_non_integer_label_falls_back_to_ip_order(cluster, caplog, bad_value):
    pgs = [cluster.add_host("10.0.0.1", 2), cluster.add_host("10.0.0.3", bad_value), cluster.add_host("10.0.0.2", 0)]

    ordered, record = sort_and_capture_log(pgs, caplog)

    assert cluster.ips(ordered) == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    assert ordered == legacy_sort_by_node_ip(pgs)
    assert record.levelno == logging.WARNING
    assert f"not a non-negative integer: [{bad_value!r}]" in record.getMessage()


def test_duplicate_ranks_are_tie_broken_by_ip(cluster, caplog):
    pgs = [
        cluster.add_host("10.0.0.9", 1),
        cluster.add_host("10.0.0.12", 0),
        cluster.add_host("10.0.0.7", 1),
        cluster.add_host("10.0.0.10", 0),
        cluster.add_host("10.0.0.11", 2),
    ]

    ordered, record = sort_and_capture_log(pgs, caplog)

    assert cluster.ips(ordered) == ["10.0.0.10", "10.0.0.12", "10.0.0.7", "10.0.0.9", "10.0.0.11"]
    assert sort_placement_group_by_node_ip(list(reversed(pgs))) == ordered
    assert record.levelno == logging.WARNING
    assert "rank(s) [0, 1] are carried by more than one node" in record.getMessage()


def test_placement_groups_on_the_same_node_keep_input_order(cluster, caplog):
    shared = cluster.add_node("10.0.0.5", 1)
    first, second = cluster.add_pg({0: shared}), cluster.add_pg({0: shared})
    other = cluster.add_host("10.0.0.6", 0)

    ordered, record = sort_and_capture_log([first, second, other], caplog)

    assert ordered == [other, first, second]
    assert sort_placement_group_by_node_ip([second, first, other]) == [other, second, first]
    assert record.levelno == logging.INFO


def test_trainer_pool_and_worker_ranks_follow_label_order(cluster, monkeypatch):
    rank_of = {"10.8.0.20": 1, "10.8.1.21": 3, "10.8.2.22": 0, "10.8.3.23": 2}
    pgs = [cluster.add_host(ip, rank, bundles=2) for ip, rank in rank_of.items()]
    created = iter(pgs)
    monkeypatch.setattr(ray_base, "placement_group", lambda **kwargs: next(created))
    monkeypatch.setattr(ray, "get", lambda refs: refs)
    workers = []
    monkeypatch.setattr(RayWorkerGroup, "_get_master_addr_port", lambda self, pg, **kwargs: None)
    monkeypatch.setattr(RayWorkerGroup, "_create_worker", lambda self, **kwargs: workers.append(kwargs))

    pool = RayResourcePool([2, 2, 2, 2], use_gpu=False, name_prefix="trainer_pool")
    pool.get_placement_groups()
    RayWorkerGroup(resource_pool=pool, ray_cls_with_init=None, name_prefix="trainer")

    # The pool's placement groups feed TPU_WORKER_HOSTNAMES / TORCH_TPU_SLICEBUILDER_ADDRESSES.
    assert [rank_of[ip] for ip in cluster.ips(pool.get_placement_groups())] == [0, 1, 2, 3]
    assert [(w["rank"], rank_of[cluster.ip_of(w["pg"])], w["local_rank"]) for w in workers] == [
        (rank, rank // 2, rank % 2) for rank in range(8)
    ]
