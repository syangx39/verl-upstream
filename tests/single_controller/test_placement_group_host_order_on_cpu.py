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
    TPU_SLICE_NAME_LABEL,
    TPU_WORKER_ID_LABEL,
    RayResourcePool,
    RayWorkerGroup,
    sort_placement_group_by_node_ip,
    split_resource_pool,
)

# Topological ranks of the hosts of an 8-host TPU 7x slice listed in IP order: the worker order that the IP sort
# produced in the port where the mesh formed but the first FSDP collective halted.
IP_ORDER_RANKS = [1, 6, 5, 0, 2, 4, 7, 3]

# (ip, rank) of four hosts in placement group creation order; creation, IP and rank order all differ.
FRESH_POOL_HOSTS = [("10.8.3.23", 2), ("10.8.0.20", 1), ("10.8.2.22", 3), ("10.8.1.21", 0)]
FRESH_POOL_IP_ORDER = ["10.8.0.20", "10.8.1.21", "10.8.2.22", "10.8.3.23"]
FRESH_POOL_RANK_ORDER = ["10.8.1.21", "10.8.0.20", "10.8.3.23", "10.8.2.22"]


class PlacedGroup(PlacementGroup):
    """A placement group that is already placed, so ``ready()`` needs no Ray cluster."""

    def ready(self):
        return None


class FakeCluster:
    """``ray.nodes()`` entries and placement group table of a fake Ray cluster."""

    def __init__(self):
        self.nodes: list[dict] = []
        self.table: dict = {}
        self.slice_of: dict[str, str] = {}  # node id -> slice name used by the test

    def add_node(self, ip: str, rank=None, worker_id=None, slice_group=None, slice_name=None, resources=None) -> str:
        """Add a node; ``str(rank)`` / ``str(worker_id)`` become its rank / ``ray.io/tpu-worker-id`` labels.

        ``slice_group=k`` adds the slice resource ``tpu-group-<k>``, ``slice_name`` the ``ray.io/tpu-slice-name``
        label, and ``resources`` further resources.
        """
        node_id = f"{len(self.nodes) + 1:056x}"
        labels = {"ray.io/node-id": node_id}
        node_resources = {"CPU": 8.0, **(resources or {})}
        if rank is not None:
            labels[TPU_NODE_RANK_LABEL] = str(rank)
        if worker_id is not None:
            labels[TPU_WORKER_ID_LABEL] = str(worker_id)
        if slice_group is not None:
            node_resources[f"tpu-group-{slice_group}"] = 1.0
        if slice_name is not None:
            labels[TPU_SLICE_NAME_LABEL] = slice_name
        self.slice_of[node_id] = f"tpu-group-{slice_group}" if slice_group is not None else slice_name
        self.nodes.append(
            {"NodeID": node_id, "NodeManagerAddress": ip, "Alive": True, "Labels": labels, "Resources": node_resources}
        )
        return node_id

    def add_pg(self, bundles_to_node_id: dict, bundle_count: int = 1) -> PlacedGroup:
        """Add a placement group whose bundles sit on the given node ids ({} = unknown to the table)."""
        pg = PlacedGroup(PlacementGroupID.from_random(), bundle_cache=[{"CPU": 1.0}] * bundle_count)
        self.table[pg.id] = (
            {"state": "CREATED", "bundles_to_node_id": dict(bundles_to_node_id)} if bundles_to_node_id else {}
        )
        return pg

    def add_host(self, ip: str, rank=None, bundles: int = 1, **node_kwargs) -> PlacedGroup:
        """Add a node (see ``add_node``) and a placement group with all of its bundles on it (one per TPU host)."""
        node_id = self.add_node(ip, rank, **node_kwargs)
        return self.add_pg(dict.fromkeys(range(bundles), node_id), bundle_count=bundles)

    def placement_group_table(self, placement_group_id=None) -> dict:
        return self.table.get(placement_group_id, {})

    def node_id_of(self, pg: PlacementGroup) -> str:
        bundles = self.table[pg.id]["bundles_to_node_id"]
        return bundles[min(bundles)]

    def ip_of(self, pg: PlacementGroup) -> str:
        node_id = self.node_id_of(pg)
        return next(node["NodeManagerAddress"] for node in self.nodes if node["NodeID"] == node_id)

    def ips(self, pgs: list[PlacementGroup]) -> list[str]:
        return [self.ip_of(pg) for pg in pgs]

    def slices_and_ranks(self, pgs: list[PlacementGroup]) -> list[tuple]:
        """(slice, rank label) of the node of each placement group."""
        labels = {node["NodeID"]: node["Labels"] for node in self.nodes}
        return [
            (self.slice_of[self.node_id_of(pg)], int(labels[self.node_id_of(pg)][TPU_NODE_RANK_LABEL])) for pg in pgs
        ]


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


def previous_order(pgs: list[PlacementGroup]) -> list[PlacementGroup]:
    """Order of ``sort_placement_group_by_node_ip`` before slices were considered (8c37b61b): (rank, ip), else ip."""
    node_by_id = {node["NodeID"]: node for node in ray.nodes()}
    ip, rank = {}, {}
    for pg in pgs:
        bundles = ray._private.state.state.placement_group_table(pg.id).get("bundles_to_node_id", {})
        node = node_by_id.get(bundles[min(bundles)]) if bundles else None
        ip[pg.id] = node["NodeManagerAddress"] if node else ""
        text = str((node or {}).get("Labels", {}).get(TPU_NODE_RANK_LABEL, "")).strip()
        rank[pg.id] = int(text) if text.isascii() and text.isdigit() else None
    if pgs and all(rank[pg.id] is not None for pg in pgs):
        return sorted(pgs, key=lambda pg: (rank[pg.id], ip[pg.id]))
    return sorted(pgs, key=lambda pg: ip[pg.id])


def sort_and_capture_log(pgs: list[PlacementGroup], caplog) -> tuple[list[PlacementGroup], logging.LogRecord]:
    """Sort ``pgs`` and return the result with the single order record logged by the call."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=ray_base.logger.name):
        ordered = sort_placement_group_by_node_ip(pgs)
    records = [r for r in caplog.records if r.getMessage().startswith("Placement group order by ")]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    return ordered, records[0]


def make_fresh_pool(cluster: FakeCluster, monkeypatch, labeled: bool, bundles: int = 2) -> RayResourcePool:
    """A RayResourcePool whose placement groups are created on FRESH_POOL_HOSTS, in that order, when first asked."""
    pgs = [cluster.add_host(ip, rank if labeled else None, bundles=bundles) for ip, rank in FRESH_POOL_HOSTS]
    created = iter(pgs)
    monkeypatch.setattr(ray_base, "placement_group", lambda **kwargs: next(created))
    monkeypatch.setattr(ray, "get", lambda refs: refs)
    return RayResourcePool([bundles] * len(pgs), use_gpu=False, name_prefix="fresh_pool")


def record_workers(monkeypatch) -> list[dict]:
    """Make RayWorkerGroup record the arguments of each worker it would create instead of starting actors."""
    workers = []
    monkeypatch.setattr(RayWorkerGroup, "_get_master_addr_port", lambda self, pg, **kwargs: None)
    monkeypatch.setattr(RayWorkerGroup, "_create_worker", lambda self, **kwargs: workers.append(kwargs))
    return workers


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


@pytest.mark.parametrize("labeled", [True, False], ids=["rank_labels", "no_labels"])
def test_first_and_later_get_placement_groups_calls_agree(cluster, monkeypatch, labeled):
    pool = make_fresh_pool(cluster, monkeypatch, labeled)

    first = pool.get_placement_groups()
    later = pool.get_placement_groups()

    assert cluster.ips(first) == (FRESH_POOL_RANK_ORDER if labeled else FRESH_POOL_IP_ORDER)
    assert later == first


@pytest.mark.parametrize("labeled", [True, False], ids=["rank_labels", "no_labels"])
def test_split_of_fresh_pool_follows_host_order(cluster, monkeypatch, labeled):
    pool = make_fresh_pool(cluster, monkeypatch, labeled)
    workers = record_workers(monkeypatch)

    sub_pools = split_resource_pool(pool, split_size=2)  # one sub-pool per host
    for idx, sub_pool in enumerate(sub_pools):
        RayWorkerGroup(resource_pool=sub_pool, ray_cls_with_init=None, name_prefix=f"replica_{idx}")

    expected = FRESH_POOL_RANK_ORDER if labeled else FRESH_POOL_IP_ORDER
    assert [cluster.ip_of(sub_pool.pgs[sub_pool.start_bundle_index // 2]) for sub_pool in sub_pools] == expected
    assert all(sub_pool.pgs == pool.get_placement_groups() for sub_pool in sub_pools)
    # Each replica's two workers sit on its host.
    assert [cluster.ip_of(w["pg"]) for w in workers] == [ip for ip in expected for _ in range(2)]


def test_log_shows_tpu_worker_id_next_to_rank(cluster, caplog):
    pgs = [
        cluster.add_host("10.0.0.2", 1, worker_id=1),
        cluster.add_host("10.0.0.1", 0, worker_id=0),
        cluster.add_host("10.0.0.3", 2),  # no ray.io/tpu-worker-id label
    ]

    ordered, record = sort_and_capture_log(pgs, caplog)

    assert cluster.ips(ordered) == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    assert record.levelno == logging.INFO
    message = record.getMessage()
    assert "10.0.0.1 node=" in message and "rank=0 tpu-worker-id=0, [1] 10.0.0.2" in message
    assert message.endswith("rank=2 tpu-worker-id=n/a")


def test_rank_differing_from_tpu_worker_id_warns_without_reordering(cluster, caplog):
    hosts = [("10.0.0.4", 0, 0), ("10.0.0.3", 1, 2), ("10.0.0.2", 2, 1), ("10.0.0.1", 3, 3)]  # (ip, rank, worker id)
    pgs = [cluster.add_host(ip, rank, worker_id=worker_id) for ip, rank, worker_id in hosts]

    ordered, record = sort_and_capture_log(pgs, caplog)

    assert cluster.ips(ordered) == ["10.0.0.4", "10.0.0.3", "10.0.0.2", "10.0.0.1"]  # by rank
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert (
        f"(Ray node labels {TPU_NODE_RANK_LABEL} and {TPU_WORKER_ID_LABEL} differ on 2 of 4 placement group nodes)"
        in message
    )
    assert "rank=1 tpu-worker-id=2" in message and "rank=2 tpu-worker-id=1" in message


def test_tpu_worker_id_without_rank_labels_is_only_logged(cluster, caplog):
    pgs = [cluster.add_host("10.0.0.2", worker_id=0), cluster.add_host("10.0.0.1", worker_id=1)]

    ordered, record = sort_and_capture_log(pgs, caplog)

    assert ordered == legacy_sort_by_node_ip(pgs)
    assert record.levelno == logging.INFO
    assert "rank=n/a tpu-worker-id=1, [1] 10.0.0.2" in record.getMessage()


@pytest.mark.parametrize("seed", range(10))
def test_tpu_worker_id_never_changes_the_order(cluster, seed):
    rng = random.Random(seed)
    labeled = rng.random() < 0.5
    pgs = [
        cluster.add_host(
            f"10.0.{rng.randint(0, 3)}.{rng.randint(0, 255)}",
            rank=rng.randint(0, 7) if labeled else None,
            worker_id=rng.choice([None, rng.randint(0, 7)]),
        )
        for _ in range(rng.randint(1, 8))
    ]

    with_worker_ids = sort_placement_group_by_node_ip(pgs)
    for node in cluster.nodes:
        node["Labels"].pop(TPU_WORKER_ID_LABEL, None)

    assert sort_placement_group_by_node_ip(pgs) == with_worker_ids


@pytest.mark.parametrize(
    "first, second",
    [({"slice_group": 2}, {"slice_group": 10}), ({"slice_name": "slice-a"}, {"slice_name": "slice-b"})],
    ids=["tpu_group_resource", "tpu_slice_name_label"],
)
def test_two_slices_are_kept_contiguous_in_slice_order(cluster, caplog, first, second):
    # Ranks 0..3 in each slice; node IPs alternate between the slices and creation order is scrambled.
    hosts = [(f"10.9.0.{10 + 2 * rank}", rank, second) for rank in range(4)]
    hosts += [(f"10.9.0.{11 + 2 * rank}", rank, first) for rank in range(4)]
    pgs = [cluster.add_host(ip, rank, **kwargs) for ip, rank, kwargs in random.Random(1).sample(hosts, len(hosts))]
    first_name = f"tpu-group-{first['slice_group']}" if "slice_group" in first else first["slice_name"]
    second_name = f"tpu-group-{second['slice_group']}" if "slice_group" in second else second["slice_name"]
    # Without slices, (rank, ip) interleaves them.
    assert [name for name, _ in cluster.slices_and_ranks(previous_order(pgs))] == [second_name, first_name] * 4

    ordered, record = sort_and_capture_log(pgs, caplog)

    expected = [(first_name, rank) for rank in range(4)] + [(second_name, rank) for rank in range(4)]
    assert cluster.slices_and_ranks(ordered) == expected  # tpu-group indices compare as numbers: 2 before 10
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert message.startswith(f"Placement group order by TPU slice, then Ray node label {TPU_NODE_RANK_LABEL}")
    assert f"the pool spans 2 TPU slices {[first_name, second_name]}: each slice is kept contiguous" in message
    assert "multi-slice pools are not validated on TPU" in message
    assert "carried by more than one node" not in message  # ranks repeat across slices only
    assert f"slice={first_name} rank=0" in message and f"slice={second_name} rank=3" in message


def test_tpu_group_slices_come_before_named_slices(cluster):
    pgs = [
        cluster.add_host("10.0.0.1", 0, slice_name="a-slice"),
        cluster.add_host("10.0.0.2", 1, slice_name="a-slice"),
        cluster.add_host("10.0.0.3", 1, slice_group=5),
        cluster.add_host("10.0.0.4", 0, slice_group=5),
    ]

    ordered = sort_placement_group_by_node_ip(pgs)

    assert cluster.slices_and_ranks(ordered) == [("tpu-group-5", 0), ("tpu-group-5", 1), ("a-slice", 0), ("a-slice", 1)]


@pytest.mark.parametrize("without_slice", ["no_slice_resource_or_label", "two_tpu_group_resources"])
def test_missing_slice_keeps_rank_order_and_warns(cluster, caplog, without_slice):
    pgs = [
        cluster.add_host("10.0.0.1", 0, slice_group=0),
        cluster.add_host("10.0.0.2", 0, slice_group=1),
        cluster.add_host("10.0.0.3", 1, slice_group=0),
    ]
    if without_slice == "no_slice_resource_or_label":
        pgs.append(cluster.add_host("10.0.0.4", 1))
    else:
        pgs.append(cluster.add_host("10.0.0.4", 1, resources={"tpu-group-0": 1.0, "tpu-group-1": 1.0}))

    ordered, record = sort_and_capture_log(pgs, caplog)

    # Today's (rank, ip) order, which interleaves the slices.
    assert ordered == previous_order(pgs)
    assert cluster.ips(ordered) == ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert message.startswith(f"Placement group order by Ray node label {TPU_NODE_RANK_LABEL}, then node ip (")
    assert "rank(s) [0, 1] are carried by more than one node" in message
    assert (
        f"no TPU slice (tpu-group-<k> resource or {TPU_SLICE_NAME_LABEL} label) on 1 of 4 placement group nodes, "
        "so hosts of different slices may be interleaved" in message
    )
    assert message.endswith("10.0.0.4 node=00000000 slice=n/a rank=1 tpu-worker-id=n/a")


@pytest.mark.parametrize("seed", range(40))
def test_single_slice_orders_unchanged_randomized(cluster, caplog, seed):
    rng = random.Random(seed)
    slice_kwargs = rng.choice([{}, {"slice_group": rng.randint(0, 20)}, {"slice_name": f"slice-{rng.randint(0, 9)}"}])
    labels = rng.choice(["all", "partial", "invalid", "none"])
    pgs = []
    for _ in range(rng.randint(1, 9)):
        rank = rng.randint(0, 5)  # may repeat inside the slice
        if labels == "none" or (labels == "partial" and rng.random() < 0.3):
            rank = None
        elif labels == "invalid" and rng.random() < 0.3:
            rank = rng.choice(["abc", "-1", "", "1.5"])
        ip = f"10.{rng.randint(0, 1)}.{rng.randint(0, 3)}.{rng.randint(0, 255)}"
        pgs.append(cluster.add_host(ip, rank, worker_id=rng.choice([None, rng.randint(0, 5)]), **slice_kwargs))
    if rng.random() < 0.3:
        pgs.append(rng.choice(pgs))  # the same placement group twice

    ordered, record = sort_and_capture_log(pgs, caplog)

    assert ordered == previous_order(pgs)
    assert "TPU slices" not in record.getMessage()
    assert "interleaved" not in record.getMessage() or "carried by more than one node" in record.getMessage()


def test_slice_resource_ignores_placement_group_resources(cluster, caplog):
    pg_resources = {"tpu-group-3_group_0123abcd": 0.0001, "tpu-group-3_group_0_0123abcd": 0.0001}
    pgs = [
        cluster.add_host("10.0.0.2", 0, slice_group=3, resources=pg_resources),
        cluster.add_host("10.0.0.1", 1, slice_group=3, resources=pg_resources),
    ]

    ordered, record = sort_and_capture_log(pgs, caplog)

    assert cluster.ips(ordered) == ["10.0.0.2", "10.0.0.1"]
    assert record.levelno == logging.INFO
    assert "slice=tpu-group-3 rank=0" in record.getMessage()


def test_ip_order_of_a_pool_spanning_slices_warns(cluster, caplog):
    pgs = [cluster.add_host(f"10.0.0.{i}", slice_group=i % 2) for i in range(1, 5)]

    ordered, record = sort_and_capture_log(pgs, caplog)

    assert ordered == legacy_sort_by_node_ip(pgs)
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert message.startswith(
        "Placement group order by node ip (the pool spans 2 TPU slices ['tpu-group-0', 'tpu-group-1'], which node ip "
        "order may interleave; multi-slice pools are not validated on TPU)"
    )
    assert "slice=tpu-group-1 rank=n/a" in message
