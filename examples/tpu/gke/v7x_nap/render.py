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
"""Render the per-run Ray JobSet for the shared GKE TPU 7x NAP cluster, and its pre-warm placeholders.

Templates (Python ``string.Template``) live next to this package in ``examples/tpu/gke/``:

* ``ray-jobset-v7x-nap.yaml``: the JobSet with the CPU ``head`` replicatedJob; ``${TPU_REPLICATED_JOBS}`` receives
  one rendering of ``ray-jobset-v7x-nap-slice.yaml`` per TPU slice (trainer slices first, then rollout slices).
* ``ray-jobset-v7x-nap-prewarm.yaml``: one placeholder JobSet per TPU slice that only holds the slice's chips, so
  that node auto-provisioning creates the node pools before the real job is applied (see the launcher's README).

Every value comes from the environment (``run_ray_jobset_v7x_nap.sh`` exports them); this module only derives the
slice layout (hosts per slice, placement policy, Ray slice resources) and checks the result against the rules of
the cluster. Standard library only, so the launcher runs with any ``python3``.

Usage: ``render.py manifest|prewarm --out PATH`` (prints the slice plan as JSON on stdout).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

TEMPLATE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_TEMPLATE = TEMPLATE_DIR / "ray-jobset-v7x-nap.yaml"
SLICE_TEMPLATE = TEMPLATE_DIR / "ray-jobset-v7x-nap-slice.yaml"
PREWARM_TEMPLATE = TEMPLATE_DIR / "ray-jobset-v7x-nap-prewarm.yaml"

# A tpu7x-standard-4t node (one TPU 7x host) has 4 chips; every chip exposes 2 TensorCores as 2 devices.
CHIPS_PER_HOST = 4
TENSORCORES_PER_CHIP = 2
# JobSet pod hostnames are `<jobset>-<replicatedJob>-<job index>-<pod index>` and must stay a DNS label (<= 63).
MAX_RUN_NAME_LEN = 30
DNS_LABEL = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
TOPOLOGY = re.compile(r"^[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*$")
# Keys that must be set (non-empty) in the environment for each template.
MANIFEST_KEYS = (
    "RUN_NAME",
    "RAY_NAMESPACE",
    "KUEUE_QUEUE",
    "PRIORITY_CLASS",
    "SERVICE_ACCOUNT",
    "RUN_OWNER",
    "CODE_SHA",
    "LAUNCH_ID",
    "IMAGE",
    "POD_TPU_ACCELERATOR_TYPE",
    "GCS_BUCKET",
    "GCSFUSE_MOUNT_OPTIONS",
    "DATA_MOUNT",
    "RUN_DIR",
    "SRC_TARBALL",
    "HEAD_NODEPOOL",
    "HEAD_NUM_CPUS",
    "HEAD_OBJECT_STORE_BYTES",
    "HEAD_DSHM_SIZE",
    "TPU_DSHM_SIZE",
    "TPU_TOPOLOGY",
    "RAY_TPU_PER_HOST",
    "JOBSET_MAX_RESTARTS",
    "JOBSET_TTL_SECONDS",
    "NODE_WAIT_TIMEOUT_S",
    "NODE_RANK_SOURCE",
    "NODE_RANK_TIMEOUT_S",
    "DRIVER_TIMEOUT_S",
    "REQUIRE_NODE_RANK",
)
PREWARM_KEYS = (
    "RUN_NAME",
    "RAY_NAMESPACE",
    "KUEUE_QUEUE",
    "PRIORITY_CLASS",
    "IMAGE",
    "TPU_TOPOLOGY",
    "PREWARM_HOLD_S",
    "LAUNCH_ID",
)


@dataclass(frozen=True)
class SlicePlan:
    """One TPU slice of the run: its replicatedJob name, role and Ray slice resource index."""

    job: str
    role: str
    index: int

    @property
    def ray_resource(self) -> str:
        """Ray resource that every host of the slice registers (verl's ``tpu-group-<k>`` convention)."""
        return f"tpu-group-{self.index}"


def topology_chips(topology: str) -> int:
    """Number of TPU chips of a slice topology such as ``2x2x4``."""
    if not TOPOLOGY.match(topology):
        raise ValueError(f"TPU topology {topology!r} is not of the form AxBxC")
    a, b, c = (int(x) for x in topology.split("x"))
    return a * b * c


def hosts_per_slice(topology: str) -> int:
    """Number of TPU 7x hosts (pods) of one slice."""
    chips = topology_chips(topology)
    if chips % CHIPS_PER_HOST:
        raise ValueError(f"TPU topology {topology} has {chips} chips, not a multiple of {CHIPS_PER_HOST} per host")
    return chips // CHIPS_PER_HOST


def placement_policy(topology: str) -> str:
    """Placement policy node auto-provisioning needs to build a multi-host slice; empty for single-host slices.

    The cluster names them ``tpu7x-<TensorCores>-<topology>-placement-policy`` (e.g. ``tpu7x-32-2x2x4-...``).
    """
    if hosts_per_slice(topology) == 1:
        return ""
    return f"tpu7x-{topology_chips(topology) * TENSORCORES_PER_CHIP}-{topology}-placement-policy"


def plan_slices(trainer_slices: int, rollout_slices: int) -> list[SlicePlan]:
    """Trainer slices take ``tpu-group-0..T-1`` and rollout slices ``tpu-group-T..T+R-1``."""
    if trainer_slices < 1 or rollout_slices < 0:
        raise ValueError(f"need >= 1 trainer slice and >= 0 rollout slices, got {trainer_slices}/{rollout_slices}")
    plan = [SlicePlan(f"trainer-{i}", "trainer", i) for i in range(trainer_slices)]
    plan += [SlicePlan(f"rollout-{i}", "rollout", trainer_slices + i) for i in range(rollout_slices)]
    return plan


def _strip_leading_comments(text: str) -> str:
    """Drop the documentation header of a snippet template (it would land mid-manifest)."""
    lines = text.splitlines(keepends=True)
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("#")):
        lines.pop(0)
    return "".join(lines)


def _require(values: Mapping[str, str], keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if not str(values.get(k, "")).strip()]
    if missing:
        raise ValueError(f"missing values: {', '.join(missing)}")


def check_run_name(name: str) -> None:
    """The run name is the JobSet name: a DNS label of at most ``MAX_RUN_NAME_LEN`` characters."""
    if len(name) > MAX_RUN_NAME_LEN or not DNS_LABEL.match(name):
        raise ValueError(
            f"RUN_NAME {name!r} must be a DNS label (lowercase letters, digits, '-') of <= {MAX_RUN_NAME_LEN} chars"
        )


def _slice_values(values: Mapping[str, str], sl: SlicePlan, topology: str) -> dict[str, str]:
    policy = placement_policy(topology)
    multi_host = bool(policy)
    pod_annotation = "                alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool"
    selector = f'                cloud.google.com/placement-policy-name: "{policy}"'
    return {
        **values,
        "SLICE_JOB": sl.job,
        "SLICE_ROLE": sl.role,
        "SLICE_INDEX": str(sl.index),
        "SLICE_RAY_RESOURCE": sl.ray_resource,
        "HOSTS_PER_SLICE": str(hosts_per_slice(topology)),
        "CHIPS_PER_HOST": str(CHIPS_PER_HOST),
        "SLICE_EXTRA_POD_ANNOTATIONS": pod_annotation if multi_host else "",
        "SLICE_EXTRA_NODE_SELECTOR": selector if multi_host else "",
    }


def _slice_counts(values: Mapping[str, str]) -> tuple[int, int]:
    return int(values.get("TRAINER_SLICES", "1")), int(values.get("ROLLOUT_SLICES", "1"))


def expected_slices(values: Mapping[str, str]) -> str:
    """``tpu-group-<k>=<hosts>`` pairs the head waits for (comma separated)."""
    topology = values["TPU_TOPOLOGY"]
    hosts = hosts_per_slice(topology)
    return ",".join(f"{sl.ray_resource}={hosts}" for sl in plan_slices(*_slice_counts(values)))


def render_manifest(values: Mapping[str, str]) -> str:
    """Render the JobSet (head plus one replicatedJob per TPU slice)."""
    _require(values, MANIFEST_KEYS)
    check_run_name(values["RUN_NAME"])
    topology = values["TPU_TOPOLOGY"]
    snippet = string.Template(_strip_leading_comments(SLICE_TEMPLATE.read_text()))
    jobs = [snippet.substitute(_slice_values(values, sl, topology)) for sl in plan_slices(*_slice_counts(values))]
    full = {
        **values,
        "EXPECTED_SLICES": expected_slices(values),
        "WORKER_NUM_CPUS": values.get("WORKER_NUM_CPUS", ""),
        "TPU_REPLICATED_JOBS": "".join(jobs).rstrip("\n"),
    }
    return string.Template(MANIFEST_TEMPLATE.read_text()).substitute(full).rstrip("\n") + "\n"


def prewarm_names(values: Mapping[str, str]) -> list[str]:
    """Names of the placeholder JobSets (``<run>-w<k>``, one per TPU slice)."""
    return [f"{values['RUN_NAME']}-w{sl.index}" for sl in plan_slices(*_slice_counts(values))]


def render_prewarm_docs(values: Mapping[str, str]) -> list[tuple[str, str]]:
    """Render one placeholder JobSet per TPU slice; returns (name, manifest) pairs."""
    _require(values, PREWARM_KEYS)
    check_run_name(values["RUN_NAME"])
    topology = values["TPU_TOPOLOGY"]
    tpl = string.Template(PREWARM_TEMPLATE.read_text())
    docs = []
    for sl, name in zip(plan_slices(*_slice_counts(values)), prewarm_names(values), strict=True):
        text = tpl.substitute({**_slice_values(values, sl, topology), "WARM_NAME": name}).rstrip("\n") + "\n"
        docs.append((name, text))
    return docs


def render_prewarm(values: Mapping[str, str]) -> str:
    """Render one placeholder JobSet per TPU slice (multi-document YAML)."""
    return "---\n".join(text for _, text in render_prewarm_docs(values))


def check_manifest(text: str, values: Mapping[str, str]) -> list[str]:
    """Check a rendered manifest against the cluster's rules; returns the violations (empty when clean).

    Structural checks need PyYAML; without it only the text-level checks run.
    """
    problems = []
    if "${" in text:
        problems.append("unsubstituted template variable left in the manifest")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - PyYAML is optional for the launcher
        return problems
    docs = [d for d in yaml.safe_load_all(text) if d]
    jobsets = [d for d in docs if d.get("kind") == "JobSet"]
    if len(jobsets) != 1:
        return problems + [f"expected exactly one JobSet, found {len(jobsets)}"]
    js = jobsets[0]
    name = js["metadata"]["name"]
    if js["metadata"].get("labels", {}).get("kueue.x-k8s.io/queue-name") != values.get("KUEUE_QUEUE"):
        problems.append("JobSet is missing the Kueue queue label")
    if js["spec"].get("successPolicy", {}).get("targetReplicatedJobs") != ["head"]:
        problems.append("successPolicy must target the head replicatedJob")
    topology = values["TPU_TOPOLOGY"]
    hosts = hosts_per_slice(topology)
    policy = placement_policy(topology)
    rjobs = {rj["name"]: rj for rj in js["spec"]["replicatedJobs"]}
    expected = ["head"] + [sl.job for sl in plan_slices(*_slice_counts(values))]
    if list(rjobs) != expected:
        problems.append(f"replicatedJobs {list(rjobs)} != {expected}")
    for rj_name, rj in rjobs.items():
        job = rj["template"]["spec"]
        pod = job["template"]
        spec = pod["spec"]
        if len(f"{name}-{rj_name}-0-{max(job.get('parallelism', 1) - 1, 0)}") > 63:
            problems.append(f"{rj_name}: pod hostname longer than 63 characters")
        for c in spec.get("containers", []):
            for kind in ("requests", "limits"):
                extra = set(c.get("resources", {}).get(kind, {})) - {"google.com/tpu"}
                if extra:
                    problems.append(f"{rj_name}/{c['name']}: {kind} {sorted(extra)} (pods carry only google.com/tpu)")
        if rj_name == "head":
            if spec.get("nodeSelector", {}).get("cloud.google.com/gke-nodepool") != values.get("HEAD_NODEPOOL"):
                problems.append("head: not pinned to the head node pool")
            continue
        if rj.get("replicas") != 1:
            problems.append(f"{rj_name}: replicas must be 1 (replicas > 1 makes GKE inject MEGASCALE_*)")
        if job.get("parallelism") != hosts or job.get("completions") != hosts:
            problems.append(f"{rj_name}: parallelism/completions must equal the {hosts} hosts of {topology}")
        sel = spec.get("nodeSelector", {})
        if sel.get("cloud.google.com/gke-tpu-topology") != topology:
            problems.append(f"{rj_name}: missing gke-tpu-topology={topology} node selector")
        if (sel.get("cloud.google.com/placement-policy-name") or "") != policy:
            problems.append(f"{rj_name}: placement-policy-name must be {policy!r}")
        ann = pod.get("metadata", {}).get("annotations", {})
        if policy and ann.get("alpha.jobset.sigs.k8s.io/exclusive-topology") != "cloud.google.com/gke-nodepool":
            problems.append(f"{rj_name}: multi-host slice without the exclusive-topology annotation")
        if any(k.startswith("kueue.x-k8s.io/podset-") for k in ann):
            problems.append(f"{rj_name}: Kueue topology annotations are rejected on this cluster")
        tpu = {c["name"]: c.get("resources", {}).get("limits", {}).get("google.com/tpu") for c in spec["containers"]}
        if tpu.get("ray-worker") != str(CHIPS_PER_HOST):
            problems.append(f"{rj_name}: ray-worker must request google.com/tpu={CHIPS_PER_HOST}")
    return problems


def slice_plan_json(values: Mapping[str, str]) -> str:
    """Human-readable summary of the slice layout (printed by the launcher)."""
    topology = values["TPU_TOPOLOGY"]
    return json.dumps(
        {
            "topology": topology,
            "hosts_per_slice": hosts_per_slice(topology),
            "placement_policy": placement_policy(topology) or None,
            "slices": [
                {"replicated_job": sl.job, "role": sl.role, "ray_resource": sl.ray_resource}
                for sl in plan_slices(*_slice_counts(values))
            ],
        }
    )


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["manifest", "prewarm", "prewarm-names", "plan"])
    ap.add_argument("--out", help="output file (default: stdout)")
    ap.add_argument("--out-dir", help="prewarm: also write one <name>.yaml per placeholder into this directory")
    args = ap.parse_args(argv)
    values = dict(os.environ)
    if args.what == "manifest":
        text = render_manifest(values)
        problems = check_manifest(text, values)
        if problems:
            for p in problems:
                print(f"[render] ERROR: {p}", file=sys.stderr)
            return 2
    elif args.what == "prewarm":
        text = render_prewarm(values)
        if args.out_dir:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            for name, doc in render_prewarm_docs(values):
                (Path(args.out_dir) / f"{name}.yaml").write_text(doc)
    elif args.what == "prewarm-names":
        text = "\n".join(prewarm_names(values)) + "\n"
    else:
        text = slice_plan_json(values) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
