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
"""CPU tests for the TPU 7x NAP JobSet launcher (examples/tpu/gke/run_ray_jobset_v7x_nap.sh and v7x_nap/)."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

LAUNCHER_DIR = Path(__file__).resolve().parents[2] / "examples" / "tpu" / "gke" / "v7x_nap"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"v7x_nap_{name}", LAUNCHER_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render = _load("render")
snapshot = _load("pod_env_snapshot")
head = _load("head_supervisor")


def _values(**overrides) -> dict:
    values = {
        "RUN_NAME": "v7x-test-1029-0800",
        "RAY_NAMESPACE": "dev-ns",
        "KUEUE_QUEUE": "multislice-queue",
        "PRIORITY_CLASS": "medium",
        "SERVICE_ACCOUNT": "default",
        "RUN_OWNER": "tester",
        "CODE_SHA": "0123456789abcdef0123456789abcdef01234567",
        "LAUNCH_ID": "0f1e2d3c4b5a6978",
        "IMAGE": "registry.example/verl-tpu@sha256:" + "0" * 64,
        "POD_TPU_ACCELERATOR_TYPE": "tpu7x",
        "GCS_BUCKET": "example-bucket",
        "GCSFUSE_MOUNT_OPTIONS": "implicit-dirs",
        "DATA_MOUNT": "/data",
        "RUN_DIR": "/data/runs/v7x-test-1029-0800",
        "SRC_TARBALL": "/data/runs/v7x-test-1029-0800/src-0123456789ab.tar.gz",
        "HEAD_NODEPOOL": "cpu-np",
        "HEAD_NUM_CPUS": "12",
        "HEAD_OBJECT_STORE_BYTES": "32000000000",
        "HEAD_DSHM_SIZE": "48Gi",
        "TPU_DSHM_SIZE": "400Gi",
        "TPU_TOPOLOGY": "2x2x4",
        "TRAINER_SLICES": "1",
        "ROLLOUT_SLICES": "1",
        "RAY_TPU_PER_HOST": "8",
        "WORKER_NUM_CPUS": "",
        "JOBSET_MAX_RESTARTS": "0",
        "JOBSET_TTL_SECONDS": "1800",
        "NODE_WAIT_TIMEOUT_S": "1200",
        "NODE_RANK_SOURCE": "job-index",
        "NODE_RANK_TIMEOUT_S": "600",
        "DRIVER_TIMEOUT_S": "6600",
        "REQUIRE_NODE_RANK": "1",
        "PREWARM_HOLD_S": "1800",
    }
    values.update(overrides)
    return values


def _env(container: dict) -> dict:
    return {e["name"]: e.get("value") for e in container.get("env", []) if "value" in e}


def test_slice_layout():
    assert render.hosts_per_slice("2x2x4") == 4
    assert render.hosts_per_slice("2x2x1") == 1
    assert render.placement_policy("2x2x4") == "tpu7x-32-2x2x4-placement-policy"
    assert render.placement_policy("2x4x4") == "tpu7x-64-2x4x4-placement-policy"
    assert render.placement_policy("2x2x1") == ""
    plan = render.plan_slices(1, 2)
    assert [(s.job, s.role, s.ray_resource) for s in plan] == [
        ("trainer-0", "trainer", "tpu-group-0"),
        ("rollout-0", "rollout", "tpu-group-1"),
        ("rollout-1", "rollout", "tpu-group-2"),
    ]
    with pytest.raises(ValueError):
        render.plan_slices(0, 1)
    with pytest.raises(ValueError):
        render.hosts_per_slice("2x2")


def test_manifest_two_slices_follows_cluster_rules():
    values = _values()
    text = render.render_manifest(values)
    assert render.check_manifest(text, values) == []
    (jobset,) = [d for d in yaml.safe_load_all(text) if d]
    assert jobset["kind"] == "JobSet"
    assert jobset["spec"]["successPolicy"] == {"operator": "All", "targetReplicatedJobs": ["head"]}
    assert jobset["metadata"]["annotations"]["verl-nap/launch-id"] == values["LAUNCH_ID"]
    assert jobset["spec"]["failurePolicy"] == {
        "maxRestarts": 0,
        "rules": [{"name": "head_failure_is_final", "action": "FailJobSet", "targetReplicatedJobs": ["head"]}],
    }
    rjobs = {rj["name"]: rj for rj in jobset["spec"]["replicatedJobs"]}
    assert list(rjobs) == ["head", "trainer-0", "rollout-0"]

    head_pod = rjobs["head"]["template"]["spec"]["template"]
    head_env = _env(head_pod["spec"]["containers"][0])
    assert head_env["VERL_NAP_EXPECTED_SLICES"] == "tpu-group-0=4,tpu-group-1=4"
    assert head_env["VERL_NAP_RUN_DIR"] == values["RUN_DIR"]
    assert head_pod["spec"]["nodeSelector"] == {"cloud.google.com/gke-nodepool": "cpu-np"}

    for name, resource in (("trainer-0", "tpu-group-0"), ("rollout-0", "tpu-group-1")):
        rj = rjobs[name]
        assert rj["replicas"] == 1
        job = rj["template"]["spec"]
        assert (job["parallelism"], job["completions"], job["completionMode"]) == (4, 4, "Indexed")
        pod = job["template"]
        assert pod["spec"]["nodeSelector"] == {
            "cloud.google.com/gke-tpu-accelerator": "tpu7x",
            "cloud.google.com/gke-tpu-topology": "2x2x4",
            "cloud.google.com/placement-policy-name": "tpu7x-32-2x2x4-placement-policy",
        }
        ann = pod["metadata"]["annotations"]
        assert ann["alpha.jobset.sigs.k8s.io/exclusive-topology"] == "cloud.google.com/gke-nodepool"
        assert ann["cluster-autoscaler.kubernetes.io/safe-to-evict"] == "false"
        (container,) = pod["spec"]["containers"]
        assert container["resources"] == {"requests": {"google.com/tpu": "4"}, "limits": {"google.com/tpu": "4"}}
        env = _env(container)
        assert env["VERL_NAP_SLICE_RESOURCE"] == resource
        assert env["VERL_NAP_HEAD_HOST"] == f"{values['RUN_NAME']}-head-0-0.{values['RUN_NAME']}"
        assert env["VERL_NAP_HOSTS_PER_SLICE"] == "4"
        assert env["VERL_NAP_NODE_RANK_SOURCE"] == "job-index"
        assert values["SRC_TARBALL"] in container["args"][0]
        volumes = {v["name"]: v for v in pod["spec"]["volumes"]}
        assert volumes["gcs-volume"]["csi"]["volumeAttributes"]["bucketName"] == "example-bucket"
        assert volumes["dshm"]["emptyDir"] == {"medium": "Memory", "sizeLimit": "400Gi"}


def test_manifest_trainer_only_and_bad_inputs():
    values = _values(ROLLOUT_SLICES="0")
    text = render.render_manifest(values)
    assert render.check_manifest(text, values) == []
    (jobset,) = [d for d in yaml.safe_load_all(text) if d]
    assert [rj["name"] for rj in jobset["spec"]["replicatedJobs"]] == ["head", "trainer-0"]

    with pytest.raises(ValueError, match="RUN_NAME"):
        render.render_manifest(_values(RUN_NAME="this-run-name-is-far-too-long-for-a-jobset"))
    with pytest.raises(ValueError, match="RUN_NAME"):
        render.render_manifest(_values(RUN_NAME="Upper-Case"))
    with pytest.raises(ValueError, match="missing values: GCS_BUCKET"):
        render.render_manifest(_values(GCS_BUCKET=""))


def test_check_manifest_reports_violations():
    values = _values()
    (jobset,) = [d for d in yaml.safe_load_all(render.render_manifest(values)) if d]
    trainer = jobset["spec"]["replicatedJobs"][1]
    trainer["replicas"] = 2
    pod = trainer["template"]["spec"]["template"]
    del pod["spec"]["nodeSelector"]["cloud.google.com/placement-policy-name"]
    pod["spec"]["containers"][0]["resources"]["requests"]["cpu"] = "8"
    problems = render.check_manifest(yaml.safe_dump(jobset), values)
    assert any("replicas must be 1" in p for p in problems)
    assert any("placement-policy-name" in p for p in problems)
    assert any("cpu" in p for p in problems)


def test_prewarm_one_placeholder_per_slice(tmp_path, monkeypatch):
    values = _values()
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    assert render.main(["prewarm", "--out", str(tmp_path / "all.yaml"), "--out-dir", str(tmp_path / "one")]) == 0
    for name in render.prewarm_names(values):
        (doc,) = [d for d in yaml.safe_load_all((tmp_path / "one" / f"{name}.yaml").read_text()) if d]
        assert doc["metadata"]["name"] == name
        assert doc["metadata"]["annotations"]["verl-nap/launch-id"] == values["LAUNCH_ID"]
    docs = [d for d in yaml.safe_load_all(render.render_prewarm(values)) if d]
    assert [d["metadata"]["name"] for d in docs] == render.prewarm_names(values)
    assert render.prewarm_names(values) == [f"{values['RUN_NAME']}-w0", f"{values['RUN_NAME']}-w1"]
    for doc in docs:
        assert doc["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "multislice-queue"
        (rj,) = doc["spec"]["replicatedJobs"]
        assert rj["replicas"] == 1 and rj["template"]["spec"]["parallelism"] == 4
        pod = rj["template"]["spec"]["template"]
        assert pod["spec"]["nodeSelector"]["cloud.google.com/placement-policy-name"].startswith("tpu7x-32-")
        assert pod["spec"]["containers"][0]["image"] == values["IMAGE"]


def test_env_snapshot_allowlist_and_redaction():
    env = {
        "LIBTPU_INIT_ARGS": "--xla_tpu_bf16_emission_mode=PROMOTE_F32",
        "TPU_WORKER_ID": "3",
        "MEGASCALE_NUM_SLICES": "2",
        "HF_TOKEN": "hf_secret",
        "VERL_UPLOAD_TOKEN": "secret",
        "VERL_NAP_RUN_DIR": "/data/run",
        "HOME": "/root",
        "PATH": "/usr/bin",
        "XLA_FLAGS": "x" * (snapshot.MAX_VALUE_LEN + 10),
    }
    out = snapshot.filter_env(env)
    assert out["LIBTPU_INIT_ARGS"] == env["LIBTPU_INIT_ARGS"]
    assert out["TPU_WORKER_ID"] == "3" and out["MEGASCALE_NUM_SLICES"] == "2"
    assert out["VERL_NAP_RUN_DIR"] == "/data/run"
    assert out["VERL_UPLOAD_TOKEN"] == snapshot.REDACTED
    assert "HF_TOKEN" not in out and "HOME" not in out and "PATH" not in out
    assert out["XLA_FLAGS"].startswith("x" * snapshot.MAX_VALUE_LEN) and out["XLA_FLAGS"].endswith("more chars>")


def test_env_snapshot_scans_only_ray_processes(tmp_path):
    def fake_proc(pid: str, cmdline: list[str], environ: dict):
        d = tmp_path / pid
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in cmdline) + b"\0")
        (d / "environ").write_bytes(b"".join(f"{k}={v}".encode() + b"\0" for k, v in environ.items()))

    fake_proc("101", ["ray::TrainerWorker.update_actor"], {"LIBTPU_INIT_ARGS": "--a", "HF_TOKEN": "x"})
    fake_proc("102", ["/usr/bin/python3", "train.py"], {"LIBTPU_INIT_ARGS": "--b"})
    fake_proc("103", ["/opt/ray/core/src/ray/raylet/raylet", "--node"], {"TPU_ACCELERATOR_TYPE": "tpu7x"})
    (tmp_path / "self").mkdir()
    seen: set = set()
    records = sorted(snapshot.scan_procs(seen, proc_root=str(tmp_path)), key=lambda r: r["pid"])
    assert [r["pid"] for r in records] == [101, 103]
    assert records[0]["env"] == {"LIBTPU_INIT_ARGS": "--a"}
    assert snapshot.scan_procs(seen, proc_root=str(tmp_path)) == []


def test_head_checks_slice_hosts_and_ranks():
    assert head.parse_expected_slices("tpu-group-0=4, tpu-group-1=4") == {"tpu-group-0": 4, "tpu-group-1": 4}

    def node(ip, group, rank=None):
        labels = {} if rank is None else {head.NODE_RANK_LABEL: str(rank)}
        return {"ip": ip, "hostname": ip, "alive": True, "resources": {"TPU": 8.0, group: 1.0}, "labels": labels}

    expected = {"tpu-group-0": 4, "tpu-group-1": 4}
    nodes = [node(f"10.0.0.{i}", "tpu-group-0", r) for i, r in enumerate([2, 0, 3, 1])]
    nodes += [node(f"10.0.1.{i}", "tpu-group-1", r) for i, r in enumerate([1, 3, 0, 2])]
    hosts_ok, ranks_ok, lines = head.check_slices(nodes, expected)
    assert hosts_ok and ranks_ok
    assert lines[0] == "tpu-group-0: 4/4 hosts"

    # Ray's worker id agrees with every rank: fine; one host whose rank disagrees with its worker id: not fine.
    agree = [
        dict(n, labels={**n["labels"], head.RAY_TPU_WORKER_ID_LABEL: n["labels"][head.NODE_RANK_LABEL]}) for n in nodes
    ]
    assert head.check_slices(agree, expected)[:2] == (True, True)
    swapped = [dict(n) for n in agree]
    swapped[0] = dict(swapped[0], labels={**swapped[0]["labels"], head.RAY_TPU_WORKER_ID_LABEL: "3"})
    hosts_ok, ranks_ok, lines = head.check_slices(swapped, expected)
    assert (hosts_ok, ranks_ok) == (True, False) and "rank != tpu-worker-id" in lines[0]

    unlabelled = nodes[:-1] + [node("10.0.1.9", "tpu-group-1")]
    assert head.check_slices(unlabelled, expected)[:2] == (True, False)
    duplicate = nodes[:-1] + [node("10.0.1.9", "tpu-group-1", 0)]
    assert head.check_slices(duplicate, expected)[:2] == (True, False)
    assert head.check_slices(nodes[:-1], expected)[0] is False


def test_rendered_from_environment(tmp_path, monkeypatch):
    for key, value in _values().items():
        monkeypatch.setenv(key, value)
    out = tmp_path / "jobset.yaml"
    assert render.main(["manifest", "--out", str(out)]) == 0
    assert "${" not in out.read_text()
    assert render.main(["prewarm-names", "--out", str(tmp_path / "names")]) == 0
    assert (tmp_path / "names").read_text().split() == render.prewarm_names(dict(os.environ))


# --- ownership of created objects (v7x_nap/owned_objects.sh) with a fake kubectl -------------------------------------

FAKE_KUBECTL = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, re, sys, uuid
    state = os.environ["FAKE_KUBE_STATE"]
    args = sys.argv[1:]
    with open(os.path.join(state, "_calls.log"), "a") as f:
        f.write(" ".join(args) + "\\n")
    a, i = [], 0
    while i < len(args):
        if args[i] in ("--context", "-n"):
            i += 2
            continue
        a.append(args[i])
        i += 1
    def path(name):
        return os.path.join(state, name + ".json")
    def fail(msg):
        sys.stderr.write(msg + "\\n")
        sys.exit(1)
    if a[0] == "create":
        text = open(a[a.index("-f") + 1]).read()
        name = re.search(r"^  name: (\\S+)", text, re.M).group(1).strip('"')
        launch = re.search(r'verl-nap/launch-id: "([^"]*)"', text)
        if os.path.exists(path(name)):
            fail(f'Error from server (AlreadyExists): jobsets.jobset.x-k8s.io "{name}" already exists')
        uid = str(uuid.uuid4())
        json.dump({"uid": uid, "launch_id": launch.group(1) if launch else ""}, open(path(name), "w"))
        if os.environ.get("FAKE_LOSE_CREATE_RESPONSE") == name:
            fail("Unable to connect to the server: net/http: request canceled")
        print(f"JobSet {name} {uid}", end="")
    elif a[0] == "get":
        name = a[2]
        if not os.path.exists(path(name)):
            fail(f'Error from server (NotFound): jobsets.jobset.x-k8s.io "{name}" not found')
        obj = json.load(open(path(name)))
        meta = {"name": name, "uid": obj["uid"], "annotations": {"verl-nap/launch-id": obj["launch_id"]}}
        print(json.dumps({"metadata": meta}))
    elif a[0] == "delete" and "--raw" in a:
        name = a[a.index("--raw") + 1].rsplit("/", 1)[1]
        want = json.load(open(a[a.index("-f") + 1]))["preconditions"]["uid"]
        if not os.path.exists(path(name)):
            fail(f'Error from server (NotFound): jobsets.jobset.x-k8s.io "{name}" not found')
        have = json.load(open(path(name)))["uid"]
        if have != want:
            fail(f"Error from server (Conflict): Precondition failed: UID in precondition: {want}, object: {have}")
        os.remove(path(name))
    else:
        fail("fake kubectl: unsupported " + " ".join(a))
    """
)


class FakeCluster:
    """A directory of JobSets (name -> uid, launch id) behind a fake kubectl, plus a bash runner for the library."""

    def __init__(self, tmp_path: Path):
        self.state = tmp_path / "state"
        self.state.mkdir(parents=True)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        kubectl = bin_dir / "kubectl"
        kubectl.write_text(FAKE_KUBECTL)
        kubectl.chmod(0o755)
        self.tmp = tmp_path
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_KUBE_STATE": str(self.state),
            "KUBE_CONTEXT": "ctx",
            "RAY_NAMESPACE": "ns",
            "LAUNCH_ID": "launch-ours",
            "OWNED_LEDGER": str(tmp_path / "ledger.log"),
        }

    def manifest(self, name: str, launch_id: str = "launch-ours") -> str:
        f = self.tmp / f"{name}.yaml"
        f.write_text(
            f'kind: JobSet\nmetadata:\n  name: {name}\n  annotations:\n    verl-nap/launch-id: "{launch_id}"\n'
        )
        return str(f)

    def preexisting(self, name: str, uid: str = "someone-elses-uid", launch_id: str = "launch-theirs"):
        (self.state / f"{name}.json").write_text(json.dumps({"uid": uid, "launch_id": launch_id}))

    def names(self) -> set:
        return {p.stem for p in self.state.glob("*.json")}

    def run(self, script: str, **extra_env) -> subprocess.CompletedProcess:
        lib = LAUNCHER_DIR / "owned_objects.sh"
        body = f'set -euo pipefail\nlog() {{ echo "LOG: $*" >&2; }}\nsource "{lib}"\n{script}\n'
        return subprocess.run(["bash", "-c", body], env={**self.env, **extra_env}, capture_output=True, text=True)


def test_owned_objects_delete_only_what_this_run_created(tmp_path):
    k = FakeCluster(tmp_path)
    script = f'owned_create {k.manifest("run-a")}\necho "UID=$OWNED_LAST_UID"\nowned_delete_all\n'
    res = k.run(script + 'echo "LEFT=${#OWNED_OBJECTS[@]}"')
    assert res.returncode == 0, res.stderr
    assert "UID=" in res.stdout and "LEFT=0" in res.stdout
    assert k.names() == set()
    assert "created jobset run-a" in (tmp_path / "ledger.log").read_text()


def test_owned_objects_never_delete_a_preexisting_object(tmp_path):
    k = FakeCluster(tmp_path)
    k.preexisting("run-a")
    script = f"owned_create {k.manifest('run-a')} || echo CREATE_FAILED\nowned_delete_all\n"
    res = k.run(script + 'echo "LEFT=${#OWNED_OBJECTS[@]}"')
    assert res.returncode == 0, res.stderr
    assert "CREATE_FAILED" in res.stdout and "already exists and is not ours" in res.stderr
    assert k.names() == {"run-a"}
    assert "delete" not in (k.state / "_calls.log").read_text()


def test_owned_objects_leave_a_replaced_object_alone(tmp_path):
    k = FakeCluster(tmp_path)
    script = f"""
owned_create {k.manifest("run-a")}
python3 - "$FAKE_KUBE_STATE/run-a.json" <<'PY'
import json, sys
json.dump({{"uid": "replacement-uid", "launch_id": "launch-theirs"}}, open(sys.argv[1], "w"))
PY
owned_delete_all
echo "LEFT=${{#OWNED_OBJECTS[@]}}"
"""
    res = k.run(script)
    assert res.returncode == 0, res.stderr
    assert "replaced by someone else" in res.stderr and "LEFT=0" in res.stdout
    assert json.loads((k.state / "run-a.json").read_text())["uid"] == "replacement-uid"


def test_owned_objects_partial_placeholder_creation(tmp_path):
    k = FakeCluster(tmp_path)
    k.preexisting("run-a-w1")
    script = f"""
owned_create {k.manifest("run-a-w0")}
owned_create {k.manifest("run-a-w1")} || echo W1_FAILED
owned_delete_all
"""
    res = k.run(script)
    assert res.returncode == 0, res.stderr
    assert "W1_FAILED" in res.stdout
    assert k.names() == {"run-a-w1"}  # ours deleted, the other one untouched


def test_owned_objects_adopt_only_with_our_launch_id(tmp_path):
    k = FakeCluster(tmp_path)
    # The create reached the server but its response was lost: the object carries our launch id, so it is ours.
    res = k.run(f"owned_create {k.manifest('run-a')}\nowned_delete_all", FAKE_LOSE_CREATE_RESPONSE="run-a")
    assert res.returncode == 0, res.stderr
    assert "adopting it" in res.stderr and k.names() == set()
    # Same failure, but the object of that name was made by another launch: not adopted, not deleted.
    k2 = FakeCluster(tmp_path / "second")
    res = k2.run(
        f"owned_create {k2.manifest('run-b', launch_id='launch-theirs')} || echo NOT_OURS\nowned_delete_all",
        FAKE_LOSE_CREATE_RESPONSE="run-b",
    )
    assert res.returncode == 0, res.stderr
    assert "NOT_OURS" in res.stdout and k2.names() == {"run-b"}
