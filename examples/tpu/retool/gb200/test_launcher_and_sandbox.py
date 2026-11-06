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
"""Unit tests for GB200 launcher UID-preconditioned cleanup, external sandbox wiring, and fail-closed privilege drop."""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import uvicorn  # noqa: F401
except ImportError:
    import types

    sys.modules["uvicorn"] = types.ModuleType("uvicorn")

try:
    import fastapi  # noqa: F401
except ImportError:
    import types

    fa = types.ModuleType("fastapi")

    class _DummyFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def middleware(self, *args, **kwargs):
            return lambda fn: fn

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

    fa.FastAPI = _DummyFastAPI
    fa.Request = object
    fa.Response = object
    sys.modules["fastapi"] = fa

try:
    import pydantic  # noqa: F401
except ImportError:
    import types

    pd = types.ModuleType("pydantic")

    class _DummyBaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    pd.BaseModel = _DummyBaseModel
    pd.Field = lambda default=None, **kwargs: default
    sys.modules["pydantic"] = pd

from examples.tpu.retool import sandbox_fusion_server  # noqa: E402


def _load_resolve_endpoint():
    """Import resolve_sandbox_fusion_endpoint from sandbox_fusion_tools.py without requiring ray/torch."""
    tools_path = REPO_ROOT / "examples" / "tpu" / "retool" / "sandbox_fusion_tools.py"
    src = tools_path.read_text(encoding="utf-8")
    start = src.index("def resolve_sandbox_fusion_endpoint")
    end = src.index("class SandboxFusionTool")
    ns = {"os": os, "urllib": __import__("urllib.parse"), "Optional": __import__("typing").Optional}
    exec(src[start:end], ns)  # noqa: S102
    return ns["resolve_sandbox_fusion_endpoint"]


resolve_sandbox_fusion_endpoint = _load_resolve_endpoint()


class TestSandboxPrivilegeDrop(unittest.TestCase):
    """Verify sandbox_fusion_server fails closed if privilege drop fails."""

    def test_preexec_raises_when_setuid_fails(self):
        preexec = sandbox_fusion_server._make_preexec_fn(1024, drop_uid_gid=(65534, 65534))
        with (
            mock.patch("os.setsid"),
            mock.patch("resource.setrlimit"),
            mock.patch("os.setgroups"),
            mock.patch("os.setgid"),
            mock.patch("os.setuid", side_effect=OSError("setuid failed")),
        ):
            with self.assertRaises(OSError):
                preexec()

    def test_preexec_raises_when_setgroups_fails(self):
        preexec = sandbox_fusion_server._make_preexec_fn(1024, drop_uid_gid=(65534, 65534))
        with (
            mock.patch("os.setsid"),
            mock.patch("resource.setrlimit"),
            mock.patch("os.setgroups", side_effect=PermissionError("setgroups denied")),
        ):
            with self.assertRaises(PermissionError):
                preexec()

    def test_preexec_rejects_zero_uid_or_remaining_root(self):
        preexec_zero = sandbox_fusion_server._make_preexec_fn(1024, drop_uid_gid=(0, 65534))
        with mock.patch("os.setsid"), mock.patch("resource.setrlimit"):
            with self.assertRaises(PermissionError):
                preexec_zero()

        preexec_still_root = sandbox_fusion_server._make_preexec_fn(1024, drop_uid_gid=(65534, 65534))
        with (
            mock.patch("os.setsid"),
            mock.patch("resource.setrlimit"),
            mock.patch("os.setgroups"),
            mock.patch("os.setgid"),
            mock.patch("os.setuid"),
            mock.patch("os.geteuid", return_value=0),
        ):
            with self.assertRaises(PermissionError):
                preexec_still_root()

    def test_run_code_fails_closed_when_chown_fails_as_root(self):
        req = sandbox_fusion_server.RunCodeRequest(code="print(123)", language="python", run_timeout=5.0)
        with (
            mock.patch("os.geteuid", return_value=0),
            mock.patch.dict(os.environ, {"SANDBOX_ALLOW_ROOT_EXEC": "0"}, clear=False),
            mock.patch("os.chown", side_effect=PermissionError("chown failed")),
            mock.patch("asyncio.create_subprocess_exec") as mock_spawn,
        ):
            res = asyncio.run(sandbox_fusion_server.run_code(req))
            self.assertEqual(res["status"], "SandboxError")
            self.assertIn("PrivilegeDropError", res["message"])
            mock_spawn.assert_not_called()

    def test_resolve_drop_uid_gid_adopts_restricted_executable_owner(self):
        fake_stat = os.stat_result((0o040750, 1, 1, 1, 1000, 100, 4096, 0, 0, 0))
        world_stat = os.stat_result((0o040755, 1, 1, 1, 0, 0, 4096, 0, 0, 0))

        def _stat_side_effect(p):
            if p == "/home/ray":
                return fake_stat
            return world_stat

        with (
            mock.patch("os.geteuid", return_value=0),
            mock.patch("sys.executable", "/home/ray/anaconda3/bin/python"),
            mock.patch("os.path.realpath", return_value="/home/ray/anaconda3/bin/python"),
            mock.patch("os.stat", side_effect=_stat_side_effect),
            mock.patch.dict(os.environ, {"SANDBOX_ALLOW_ROOT_EXEC": "0"}, clear=True),
        ):
            self.assertEqual(sandbox_fusion_server._resolve_drop_uid_gid(), (1000, 100))

    def test_run_code_handles_subprocess_exec_permission_error_without_500(self):
        req = sandbox_fusion_server.RunCodeRequest(code="print(123)", language="python", run_timeout=5.0)
        with (
            mock.patch("os.geteuid", return_value=1000),
            mock.patch("asyncio.create_subprocess_exec", side_effect=PermissionError(13, "Permission denied")),
        ):
            res = asyncio.run(sandbox_fusion_server.run_code(req))
            self.assertEqual(res["status"], "Failed")
            self.assertEqual(res["message"], "ExecutionError")
            self.assertEqual(res["run_result"]["return_code"], -1)
            self.assertIn("PermissionError", res["run_result"]["stderr"])


class TestSandboxEndpointResolution(unittest.TestCase):
    """Verify external SANDBOX_FUSION_HOST / SANDBOX_FUSION_URL disables local server auto-start."""

    def test_default_loopback(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            url, local_port = resolve_sandbox_fusion_endpoint("http://127.0.0.1:18080/run_code")
            self.assertEqual(url, "http://127.0.0.1:18080/run_code")
            self.assertEqual(local_port, 18080)

    def test_external_host_disables_local_port(self):
        with mock.patch.dict(
            os.environ,
            {"SANDBOX_FUSION_HOST": "sandbox.isolated.svc.cluster.local", "SANDBOX_FUSION_PORT": "19090"},
            clear=True,
        ):
            url, local_port = resolve_sandbox_fusion_endpoint("http://127.0.0.1:18080/run_code")
            self.assertEqual(url, "http://sandbox.isolated.svc.cluster.local:19090/run_code")
            self.assertIsNone(local_port)

    def test_explicit_external_url_overrides_config(self):
        with mock.patch.dict(
            os.environ,
            {"SANDBOX_FUSION_URL": "http://10.24.0.99:28080/run_code"},
            clear=True,
        ):
            url, local_port = resolve_sandbox_fusion_endpoint("http://127.0.0.1:18080/run_code")
            self.assertEqual(url, "http://10.24.0.99:28080/run_code")
            self.assertIsNone(local_port)

    def test_disable_auto_start_on_loopback(self):
        with mock.patch.dict(os.environ, {"SANDBOX_DISABLE_AUTO_START": "1"}, clear=True):
            url, local_port = resolve_sandbox_fusion_endpoint("http://127.0.0.1:18080/run_code")
            self.assertEqual(url, "http://127.0.0.1:18080/run_code")
            self.assertIsNone(local_port)

    def test_external_mode_in_dapo_script_refuses_local_server_auto_start(self):
        dapo_script = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh"
        env = os.environ.copy()
        # Use an unreachable non-loopback address (127.0.0.2 is non-127.0.0.1/localhost/::1) on an unused port
        env.update(
            {
                "SANDBOX_FUSION_HOST": "127.0.0.2",
                "SANDBOX_FUSION_PORT": "59999",
            }
        )
        res = subprocess.run(["bash", str(dapo_script)], env=env, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("external SandboxFusion endpoint", res.stderr)
        self.assertIn("local server auto-start is disabled in external mode", res.stderr)


class _MockKubeProxyHandler(BaseHTTPRequestHandler):
    deletes: list[tuple[str, dict]] = []
    status_code: int = 200

    def do_DELETE(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        self.__class__.deletes.append((self.path, body))
        code = self.__class__.status_code
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if code == 200:
            self.wfile.write(b'{"kind":"Status","status":"Success"}')
        else:
            self.wfile.write(b'{"kind":"Status","status":"Failure","message":"simulated delete failure"}')

    def log_message(self, format, *args):  # noqa: A002
        return


class TestLauncherResourceLifecycle(unittest.TestCase):
    """Verify run_ray_jobset_gb200.sh per-object UID capture, ambiguous create cleanup, and fail-closed GCS checks."""

    def setUp(self):
        _MockKubeProxyHandler.deletes = []
        _MockKubeProxyHandler.status_code = 200
        self.server = HTTPServer(("127.0.0.1", 0), _MockKubeProxyHandler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _make_mock_bins(
        self,
        tmp_dir: Path,
        fail_on_create_idx: int = 3,
        existing_k8s: bool = False,
        gcs_mode: str = "not_found",
        ambiguous_get_response: str = "Error from server (NotFound): not found",
        ambiguous_get_rc: int = 1,
    ):
        bin_dir = tmp_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        state_file = tmp_dir / "create_count.txt"
        state_file.write_text("0", encoding="utf-8")

        kubectl_script = bin_dir / "kubectl"
        kubectl_script.write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail
args=("$@")
if [[ "${{args[*]}}" == *" get "* &&
      "${{args[*]}}" == *"jsonpath={{.metadata.labels.retool-launch-id}}:{{.metadata.uid}}"* ]]; then
    echo "{ambiguous_get_response}"
    exit {ambiguous_get_rc}
fi
if [[ "${{args[*]}}" == *" get "* ]]; then
    if [[ "{1 if existing_k8s else 0}" == "1" ]]; then
        exit 0
    fi
    echo "Error from server (NotFound): not found" >&2
    exit 1
fi
if [[ "${{args[*]}}" == *" create -f - "* ]]; then
    cat >/dev/null
    cnt=$(cat "{state_file}")
    cnt=$((cnt + 1))
    echo "${{cnt}}" > "{state_file}"
    if [[ "${{cnt}}" -eq {fail_on_create_idx} ]]; then
        echo "Error from server: simulated failure on object ${{cnt}}" >&2
        exit 1
    fi
    printf 'uid-obj-%d' "${{cnt}}"
    exit 0
fi
if [[ "${{args[*]}}" == *" proxy --port=0"* ]]; then
    echo "Starting to serve on 127.0.0.1:{self.port}"
    sleep 30
    exit 0
fi
exit 0
""",
            encoding="utf-8",
        )
        kubectl_script.chmod(0o755)

        gsutil_script = bin_dir / "gsutil"
        if gcs_mode == "exists":
            gsutil_body = "exit 0"
            gsutil_lock_body = "exit 0"
        elif gcs_mode == "not_found":
            gsutil_body = 'echo "No URLs matched: gs://mock-bucket/mock-runs" >&2\nexit 1'
            gsutil_lock_body = "cat >/dev/null\nexit 0"
        elif gcs_mode == "lock_race":
            gsutil_body = 'echo "No URLs matched: gs://mock-bucket/mock-runs" >&2\nexit 1'
            gsutil_lock_body = 'cat >/dev/null\necho "PreconditionException: 412 Precondition Failed" >&2\nexit 1'
        else:
            gsutil_body = (
                'echo "AccessDeniedException: 403 Caller does not have storage.objects.list access" >&2\nexit 1'
            )
            gsutil_lock_body = "exit 1"
        gsutil_script.write_text(
            f"""#!/usr/bin/env bash
if [[ "$*" == *"stat"* ]]; then
    {gsutil_body}
fi
if [[ "$*" == *"x-goog-if-generation-match:0"* ]]; then
    {gsutil_lock_body}
fi
exit 0
""",
            encoding="utf-8",
        )
        gsutil_script.chmod(0o755)
        return bin_dir

    def test_partial_create_failure_cleans_up_created_objects_with_uid_preconditions(self):
        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        with tempfile.TemporaryDirectory() as td:
            bin_dir = self._make_mock_bins(Path(td), fail_on_create_idx=3)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env.get('PATH', '')}",
                    "RUN_NAME": "unit-test-run",
                    "LAUNCH_ID": "launch-abc123",
                    "KUBE_CONTEXT": "mock-ctx",
                    "NODEPOOL": "mock-pool",
                    "GCS_BUCKET": "mock-bucket",
                    "RUN_ROOT": "mock-runs",
                    "INPUTS_ROOT": "mock-inputs",
                    "IMAGE": "mock-image:latest",
                }
            )
            res = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            deleted = dict(_MockKubeProxyHandler.deletes)
            self.assertEqual(len(deleted), 2, f"Expected 2 atomic deletes, got: {_MockKubeProxyHandler.deletes}")
            self.assertEqual(
                deleted["/api/v1/namespaces/default/services/unit-test-run-head"]["preconditions"]["uid"],
                "uid-obj-2",
            )
            self.assertEqual(
                deleted["/apis/resource.nvidia.com/v1beta1/namespaces/default/computedomains/unit-test-run-cd"][
                    "preconditions"
                ]["uid"],
                "uid-obj-1",
            )

    def test_ambiguous_create_response_loss_cleans_up_when_launch_id_matches(self):
        """Cleanup resolves UID by launch-id when Service creation succeeds but the response is lost."""
        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        with tempfile.TemporaryDirectory() as td:
            # Fail on object 2 (Service), but simulate that the Service WAS created on the server with our launch-id
            bin_dir = self._make_mock_bins(
                Path(td),
                fail_on_create_idx=2,
                ambiguous_get_response="launch-svc999:uid-ambiguous-svc-2",
                ambiguous_get_rc=0,
            )
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env.get('PATH', '')}",
                    "RUN_NAME": "ambig-svc-run",
                    "LAUNCH_ID": "launch-svc999",
                    "KUBE_CONTEXT": "mock-ctx",
                    "NODEPOOL": "mock-pool",
                    "GCS_BUCKET": "mock-bucket",
                    "RUN_ROOT": "mock-runs",
                    "INPUTS_ROOT": "mock-inputs",
                    "IMAGE": "mock-image:latest",
                }
            )
            res = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            deleted = dict(_MockKubeProxyHandler.deletes)
            self.assertEqual(len(deleted), 2, f"Expected both CD and ambiguous Service to be deleted, got: {deleted}")
            self.assertEqual(
                deleted["/api/v1/namespaces/default/services/ambig-svc-run-head"]["preconditions"]["uid"],
                "uid-ambiguous-svc-2",
            )
            self.assertEqual(
                deleted["/apis/resource.nvidia.com/v1beta1/namespaces/default/computedomains/ambig-svc-run-cd"][
                    "preconditions"
                ]["uid"],
                "uid-obj-1",
            )

    def test_ambiguous_create_does_not_delete_foreign_launch_id(self):
        """If an object with a different retool-launch-id exists after ambiguous create, cleanup must not delete it."""
        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        with tempfile.TemporaryDirectory() as td:
            bin_dir = self._make_mock_bins(
                Path(td),
                fail_on_create_idx=2,
                ambiguous_get_response="foreign-launch-id:uid-foreign-svc",
                ambiguous_get_rc=0,
            )
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env.get('PATH', '')}",
                    "RUN_NAME": "foreign-svc-run",
                    "LAUNCH_ID": "our-launch-id",
                    "KUBE_CONTEXT": "mock-ctx",
                    "NODEPOOL": "mock-pool",
                    "GCS_BUCKET": "mock-bucket",
                    "RUN_ROOT": "mock-runs",
                    "INPUTS_ROOT": "mock-inputs",
                    "IMAGE": "mock-image:latest",
                }
            )
            res = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            deleted = dict(_MockKubeProxyHandler.deletes)
            self.assertNotIn("/api/v1/namespaces/default/services/foreign-svc-run-head", deleted)
            self.assertIn(
                "/apis/resource.nvidia.com/v1beta1/namespaces/default/computedomains/foreign-svc-run-cd",
                deleted,
            )

    def test_delete_failure_is_logged_to_stderr(self):
        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        _MockKubeProxyHandler.status_code = 500
        with tempfile.TemporaryDirectory() as td:
            bin_dir = self._make_mock_bins(Path(td), fail_on_create_idx=2)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env.get('PATH', '')}",
                    "RUN_NAME": "del-fail-run",
                    "KUBE_CONTEXT": "mock-ctx",
                    "NODEPOOL": "mock-pool",
                    "GCS_BUCKET": "mock-bucket",
                    "RUN_ROOT": "mock-runs",
                    "INPUTS_ROOT": "mock-inputs",
                    "IMAGE": "mock-image:latest",
                }
            )
            res = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("ERROR: Kubernetes DELETE failed", res.stderr)

    def test_collision_or_gcs_permission_error_fails_closed_without_deletes(self):
        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        for existing_k8s, gcs_mode, expected_err in (
            (True, "not_found", "already exists in namespace"),
            (False, "exists", "already contains artifacts from a previous run"),
            (False, "perm_denied", "failing closed to prevent accidental overwrite"),
            (False, "lock_race", "failed to acquire atomic GCS launch lock"),
        ):
            with tempfile.TemporaryDirectory() as td:
                bin_dir = self._make_mock_bins(
                    Path(td), fail_on_create_idx=99, existing_k8s=existing_k8s, gcs_mode=gcs_mode
                )
                env = os.environ.copy()
                env.update(
                    {
                        "PATH": f"{bin_dir}:{env.get('PATH', '')}",
                        "RUN_NAME": "collision-run",
                        "KUBE_CONTEXT": "mock-ctx",
                        "NODEPOOL": "mock-pool",
                        "GCS_BUCKET": "mock-bucket",
                        "RUN_ROOT": "mock-runs",
                        "INPUTS_ROOT": "mock-inputs",
                        "IMAGE": "mock-image:latest",
                    }
                )
                res = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
                self.assertNotEqual(res.returncode, 0)
                self.assertIn(expected_err, res.stderr)
                self.assertEqual(_MockKubeProxyHandler.deletes, [])

    def test_multinode_16_nodes_partial_create_cleans_up_all_created_workers_by_uid(self):
        """Verify NUM_NODES=16 (64 GPUs) creates worker-0..worker-14 and cleans up all created pods on failure."""
        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        with tempfile.TemporaryDirectory() as td:
            # Objects: 1=CD, 2=Service, 3..17=worker-0..worker-14, 18=head. Fail on 18 (head).
            bin_dir = self._make_mock_bins(Path(td), fail_on_create_idx=18)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env.get('PATH', '')}",
                    "RUN_NAME": "scale16-run",
                    "LAUNCH_ID": "launch-scale16",
                    "NUM_NODES": "16",
                    "KUBE_CONTEXT": "mock-ctx",
                    "NODEPOOL": "mock-pool",
                    "GCS_BUCKET": "mock-bucket",
                    "RUN_ROOT": "mock-runs",
                    "INPUTS_ROOT": "mock-inputs",
                    "IMAGE": "mock-image:latest",
                }
            )
            res = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            deleted = dict(_MockKubeProxyHandler.deletes)
            # Delete CD (uid-obj-1), Service (uid-obj-2), and 15 workers worker-0..worker-14 (uid-obj-3..uid-obj-17).
            self.assertEqual(len(deleted), 17, f"Expected 17 atomic deletes for NUM_NODES=16, got {len(deleted)}")
            for w in range(15):
                self.assertEqual(
                    deleted[f"/api/v1/namespaces/default/pods/scale16-run-worker-{w}"]["preconditions"]["uid"],
                    f"uid-obj-{w + 3}",
                )

    def test_gpus_per_node_validation_and_pod_resource_matching(self):
        """Verify GPUS_PER_NODE rejects >4 or <1 and renders matching Pod GPU requests/limits and EXPECT_GIT_SHA."""
        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        with tempfile.TemporaryDirectory() as td:
            bin_dir = self._make_mock_bins(Path(td), fail_on_create_idx=99)
            for bad_gpus in ("0", "5", "8", "abc"):
                env = os.environ.copy()
                env.update(
                    {
                        "PATH": f"{bin_dir}:{env.get('PATH', '')}",
                        "RUN_NAME": "bad-gpu-run",
                        "GPUS_PER_NODE": bad_gpus,
                        "KUBE_CONTEXT": "mock-ctx",
                        "NODEPOOL": "mock-pool",
                        "GCS_BUCKET": "mock-bucket",
                        "RUN_ROOT": "mock-runs",
                        "INPUTS_ROOT": "mock-inputs",
                        "IMAGE": "mock-image:latest",
                    }
                )
                res = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
                self.assertNotEqual(res.returncode, 0)
                self.assertIn("GPUS_PER_NODE must be an integer between 1 and 4", res.stderr)

        src = launcher.read_text(encoding="utf-8")
        self.assertEqual(src.count('nvidia.com/gpu: "${GPUS_PER_NODE}"'), 4)
        self.assertNotIn('nvidia.com/gpu: "4"', src)
        self.assertEqual(src.count("- name: EXPECT_GIT_SHA"), 2)

    def test_wandb_secret_ref_and_plaintext_rejection(self):
        """Verify WANDB_SECRET_NAME validation/check and rejection of plaintext JOBENV_WANDB_API_KEY."""
        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        with tempfile.TemporaryDirectory() as td:
            bin_dir = self._make_mock_bins(Path(td), fail_on_create_idx=99)
            base_env = os.environ.copy()
            base_env.update(
                {
                    "PATH": f"{bin_dir}:{base_env.get('PATH', '')}",
                    "RUN_NAME": "wandb-test-run",
                    "KUBE_CONTEXT": "mock-ctx",
                    "NODEPOOL": "mock-pool",
                    "GCS_BUCKET": "mock-bucket",
                    "RUN_ROOT": "mock-runs",
                    "INPUTS_ROOT": "mock-inputs",
                    "IMAGE": "mock-image:latest",
                }
            )
            # 1. Plaintext JOBENV_WANDB_API_KEY must be rejected
            env_plain = dict(base_env, JOBENV_WANDB_API_KEY="secret-value")
            res_plain = subprocess.run(["bash", str(launcher)], env=env_plain, capture_output=True, text=True)
            self.assertNotEqual(res_plain.returncode, 0)
            self.assertIn("refusing to inject plaintext JOBENV_WANDB_API_KEY", res_plain.stderr)

            # 2. Missing K8s Secret when WANDB_SECRET_NAME is set must fail closed
            env_missing = dict(base_env, WANDB_SECRET_NAME="xiaotongyang-wandb")
            res_missing = subprocess.run(["bash", str(launcher)], env=env_missing, capture_output=True, text=True)
            self.assertNotEqual(res_missing.returncode, 0)
            self.assertIn("Secret xiaotongyang-wandb (key WANDB_API_KEY) not found or empty", res_missing.stderr)

    def test_precheck_json_emission_and_cross_node_readiness_parsing(self):
        """Verify verify_venv_lock writes structured JSON and the launcher readiness snippet parses it end-to-end."""
        from examples.tpu.retool.gb200 import verify_venv_lock

        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        src = launcher.read_text(encoding="utf-8")
        self.assertEqual(
            src.count('--output-json "${RUN_DIR}/env/${LAUNCH_ID}-\\${HOST_TAG}-precheck.json"'),
            2,
        )
        m = re.search(r"python3 -c '([^']+)' \"\$\{RUN_DIR\}/env\"", src)
        self.assertIsNotNone(m, "Failed to extract cross-node readiness Python snippet from launcher")
        readiness_code = m.group(1)

        with tempfile.TemporaryDirectory() as td:
            env_dir = Path(td)
            lid = "launch-001"
            expected_sha = "a2cbde47ba729d0cba7e881da1692df20d8e05e4"
            lock_sha = "df508db0b957fa181c229fb75885cd3863cb651fd5233d0fa5af95e990bda0be"
            comp_sha = "1" * 64

            for host in ("head", "worker-0"):
                verify_venv_lock.write_precheck_json(
                    str(env_dir / f"{lid}-{host}-precheck.json"),
                    repo="/workspace/verl-pin",
                    head_sha=expected_sha,
                    lock_sha=lock_sha,
                    expect_git_sha=expected_sha,
                    expect_lock_sha256=lock_sha,
                    checked_packages={"torch": "2.13.0+cu130", "vllm": "0.32.0"},
                )
                (env_dir / f"{lid}-{host}-g0-verify.json").write_text(
                    json.dumps({"g0_checkpoint_manifest": {"composite_sha256": comp_sha}}),
                    encoding="utf-8",
                )

            ok_res = subprocess.run(
                [sys.executable, "-c", readiness_code, str(env_dir), lid, "2", expected_sha],
                capture_output=True,
                text=True,
            )
            self.assertEqual(ok_res.returncode, 0, msg=ok_res.stderr)
            self.assertIn(expected_sha, ok_res.stdout)

            # Mismatched EXPECT_GIT_SHA must fail
            bad_expect = subprocess.run(
                [sys.executable, "-c", readiness_code, str(env_dir), lid, "2", "deadbeef" * 5],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(bad_expect.returncode, 0)

            # Plain-text non-JSON in *-precheck.json must fail with JSONDecodeError
            (env_dir / f"{lid}-worker-0-precheck.json").write_text("plain text output\n", encoding="utf-8")
            bad_json = subprocess.run(
                [sys.executable, "-c", readiness_code, str(env_dir), lid, "2", expected_sha],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(bad_json.returncode, 0)


class TestG0ManifestCompositeHashCheck(unittest.TestCase):
    """Verify verify_g0_gb200.verify_checkpoint_manifest checks --expect-composite-sha256."""

    def test_verify_checkpoint_manifest_expected_composite_sha256(self):
        from examples.tpu.retool.gb200 import verify_g0_gb200

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
                (model_dir / name).write_text("{}", encoding="utf-8")
            for idx in range(1, 17):
                (model_dir / f"model-{idx:05d}-of-00016.safetensors").write_bytes(f"shard-{idx}".encode())

            rep = verify_g0_gb200.verify_checkpoint_manifest(model_dir)
            actual = rep["composite_sha256"]
            self.assertIsNone(rep["expected_composite_sha256"])
            self.assertIsNone(rep["composite_sha256_match"])

            rep_ok = verify_g0_gb200.verify_checkpoint_manifest(model_dir, expect_composite_sha256=actual)
            self.assertEqual(rep_ok["expected_composite_sha256"], actual)
            self.assertTrue(rep_ok["composite_sha256_match"])

            rep_bad = verify_g0_gb200.verify_checkpoint_manifest(model_dir, expect_composite_sha256="0" * 64)
            self.assertEqual(rep_bad["expected_composite_sha256"], "0" * 64)
            self.assertFalse(rep_bad["composite_sha256_match"])

            ref_dir = Path(td) / "ref"
            ref_dir.mkdir()
            for p in model_dir.iterdir():
                if p.is_file():
                    (ref_dir / p.name).write_bytes(p.read_bytes())

            rep_ref_ok = verify_g0_gb200.verify_checkpoint_manifest(
                model_dir, ref_dir=ref_dir, expect_composite_sha256=actual
            )
            # Pinned fast path: the staged copy is fully hashed, the reference is only size-checked,
            # so no reference hash may be reported.
            self.assertTrue(rep_ref_ok["ref_match"])
            self.assertEqual(rep_ref_ok["ref_check_mode"], "size_only")
            self.assertFalse(rep_ref_ok["ref_sha256_verified"])
            self.assertIsNone(rep_ref_ok["ref_composite_sha256"])

            rep_ref_full = verify_g0_gb200.verify_checkpoint_manifest(
                model_dir, ref_dir=ref_dir, expect_composite_sha256=actual, full_ref_hash=True
            )
            self.assertEqual(rep_ref_full["ref_check_mode"], "sha256")
            self.assertTrue(rep_ref_full["ref_match"])
            self.assertTrue(rep_ref_full["ref_sha256_verified"])
            self.assertEqual(rep_ref_full["ref_composite_sha256"], actual)

            # Same size, different content: undetectable by the size-only check (and therefore never
            # reported as hash-verified), detected by the full reference hash.
            shard = ref_dir / "model-00001-of-00016.safetensors"
            original = shard.read_bytes()
            shard.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
            rep_same_size = verify_g0_gb200.verify_checkpoint_manifest(
                model_dir, ref_dir=ref_dir, expect_composite_sha256=actual
            )
            self.assertEqual(rep_same_size["ref_check_mode"], "size_only")
            self.assertFalse(rep_same_size["ref_sha256_verified"])
            self.assertIsNone(rep_same_size["ref_composite_sha256"])
            rep_same_size_full = verify_g0_gb200.verify_checkpoint_manifest(
                model_dir, ref_dir=ref_dir, expect_composite_sha256=actual, full_ref_hash=True
            )
            self.assertFalse(rep_same_size_full["ref_match"])
            self.assertFalse(rep_same_size_full["ref_sha256_verified"])
            self.assertNotEqual(rep_same_size_full["ref_composite_sha256"], actual)
            self.assertIn("model-00001-of-00016.safetensors", rep_same_size_full["mismatches"])

            # Without a matched pinned composite the reference is always fully hashed.
            rep_unpinned = verify_g0_gb200.verify_checkpoint_manifest(model_dir, ref_dir=ref_dir)
            self.assertEqual(rep_unpinned["ref_check_mode"], "sha256")
            self.assertFalse(rep_unpinned["ref_match"])

            shard.write_bytes(b"corrupted-size")
            rep_ref_bad = verify_g0_gb200.verify_checkpoint_manifest(
                model_dir, ref_dir=ref_dir, expect_composite_sha256=actual
            )
            self.assertFalse(rep_ref_bad["ref_match"])
            self.assertIn("model-00001-of-00016.safetensors", rep_ref_bad["mismatches"])


class TestDistributedTpuGuards(unittest.TestCase):
    """The GB200 IPv4 socket fix in verl/utils/distributed.py must not change the shared TPU path."""

    @staticmethod
    def _functions():
        import ast

        src = (REPO_ROOT / "verl" / "utils" / "distributed.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        return src, {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def test_tpu_branches_are_kept(self):
        import ast

        src, funcs = self._functions()
        numa = ast.get_source_segment(src, funcs["set_numa_affinity"])
        self.assertLess(numa.index('if device_name == "TPU":'), numa.index('ctypes.CDLL("libnuma.so")'))
        ray_init = ast.get_source_segment(src, funcs["initialize_global_process_group_ray"])
        self.assertIn("import torch_tpu", ray_init)
        self.assertLess(ray_init.index('get_platform().device_name == "tpu"'), ray_init.index("init_process_group("))
        self.assertLess(ray_init.index("import torch_tpu"), ray_init.index("_configure_ipv4_socket_env_if_needed()"))

    def test_ipv4_socket_env_is_noop_on_tpu_and_set_on_gpu(self):
        import ast
        import types

        src, funcs = self._functions()
        fn_src = ast.get_source_segment(src, funcs["_configure_ipv4_socket_env_if_needed"])
        keys = ("MASTER_ADDR", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "NCCL_SOCKET_FAMILY")
        for device_name, expect_set in (("tpu", False), ("cuda", True)):
            ns = {
                "os": os,
                "get_platform": lambda device_name=device_name: types.SimpleNamespace(device_name=device_name),
                "is_ipv6": lambda addr: ":" in addr,
            }
            exec(fn_src, ns)
            env = {k: v for k, v in os.environ.items() if k not in keys}
            env["MASTER_ADDR"] = "10.0.0.2"
            with mock.patch.dict(os.environ, env, clear=True), mock.patch("os.path.exists", return_value=True):
                ns["_configure_ipv4_socket_env_if_needed"]()
                self.assertEqual("GLOO_SOCKET_IFNAME" in os.environ, expect_set, device_name)
                self.assertEqual(os.environ.get("NCCL_SOCKET_FAMILY") == "AF_INET", expect_set, device_name)


class TestG64ReportTables(unittest.TestCase):
    """Archived GB200 reports' G64-D tables must be the verbatim output of logs/gb200/gen_g64_report_tables.py."""

    LOGS_DIR = REPO_ROOT / "examples" / "tpu" / "retool" / "logs" / "gb200"
    DOCS = (
        REPO_ROOT / "examples" / "tpu" / "retool" / "exp" / "gb200" / "GB200_REPRO_GUIDE_V1.md",
        REPO_ROOT / "examples" / "tpu" / "retool" / "exp" / "gb200" / "REPRO_RESULTS_GB200_ZH.md",
    )

    @classmethod
    def setUpClass(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location("gen_g64", cls.LOGS_DIR / "gen_g64_report_tables.py")
        cls.gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gen)
        cls.out = {run: cls.gen.build_all(run) for run in cls.gen.RUNS}

    def test_head_log_matches_metrics_json_and_reviewer_spot_checks(self):
        self.assertGreater(self.out["g64-retool-smoke-04"]["head_log_vs_metrics_json_values_compared"], 900)
        self.assertGreater(self.out["g64-retool-25s-01"]["head_log_vs_metrics_json_values_compared"], 2500)
        steps = self.gen.parse_head_log_steps(self.LOGS_DIR / "g64-retool-smoke-04" / "head.log")
        self.assertAlmostEqual(steps[1]["metrics"]["actor/grad_norm"], 0.17162081599235535, places=12)
        self.assertAlmostEqual(steps[5]["metrics"]["critic/score/mean"], -0.008007812313735485, places=12)
        smoke = self.out["g64-retool-smoke-04"]["training_table"]
        self.assertIn("`0.1716`", smoke)
        self.assertIn("`-0.0080`", smoke)
        self.assertIn("Steps 2–5 mean (4 steps)", smoke)
        self.assertIn("Steps 2–25 mean (24 steps)", self.out["g64-retool-25s-01"]["training_table"])
        self.assertIn("askhat_table", self.out["g64-retool-25s-01"])

    def test_doc_blocks_match_generator(self):
        for doc in self.DOCS:
            text = doc.read_text(encoding="utf-8")
            for run, tables in self.out.items():
                for name in self.gen.TABLES:
                    if name not in tables:
                        continue
                    m = re.search(
                        rf"<!-- BEGIN AUTO-GENERATED {run}/{name} \(gen_g64_report_tables\.py\) -->\n(.*?)\n"
                        rf"<!-- END AUTO-GENERATED {run}/{name} -->",
                        text,
                        re.S,
                    )
                    self.assertIsNotNone(m, f"{doc.name}: missing AUTO-GENERATED {run}/{name} block")
                    self.assertEqual(m.group(1), tables[name], f"{doc.name}: {run}/{name} differs from generator")

    def test_25_step_run_config_matches_smoke_except_step_count_and_paths(self):
        changed = {
            d[1:].strip().split("'")[1]
            for d in self.gen.resolved_config_diff("g64-retool-smoke-04", "g64-retool-25s-01")
        }
        self.assertEqual(
            changed,
            {
                "experiment_name",
                "TENSORBOARD_DIR",
                "default_local_dir",
                "rollout_audit_dir",
                "rollout_data_dir",
                "test_freq",
                "total_training_steps",
                "validation_data_dir",
            },
        )

    def test_recompute_script_uses_runtime_scorer(self):
        script = self.LOGS_DIR / "recompute_g64_metrics.py"
        self.assertNotIn("def compute_expected_shaped_score", script.read_text(encoding="utf-8"))
        for run, final in (("g64-retool-smoke-04", 5), ("g64-retool-25s-01", 25)):
            res = subprocess.run(
                [sys.executable, str(script), "--run", run], capture_output=True, text=True, cwd=str(REPO_ROOT)
            )
            self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            summary = json.loads(res.stdout)
            self.assertEqual(summary["env_verification"]["num_nodes_verified"], 16)
            self.assertIn("retool_tpu.py::compute_score", summary["validation_scorer"]["function"])
            for step_key in ("validation_step_0", f"validation_step_{final}"):
                for ds in ("aime_2024", "aime_2025"):
                    self.assertEqual(
                        summary[step_key][ds]["rescored_samples_matching_logged_pred_acc_score"], "900 / 900"
                    )
            self.assertEqual(summary["training_rollouts"]["per_sample_rescoring"].split(" ")[0], "not")

    def _build_ev(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("build_ev", self.LOGS_DIR / "build_g64_run_evidence.py")
        build_ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_ev)
        return build_ev

    def test_mask_audit_flags_a_tool_response_straddling_segments(self):
        a = self._build_ev().mask_audit
        text = {a.IM_START: "<|im_start|>", a.IM_END: "<|im_end|>", a.NEWLINE: "\n", a.USER: "user"}
        text |= {a.ASSISTANT: "assistant", a.TOOL_CALL_OPEN: "<tool_call>", a.TOOL_CALL_CLOSE: "</tool_call>"}
        text |= {a.TOOL_RESPONSE_OPEN: "<tool_response>", a.TOOL_RESPONSE_CLOSE: "</tool_response>"}
        model = [1, a.TOOL_CALL_OPEN, 2, a.TOOL_CALL_CLOSE, a.IM_END]
        turn = [a.NEWLINE, a.IM_START, a.USER, a.NEWLINE, a.TOOL_RESPONSE_OPEN, 5, 6, a.TOOL_RESPONSE_CLOSE]
        turn += [a.IM_END, a.NEWLINE, a.IM_START, a.ASSISTANT, a.NEWLINE]
        for leak_from, expect_violation in ((len(turn), False), (6, True)):
            ids = model + turn + [7, a.IM_END]
            mask = [1] * len(model) + [0 if i < leak_from else 1 for i in range(len(turn))] + [1, 1]
            segs = [
                {"mask": m, "start": i, "end": j, "text": "".join(text.get(t, "w") for t in ids[i:j])}
                for m, i, j in a._runs(mask)
            ]
            rec = {"response_ids": ids, "response_mask": mask, "segments": segs, "mask_length_matches": True}
            res = a.audit_session(rec)
            self.assertEqual(bool(res["violations"]), expect_violation, res)
            self.assertFalse(a.per_segment_check(rec), "the earlier per-segment check misses both cases")

    def test_mask_audit_recomputed_from_archived_raw_sessions(self):
        import base64
        import gzip
        import hashlib

        build_ev = self._build_ev()
        for run, final in (("g64-retool-smoke-04", 5), ("g64-retool-25s-01", 25)):
            ev_dir = self.LOGS_DIR / run
            manifest = json.loads((ev_dir / "raw_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["files"]), final + 1)
            for f in manifest["files"]:
                data = (ev_dir / f["path"]).read_bytes()
                data = gzip.decompress(data) if f["path"].endswith(".gz") else data
                self.assertEqual(hashlib.sha256(data).hexdigest(), f["sha256"], f["path"])
                self.assertEqual(base64.b64encode(hashlib.md5(data).digest()).decode(), f["md5_base64"], f["path"])
            summary = json.loads((ev_dir / "rollout_and_audit_summary.json").read_text(encoding="utf-8"))
            for s, audit in build_ev.audit_steps(ev_dir / "rollout_audit", final, ".jsonl.gz").items():
                self.assertEqual({k: summary["steps"][s][k] for k in audit}, audit, f"{run} step {s}")
            t = summary["totals"]
            self.assertEqual(t["total_audit_sessions"], 8 * final)
            for key in (
                "audit_mask_violations",
                "audit_tool_response_tokens_in_loss",
                "audit_tool_call_tokens_not_in_loss",
                "audit_mask0_runs_not_tool_turn",
                "audit_im_start_in_loss",
            ):
                self.assertEqual(t[key], 0, f"{run}: {key}")
        # Two sessions of the 25-step run have malformed tool-call tags written by the model, all in the loss.
        self.assertEqual(t["audit_sessions_malformed_tool_call_tags"], 2)
        self.assertEqual((t["audit_tool_response_blocks"], t["audit_mask0_runs"]), (706, 706))

    def test_timing_breakdown_recomputed_from_tensorboard_events(self):
        build_ev = self._build_ev()
        for run, spec in self.gen.RUNS.items():
            ev_dir = self.LOGS_DIR / run
            metrics = json.loads((ev_dir / spec["metrics"]).read_text(encoding="utf-8"))
            archived = json.loads((ev_dir / "step_timing_breakdown.json").read_text(encoding="utf-8"))
            fresh = build_ev.timing_breakdown(ev_dir, metrics, spec["final_step"])
            self.assertEqual(json.loads(json.dumps(fresh)), archived, run)
            self.assertEqual(archived["log_utc_offset_hours"], 0.0)
        s = archived["summary"]
        self.assertEqual((s["steps"], s["n"]), ([2, 24], 23))
        self.assertAlmostEqual(s["mean_wall_s"], 73.171, places=3)
        self.assertAlmostEqual(s["mean_tb_span_prev_step_s"], 39.956, places=3)
        self.assertLess(abs(s["mean_residual_s"] - s["mean_post_sync_to_log_s"]), 0.1)
        step25 = archived["rows"][-1]
        self.assertAlmostEqual(step25["timing_s_testing"], 126.531, places=3)


class TestV1ValidateDumpNumTurns(unittest.TestCase):
    """Verify V1 PPOTrainer._validate writes session num_turns for shaped reward reconciliation."""

    def test_v1_validate_dump_writes_num_turns_and_reconciles_shaped_reward(self):
        import ast
        import importlib
        import importlib.util
        import math
        import types

        test_path = REPO_ROOT / "tests/trainer/ppo/v1/test_rollout_audit_on_cpu.py"
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("_collate", "test_validate_dump_writes_num_turns_and_reconciles_shaped_reward")
        ]
        mod = ast.Module(body=nodes, type_ignores=[])
        ast.fix_missing_locations(mod)

        ra_spec = importlib.util.spec_from_file_location(
            "verl.trainer.ppo.v1.rollout_audit", REPO_ROOT / "verl/trainer/ppo/v1/rollout_audit.py"
        )
        ra_mod = importlib.util.module_from_spec(ra_spec)
        ra_spec.loader.exec_module(ra_mod)

        pytest_mod = types.ModuleType("pytest")

        class _Approx:
            def __init__(self, val, rel=1e-6, abs_tol=1e-12):
                self.val = val
                self.rel = rel
                self.abs_tol = abs_tol

            def __eq__(self, other):
                return math.isclose(self.val, other, rel_tol=self.rel, abs_tol=self.abs_tol)

        pytest_mod.approx = lambda val, **_kw: _Approx(val)
        verl_pkg = types.ModuleType("verl")
        verl_pkg.__file__ = str(REPO_ROOT / "verl/__init__.py")
        ns = {
            "Path": Path,
            "verl": verl_pkg,
            "pytest": pytest_mod,
            "json": json,
            "np": types.SimpleNamespace(
                array=lambda x, dtype=object: list(x),
                fromiter=lambda it, dtype=object, count=None: list(it),
            ),
            "importlib": importlib,
            "split_trajectory_key": ra_mod.split_trajectory_key,
            "summarize_validation_dump": ra_mod.summarize_validation_dump,
            "validation_dump_identities": ra_mod.validation_dump_identities,
            "validation_dump_identity_columns": ra_mod.validation_dump_identity_columns,
            "validation_question_id": ra_mod.validation_question_id,
        }
        exec(compile(mod, str(test_path), "exec"), ns)
        with tempfile.TemporaryDirectory() as td:
            ns["test_validate_dump_writes_num_turns_and_reconciles_shaped_reward"](Path(td))

    def test_dapo_script_scales_max_token_len_per_gpu_for_16k_response(self):
        dapo_script = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh"
        src = dapo_script.read_text(encoding="utf-8")
        self.assertIn(
            "MIN_TOKEN_LEN_PER_GPU=$(( (max_model_len + CONTEXT_PARALLEL_SIZE - 1) / CONTEXT_PARALLEL_SIZE ))",
            src,
        )
        self.assertIn(
            "DEFAULT_MAX_TOKEN_LEN_PER_GPU=$(( MIN_TOKEN_LEN_PER_GPU > 16384 ? MIN_TOKEN_LEN_PER_GPU : 16384 ))",
            src,
        )

    def test_dapo_script_auto_enables_wandb_when_wandb_api_key_present(self):
        dapo_script = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_qwen3_30b_a3b_dapo_torchtitan_gb200.sh"
        src = dapo_script.read_text(encoding="utf-8")
        self.assertIn('if [[ -z "${LOGGER_LIST:-}" ]]; then', src)
        self.assertIn('if [[ -n "${WANDB_API_KEY:-}" ]]; then', src)
        self.assertIn("LOGGER_LIST=\"['console','tensorboard','wandb']\"", src)

    def test_launcher_exclude_nodes_renders_node_affinity(self):
        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        src = launcher.read_text(encoding="utf-8")
        self.assertIn('if [[ -n "${EXCLUDE_NODES:-}" ]]; then', src)
        self.assertEqual(src.count("${NODE_AFFINITY_YAML}"), 4)
        snippet = """
import sys
nodes = [n.strip() for n in sys.argv[1].split(",") if n.strip()]
if nodes:
    print("    nodeAffinity:")
    print("      requiredDuringSchedulingIgnoredDuringExecution:")
    print("        nodeSelectorTerms:")
    print("        - matchExpressions:")
    print("          - key: kubernetes.io/hostname")
    print("            operator: NotIn")
    print("            values:")
    for n in nodes:
        print(f"            - {n}")
"""
        res = subprocess.run([sys.executable, "-c", snippet, "bad-node-1, bad-node-2 "], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("key: kubernetes.io/hostname", res.stdout)
        self.assertIn("operator: NotIn", res.stdout)
        self.assertIn("- bad-node-1", res.stdout)
        self.assertIn("- bad-node-2", res.stdout)

    def test_recompute_g64_smoke_metrics_script(self):
        recompute_script = REPO_ROOT / "examples" / "tpu" / "retool" / "logs" / "gb200" / "recompute_g64_metrics.py"
        res = subprocess.run(
            [sys.executable, str(recompute_script), "--run", "g64-retool-smoke-04"], capture_output=True, text=True
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        payload = json.loads(res.stdout)
        self.assertEqual(payload["run_name"], "g64-retool-smoke-04")
        self.assertEqual(payload["env_verification"]["num_nodes_verified"], 16)
        self.assertEqual(payload["env_verification"]["total_gpus_verified"], 64)
        self.assertEqual(
            payload["env_verification"]["staged_checkpoint_composite_sha256_all_nodes"],
            "59247a00c13f2364a12aab5ab6b88b6d971d59ff643c57aad8ac4e55a71231b0",
        )
        self.assertEqual(
            payload["validation_step_0"]["aime_2025"]["rescored_samples_matching_logged_pred_acc_score"], "900 / 900"
        )
        self.assertEqual(
            payload["validation_step_5"]["aime_2025"]["rescored_samples_matching_logged_pred_acc_score"], "900 / 900"
        )


if __name__ == "__main__":
    unittest.main()
