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


if __name__ == "__main__":
    unittest.main()
