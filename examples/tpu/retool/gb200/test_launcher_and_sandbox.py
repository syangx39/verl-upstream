#!/usr/bin/env python3
"""Unit tests for GB200 launcher UID-preconditioned cleanup, external sandbox wiring, and fail-closed privilege drop."""

import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

    def do_DELETE(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        self.__class__.deletes.append((self.path, body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"kind":"Status","status":"Success"}')

    def log_message(self, format, *args):  # noqa: A002
        return


class TestLauncherResourceLifecycle(unittest.TestCase):
    """Verify run_ray_jobset_gb200.sh per-object UID capture, partial failure cleanup, and collision checks."""

    def setUp(self):
        _MockKubeProxyHandler.deletes = []
        self.server = HTTPServer(("127.0.0.1", 0), _MockKubeProxyHandler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _make_mock_bins(self, tmp_dir: Path, fail_on_create_idx: int = 3, existing_k8s: bool = False, existing_gcs: bool = False):
        bin_dir = tmp_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        state_file = tmp_dir / "create_count.txt"
        state_file.write_text("0", encoding="utf-8")

        kubectl_script = bin_dir / "kubectl"
        kubectl_script.write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail
args=("$@")
if [[ "${{args[*]}}" == *" get "* ]]; then
    if [[ "{1 if existing_k8s else 0}" == "1" ]]; then
        exit 0
    fi
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
        gsutil_script.write_text(
            f"""#!/usr/bin/env bash
if [[ "$*" == *"stat"* ]]; then
    exit {0 if existing_gcs else 1}
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
            # Object 1 (ComputeDomain: uid-obj-1) and Object 2 (Service: uid-obj-2) succeeded before Object 3 failed.
            # Cleanup must have issued atomic DELETEs with DeleteOptions.preconditions.uid for exactly those two!
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

    def test_collision_on_existing_k8s_or_gcs_aborts_without_deletes(self):
        launcher = REPO_ROOT / "examples" / "tpu" / "retool" / "gb200" / "run_ray_jobset_gb200.sh"
        for existing_k8s, existing_gcs in ((True, False), (False, True)):
            with tempfile.TemporaryDirectory() as td:
                bin_dir = self._make_mock_bins(
                    Path(td), fail_on_create_idx=99, existing_k8s=existing_k8s, existing_gcs=existing_gcs
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
                self.assertEqual(_MockKubeProxyHandler.deletes, [])


if __name__ == "__main__":
    unittest.main()
