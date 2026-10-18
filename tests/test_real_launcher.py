"""Exercise the real launcher without installing packages or starting any servers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
POWERSHELL = shutil.which("powershell.exe")
VENV = Path(sys.executable).resolve().parent.parent
WINDOWS_VENV = os.name == "nt" and Path(sys.executable).parent.name.lower() == "scripts"


def ps_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(POWERSHELL and WINDOWS_VENV, "Requires Windows PowerShell and a Python virtual environment")
class RealLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="forte-real-launch-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "scripts").mkdir()
        (self.root / "backend").mkdir()
        for name in ("common.ps1", "real.ps1"):
            shutil.copyfile(ROOT / "scripts" / name, self.root / "scripts" / name)
        (self.root / "backend" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "backend" / "doctor.py").write_text(
            "import json, os, pathlib, sys\n"
            "payload = {'arguments': sys.argv[1:], 'settings': {k: v for k, v in os.environ.items() if k.startswith('CASH_AUDIT_')}}\n"
            "pathlib.Path(os.environ['AUDIT_TEST_TRACE']).write_text(json.dumps(payload), encoding='utf-8')\n"
            "raise SystemExit(int(os.environ.get('AUDIT_TEST_DOCTOR_EXIT', '0')))\n",
            encoding="utf-8",
        )
        (self.root / "uvicorn.py").write_text(
            "import os, pathlib\n"
            "pathlib.Path(os.environ['AUDIT_TEST_APP_TRACE']).write_text('unexpected start')\n"
            "raise SystemExit(77)\n",
            encoding="utf-8",
        )
        self.trace = self.root / "doctor-trace.json"
        self.app_trace = self.root / "application-trace.txt"

    def run_launcher(self, *, doctor_exit=0, venv=VENV):
        env = os.environ.copy()
        env.update({
            "CASH_AUDIT_MODE": "mock", "CASH_AUDIT_PUBLIC_ORIGIN": "https://stale.example.com",
            "CASH_AUDIT_API_KEY": "launcher-test-sentinel-no-real-secret",
            "AUDIT_TEST_TRACE": str(self.trace), "AUDIT_TEST_APP_TRACE": str(self.app_trace),
            "AUDIT_TEST_DOCTOR_EXIT": str(doctor_exit),
        })
        # The wrapper records restoration even when the launcher raises an error.
        script = (
            "$ErrorActionPreference = 'Stop'\n"
            "$failed = $false\n"
            "try {\n"
            f"  & {ps_quote(self.root / 'scripts' / 'real.ps1')} -CheckOnly -VenvPath {ps_quote(venv)}\n"
            "} catch { $failed = $true; Write-Output $_.Exception.Message }\n"
            "Write-Output ('restored-mode=' + $env:CASH_AUDIT_MODE)\n"
            "Write-Output ('restored-origin=' + $env:CASH_AUDIT_PUBLIC_ORIGIN)\n"
            "if ($failed) { exit 1 }\n"
        )
        return subprocess.run(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
            cwd=self.root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def assert_restored(self, output):
        self.assertIn("restored-mode=mock", output)
        self.assertIn("restored-origin=https://stale.example.com", output)
        self.assertNotIn("launcher-test-sentinel-no-real-secret", output)

    def test_preflight_uses_real_isolated_settings_and_check_only_does_not_start_app(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        trace = json.loads(self.trace.read_text(encoding="utf-8"))
        self.assertEqual(trace["settings"], {
            "CASH_AUDIT_MODE": "real", "CASH_AUDIT_DATA_DIR": str(self.root / "backend" / ".local" / "real"),
        })
        self.assertEqual(trace["arguments"][:4], ["--mode", "real", "--port", "8000"])
        self.assertFalse(self.app_trace.exists())
        self.assert_restored(result.stdout)

    def test_failed_preflight_does_not_launch_application_or_leave_real_environment(self):
        result = self.run_launcher(doctor_exit=7)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Real application preflight failed (exit code 7)", result.stdout)
        self.assertFalse(self.app_trace.exists())
        self.assert_restored(result.stdout)

    def test_research_environment_is_rejected_before_preflight(self):
        protected = self.root / ".venv"
        protected.mkdir()
        marker = protected / "unchanged.txt"
        marker.write_bytes(b"research environment remains unchanged")
        result = self.run_launcher(venv=protected)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Use a separate application environment", result.stdout)
        self.assertEqual(marker.read_bytes(), b"research environment remains unchanged")
        self.assertFalse(self.trace.exists())
        self.assertFalse(self.app_trace.exists())
        self.assert_restored(result.stdout)


if __name__ == "__main__":
    unittest.main()
