"""Runs the real-browser checks in their own process, like the dashboard they test.

Needs Playwright (src/scripts/setup.ps1 -BrowserTests) and installed Edge. It fails if they are
missing, so a passing suite always means the browser checks ran.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_browser_checks_pass():
    done = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests" / "frontend_checks.py"), "-q", "--tb=short"],
                          cwd=ROOT, capture_output=True, text=True, timeout=570)
    assert done.returncode == 0, done.stdout[-4000:]
