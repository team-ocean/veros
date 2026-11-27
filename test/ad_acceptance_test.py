"""Opt-in scientific acceptance lane. Unmet requirements are ordinary failures."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.ad_demos import DEMOS


@pytest.mark.skipif(os.environ.get("VEROS_RUN_AD_ACCEPTANCE") != "1", reason="set VEROS_RUN_AD_ACCEPTANCE=1")
@pytest.mark.parametrize("demo", DEMOS)
def test_research_acceptance(demo, tmp_path):
    timeout = float(os.environ.get("VEROS_AD_TIMEOUT", "600"))
    command = [
        sys.executable,
        "-m",
        "benchmarks.ad_demos",
        "--demo",
        demo,
        "--mode",
        "acceptance",
        "--setup",
        os.environ.get("VEROS_AD_SETUP", "acc"),
        "--physics",
        os.environ.get("VEROS_AD_PHYSICS", "default"),
        "--timeout",
        str(timeout),
        "--output",
        str(tmp_path),
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=timeout + 30
    )
    report = json.loads((tmp_path / f"{demo}.json").read_text())
    assert result.returncode == 0 and report["status"] == "passed", json.dumps(report, indent=2)
