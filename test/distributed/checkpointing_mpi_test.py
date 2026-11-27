import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from veros import runtime_settings


def test_distributed_checkpoint_replay():
    if runtime_settings.backend != "jax":
        pytest.skip("requires JAX")
    pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4jax")
    launcher = shutil.which("mpiexec")
    if launcher is None:
        pytest.skip("requires mpiexec")
    kernel = Path(__file__).with_name("checkpointing_kernel.py")
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = str(kernel.parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    for ranks in (1, 2, 4):
        completed = subprocess.run(
            [launcher, "-n", str(ranks), sys.executable, str(kernel)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        line = next(line for line in completed.stdout.splitlines() if line.startswith("CHECKPOINT_RESULT "))
        results.append(json.loads(line.split(" ", 1)[1]))
    for result in results[1:]:
        for name in results[0]:
            np.testing.assert_allclose(result[name], results[0][name], rtol=1e-11, atol=1e-11)
