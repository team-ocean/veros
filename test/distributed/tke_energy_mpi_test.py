"""Global TKE allocation and derivatives agree across process partitions."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("backend", ("numpy", "jax"))
def test_tke_energy_partition_independence(backend):
    pytest.importorskip("mpi4py")
    if backend == "jax":
        pytest.importorskip("mpi4jax")
    launcher = shutil.which("mpiexec")
    if launcher is None:
        installed = Path(sys.executable).with_name("mpiexec")
        if not installed.exists():
            pytest.skip("requires mpiexec")
        launcher = str(installed)
    kernel = Path(__file__).with_name("tke_energy_kernel.py")
    environment = dict(
        os.environ,
        PYTHONPATH=str(kernel.parents[2]),
        JAX_PLATFORMS="cpu",
        VEROS_DEVICE="cpu",
        VEROS_USE_SPECIAL_TDMA="0",
        UCX_TLS="tcp,self",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    results = []
    for ranks in (1, 2, 4):
        completed = subprocess.run(
            [launcher, "-n", str(ranks), sys.executable, str(kernel), backend],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        line = next(line for line in completed.stdout.splitlines() if line.startswith("TKE_ENERGY_RESULT "))
        results.append(json.loads(line.split(" ", 1)[1]))
    for result in results[1:]:
        for name in ("value", "jvp", "vjp") if backend == "jax" else ("value",):
            np.testing.assert_allclose(result[name], results[0][name], rtol=1e-10, atol=1e-11)


def test_positive_tke_compiled_ocean_and_checkpoint_mpi():
    pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4jax")
    launcher = str(Path(sys.executable).with_name("mpiexec"))
    if not Path(launcher).exists():
        pytest.skip("requires mpiexec")
    kernel = Path(__file__).with_name("integration_kernel.py")
    environment = dict(
        os.environ,
        PYTHONPATH=str(kernel.parents[2]),
        JAX_PLATFORMS="cpu",
        VEROS_DEVICE="cpu",
        VEROS_USE_SPECIAL_TDMA="0",
        VEROS_LINEAR_SOLVER="scipy_jax",
        UCX_TLS="tcp,self",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    results = []
    for ranks in (1, 2):
        completed = subprocess.run(
            [launcher, "-n", str(ranks), sys.executable, str(kernel), "pressure", "scipy_jax", "positive_diffusion"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=450,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        line = next(line for line in completed.stdout.splitlines() if line.startswith("INTEGRATION_RESULT "))
        results.append(json.loads(line.split(" ", 1)[1]))
    for name in ("value", "gradient", "parameter_gradient"):
        np.testing.assert_allclose(results[1][name], results[0][name], rtol=2e-5, atol=1e-5)
