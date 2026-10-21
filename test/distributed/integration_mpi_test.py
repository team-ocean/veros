import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from veros import runtime_settings


@pytest.mark.parametrize("solver", ("streamfunction", "pressure"))
@pytest.mark.parametrize("implementation", ("scipy_jax", "petsc"))
def test_compiled_mpi(solver, implementation):
    if runtime_settings.backend != "jax":
        pytest.skip("compiled integration requires JAX")
    pytest.importorskip("mpi4py")
    pytest.importorskip("mpi4jax")
    if implementation == "petsc":
        pytest.importorskip("petsc4py.PETSc")
    launcher = shutil.which("mpiexec")
    if launcher is None:
        pytest.skip("mpiexec is required")
    kernel = Path(__file__).with_name("integration_kernel.py")
    env = dict(
        os.environ,
        VEROS_DEVICE="cpu",
        VEROS_USE_SPECIAL_TDMA="0",
        JAX_PLATFORMS="cpu",
        VEROS_LINEAR_SOLVER=implementation,
    )
    env["PYTHONPATH"] = str(kernel.parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    for ranks in (1, 2, 4):
        print(f"Compiled MPI {solver}: {ranks} ranks", flush=True)
        completed = subprocess.run(
            [launcher, "-n", str(ranks), sys.executable, str(kernel), solver],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        line = next(line for line in completed.stdout.splitlines() if line.startswith("INTEGRATION_RESULT "))
        results.append(json.loads(line.split(" ", 1)[1]))
    for result in results[1:]:
        np.testing.assert_allclose(result["value"], results[0]["value"], rtol=1e-8, atol=1e-6)
        np.testing.assert_allclose(result["gradient"], results[0]["gradient"], rtol=2e-5, atol=1e-5)
        np.testing.assert_allclose(result["parameter_gradient"], results[0]["parameter_gradient"], rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("solver", ("streamfunction", "pressure"))
@pytest.mark.parametrize("implementation", ("scipy_jax", "petsc"))
def test_compiled_gpu_mpi(solver, implementation):
    if runtime_settings.backend != "jax" or runtime_settings.device != "gpu":
        pytest.skip("requires the JAX GPU backend")
    if implementation == "petsc":
        pytest.importorskip("petsc4py.PETSc")
    mpi = pytest.importorskip("mpi4jax")
    if not mpi.has_cuda_support():
        pytest.skip("mpi4jax CUDA support is required")
    launcher = shutil.which("mpiexec")
    if launcher is None:
        pytest.skip("mpiexec is required")
    kernel = Path(__file__).with_name("integration_kernel.py")
    env = dict(
        os.environ,
        VEROS_DEVICE="gpu",
        VEROS_USE_SPECIAL_TDMA="0",
        JAX_PLATFORMS="cuda",
        VEROS_LINEAR_SOLVER="scipy_jax",
        MPI4JAX_USE_CUDA_MPI="0",
    )
    env["PYTHONPATH"] = str(kernel.parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    for ranks in (1, 2):
        print(f"Compiled GPU MPI {solver}: {ranks} ranks", flush=True)
        completed = subprocess.run(
            [launcher, "-n", str(ranks), sys.executable, str(kernel), solver, implementation],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        line = next(line for line in completed.stdout.splitlines() if line.startswith("INTEGRATION_RESULT "))
        results.append(json.loads(line.split(" ", 1)[1]))
    for name in ("value", "gradient", "parameter_gradient"):
        assert np.isfinite(results[0][name])
        np.testing.assert_allclose(results[1][name], results[0][name], rtol=2e-5, atol=1e-5)
