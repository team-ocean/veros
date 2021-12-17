import os
import sys
import subprocess

import pytest


def run_dist_kernel(kernel):
    pytest.importorskip("mpi4py")

    here = os.path.dirname(__file__)
    return subprocess.check_call(
        [sys.executable, "-m", "mpi4py", os.path.join(here, kernel)], stderr=subprocess.STDOUT, timeout=300
    )


def test_gather():
    run_dist_kernel("gather_kernel.py")


def test_scatter():
    run_dist_kernel("scatter_kernel.py")


def test_acc():
    run_dist_kernel("acc_kernel.py")


@pytest.mark.parametrize("solver", ["scipy", "scipy_jax", "petsc"])
@pytest.mark.parametrize("streamfunction", [True, False])
def test_linear_solver(solver, streamfunction):
    from veros import runtime_settings

    if solver == "scipy_jax" and runtime_settings.backend != "jax":
        pytest.skip("scipy_jax solver requires JAX")

    kernel = "streamfunction_kernel.py" if streamfunction else "pressure_kernel.py"
    orig_solver = os.environ.get("VEROS_LINEAR_SOLVER", "best")
    try:
        os.environ["VEROS_LINEAR_SOLVER"] = solver
        run_dist_kernel(kernel)
    finally:
        os.environ["VEROS_LINEAR_SOLVER"] = orig_solver
