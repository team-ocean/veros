import os
import sys
import subprocess

import pytest


def run_dist_kernel(kernel):
    pytest.mark.importorskip("mpi4py")

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
