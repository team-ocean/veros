"""Owned-cell turbulence kernels with global nonlinear work allocation."""

import json
import os
from pathlib import Path
import sys

backend = sys.argv[1]
os.environ.update(VEROS_BACKEND=backend, VEROS_DEVICE="cpu", VEROS_DISKLESS_MODE="1")
from mpi4py import MPI  # noqa: E402
import numpy as np  # noqa: E402
from veros import runtime_settings as rs, runtime_state as rst  # noqa: E402

rs.num_proc = (MPI.COMM_WORLD.size, 1)
if backend == "jax":
    import jax

    jax.config.update("jax_enable_x64", True)
    if rst.proc_num > 1:
        from veros._distributed_jax import prepare_communicator

        prepare_communicator(rs.mpi_comm)
from veros.core.operators import numpy as npx  # noqa: E402
from veros.core.tke_energy import redistribute, positive_diffusion_column  # noqa: E402
from veros.distributed import global_sum  # noqa: E402
from veros.routines import tracing_context  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tke_energy_test import columns  # noqa: E402

available = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {available[rst.proc_rank % len(available)]})
rank, count = rst.proc_rank, rst.proc_num
arguments, width = columns()
args = [npx.asarray(np.split(value, count)[rank][:, None]) for value in arguments]
width = npx.asarray(np.split(width, count)[rank][:, None])
energy = np.linspace(-0.19, 0.51, 120).reshape(4, 3, 10)
volume = np.linspace(0.1, 3, energy.size).reshape(energy.shape)
energy, volume = (npx.asarray(np.split(value, count)[rank]) for value in (energy, volume))


def function(z):
    a, b, c, rhs, water, edge, b_edge, rates = args
    factor = npx.exp(z[1])
    with tracing_context():
        work = redistribute(1.3 + z[0], energy + 0.01 * z[1], volume * npx.exp(z[2]), 2.0)
        updated, correction = positive_diffusion_column(
            a * factor,
            1 + (b - 1) * factor,
            c * factor,
            rhs * npx.exp(z[0]),
            water,
            edge,
            1 + (b_edge - 1) * factor,
            rates * npx.exp(z[2]),
        )
        return npx.stack(
            (
                global_sum(npx.sum(work * volume * npx.exp(z[2]))),
                global_sum(npx.sum(work**2 * volume)),
                global_sum(npx.sum(updated * width) + npx.sum(correction)),
            )
        )


point = npx.asarray([0.03, 0.01, -0.02])
report = dict(backend=backend, ranks=count)
if backend == "jax":
    direction = npx.asarray([0.2, -0.3, 0.1])
    seed = npx.asarray([0.1, -0.2, 0.3])
    value, product = jax.jit(lambda z: jax.jvp(function, (z,), (direction,)))(point)
    local_transpose = jax.jit(lambda z: jax.vjp(function, z)[1](seed / count)[0])(point)
    jax.block_until_ready((value, product, local_transpose))
    transpose = MPI.COMM_WORLD.allreduce(np.asarray(local_transpose))
    np.testing.assert_allclose(np.vdot(product, seed), np.vdot(transpose, direction), rtol=1e-10, atol=1e-11)
    epsilon = 1e-5
    difference = (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon)
    np.testing.assert_allclose(difference, product, rtol=2e-3, atol=1e-8)
    report.update(jvp=np.asarray(product).tolist(), vjp=np.asarray(transpose).tolist())
else:
    value = function(point)
np.testing.assert_allclose(np.asarray(value)[0], 1.33, rtol=1e-12, atol=1e-12)
report["value"] = np.asarray(value).tolist()
if rank == 0:
    print("TKE_ENERGY_RESULT " + json.dumps(report), flush=True)
