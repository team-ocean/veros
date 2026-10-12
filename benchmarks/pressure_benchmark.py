"""Measure PETSc solve cost and ordered-callback overhead on CPU, GPU, and MPI.

Native KSP timing excludes copying, normalization, and independent residual
checks to provide a reference for callback overhead.
"""

import argparse
import json
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from veros import runtime_settings as rs, runtime_state as rst


def timed(function):
    jax.effects_barrier()
    rs.mpi_comm.Barrier()
    start = time.perf_counter()
    result = function()
    jax.block_until_ready(result)
    jax.effects_barrier()
    seconds = rs.mpi_comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
    return result, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--num-proc", type=int, nargs=2, default=(1, 1))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--solves", type=int, default=100)
    parser.add_argument("--pressure", action="store_true")
    args = parser.parse_args()
    if min(args.size, args.repeats, args.solves) < 1:
        parser.error("size, repeats, and solves must be positive")
    if min(args.num_proc) < 1 or any(args.size % count for count in args.num_proc):
        parser.error("size must be divisible by each positive process-grid dimension")
    rs.backend = "jax"
    rs.diskless_mode = True
    rs.linear_solver = "scipy_jax"
    rs.num_proc = tuple(args.num_proc)
    PETSc.Log.begin()
    if __package__:
        from .integration_benchmark import make_acc_setup
    else:
        from integration_benchmark import make_acc_setup

    sim = make_acc_setup(
        dict(
            nx=args.size,
            ny=args.size,
            nz=15,
            dt_mom=600,
            dt_tracer=600,
            enable_tke=False,
            enable_streamfunction=not args.pressure,
        ),
        rescale_grid=True,
    )
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=(), linear_solver="petsc")
    solver = prepared._template._linear_solver
    from veros.distributed import get_chunk_slices

    index, _ = get_chunk_slices(args.size, args.size, ("xt", "yt"))
    # Keep the global input identical when comparing process topologies.
    host_rhs = np.random.default_rng(42).normal(size=(args.size, args.size))[index].astype(rs.float_type)
    rhs = jnp.asarray(host_rhs)

    def repeated(value):
        return jax.lax.fori_loop(0, args.solves, lambda _, x: solver._compiled_solve(value)[0], jnp.zeros_like(value))

    reference, converged = solver._callback(host_rhs)
    assert bool(converged), dict(
        reason=solver._ksp.getConvergedReason(),
        iterations=solver._ksp.getIterationNumber(),
        residual=solver._residual_petsc.norm(),
        rhs_norm=solver._rhs_petsc.norm(),
    )
    repeated = jax.jit(repeated)
    for rank in range(rst.proc_num):
        if rst.proc_rank == rank:
            executable = repeated.lower(rhs).compile()
        rs.mpi_comm.Barrier()
    result, _ = timed(lambda: executable(rhs))
    np.testing.assert_allclose(result, reference, rtol=1e-8, atol=1e-8)
    _, pullback = jax.vjp(jax.jit(lambda x: solver._compiled_solve(x)[0]), rhs)
    weight = jnp.asarray(np.random.default_rng(15).normal(size=(args.size, args.size))[index])
    transpose = pullback(weight)[0]
    lhs = rs.mpi_comm.allreduce(float(jnp.vdot(result, weight)), op=MPI.SUM)
    right = rs.mpi_comm.allreduce(float(jnp.vdot(rhs, transpose)), op=MPI.SUM)
    np.testing.assert_allclose(lhs, right, rtol=1e-8, atol=1e-7)
    jax.effects_barrier()

    events = ("KSPSolve", "MatMult", "VecNorm", "VecSet", "VecScale")

    def event_info():
        return {name: PETSc.Log.Event(name).getPerfInfo() for name in events}

    def measure(function):
        before = event_info()
        durations = [timed(function)[1] / args.solves for _ in range(args.repeats)]
        after = event_info()
        event_times = {
            name: rs.mpi_comm.allreduce(after[name]["time"] - before[name]["time"], op=MPI.MAX)
            / (args.solves * args.repeats)
            for name in events
        }
        return dict(seconds=float(np.median(durations)), samples=durations, events=event_times)

    compiled = measure(lambda: executable(rhs))

    def host_callbacks():
        for _ in range(args.solves):
            result, converged = solver._callback(host_rhs)
            assert bool(converged)
        return result

    callbacks = measure(host_callbacks)

    def native_solves():
        for _ in range(args.solves):
            solver._ksp.solve(solver._rhs_petsc, solver._sol_petsc)

    native = measure(native_solves)
    assert solver._ksp.getConvergedReason() > 0
    if rst.proc_rank == 0:
        print(
            json.dumps(
                dict(
                    size=args.size,
                    extent_degrees=(60, 84),
                    formulation="pressure" if args.pressure else "streamfunction",
                    device=str(jax.devices()[0]),
                    num_proc=rs.num_proc,
                    rhs_seed=42,
                    topology_independent_rhs=True,
                    dtype=rs.float_type,
                    jax_version=jax.__version__,
                    petsc_version=PETSc.Sys.getVersion(),
                    petsc_options=rs.petsc_options,
                    ksp_type=solver._ksp.getType(),
                    ksp_tolerances=solver._ksp.getTolerances(),
                    repeats=args.repeats,
                    solves=args.solves,
                    ksp_iterations=solver._ksp.getIterationNumber(),
                    compiled=compiled,
                    callback=callbacks,
                    native=native,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10,
                ),
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
