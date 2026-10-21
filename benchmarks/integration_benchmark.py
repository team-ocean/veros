"""Compare eager and compiled integration, separating compilation and execution.

Run from the repository with PYTHONPATH=. and the JAX runtime environment set.
Use a fresh process for each checkpoint configuration when comparing peak RSS.
"""

import argparse
import json
import resource
import time

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np

from veros import logger, runtime_settings as rs


def timed(function):
    from veros import runtime_state

    jax.effects_barrier()
    if runtime_state.proc_num > 1:
        rs.mpi_comm.Barrier()
    start = time.perf_counter()
    result = function()
    jax.block_until_ready(result)
    jax.effects_barrier()
    seconds = time.perf_counter() - start
    if runtime_state.proc_num > 1:
        from mpi4py import MPI

        seconds = rs.mpi_comm.allreduce(seconds, op=MPI.MAX)
    return result, seconds


def compile_serially(function, *args):
    from veros import runtime_state

    for rank in range(runtime_state.proc_num):
        if runtime_state.proc_rank == rank:
            executable = function.lower(*args).compile()
        if runtime_state.proc_num > 1:
            rs.mpi_comm.Barrier()
    # Invoke the jitted function so ordered MPI tokens use normal JAX dispatch.
    return function, executable


def samples(function, repeats):
    return [timed(function)[1] for _ in range(repeats)]


def make_acc_setup(override, rescale_grid=False):
    from veros.setups.acc_basic import ACCBasicSetup

    if not rescale_grid:
        return ACCBasicSetup(override=override)

    from veros import veros_routine
    from veros.core.operators import at, update

    class BenchmarkSetup(ACCBasicSetup):
        @veros_routine
        def set_grid(self, state):
            super().set_grid(state)
            state.variables.dxt = update(state.variables.dxt, at[:], 60 / state.settings.nx)
            state.variables.dyt = update(state.variables.dyt, at[:], 84 / state.settings.ny)

    return BenchmarkSetup(override=override)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", choices=("acc", "global_4deg"), default="acc")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--fixed-mixing", action="store_true", help="Disable TKE and use fixed vertical mixing")
    parser.add_argument("--compiled-solver", choices=("petsc", "scipy_jax"), default="scipy_jax")
    parser.add_argument("--eager-solver", choices=("petsc", "scipy_jax", "best"), default="scipy_jax")
    parser.add_argument("--shape", type=int, nargs=3, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--num-proc", type=int, nargs=2, default=(1, 1))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--profile-petsc", action="store_true", help="Report KSP time within warm forward execution")
    parser.add_argument("--rescale-grid", action="store_true", help="Keep the ACC domain at 60 by 84 degrees")
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    if args.setup == "acc" and args.shape and args.shape[2] != 15:
        parser.error("ACC basic requires nz=15")
    if args.profile_petsc and args.compiled_solver != "petsc":
        parser.error("PETSc profiling requires --compiled-solver petsc")
    if args.profile_petsc:
        from petsc4py import PETSc

        PETSc.Log.begin()
    rs.backend = "jax"
    rs.diskless_mode = True
    rs.linear_solver = args.eager_solver
    rs.num_proc = tuple(args.num_proc)
    if args.setup == "acc":
        override = dict(nx=16, ny=16, nz=15, dt_mom=600, dt_tracer=600)
        forcing = "t_star"
    else:
        from veros.setups.global_4deg import GlobalFourDegreeSetup as Setup

        override = dict(dt_mom=1800, dt_tracer=3600)
        forcing = "qnet"
    if args.fixed_mixing:
        override["enable_tke"] = False
    if args.shape:
        override.update(zip(("nx", "ny", "nz"), args.shape))
    sim = make_acc_setup(override, args.rescale_grid) if args.setup == "acc" else Setup(override=override)
    sim.setup()
    prepared, prepare_seconds = timed(
        lambda: sim.prepare_integrator(checkpoint=args.checkpoint, linear_solver=args.compiled_solver)
    )
    initial = prepared.pack(sim.state)

    def advance(carry):
        return prepared.advance(carry, n_steps=args.steps, final_only=True)

    def first_forward():
        compile_serially(prepared._compiled_step, initial, {}, {})
        return advance(initial)

    logger.info("Benchmark: compiling and measuring forward execution")
    (final, _), compile_forward_seconds = timed(first_forward)
    prepared.check_status(final)
    solver_before = PETSc.Log.Event("KSPSolve").getPerfInfo() if args.profile_petsc else None
    forward_samples = samples(lambda: advance(initial), args.repeats)
    forward_seconds = float(np.median(forward_samples))
    solver_profile = None
    if args.profile_petsc:
        from mpi4py import MPI

        solver_after = PETSc.Log.Event("KSPSolve").getPerfInfo()
        solver_profile = {
            key: rs.mpi_comm.allreduce(solver_after[key] - solver_before[key], op=MPI.MAX) for key in ("time", "count")
        }
        solver_profile["wall_seconds"] = sum(forward_samples)

    def loss(scale, initial):
        carry = prepared.replace_variables(initial, **{forcing: scale * initial.variables.get(forcing)})
        result, _ = prepared.advance(carry, n_steps=args.steps, final_only=True)
        from veros.distributed import global_sum
        from veros.routines import tracing_context

        with tracing_context():
            return global_sum(jnp.sum(result.variables.temp[2:-2, 2:-2, :, result.variables.tau]))

    backward = None
    gradient = tangent = value = compile_backward_seconds = backward_seconds = None
    backward_samples = []
    if not args.forward_only:
        logger.info("Benchmark: compiling and measuring value plus gradient")
        from veros import runtime_state

        def first_backward():
            nonlocal backward
            backward = jax.value_and_grad(lambda scale, carry: loss(scale, carry) / runtime_state.proc_num)
            return backward(1.0, initial)

        (value, gradient), compile_backward_seconds = timed(first_backward)
        value = value * runtime_state.proc_num
        backward_samples = samples(lambda: backward(1.0, initial), args.repeats)
        backward_seconds = float(np.median(backward_samples))

        if runtime_state.proc_num > 1:
            from mpi4py import MPI

            gradient = rs.mpi_comm.allreduce(float(gradient), op=MPI.SUM)
    if backward is not None:
        assert np.isfinite(gradient), gradient
        if rs.float_type == "float64":
            for step in (1e-3, 1e-4):
                plus, _ = timed(lambda: backward(1 + step, initial))[0]
                minus, _ = timed(lambda: backward(1 - step, initial))[0]
                np.testing.assert_allclose(
                    gradient, runtime_state.proc_num * (plus - minus) / (2 * step), rtol=1e-4, atol=1e-5
                )
    jax.clear_caches()
    if backward is not None:
        del backward
        logger.info("Benchmark: checking forward/reverse derivative agreement")

        def differentiate(scale, carry):
            return jax.jvp(lambda value: loss(value, carry), (scale,), (jnp.ones_like(scale),))

        (_, tangent), _ = timed(lambda: differentiate(1.0, initial))
        derivative_rtol = 1e-4 if rs.float_type == "float32" else 1e-6
        np.testing.assert_allclose(tangent, gradient, rtol=derivative_rtol, atol=1e-5)
        del differentiate
    logger.info("Benchmark: warming and measuring ordinary execution")
    prepared.install(initial)
    _, eager_cold_seconds = timed(lambda: sim.step(sim.state))
    prepared.install(initial)

    def eager():
        for _ in range(args.steps):
            sim.step(sim.state)
        return sim.state.variables.temp

    # Warm diagnostics and iteration-dependent branches across the full horizon.
    timed(eager)
    eager_samples = []
    for _ in range(args.repeats):
        prepared.install(initial)
        eager_samples.append(timed(eager)[1])
    eager_seconds = float(np.median(eager_samples))
    for name in final.variables.fields():
        expected = sim.state.variables.get(name)
        absolute_tolerance = 5e-6 * max(1, float(jnp.max(jnp.abs(expected)))) if rs.float_type == "float32" else 1e-6
        np.testing.assert_allclose(
            final.variables.get(name), expected, rtol=1e-6, atol=absolute_tolerance, equal_nan=False, err_msg=name
        )
    stats = jax.devices()[0].memory_stats() or {}
    from veros import runtime_state

    if runtime_state.proc_rank:
        return
    print(
        json.dumps(
            dict(
                setup=args.setup,
                execution_model="single_step_jit",
                shape=(sim.state.settings.nx, sim.state.settings.ny, sim.state.settings.nz),
                steps=args.steps,
                fixed_mixing=args.fixed_mixing,
                device=str(jax.devices()[0]),
                jax_version=jax.__version__,
                jaxlib_version=jaxlib.__version__,
                dtype=rs.float_type,
                tdma_accelerated=rs.use_special_tdma,
                checkpoint=args.checkpoint,
                compiled_solver=args.compiled_solver,
                eager_solver=args.eager_solver,
                num_proc=args.num_proc,
                petsc_options=rs.petsc_options,
                forward_solver_profile=solver_profile,
                repeats=args.repeats,
                rescale_grid=args.rescale_grid,
                prepare_seconds=prepare_seconds,
                compile_forward_seconds=compile_forward_seconds,
                forward_seconds=forward_seconds,
                compile_backward_seconds=compile_backward_seconds,
                backward_seconds=backward_seconds,
                eager_cold_seconds=eager_cold_seconds,
                eager_seconds=eager_seconds,
                forward_samples=forward_samples,
                backward_samples=backward_samples,
                eager_samples=eager_samples,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                device_peak_bytes=stats.get("peak_bytes_in_use"),
                value=float(value) if value is not None else None,
                gradient=float(gradient) if gradient is not None else None,
                forward_derivative=float(tangent) if tangent is not None else None,
            ),
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
