"""Check distributed forward parity and derivatives under mpiexec."""

import json
import os
import sys
import gc
import ctypes
from copy import copy

os.environ["VEROS_BACKEND"] = "jax"
os.environ.setdefault("VEROS_LINEAR_SOLVER", "scipy_jax")
os.environ["VEROS_DISKLESS_MODE"] = "1"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from veros import runtime_settings as rs, runtime_state as rst, veros_routine  # noqa: E402
from veros.distributed import exchange_overlap, global_sum, global_max, global_min  # noqa: E402
from veros.routines import tracing_context  # noqa: E402

rs.num_proc = {1: (1, 1), 2: (2, 1), 4: (2, 2)}[MPI.COMM_WORLD.size]
implementation = sys.argv[2] if len(sys.argv) > 2 else rs.linear_solver
tke_policy = sys.argv[3] if len(sys.argv) > 3 else "default"
assert tke_policy in ("default", "positive_diffusion")

from veros.setups.acc_basic import ACCBasicSetup  # noqa: E402


def compile_serially(function, *args):
    # Avoid concurrent compiler peaks when all ranks share a small test host.
    jax.effects_barrier()
    jax.clear_caches()
    gc.collect()
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim(0)
    compiled = jax.jit(function)
    for rank in range(rst.proc_num):
        if rank == rst.proc_rank:
            compiled.lower(*args).compile()
        rs.mpi_comm.Barrier()
    return compiled


def phase(name):
    jax.effects_barrier()
    rs.mpi_comm.Barrier()
    if rst.proc_rank == 0:
        print(name, flush=True)


class Simulation(ACCBasicSetup):
    @veros_routine
    def set_initial_conditions(self, state):
        super().set_initial_conditions(state)
        if tke_policy == "positive_diffusion":
            vs = state.variables
            vs.tke = jnp.broadcast_to(1e-3 * vs.maskW[..., None], vs.tke.shape)

    @veros_routine
    def set_diagnostics(self, state):
        for diagnostic in state.diagnostics.values():
            diagnostic.output_frequency = None
            diagnostic.sampling_frequency = None
        averages = state.diagnostics["averages"]
        averages.output_variables = ("temp",)
        averages.sampling_frequency = state.settings.dt_tracer
        averages.output_frequency = 3 * state.settings.dt_tracer


sim = Simulation(
    override=dict(
        nx=8,
        ny=8,
        dt_mom=600,
        dt_tracer=600,
        enable_streamfunction=sys.argv[1] == "streamfunction",
        enable_tke=tke_policy == "positive_diffusion",
        enable_tke_positive_diffusion=tke_policy == "positive_diffusion",
        kappaH_0=1e-4,
        enable_neutral_diffusion=False,
        enable_skew_diffusion=False,
    )
)
sim.setup()
mixing_parameter = "c_k" if tke_policy == "positive_diffusion" else "kappaH_0"
mixing_value = getattr(sim.state.settings, mixing_parameter)
p = sim.prepare_integrator(record_diagnostics=("averages",), checkpoint=True, linear_solver=implementation)
c = p.pack(sim.state)
if implementation == "petsc":
    from veros.core.external.solvers.petsc_jax import CompiledPETScSolver

    assert isinstance(p._template._linear_solver, CompiledPETScSolver)
    if rst.proc_num > 1:
        from veros._distributed_jax import numerical_communicator

        # PETSc callbacks and mpi4jax transfers share this ordered communicator.
        assert p._template._linear_solver._callback_comm == numerical_communicator(rs.mpi_comm)
phase("Forward trajectory")
advance = compile_serially(lambda carry: p.advance(carry, n_steps=3), c)
final, records = advance(c)
p.check_status(final)
del advance
if rst.proc_num == 1:
    for _ in range(3):
        sim.step(sim.state)
    for key in c.variables.fields():
        np.testing.assert_allclose(
            final.variables.get(key), sim.state.variables.get(key), rtol=2e-7, atol=2e-7, err_msg=key
        )


def loss(controls):
    carry = p.replace_variables(c, t_star=controls["forcing"] * c.variables.t_star)
    _, records = p.advance(carry, n_steps=3, parameters={mixing_parameter: controls["mixing"] * mixing_value})
    with tracing_context():
        return global_sum(jnp.sum(records["averages"].values["temp"][-1, 2:-2, 2:-2] ** 2))


point = jnp.array(1.0)
controls = dict(forcing=point, mixing=point)
direction = jax.tree.map(jnp.ones_like, controls)
phase("Reverse derivative")
backward = compile_serially(jax.value_and_grad(lambda controls: loss(controls) / rst.proc_num), controls)
value, gradient = backward(controls)
value = value * rst.proc_num
jax.block_until_ready(gradient)
del backward
phase("Forward derivative")
forward = compile_serially(lambda a: jax.jvp(loss, (a,), (direction,)), controls)
_, tangent = forward(controls)
jax.block_until_ready(tangent)
del forward
# A replicated global loss is seeded once across ranks. Shared-control sensitivities sum over ranks.
shared_gradients = {
    name: rs.mpi_comm.allreduce(float(coefficient), op=MPI.SUM) for name, coefficient in gradient.items()
}
shared_gradient = shared_gradients["forcing"]
assert abs(shared_gradients["mixing"]) > 1e-8
phase("Checkpointed ocean replay")
from veros.checkpointing import CheckpointedIntegrator, PeriodicDiskSchedule  # noqa: E402
import tempfile  # noqa: E402


def replay_record_loss(records):
    record = records["averages"]
    with tracing_context():
        return global_sum(jnp.sum(jnp.where(record.valid[-1], record.values["temp"][-1, 2:-2, 2:-2] ** 2, 0)))


jax.clear_caches()
gc.collect()
with tempfile.TemporaryDirectory(prefix="veros-ocean-replay-") as directory:
    driver = CheckpointedIntegrator(p, record_objective=replay_record_loss)
    result = driver.value_and_grad(
        c,
        n_steps=3,
        parameters={mixing_parameter: mixing_value},
        schedule=PeriodicDiskSchedule(3, disk_interval=2, ram_checkpoints=1),
        directory=directory,
    )
    sensitivity = rs.mpi_comm.allreduce(
        float(jnp.sum(result.gradient.variables.t_star * c.variables.t_star)), op=MPI.SUM
    )
    np.testing.assert_allclose(result.value, value, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(sensitivity, shared_gradients["forcing"], rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(
        result.parameter_gradient[mixing_parameter] * mixing_value,
        shared_gradients["mixing"],
        rtol=2e-5,
        atol=1e-5,
    )
    del driver, result
jax.clear_caches()
gc.collect()

phase("Finite differences")
evaluate = compile_serially(loss, controls)
plus = evaluate(jax.tree.map(lambda coefficient: coefficient + 0.001, controls))
jax.effects_barrier()
minus = evaluate(jax.tree.map(lambda coefficient: coefficient - 0.001, controls))
finite_difference = float((plus - minus) / 0.002)
for name in controls:
    plus = evaluate(dict(controls, **{name: jnp.array(1.001)}))
    jax.effects_barrier()
    minus = evaluate(dict(controls, **{name: jnp.array(0.999)}))
    np.testing.assert_allclose(shared_gradients[name], (plus - minus) / 0.002, rtol=2e-4, atol=1e-5)
del evaluate
np.testing.assert_allclose(sum(shared_gradients.values()), tangent, rtol=1e-7, atol=1e-7)
np.testing.assert_allclose(sum(shared_gradients.values()), finite_difference, rtol=2e-4, atol=1e-5)

if implementation == "petsc":
    phase("Distributed PETSc transpose and zero local tangents")
    solver = p._template._linear_solver
    shape = (sim.state.settings.nx // rs.num_proc[0], sim.state.settings.ny // rs.num_proc[1])
    rng = np.random.default_rng(300 + rst.proc_rank)
    rhs, direction, weight = (jnp.asarray(rng.normal(size=shape)) for _ in range(3))
    if rst.proc_rank:
        direction = jnp.zeros_like(direction)

    def solve(value):
        return solver._compiled_solve(value)[0]

    transformed = compile_serially(lambda a, b: jax.jvp(solve, (a,), (b,)), rhs, direction)
    _, tangent = transformed(rhs, direction)
    jax.block_until_ready(tangent)
    _, pullback = jax.vjp(jax.jit(solve), rhs)
    cotangent = pullback(weight)[0]
    jax.block_until_ready(cotangent)
    lhs = rs.mpi_comm.allreduce(float(jnp.vdot(tangent, weight)), op=MPI.SUM)
    rhs_dot = rs.mpi_comm.allreduce(float(jnp.vdot(direction, cotangent)), op=MPI.SUM)
    np.testing.assert_allclose(lhs, rhs_dot, rtol=1e-9, atol=1e-8)
    zero, converged = jax.jit(solver._compiled_solve)(jnp.zeros_like(rhs))
    assert bool(converged)
    np.testing.assert_array_equal(zero, np.zeros(shape))
    bad_rhs = rhs.at[0, 0].set(jnp.nan) if rst.proc_rank == 0 else rhs
    invalid, converged = jax.jit(solver._compiled_solve)(bad_rhs)
    assert not bool(converged) and np.isnan(invalid).all()
    if rst.proc_num > 1:
        from veros.core.external.solvers import get_compiled_solver

        selected = get_compiled_solver(sim.state, "best")
        assert isinstance(selected, CompiledPETScSolver)

# The halo adjoint includes boundaries, all four corners, and independent rank inputs.
for cyclic in (False, True):
    phase(f"Halo adjoint, cyclic={cyclic}")
    shape = c.variables.surface_taux.shape
    rng = np.random.default_rng(100 + rst.proc_rank)
    x, direction, weight = (jnp.asarray(rng.normal(size=shape)) for _ in range(3))

    def halo(value):
        with tracing_context():
            return exchange_overlap(value, ("xt", "yt"), cyclic)

    exchanged, dx = jax.jit(lambda a, b: jax.jvp(halo, (a,), (b,)))(x, direction)
    _, pullback = jax.vjp(jax.jit(halo), x)
    cotangent = pullback(weight)[0]
    lhs = rs.mpi_comm.allreduce(float(jnp.vdot(dx, weight)), op=MPI.SUM)
    rhs = rs.mpi_comm.allreduce(float(jnp.vdot(direction, cotangent)), op=MPI.SUM)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)
    eager = exchange_overlap(x, ("xt", "yt"), cyclic)
    np.testing.assert_array_equal(exchanged, eager)


def diagnostic_probe(carry):
    state = p._make_state(carry.variables)
    values = {}
    for name, original in sim.state.diagnostics.items():
        diagnostic = copy(original)
        diagnostic.variables = copy(carry.diagnostics.get(name)) if name != "snapshot" else state.variables
        diagnostic.diagnose(state)
        values[name] = diagnostic.finalize(state)
    return values


def traced_diagnostics(carry):
    with tracing_context():
        return diagnostic_probe(carry)


phase("Numerical diagnostics")
probe = compile_serially(traced_diagnostics, final)
compiled_values = probe(final)
jax.block_until_ready(compiled_values)
del probe
jax.clear_caches()
eager_values = diagnostic_probe(final)
for name in compiled_values:
    for key in compiled_values[name]:
        np.testing.assert_allclose(
            compiled_values[name][key], eager_values[name][key], rtol=2e-7, atol=2e-6, err_msg=f"{name}.{key}"
        )

phase("Extremum derivatives")
for reduction in (global_max, global_min):
    for tied in (False, True):

        def extremum(value):
            with tracing_context():
                return reduction(value + (0 if tied else rst.proc_rank))

        derivative = jax.jit(jax.grad(lambda value: extremum(value) / rst.proc_num))(point)
        total = rs.mpi_comm.allreduce(float(derivative), op=MPI.SUM)
        np.testing.assert_allclose(total, 1)

if rst.proc_rank == 0:
    print(
        "INTEGRATION_RESULT "
        + json.dumps(dict(value=float(value), gradient=shared_gradient, parameter_gradient=shared_gradients["mixing"])),
        flush=True,
    )
