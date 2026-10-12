"""Small distributed recurrence exercising halo adjoints and checkpoint replay."""

import json
import os
import tempfile
from pathlib import Path

os.environ["VEROS_BACKEND"] = "jax"
os.environ["VEROS_DEVICE"] = "cpu"
os.environ["VEROS_DISKLESS_MODE"] = "1"

from mpi4py import MPI  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from veros import runtime_settings as rs, runtime_state as rst  # noqa: E402
from veros.integration import IntegrationState  # noqa: E402
from veros.checkpointing import CheckpointedIntegrator, BinomialSchedule, PeriodicDiskSchedule  # noqa: E402
from veros.routines import tracing_context  # noqa: E402
from veros.distributed import exchange_overlap, global_sum, proc_rank_to_index  # noqa: E402

rs.num_proc = {1: (1, 1), 2: (2, 1), 4: (2, 2)}[MPI.COMM_WORLD.size]
rst.backend_module
if rst.proc_num > 1:
    from veros._distributed_jax import prepare_communicator

    prepare_communicator(rs.mpi_comm)

rank_x, rank_y = proc_rank_to_index(rst.proc_rank)
nx, ny = 8 // rs.num_proc[0], 8 // rs.num_proc[1]
x = np.zeros((nx + 4, ny + 4))
gx = np.arange(nx)[:, None] + nx * rank_x
gy = np.arange(ny)[None, :] + ny * rank_y
x[2:-2, 2:-2] = np.sin(gx / 8) + np.cos(gy / 8)
initial = IntegrationState(
    dict(x=jnp.asarray(x), forcing=jnp.array(0.3, dtype="float64")), {}, jnp.int32(0), jnp.int32(-1)
)
steps = 5


class Prepared:
    def validate_parameters(self, values):
        return {name: jnp.asarray(value, dtype="float64") for name, value in values.items()}

    def validate_step_inputs(self, values):
        return {name: jnp.asarray(value, dtype="float64") for name, value in values.items()}

    def step(self, carry, *, parameters, step_inputs=None):
        with tracing_context():
            x = exchange_overlap(carry.variables["x"], ("xt", "yt"), False)
            center = x[2:-2, 2:-2]
            laplacian = x[1:-3, 2:-2] + x[3:-1, 2:-2] + x[2:-2, 1:-3] + x[2:-2, 3:-1] - 4 * center
            forcing = carry.variables["forcing"] + (step_inputs or {}).get("forcing", 0)
            updated = x.at[2:-2, 2:-2].set(center + 0.05 * laplacian + forcing * parameters["scale"])
        return carry._replace(variables=dict(carry.variables, x=updated)), {"x": updated[2:-2, 2:-2]}

    def check_status(self, carry):
        assert not int(carry.failure_code)


prepared = Prepared()
parameters = {"scale": jnp.array(0.7)}


def terminal(carry, *, parameters):
    with tracing_context():
        return global_sum(jnp.sum(carry.variables["x"][2:-2, 2:-2] ** 2) + parameters["scale"] ** 2 / rst.proc_num)


def record_loss(records):
    with tracing_context():
        return 0.01 * global_sum(jnp.sum(records["x"] ** 2))


input_values = [{"forcing": jnp.array(0.02 * index, dtype="float64")} for index in range(steps)]


def reference(carry, parameters):
    value = jnp.array(0.0)
    for inputs in input_values:
        carry, records = prepared.step(carry, parameters=parameters, step_inputs=inputs)
        value = value + record_loss(records)
    return value + terminal(carry, parameters=parameters)


def serial_compile(function, *args):
    compiled = jax.jit(function)
    for rank in range(rst.proc_num):
        if rst.proc_rank == rank:
            compiled.lower(*args).compile()
        rs.mpi_comm.Barrier()
    return compiled


observer_calls = []
with tempfile.TemporaryDirectory(prefix="veros-checkpoint-test-") as directory:
    driver = CheckpointedIntegrator(prepared, terminal_objective=terminal, record_objective=record_loss)
    results = []
    for schedule in (
        BinomialSchedule(steps, ram_checkpoints=1),
        PeriodicDiskSchedule(steps, disk_interval=2, ram_checkpoints=1),
    ):
        observer_calls.clear()
        result = driver.value_and_grad(
            initial,
            n_steps=steps,
            schedule=schedule,
            parameters=parameters,
            step_inputs=iter(input_values),
            input_storage="disk",
            directory=directory,
            shared_variables=("forcing",),
            observer=lambda index, carry: observer_calls.append(index),
        )
        assert observer_calls == list(range(steps))
        assert not list(Path(directory).iterdir())
        results.append(result)
    np.testing.assert_allclose(results[0].value, results[1].value)
    np.testing.assert_allclose(results[0].gradient.variables["x"], results[1].gradient.variables["x"])
    result = results[0]
    evaluate = serial_compile(reference, initial, parameters)
    epsilon = 1e-4
    finite_scale = (
        evaluate(initial, {"scale": parameters["scale"] + epsilon})
        - evaluate(initial, {"scale": parameters["scale"] - epsilon})
    ) / (2 * epsilon)
    plus = initial._replace(variables=dict(initial.variables, forcing=initial.variables["forcing"] + epsilon))
    minus = initial._replace(variables=dict(initial.variables, forcing=initial.variables["forcing"] - epsilon))
    finite_forcing = (evaluate(plus, parameters) - evaluate(minus, parameters)) / (2 * epsilon)
    np.testing.assert_allclose(result.parameter_gradient["scale"], finite_scale, rtol=1e-8)
    np.testing.assert_allclose(result.gradient.variables["forcing"], finite_forcing, rtol=1e-8)
    # Independent rank-local initial arrays have distinct cotangents.
    derivative = serial_compile(jax.grad(lambda c: reference(c, parameters) / rst.proc_num, allow_int=True), initial)
    gradient = derivative(initial)
    np.testing.assert_allclose(result.gradient.variables["x"], gradient.variables["x"], rtol=1e-11, atol=1e-11)

    # Check rank-local parameters by perturbing one rank at a time.
    local_parameters = {"scale": parameters["scale"] + 0.01 * rst.proc_rank}
    local_result = driver.value_and_grad(
        initial, n_steps=steps, parameters=local_parameters, step_inputs=input_values, shared_parameters=()
    )
    for rank in range(rst.proc_num):
        perturbation = epsilon if rst.proc_rank == rank else 0.0
        plus_value = evaluate(initial, {"scale": local_parameters["scale"] + perturbation})
        minus_value = evaluate(initial, {"scale": local_parameters["scale"] - perturbation})
        if rst.proc_rank == rank:
            np.testing.assert_allclose(
                local_result.parameter_gradient["scale"], (plus_value - minus_value) / (2 * epsilon), rtol=1e-8
            )

    # Propagate a rank-local host failure and clean all owned files.
    def fail(index, carry):
        if index == 1 and rst.proc_rank == 0:
            raise ValueError("injected observer failure")

    try:
        driver.value_and_grad(
            initial,
            n_steps=steps,
            parameters=parameters,
            step_inputs=input_values,
            directory=directory,
            input_storage="disk",
            observer=fail,
        )
    except (RuntimeError, ValueError) as error:
        assert "injected observer failure" in str(error)
    else:
        raise AssertionError("observer failure was ignored")
    assert not list(Path(directory).iterdir())
    # Schedule divergence is rejected before numerical collectives.
    if rst.proc_num > 1:
        try:
            driver.value_and_grad(
                initial,
                n_steps=steps,
                schedule=BinomialSchedule(steps, ram_checkpoints=1 + rst.proc_rank),
                parameters=parameters,
            )
        except ValueError as error:
            assert "MPI ranks must agree" in str(error)
        else:
            raise AssertionError("divergent schedules were accepted")
        for field in ("_runtime_configuration", "_diagnostic_configuration", "record_diagnostics"):
            setattr(prepared, field, (rst.proc_rank,))
            try:
                driver.value_and_grad(initial, n_steps=steps, parameters=parameters)
            except ValueError as error:
                assert "MPI ranks must agree" in str(error)
            else:
                raise AssertionError(f"divergent {field} was accepted")
            finally:
                delattr(prepared, field)
        inconsistent = initial._replace(
            variables=dict(initial.variables, forcing=initial.variables["forcing"] + rst.proc_rank)
        )
        try:
            driver.value_and_grad(inconsistent, n_steps=steps, parameters=parameters, shared_variables=("forcing",))
        except ValueError as error:
            assert "MPI ranks must agree" in str(error)
        else:
            raise AssertionError("inconsistent shared initial controls were accepted")

if rst.proc_rank == 0:
    print(
        "CHECKPOINT_RESULT "
        + json.dumps(
            dict(
                value=float(result.value),
                scale=float(result.parameter_gradient["scale"]),
                forcing=float(result.gradient.variables["forcing"]),
            )
        ),
        flush=True,
    )
