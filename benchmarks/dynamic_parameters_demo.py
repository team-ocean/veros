"""Estimate mixing settings from synthetic final-state TKE observations.

Run with PYTHONPATH=. and VEROS_BACKEND=jax. Initial conditions and geometry
are held fixed. This validates the numerical parameter path, not parameter
identifiability or the quality of the model's mixing parameterization.
"""

import argparse
import json
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from veros import runtime_settings, veros_routine, logger
from veros.checkpointing import CheckpointedIntegrator, BinomialSchedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", choices=("acc", "global_4deg"), default="acc")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--solver", choices=("scipy_jax", "petsc"), default="scipy_jax")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    runtime_settings.backend = "jax"
    runtime_settings.diskless_mode = True
    runtime_settings.linear_solver = args.solver
    if args.setup == "acc":
        from veros.setups.acc_basic import ACCBasicSetup as Setup

        overrides = dict(
            nx=8, ny=8, dt_mom=600, dt_tracer=600, enable_neutral_diffusion=False, enable_skew_diffusion=False
        )
    else:
        from veros.setups.global_4deg import GlobalFourDegreeSetup as Setup

        overrides = dict(dt_mom=1800, dt_tracer=3600)

    class Simulation(Setup):
        @veros_routine
        def set_diagnostics(self, state):
            for diagnostic in state.diagnostics.values():
                diagnostic.output_frequency = diagnostic.sampling_frequency = None

    sim = Simulation(override=overrides)
    sim.setup()
    prepared = sim.prepare_integrator(record_diagnostics=(), checkpoint=True)
    initial = prepared.pack(sim.state)
    names = ("c_k", "c_eps") + (("eke_c_k",) if sim.state.settings.enable_eke else ())
    point = {name: jnp.asarray(getattr(sim.state.settings, name)) for name in names}
    truth = {name: value * (1.2 if name != "c_eps" else 0.9) for name, value in point.items()}

    def observe(final):
        vs = final.variables
        return vs.tke[2:-2, 2:-2, :, vs.tau]

    logger.info("Generating synthetic observations")
    target_state, _ = prepared.advance(initial, n_steps=args.steps, parameters=truth, final_only=True)
    prepared.check_status(target_state)
    target = observe(target_state)
    scale = jnp.maximum(jnp.mean(jnp.abs(target)), 1e-10)
    jax.block_until_ready((target, scale))
    del target_state
    jax.clear_caches()

    def terminal(final, *, parameters):
        return jnp.mean(((observe(final) - target) / scale) ** 2)

    def loss(values, carry):
        final, _ = prepared.advance(carry, n_steps=args.steps, parameters=values, final_only=True)
        return terminal(final, parameters=values), final

    logger.info("Checking reverse and forward derivatives and finite differences")
    backward = jax.value_and_grad(loss, has_aux=True)
    (value, final), gradient = backward(point, initial)
    jax.block_until_ready(gradient)
    prepared.check_status(final)
    direction = jax.tree.map(lambda parameter: parameter * 0.1, point)
    expected = sum(gradient[name] * direction[name] for name in names)
    assert np.isfinite(value) and np.isfinite(expected) and abs(float(expected)) > 1e-12

    def scalar_loss(values):
        return loss(values, initial)[0]

    _, tangent = jax.jvp(scalar_loss, (point,), (direction,))
    np.testing.assert_allclose(tangent, expected, rtol=1e-5, atol=1e-9)
    evaluate = loss
    differences = []
    for epsilon in (1e-5, 1e-6):
        plus = jax.tree.map(lambda parameter, delta: parameter + epsilon * delta, point, direction)
        minus = jax.tree.map(lambda parameter, delta: parameter - epsilon * delta, point, direction)
        difference = (evaluate(plus, initial)[0] - evaluate(minus, initial)[0]) / (2 * epsilon)
        logger.info("Finite difference at {}: {}, AD: {}", epsilon, float(difference), float(expected))
        np.testing.assert_allclose(difference, expected, rtol=2e-4, atol=1e-8)
        differences.append(float(difference))

    logger.info("Checking RAM/disk replay against the composed gradient")
    driver = CheckpointedIntegrator(prepared, terminal_objective=terminal)
    schedule = BinomialSchedule(args.steps, ram_checkpoints=1, disk_checkpoints=1)
    with tempfile.TemporaryDirectory(prefix="veros-parameter-demo-") as directory:
        replay = driver.value_and_grad(
            initial, n_steps=args.steps, parameters=point, schedule=schedule, directory=directory
        )
    np.testing.assert_allclose(replay.value, value, rtol=1e-9, atol=1e-10)
    for name in names:
        np.testing.assert_allclose(replay.parameter_gradient[name], gradient[name], rtol=2e-6, atol=1e-8)

    logger.info("Taking a gradient step in log parameter coordinates")
    log_gradient = {name: gradient[name] * point[name] for name in names}
    norm = jnp.sqrt(sum(coefficient**2 for coefficient in log_gradient.values()))
    for step in (0.1, 0.05, 0.025, 0.0125, 0.00625):
        updated = {name: point[name] * jnp.exp(-step * log_gradient[name] / norm) for name in names}
        improved, improved_state = evaluate(updated, initial)
        prepared.check_status(improved_state)
        if float(improved) < float(value):
            break
    assert float(improved) < float(value), "Gradient update did not decrease the objective"
    result = dict(
        setup=args.setup,
        device=str(jax.devices()[0]),
        steps=args.steps,
        simulated_days=args.steps * sim.state.settings.dt_tracer / 86400,
        parameters={name: float(parameter) for name, parameter in point.items()},
        gradients={name: float(coefficient) for name, coefficient in gradient.items()},
        directional_reverse=float(expected),
        directional_forward=float(tangent),
        finite_differences=differences,
        finite_difference_steps=[1e-5, 1e-6],
        loss=float(value),
        updated_loss=float(improved),
        updated_parameters={name: float(parameter) for name, parameter in updated.items()},
        replay_statistics=replay.statistics,
    )
    print("DYNAMIC_PARAMETERS_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
