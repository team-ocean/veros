"""Trace neutral-coefficient parity and replay the first sensitive surface face.

Use a fresh process with the desired JAX device and PETSc library selected.
--zero-surface-slopes applies an experimental boundary policy in this process.
It does not change Veros defaults.
"""

import argparse
import importlib
import json
from pathlib import Path
import resource

import jax
import jax.numpy as jnp
import numpy as np

from veros import runtime_settings
from benchmarks.integration_benchmark import make_acc_setup


def comparison(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    difference = np.abs(actual.astype(float) - expected.astype(float))
    mismatch = (difference > 1e-6 + 1e-6 * np.abs(expected)) | ~np.isfinite(difference)
    index = np.unravel_index(np.argmax(difference), actual.shape) if actual.shape else ()
    return dict(
        max_abs=float(difference[index]),
        mismatches=int(mismatch.sum()),
        index=[int(i) for i in index],
        actual=float(actual[index]),
        expected=float(expected[index]),
    )


def snapshot(carry):
    vs = carry.variables
    return {name: getattr(vs, name) for name in ("K_11", "K_22", "Ai_ez", "psi", "K_iso")} | {
        "temp": vs.temp[..., vs.tau],
        "salt": vs.salt[..., vs.tau],
    }


def surface_face(state, values, index):
    from veros.core import density

    i, j, k = index
    if k != state.settings.nz - 1:
        raise ValueError("Surface analysis requires the top tracer level")
    vs = state.variables
    temp, salt = values["temp"], values["salt"]
    temperatures = temp[i : i + 2, j, k]
    salinities = salt[i : i + 2, j, k]
    drdT = np.asarray(density.get_drhodT(state, jnp.asarray(salinities), jnp.asarray(temperatures), jnp.abs(vs.zt[k])))
    drdS = np.asarray(density.get_drhodS(state, jnp.asarray(salinities), jnp.asarray(temperatures), jnp.abs(vs.zt[k])))
    spacing = float(vs.dxu[i] * vs.cost[j])
    delta_temp = float(temperatures[1] - temperatures[0])
    delta_salt = float(salinities[1] - salinities[0])
    horizontal = (drdT * delta_temp + drdS * delta_salt) / spacing
    slopes = horizontal / 1e-20
    taper = 0.5 * (1 + np.tanh((state.settings.iso_slopec - np.abs(slopes)) / state.settings.iso_dslope))
    vertical = (
        drdT * (temp[i : i + 2, j, k] - temp[i : i + 2, j, k - 1])
        + drdS * (salt[i : i + 2, j, k] - salt[i : i + 2, j, k - 1])
    ) / float(vs.dzw[k - 1])
    return dict(
        temperatures=temperatures.tolist(),
        salinities=salinities.tolist(),
        delta_temperature=delta_temp,
        delta_salinity=delta_salt,
        spacing_m=spacing,
        horizontal_density_gradients=np.broadcast_to(horizontal, (2,)).tolist(),
        vertical_density_gradients_below_surface=np.broadcast_to(vertical, (2,)).tolist(),
        surface_vertical_density_gradient=0.0,
        denominator=-1e-20,
        slopes=np.broadcast_to(slopes, (2,)).tolist(),
        tapers=np.broadcast_to(taper, (2,)).tolist(),
    )


def replay(prepared, initial, histories, step, face):
    from veros.core.isoneutral.isoneutral import isoneutral_diffusion_pre
    from veros.routines import tracing_context

    def inline(carry):
        with tracing_context():
            return isoneutral_diffusion_pre(prepared._make_state(carry.variables))

    inline = jax.jit(inline)
    result = dict(step=step, face=list(face), inputs={})
    for mode, history in histories.items():
        values = {name: value[step - 2] for name, value in history.items()}
        carry = prepared.replace_variables(
            initial,
            temp=initial.variables.temp.at[..., initial.variables.tau].set(values["temp"]),
            salt=initial.variables.salt.at[..., initial.variables.tau].set(values["salt"]),
            K_iso=jnp.asarray(history["K_iso"][step - 1]),
        )
        state = prepared._make_state(carry.variables)
        ordinary = isoneutral_diffusion_pre(state)
        compiled = inline(carry)
        jax.block_until_ready((ordinary, compiled))
        result["inputs"][mode] = surface_face(state, values, face) | dict(
            ordinary_K11=float(ordinary.K_11[face]),
            inline_K11=float(compiled.K_11[face]),
            recorded_K11=float(history["K_11"][(step - 1, *face)]),
            same_input_comparison=comparison(compiled.K_11, ordinary.K_11),
        )
        np.testing.assert_allclose(compiled.K_11, ordinary.K_11, rtol=1e-6, atol=1e-6, equal_nan=False)
        np.testing.assert_allclose(compiled.K_11[face], history["K_11"][(step - 1, *face)], rtol=1e-6, atol=1e-6)
        if values["temp"][face] == values["temp"][face[0] + 1, face[1], face[2]]:
            i, j, k = face
            east = carry.variables.temp[i + 1, j, k, initial.variables.tau]
            changed_temp = carry.variables.temp.at[i + 1, j, k, initial.variables.tau].set(
                jnp.nextafter(east, -jnp.inf)
            )
            changed = inline(prepared.replace_variables(carry, temp=changed_temp))
            result["one_ulp_perturbation"] = dict(
                input_mode=mode,
                delta_temperature=float(jnp.nextafter(east, -jnp.inf) - east),
                original_K11=float(compiled.K_11[face]),
                perturbed_K11=float(changed.K_11[face]),
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--rescale-grid", action="store_true")
    parser.add_argument("--compiled-solver", choices=("petsc", "scipy_jax"), default="petsc")
    parser.add_argument("--eager-solver", choices=("petsc", "scipy_jax"), default="petsc")
    parser.add_argument("--zero-surface-slopes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 4 or args.steps < 2:
        parser.error("size must be at least four and steps at least two")
    runtime_settings.backend = "jax"
    runtime_settings.diskless_mode = True
    runtime_settings.linear_solver = args.eager_solver
    if args.zero_surface_slopes:
        neutral = importlib.import_module("veros.core.isoneutral.isoneutral")
        divide = neutral.stable_divide

        def zero_surface_slopes(numerator, denominator):
            # Only the kr=1 east/north stencils include all 15 tracer levels.
            if numerator.shape[-1] == 15:
                numerator = numerator.at[..., -1].set(0)
                denominator = denominator.at[..., -1].set(-1)
            return divide(numerator, denominator)

        neutral.stable_divide = zero_surface_slopes

    sim = make_acc_setup(dict(nx=args.size, ny=args.size, nz=15, dt_mom=600, dt_tracer=600), args.rescale_grid)
    sim.setup()
    prepared = sim.prepare_integrator(linear_solver=args.compiled_solver)
    initial = prepared.pack(sim.state)

    print("Running uninstrumented forward", flush=True)
    final, _ = prepared.advance(initial, n_steps=args.steps, final_only=True)
    jax.block_until_ready(final)
    prepared.check_status(final)
    plain_final = {name: np.asarray(final.variables.get(name)) for name in final.variables.fields()}
    del final
    jax.clear_caches()

    @jax.jit
    def instrumented_step(carry):
        following, _ = prepared._step(carry)
        return following, snapshot(following)

    print("Running instrumented forward", flush=True)
    final = initial
    history = []
    for _ in range(args.steps):
        final, values = instrumented_step(final)
        history.append(values)
    history = jax.tree.map(lambda *values: jnp.stack(values), *history)
    jax.block_until_ready((final, history))
    prepared.check_status(final)
    compiled_history = jax.tree.map(np.asarray, history)
    instrumentation = {name: comparison(final.variables.get(name), expected) for name, expected in plain_final.items()}
    del final, history
    jax.clear_caches()

    print("Running ordinary trajectory", flush=True)
    prepared.install(initial)
    eager_history = {name: [] for name in compiled_history}
    step_comparisons = []
    for step in range(args.steps):
        sim.step(sim.state)
        values = snapshot(prepared.pack(sim.state))
        current = {}
        for name, value in values.items():
            value = np.asarray(value)
            eager_history[name].append(value)
            current[name] = comparison(compiled_history[name][step], value)
        step_comparisons.append(current)
        print("STEP " + json.dumps(dict(step=step + 1, comparisons=current)), flush=True)
    eager_history = {name: np.stack(values) for name, values in eager_history.items()}
    final_comparison = {name: comparison(actual, sim.state.variables.get(name)) for name, actual in plain_final.items()}
    replays = []
    for step, values in enumerate(step_comparisons, start=1):
        if values["K_11"]["mismatches"]:
            face = tuple(values["K_11"]["index"])
            if face[-1] == 14 and step > 1:
                replays.append(
                    replay(prepared, initial, dict(compiled=compiled_history, eager=eager_history), step, face)
                )
            break
    result = dict(
        shape=[args.size, args.size, 15],
        steps=args.steps,
        rescale_grid=args.rescale_grid,
        device=str(jax.devices()[0]),
        jax_version=jax.__version__,
        dtype=runtime_settings.float_type,
        compiled_solver=args.compiled_solver,
        eager_solver=args.eager_solver,
        petsc_options=runtime_settings.petsc_options,
        zero_surface_slopes=args.zero_surface_slopes,
        instrumentation_comparison=instrumentation,
        final_comparison=final_comparison,
        step_comparisons=step_comparisons,
        surface_replays=replays,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Final parity failures " + json.dumps({k: v for k, v in final_comparison.items() if v["mismatches"]}),
        flush=True,
    )
    if any(value["mismatches"] for value in instrumentation.values()):
        raise RuntimeError("Instrumentation changed the trajectory beyond the parity threshold")


if __name__ == "__main__":
    main()
