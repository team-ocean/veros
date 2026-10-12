"""Probe ACC basic derivatives through the onset of unstable stratification.

--zero-unstable-slopes applies an experimental neutral-slope policy.
"""

import argparse
import importlib
import json
import resource

import numpy as np

from veros import runtime_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--zero-unstable-slopes", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    runtime_settings.backend = "jax"
    runtime_settings.diskless_mode = True
    runtime_settings.linear_solver = "scipy_jax"

    import jax
    import jax.numpy as jnp
    from veros.setups.acc_basic import ACCBasicSetup

    if args.zero_unstable_slopes:
        neutral = importlib.import_module("veros.core.isoneutral.isoneutral")
        divide = neutral.stable_divide

        def zero_unstable_slopes(numerator, denominator):
            # The caller passes min(0, vertical density gradient) - 1e-20.
            stable = denominator < -1e-20
            return divide(jnp.where(stable, numerator, 0), jnp.where(stable, denominator, -1))

        neutral.stable_divide = zero_unstable_slopes

    simulation = ACCBasicSetup(override=dict(nx=16, ny=16, nz=15, dt_mom=600, dt_tracer=600))
    simulation.setup()
    prepared = simulation.prepare_integrator(record_diagnostics=(), checkpoint=True)
    initial = prepared.pack(simulation.state)

    def objective(scale, initial):
        carry = prepared.replace_variables(initial, t_star=scale * initial.variables.t_star)
        final, _ = prepared.advance(carry, n_steps=args.steps, final_only=True)
        return jnp.sum(final.variables.temp[2:-2, 2:-2, :, final.variables.tau]), final.failure_code

    backward = jax.value_and_grad(objective, has_aux=True)
    (value, code), gradient = backward(1.0, initial)
    if int(code):
        raise RuntimeError(f"Reverse-mode primal failed with status {int(code)}")
    result = dict(
        zero_unstable_slopes=args.zero_unstable_slopes,
        steps=args.steps,
        shape=(16, 16, 15),
        device=str(jax.devices()[0]),
        dtype=runtime_settings.float_type,
        jax_version=jax.__version__,
        tdma_accelerated=runtime_settings.use_special_tdma,
        reverse_value=float(value),
        reverse_derivative=float(gradient),
        finite_differences=[],
    )
    print("Reverse derivative:", result["reverse_derivative"], flush=True)
    for scale in (1e-2, 1e-3, 1e-4):
        (plus, plus_code), _ = backward(1 + scale, initial)
        (minus, minus_code), _ = backward(1 - scale, initial)
        if int(plus_code) or int(minus_code):
            raise RuntimeError("A finite-difference primal failed")
        result["finite_differences"].append(dict(scale=scale, derivative=float((plus - minus) / (2 * scale))))

    jax.clear_caches()
    value, tangent, code = jax.jvp(
        lambda scale: objective(scale, initial), (jnp.array(1.0),), (jnp.array(1.0),), has_aux=True
    )
    if int(code):
        raise RuntimeError(f"Forward-mode primal failed with status {int(code)}")
    result.update(
        forward_value=float(value),
        forward_derivative=float(tangent),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    print(json.dumps(result, indent=2), flush=True)
    if not np.isfinite([result["reverse_derivative"], result["forward_derivative"]]).all():
        raise RuntimeError("Nonfinite derivative")


if __name__ == "__main__":
    main()
