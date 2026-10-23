"""Density anomalies retain the rational EOS without density-scale cancellation."""

import ast
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest


def original_polynomials():
    from veros.core.density import gsw

    path = Path(gsw.__file__)
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "gsw_drhodT"
    )
    return {
        node.targets[0].id: compile(ast.Expression(node.value), str(path), "eval")
        for node in function.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("v_hat_denominator", "v_hat_numerator")
    }


def decimal_anomalies(salinity, temperature, pressure):
    from veros.core.density import gsw

    expressions = original_polynomials()
    with localcontext() as context:
        context.prec = 80
        coefficients = {f"v{i:02}": Decimal.from_float(getattr(gsw, f"v{i:02}")) for i in range(1, 49)}
        values = []
        for sa, ct, p in zip(salinity, temperature, pressure):
            point = dict(
                coefficients,
                sa=Decimal.from_float(float(sa)),
                ct=Decimal.from_float(float(ct)),
                p=Decimal.from_float(float(p)),
            )
            point["sqrtsa"] = point["sa"].sqrt()
            values.append(
                float(
                    eval(expressions["v_hat_denominator"], {}, point) / eval(expressions["v_hat_numerator"], {}, point)
                    - Decimal(1024)
                )
            )
    return np.asarray(values)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("domain", ("basin", "physical"))
def test_density_anomaly_against_high_precision_original_eos(dtype, domain):
    from veros.core.density import gsw

    axes = (
        ((34.99, 35, 35.01), (6, 8, 10), tuple(range(125, 2000, 250)))
        if domain == "basin"
        else ((0, 1, 20, 35, 40), (-2, 0, 10, 25, 40), (0, 100, 1000, 5000, 10000))
    )
    salinity, temperature, pressure = [a.ravel().astype(dtype) for a in np.meshgrid(*axes, indexing="ij")]
    expected = decimal_anomalies(salinity, temperature, pressure)
    actual = np.asarray(gsw.gsw_rho(salinity, temperature, pressure))
    tolerance = 5e-14 if domain == "basin" else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance if dtype == np.float64 else 5e-4)
    if dtype == np.float64:
        expressions = original_polynomials()
        namespace = {f"v{i:02}": getattr(gsw, f"v{i:02}") for i in range(1, 49)}
        namespace.update(sa=salinity, ct=temperature, p=pressure, sqrtsa=np.sqrt(salinity))
        legacy = (
            eval(expressions["v_hat_denominator"], {}, namespace) / eval(expressions["v_hat_numerator"], {}, namespace)
            - 1024
        )
        assert np.linalg.norm(actual - expected) < np.linalg.norm(legacy - expected)


@pytest.mark.parametrize("variable", (0, 1, 2))
def test_density_anomaly_jvp_matches_analytic_eos_derivatives(variable):
    from veros import runtime_settings

    if runtime_settings.backend != "jax":
        pytest.skip("Density derivatives require the JAX backend")
    import jax
    import jax.numpy as jnp
    from veros.core.density import gsw

    jax.config.update("jax_enable_x64", True)
    args = tuple(jnp.asarray(a, dtype=jnp.float64) for a in ((1, 20, 35, 40), (-2, 10, 25, 40), (0, 1000, 5000, 10000)))
    direction = tuple(jnp.ones_like(a) if i == variable else jnp.zeros_like(a) for i, a in enumerate(args))
    _, tangent = jax.jvp(gsw.gsw_rho, args, direction)
    analytic = (gsw.gsw_drhodS, gsw.gsw_drhodT, gsw.gsw_drhodP)[variable](*args)
    if variable == 2:
        analytic *= 1e4  # The analytic pressure derivative is per Pa; input pressure is dbar.
    np.testing.assert_allclose(tangent, analytic, rtol=1e-10, atol=1e-12)
