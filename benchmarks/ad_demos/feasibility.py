"""Reference-data and statistical-response benchmarks."""

import numpy as np


def segmented_shadowing(forcing, segment_length):
    """Condense one expanding direction into segment-boundary corrections."""
    from scipy.linalg import solve_banded

    length = forcing.size
    if length % segment_length:
        raise ValueError("Use complete shadowing segments")
    segments = length // segment_length
    homogeneous = 2.0 ** np.arange(segment_length + 1)
    particular = np.zeros((segments, segment_length + 1))
    for index in range(segment_length):
        particular[:, index + 1] = 2 * particular[:, index] + forcing[index::segment_length]
    weights = np.append(np.full(segments, np.sum(homogeneous[:-1] ** 2)), 1.0)
    linear = np.append(particular[:, :-1] @ homogeneous[:-1], 0.0)
    inverse = 1 / weights
    end_growth = homogeneous[-1]
    band = np.zeros((3, segments))
    band[1] = end_growth**2 * inverse[:-1] + inverse[1:]
    band[0, 1:] = band[2, :-1] = -end_growth * inverse[1:-1]
    rhs = particular[:, -1] - end_growth * inverse[:-1] * linear[:-1] + inverse[1:] * linear[1:]
    multiplier = solve_banded((1, 1), band, rhs)
    boundaries = -linear * inverse
    boundaries[:-1] -= end_growth * multiplier * inverse[:-1]
    boundaries[1:] += multiplier * inverse[1:]
    return np.append((boundaries[:-1, None] * homogeneous[:-1] + particular[:, :-1]).reshape(-1), boundaries[-1])


def conservative_project(values, fine_volume, coarse_shape):
    """Wet-volume averages on aligned nested Cartesian cells."""
    shape = tuple(coarse_shape)
    ratios = tuple(fine // coarse for fine, coarse in zip(values.shape, shape))
    if (
        len(shape) != 3
        or values.shape != fine_volume.shape
        or any(fine != coarse * ratio for fine, coarse, ratio in zip(values.shape, shape, ratios))
    ):
        raise ValueError("Projection requires aligned, nested three-dimensional cells")
    nested = tuple(component for pair in zip(shape, ratios) for component in pair)
    volume = fine_volume.reshape(nested).sum(axis=(1, 3, 5))
    inventory = (values * fine_volume).reshape(nested).sum(axis=(1, 3, 5))
    return np.divide(inventory, volume, out=np.zeros_like(inventory), where=volume > 0), volume


def sampling_summary(samples):
    """Summarize autocorrelation and block errors with a separate stationarity check."""
    values = np.asarray(samples, dtype=float).reshape(-1)
    centered = values - values.mean()
    n = values.size
    spectrum = np.fft.rfft(centered, n=2 * n)
    covariance = np.fft.irfft(spectrum * spectrum.conj(), n=2 * n)[:n] / np.arange(n, 0, -1)
    correlation = covariance / max(covariance[0], 1e-30)
    pairs = correlation[1 : n // 2 * 2 - 1].reshape(-1, 2).sum(axis=1)
    positive = np.flatnonzero(pairs <= 0)
    cutoff = int(positive[0]) if positive.size else len(pairs)
    correlation_time = max(1.0, 1 + 2 * pairs[:cutoff].sum())
    blocks = min(16, n // 2)
    means = np.array([part.mean() for part in np.array_split(values, blocks)])
    error = float(means.std(ddof=1) / np.sqrt(blocks))
    drift = float(values[n // 2 :].mean() - values[: n // 2].mean())
    return dict(
        mean=float(values.mean()),
        variance=float(values.var()),
        standard_error=error,
        correlation_time_steps=float(correlation_time),
        effective_sample_size=float(n / correlation_time),
        split_drift=drift,
        stationary_within_block_error=abs(drift) <= 3 * error,
        correlation=correlation[: min(n, 60)].tolist(),
        scope="descriptive estimates; effective sample size requires a stationary series",
    )


def expanding_shadowing(length=4096, seed=12):
    """Tangent/adjoint least-squares shadowing on a one-dimensional expanding map.

    At a=0, F_a(x)=2x+a*sin(2*pi*x)/(2*pi) mod 1 preserves uniform measure.
    The two inverse branches cancel its first-order density perturbation.
    Thus the full response of cos(2*pi*x) is zero. The bounded shadowing
    contribution is 1/4, with an unstable contribution of -1/4.
    A symbolic Bernoulli orbit avoids finite-precision collapse of doubling.
    All directions are unstable here, so the nonintrusive subspace is full.
    """
    from scipy.linalg import solve_banded

    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length + 54)
    x = np.convolve(bits, 2.0 ** -np.arange(53, 0, -1), mode="valid")[: length + 1]
    forcing = np.sin(2 * np.pi * x[:-1]) / (2 * np.pi)
    # D v=b, D=(-2,1). Minimize ||v||² through the well-conditioned dual system.
    band = np.zeros((3, length))
    band[1] = 5
    band[0, 1:] = band[2, :-1] = -2
    dual = solve_banded((1, 1), band, forcing)
    tangent = np.zeros(length + 1)
    tangent[:-1] -= 2 * dual
    tangent[1:] += dual
    observable_gradient = -2 * np.pi * np.sin(2 * np.pi * x) / (length + 1)
    adjoint = solve_banded((1, 1), band, -2 * observable_gradient[:-1] + observable_gradient[1:])
    segmented = []
    for size in (4, 8, 16):
        values = segmented_shadowing(forcing, size)
        segmented.append(
            dict(
                segment_length=size,
                estimate=float(observable_gradient @ values),
                maximum_full_lss_difference=float(np.max(np.abs(values - tangent))),
                qr_log_growth_per_step=float(np.log(2.0**size) / size),
            )
        )
    return dict(
        tangent_estimate=float(observable_gradient @ tangent),
        adjoint_estimate=float(forcing @ adjoint),
        continuity_residual=float(np.linalg.norm(tangent[1:] - 2 * tangent[:-1] - forcing)),
        bounded_tangent_rms=float(np.sqrt(np.mean(tangent**2))),
        lyapunov_exponent=float(np.log(2)),
        unstable_dimension=1,
        expected_shadowing_contribution=0.25,
        exact_full_response=0.0,
        expected_unstable_contribution=-0.25,
        length=length,
        segmented_tangent=segmented,
        candidate_scope="one-dimensional full unstable subspace; tangent/adjoint least-squares prototypes",
    )


def statistical_benchmarks(report, save):
    from .common import Checks

    checks = Checks(on_update=save)
    report.update(checks=checks.items, metrics=checks.metrics)
    rng = np.random.default_rng(19)
    samples = []
    x, tangent, rate = 0.0, 0.0, 0.8
    for step in range(8192 + 128):
        x = rate * x + 0.2 + 0.1 * rng.normal()
        tangent = rate * tangent + 1
        if step >= 128:
            samples.append(x)
    stable = sampling_summary(samples)
    stable.update(
        analytic_mean=1.0, analytic_variance=0.01 / (1 - rate**2), analytic_response=5.0, streamed_tangent=tangent
    )
    checks.metrics["stable"] = stable
    checks.close("stable streamed response", tangent, 5.0, rtol=1e-10)
    checks.require("stable invariant mean", abs(stable["mean"] - 1) < 3 * stable["standard_error"], stable)
    estimates = [expanding_shadowing(length) for length in (1024, 4096, 16384)]
    checks.metrics["chaotic_shadowing"] = estimates
    for result in estimates:
        checks.close(
            f"shadowing transpose {result['length']}",
            result["tangent_estimate"],
            result["adjoint_estimate"],
            rtol=1e-11,
        )
        checks.require(f"shadowing continuity {result['length']}", result["continuity_residual"] < 1e-10, result)
        for segment in result["segmented_tangent"]:
            checks.close(
                f"segmented shadowing {result['length']}/{segment['segment_length']}",
                segment["estimate"],
                result["adjoint_estimate"],
                rtol=1e-8,
            )
    differences = []
    for epsilon in (0.1, 0.03):
        responses = []
        for seed in range(64):
            initial = np.random.default_rng(seed).uniform(size=4096)
            means = []
            for parameter in (-epsilon, epsilon):
                x, total = initial.copy(), np.zeros_like(initial)
                for step in range(64):
                    x = (2 * x + parameter * np.sin(2 * np.pi * x) / (2 * np.pi)) % 1
                    if step >= 32:
                        total += np.cos(2 * np.pi * x) / 32
                means.append(float(total.mean()))
            responses.append((means[1] - means[0]) / (2 * epsilon))
        mean, error = float(np.mean(responses)), float(np.std(responses, ddof=1) / 8)
        result = dict(
            epsilon=epsilon, derivative=mean, standard_error=error, ensembles=64, samples_per_ensemble=4096 * 32
        )
        differences.append(result)
        checks.require(f"independent chaotic response {epsilon}", abs(mean) < 3 * error + 1e-3, result)
        checks.require(
            f"resolved shadowing bias {epsilon}", abs(mean - estimates[-1]["tangent_estimate"]) > 3 * error, result
        )
    checks.metrics.update(
        chaotic_finite_differences=differences,
        decision="Bounded shadowing omits a resolved unstable contribution in this test; no ocean estimator adopted.",
        next_gate="full response including unstable contribution, then independent stationary ocean ensembles",
    )


def ocean_feasibility(config, report, save):
    """Measure finite-window growth and sampling limits before selecting an estimator."""
    import jax
    import jax.numpy as jnp
    from .common import Checks
    from .research_ocean import ResearchOcean, CONTROL_NAMES
    from .matrix import CASES

    checks = Checks(on_update=save)
    report.update(checks=checks.items, metrics=checks.metrics)
    ocean = ResearchOcean(config, CASES["sensitivity_trajectory"])
    steps = config["steps"]
    stride = max(1, round(30 * 86400 / ocean.dt))
    products = jax.jit(
        lambda c, p, dc, dp: jax.jvp(lambda c, p: ocean.prepared.step(c, parameters=p)[0], (c, p), (dc, dp))
    )
    full = jnp.zeros(len(CONTROL_NAMES))
    (carry, parameters), (tangent, _) = jax.jvp(ocean.controls, (full,), (jnp.eye(full.size)[0],))
    dp = jax.tree.map(jnp.zeros_like, parameters)
    scales = {"temp": 1, "salt": 0.1, "u": 0.1, "v": 0.1, "tke": 1e-3, "eke": 1e-3}

    def norm(dc):
        return jnp.sqrt(
            sum(
                jnp.sum((dc.variables.get(name)[2:-2, 2:-2] / scale) ** 2)
                for name, scale in scales.items()
                if name in dc.variables
            )
        )

    growth, observations = [], []
    initial_norm = float(norm(tangent))
    tangent = jax.tree.map(lambda a: a if a.dtype == jax.dtypes.float0 else a / initial_norm, tangent)
    for step in range(steps):
        carry, tangent = products(carry, parameters, tangent, dp)
        ocean.prepared.check_status(carry)
        magnitude = float(norm(tangent))
        growth.append(float(np.log(max(magnitude, 1e-300))))
        tangent = jax.tree.map(lambda a: a if a.dtype == jax.dtypes.float0 else a / max(magnitude, 1e-300), tangent)
        if (step + 1) % stride == 0:
            observations.append(np.asarray(ocean.observe(carry)))
    summary = {
        name: sampling_summary(np.asarray(observations)[:, index])
        for name, index in (("heat", ocean.heat_index), ("acc", ocean.acc_index), ("overturning", ocean.amoc_index))
    }
    checks.metrics.update(
        sampling=summary,
        monthly_samples=len(observations),
        growth_log=growth,
        finite_window_growth_per_day=float(np.mean(growth) * 86400 / ocean.dt),
        growth_scope="one initial tracer direction, with multistep carry and fixed phase; finite transient growth, not an asymptotic Lyapunov spectrum",
        adjustment_scope="fixed forward preparation; no established invariant distribution",
        prognostic_norm_scales=scales,
        control="heat_mean",
        settings=dict(ocean.sim.state.settings.items()),
        prepared_days=float(ocean.initial.variables.time) / 86400,
    )
    checks.require("finite ocean growth", np.isfinite(growth).all(), checks.metrics["finite_window_growth_per_day"])
    del products, carry, tangent
    jax.clear_caches()
    differences = []
    for epsilon in (0.1, 0.03):
        ensemble = []
        for phase in (0, 90, 180, 270):
            means = []
            for sign in (-1, 1):
                carry, parameters = ocean.controls(full.at[2].set(sign * epsilon), phase_days=phase)
                samples = []
                for step in range(steps):
                    carry, _ = ocean.prepared.advance(carry, n_steps=1, parameters=parameters, final_only=True)
                    ocean.prepared.check_status(carry)
                    if (step + 1) % stride == 0:
                        samples.append(float(ocean.observe(carry)[ocean.amoc_index]))
                means.append(np.mean(samples))
            ensemble.append((means[1] - means[0]) / (2 * epsilon))
        mean, error = float(np.mean(ensemble)), float(np.std(ensemble, ddof=1) / 2)
        differences.append(
            dict(
                epsilon=epsilon,
                mean_response=mean,
                phase_standard_error=error,
                resolved=abs(mean) > 3 * error,
                phase_responses=ensemble,
            )
        )
        checks.metrics["finite_window_ensemble_differences"] = differences
        save()
    checks.metrics.update(
        decision="Stationarity, adjustment and unstable dimension remain unresolved; no chaotic-ocean estimator selected.",
        next_gate="longer independent runs, constrained prognostic QR spectrum, stationary sampling and unstable-response correction",
    )


def jacobian_profile(config, report, save):
    """Compare grouped per-step linearization with independent streamed JVPs."""
    import time
    import jax
    import jax.numpy as jnp
    from .common import Checks
    from .research_ocean import ResearchOcean
    from .responses import Response
    from .research import control_indices
    from .matrix import CASES

    checks = Checks(on_update=save)
    report.update(checks=checks.items, metrics=checks.metrics)
    case = CASES["estimation_trajectory"]
    ocean = ResearchOcean(config, case)
    response = Response(ocean, control_indices(case), config)
    point = jnp.zeros(response.indices.size)
    if config.get("comparison_report"):
        import json
        from pathlib import Path
        from .provenance import file_hash

        path = Path(config["comparison_report"])
        reference = json.loads(path.read_text())
        for key in ("setup", "physics", "neutral_policy", "steps", "spinup_steps"):
            if config[key] != reference["config"][key]:
                raise ValueError(f"Jacobian reference configuration differs at {key}")
        current = report["provenance"]["source_sha256"]
        response_sources = {
            "benchmarks/ad_demos/research_ocean.py",
            "benchmarks/ad_demos/research.py",
            "benchmarks/ad_demos/responses.py",
        }
        for name, fingerprint in reference["provenance"]["source_sha256"].items():
            if (name.startswith("veros/") or name in response_sources) and current[name] != fingerprint:
                raise ValueError(f"Jacobian reference model source differs at {name}")
        for key in ("versions", "native_extensions", "runtime", "cpu_affinity"):
            if report["provenance"][key] != reference["provenance"][key]:
                raise ValueError(f"Jacobian reference environment differs at {key}")
        product_check = next(check for check in reference["checks"] if check["name"] == "grouped/individual Jacobian")
        if not product_check["passed"]:
            raise ValueError("Reference streamed columns did not pass their numerical comparison")
        expected = jnp.asarray(product_check["detail"]["expected"])
        checks.metrics.update(
            individual_seconds=reference["metrics"]["individual_seconds"],
            individual_reference=dict(
                path=str(path), sha256=file_hash(path), source_snapshot=reference["provenance"]["source_snapshot"]
            ),
        )
    else:
        started = time.perf_counter()
        individual, measured = [], []
        report["stage"] = "individual control columns"
        for index, basis in enumerate(jnp.eye(point.size)):
            value, product = response.jvp(point, basis)
            jax.block_until_ready((value, product))
            individual.append((value, product))
            measured.append(dict(index=index, product=np.asarray(product)))
            checks.metrics.update(individual_columns=measured, individual_seconds=time.perf_counter() - started)
            save()
        expected = jnp.stack([product for _, product in individual])
        jax.block_until_ready(expected)
        checks.metrics["individual_seconds"] = time.perf_counter() - started
    checks.require("finite individual products", np.isfinite(expected).all(), dict(shape=expected.shape))
    canonical = response.values(point)
    report["stage"] = "grouped control columns"
    save()
    started = time.perf_counter()
    values, grouped = response.finite_provider().jacobian(point)
    jax.block_until_ready((values, grouped))
    durations = [time.perf_counter() - started]
    checks.close("grouped/canonical primal", values, canonical, rtol=1e-10)
    checks.close("grouped/individual Jacobian", grouped, expected, rtol=1e-5)
    for repeat in range(1, config.get("jacobian_repeats", 1)):
        report["stage"] = f"grouped repeated call {repeat + 1}"
        save()
        started = time.perf_counter()
        repeated_values, repeated = response.finite_provider().jacobian(point)
        jax.block_until_ready((repeated_values, repeated))
        durations.append(time.perf_counter() - started)
        checks.close(f"repeated primal {repeat + 1}", repeated_values, canonical, rtol=1e-10)
        checks.close(f"repeated Jacobian {repeat + 1}", repeated, expected, rtol=1e-5)
    checks.metrics.update(grouped_call_seconds=durations, grouped_seconds=float(np.mean(durations)))
    report["stage"] = "checkpoint adjoint comparison"
    save()
    seed = jnp.sin(jnp.arange(values.size)).reshape(values.shape)
    adjoint = response.vjp(point, seed)
    contraction = jnp.sum(grouped * seed, axis=(1, 2))
    checks.close("grouped Jacobian/adjoint", contraction, adjoint, rtol=1e-5)
    checks.metrics.update(
        speedup=checks.metrics["individual_seconds"] / checks.metrics["grouped_seconds"],
        steps=response.steps,
        horizon_days=response.steps * ocean.dt / 86400,
        controls=point.size,
        settings=dict(ocean.sim.state.settings.items()),
        checkpoint_windows=response.finite.statistics,
        scope="one shared primal linearization per timestep; control columns mapped without batching numerical primitives",
    )
    checks.require(
        "grouped products reduce end-to-end Jacobian time by 20%",
        checks.metrics["grouped_seconds"] <= 0.8 * checks.metrics["individual_seconds"],
        dict(individual=checks.metrics["individual_seconds"], grouped=checks.metrics["grouped_seconds"]),
    )


def physics_profile(config, report, save, destination):
    """Forward circulation, tracer inventories, and energy bookkeeping under a declared policy."""
    import jax.numpy as jnp
    from .common import Checks
    from .research_ocean import ResearchOcean, CONTROL_NAMES
    from .matrix import CASES
    from veros.diagnostics.energy import diagnose_kernel

    checks = Checks(on_update=save)
    report.update(checks=checks.items, metrics=checks.metrics)
    ocean = ResearchOcean(config, CASES["sensitivity_trajectory"])
    carry, parameters = ocean.controls(jnp.zeros(len(CONTROL_NAMES)))
    volume = ocean.volume(carry)
    area = ocean.area
    initial = np.array(
        [
            float(jnp.sum(getattr(carry.variables, name)[2:-2, 2:-2, :, carry.variables.tau] * volume))
            for name in ("temp", "salt")
        ]
    )
    inputs = np.zeros(2)
    records = []
    fields = {}
    for step in range(config["steps"]):
        carry, _ = ocean.prepared.advance(carry, n_steps=1, parameters=parameters, final_only=True)
        ocean.prepared.check_status(carry)
        inputs += ocean.dt * np.array(
            [
                float(jnp.sum(getattr(carry.variables, "forc_" + name + "_surface")[2:-2, 2:-2] * area))
                for name in ("temp", "salt")
            ]
        )
        if step + 1 in sorted({1, min(30, config["steps"]), config["steps"]}):
            vs = carry.variables
            inventory = np.array(
                [float(jnp.sum(getattr(vs, name)[2:-2, 2:-2, :, vs.tau] * volume)) for name in ("temp", "salt")]
            )
            residual = inventory - initial - inputs
            scale = np.maximum(np.abs(inputs), 1e-12 * np.abs(initial))
            checks.require(
                f"tracer inventory budget {step + 1}",
                np.all(np.abs(residual) <= 1e-7 * scale + 1e-12 * np.abs(initial)),
                dict(
                    inventory_change=(inventory - initial),
                    integrated_surface_input=inputs.copy(),
                    residual=residual,
                    scale=scale,
                ),
            )
            state = ocean.prepared._make_state(vs, parameters)
            energy = diagnose_kernel(state)._asdict()
            checks.require(
                f"finite energy bookkeeping {step + 1}", all(np.isfinite(value) for value in energy.values()), energy
            )
            records.append(
                dict(
                    step=step + 1,
                    observations=ocean.observe(carry),
                    energy=energy,
                    tracer_range={
                        name: [float(jnp.min(getattr(vs, name))), float(jnp.max(getattr(vs, name)))]
                        for name in ("temp", "salt")
                    },
                    unstable_wet_faces=int(jnp.sum((vs.Nsqr[..., vs.tau] < 0) & vs.maskW)),
                )
            )
            for name in ("temp", "salt", "u", "v", "tke", "eke", "rho", "Nsqr", "kappaH"):
                if name not in vs:
                    continue
                value = vs.get(name)
                value = value[..., vs.tau] if value.ndim == 4 else value
                fields[f"step_{step + 1}_{name}"] = np.asarray(value)
                checks.require(
                    f"finite {name} field {step + 1}",
                    np.isfinite(fields[f"step_{step + 1}_{name}"]).all(),
                    dict(minimum=float(np.min(value)), maximum=float(np.max(value))),
                )
            checks.metrics["forward_records"] = records
            save()
    checks.metrics.update(
        settings=dict(ocean.sim.state.settings.items()),
        initial_inventory=initial,
        policy=config["neutral_policy"],
        scope="surface tracer budgets and resolved forward circulation; isolated neutral/skew energy exchange has separate regression coverage",
    )
    path = destination.with_suffix(".npz")
    np.savez_compressed(path, **fields)
    from .provenance import file_hash

    checks.metrics.update(fields=str(path), fields_sha256=file_hash(path))


def reference_data(report, save, destination):
    """Refined basin runs and conservative tracer projection for early closure work."""
    import jax.numpy as jnp
    from veros import VerosSetup, veros_routine
    from .common import Checks

    checks = Checks(on_update=save)
    report.update(checks=checks.items, metrics=checks.metrics)

    class Basin(VerosSetup):
        @veros_routine
        def set_parameter(self, state):
            state.settings.update(
                nx=8,
                ny=10,
                nz=4,
                dt_tracer=300,
                dt_mom=300,
                runlen=0,
                enable_streamfunction=False,
                enable_conserve_energy=False,
                enable_hor_diffusion=True,
                K_h=10,
                kappaH_0=1e-4,
                kappaM_0=1e-3,
                enable_implicit_vert_friction=True,
                eq_of_state_type=1,
            )

        @veros_routine
        def set_grid(self, state):
            vs, settings = state.variables, state.settings
            vs.dxt = jnp.full_like(vs.dxt, 160000 / settings.nx)
            vs.dyt = jnp.full_like(vs.dyt, 200000 / settings.ny)
            vs.dzt = jnp.full_like(vs.dzt, 400 / settings.nz)

        @veros_routine
        def set_coriolis(self, state):
            pass

        @veros_routine
        def set_topography(self, state):
            state.variables.kbot = state.variables.kbot.at[2:-2, 2:-2].set(1)

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            anomaly = jnp.sin(2 * jnp.pi * vs.xt[:, None, None] / 160000) * jnp.cos(
                jnp.pi * vs.yt[None, :, None] / 200000
            )
            vs.temp = ((10 + 0.002 * vs.zt[None, None, :] + anomaly) * vs.maskT)[..., None] * jnp.ones(3)
            vs.salt = jnp.full_like(vs.salt, 35) * vs.maskT[..., None]

        @veros_routine(supports_compilation=True)
        def set_forcing(self, state):
            pass

        @veros_routine
        def set_diagnostics(self, state):
            for diagnostic in state.diagnostics.values():
                diagnostic.output_frequency = diagnostic.sampling_frequency = None

        @veros_routine(supports_compilation=True)
        def after_timestep(self, state):
            pass

    simulations = [Basin(), Basin(override=dict(nx=16, ny=20, nz=8))]
    prepared, carries = [], []
    for sim in simulations:
        sim.setup()
        integrator = sim.prepare_integrator(record_diagnostics=())
        prepared.append(integrator)
        carries.append(integrator.pack(sim.state))
    cv, fv = [
        np.asarray(c.variables.area_t[2:-2, 2:-2, None] * c.variables.dzt * c.variables.maskT[2:-2, 2:-2])
        for c in carries
    ]
    data = {"fine_volume": fv, "coarse_volume": cv}
    for name in ("temp", "salt"):
        projected, volume = conservative_project(
            np.asarray(getattr(carries[1].variables, name)[2:-2, 2:-2, :, 0]), fv, cv.shape
        )
        checks.close(
            name + " projected inventory",
            np.sum(projected * volume),
            np.sum(getattr(carries[1].variables, name)[2:-2, 2:-2, :, 0] * fv),
            rtol=1e-12,
        )
        checks.close("aligned reference volume", volume, cv, rtol=1e-12)
        # Matching coarse initial tracer inventories isolates subsequent resolution differences.
        values = getattr(carries[0].variables, name).at[2:-2, 2:-2, :, :].set(jnp.asarray(projected)[..., None])
        carries[0] = prepared[0].replace_variables(carries[0], **{name: values})
    from veros.core.numerics import calc_initial_conditions_kernel
    from veros.routines import tracing_context

    state = prepared[0]._make_state(carries[0].variables)
    with tracing_context(), state.variables.unlock():
        state.variables.update(calc_initial_conditions_kernel(state))
    carries[0] = carries[0]._replace(variables=state.variables)
    trajectories = [[], []]
    for index, (integrator, carry) in enumerate(zip(prepared, carries)):
        for step in range(25):
            if step:
                carry, _ = integrator.advance(carry, n_steps=1, final_only=True)
                integrator.check_status(carry)
            trajectories[index].append(
                np.stack(
                    [
                        np.asarray(getattr(carry.variables, name)[2:-2, 2:-2, :, carry.variables.tau])
                        for name in ("temp", "salt")
                    ]
                )
            )
    coarse, fine = [np.stack(values) for values in trajectories]
    projected = np.stack([[conservative_project(tracer, fv, cv.shape)[0] for tracer in state] for state in fine])
    coarse_tendency, fine_tendency = np.diff(coarse, axis=0) / 300, np.diff(projected, axis=0) / 300
    data.update(
        coarse_tracers=coarse,
        projected_tracers=projected,
        projected_total_tendency=fine_tendency,
        coarse_total_tendency=coarse_tendency,
        total_resolution_correction=fine_tendency - coarse_tendency,
        normalization=np.maximum(np.sqrt(np.mean(fine_tendency[:12] ** 2, axis=(0, 2, 3, 4))), 1e-12),
        training_steps=np.arange(12),
        validation_steps=np.arange(12, 24),
    )
    for name, trajectory in (("coarse", coarse), ("refined projected", projected)):
        inventory = np.sum(trajectory * cv, axis=(2, 3, 4))
        checks.close(
            name + " closed-basin tracer budget", inventory, np.broadcast_to(inventory[0], inventory.shape), rtol=1e-11
        )
    path = destination.with_suffix(".npz")
    np.savez_compressed(path, **data)
    from .provenance import file_hash

    checks.metrics.update(
        dataset=str(path),
        sha256=file_hash(path),
        coarse_shape=cv.shape,
        fine_shape=fv.shape,
        dt_seconds=300,
        steps=24,
        boundary_tracer_flux=0,
        normalization_scope="training projected tendency RMS, floored at 1e-12 tracer-units/s",
        correction_scope="total resolution discrepancy between separate rollouts; includes state divergence",
        next_gate="matched-state process tendencies, momentum projection, refined circulation targets and vertical-flux representability",
    )
