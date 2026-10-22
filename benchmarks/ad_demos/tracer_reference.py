"""Independent refined rollouts and matched vertical-mixing corrections."""

from pathlib import Path
import tempfile

import numpy as np


def nested_inventory(values, volumes, shape):
    import jax.numpy as jnp

    ratios = tuple(fine // coarse for fine, coarse in zip(values.shape, shape))
    if any(fine != coarse * ratio for fine, coarse, ratio in zip(values.shape, shape, ratios)):
        raise ValueError("Reference projection requires aligned nested cells")
    layout = tuple(component for pair in zip(shape, ratios) for component in pair)
    return jnp.sum((values * volumes).reshape(layout), axis=(1, 3, 5))


def vertical_flux_tendencies(variables, diffusivity):
    """Zero boundary fluxes on the native vertical finite-volume stencil."""
    import jax.numpy as jnp

    thickness = variables.dzt[None, None, :] * (variables.hfacT if "hfacT" in variables else variables.maskT)
    values = []
    for name in ("temp", "salt"):
        tracer = getattr(variables, name)[..., variables.tau]
        flux = (
            jnp.zeros_like(tracer)
            .at[..., :-1]
            .set(
                diffusivity[..., :-1]
                * (tracer[..., 1:] - tracer[..., :-1])
                / variables.dzw[None, None, :-1]
                * variables.maskW[..., :-1]
            )
        )
        tendency = flux - jnp.concatenate((jnp.zeros_like(flux[..., :1]), flux[..., :-1]), axis=-1)
        tendency = jnp.where(variables.maskT, tendency / jnp.maximum(thickness, 1e-20), 0)
        tendency = tendency.at[:2].set(0).at[-2:].set(0).at[:, :2].set(0).at[:, -2:].set(0)
        values.append(tendency)
    return jnp.stack(values)


def positive_flux_lower_bound(states, targets, scales, *, diagnostics=None):
    """Relax the law to arbitrary nonnegative diffusivity at every wet face."""
    from scipy.optimize import nnls

    squared_error, dual_bound, columns = 0.0, 0.0, 0
    for carry, target in zip(states, targets):
        vs = carry.variables
        tracers = np.stack([np.asarray(getattr(vs, name)[2:-2, 2:-2, :, vs.tau]) for name in ("temp", "salt")])
        dz, dw, wet = np.asarray(vs.dzt), np.asarray(vs.dzw), np.asarray(vs.maskT[2:-2, 2:-2])
        for i, j in np.ndindex(wet.shape[:2]):
            levels = np.flatnonzero(wet[i, j])
            faces = levels[:-1]
            matrix = np.zeros((2 * dz.size, faces.size))
            for column, k in enumerate(faces):
                gradient = (tracers[:, i, j, k + 1] - tracers[:, i, j, k]) / dw[k] / scales
                for tracer in range(2):
                    matrix[tracer * dz.size + k, column] = gradient[tracer] / dz[k]
                    matrix[tracer * dz.size + k + 1, column] = -gradient[tracer] / dz[k + 1]
            vector = (target[:, i + 2, j + 2] / scales[:, None]).reshape(-1)
            if not faces.size:
                error = np.linalg.norm(vector)
                certificate = error**2
            else:
                coefficients, error = nnls(matrix, vector)
                polar = vector - matrix @ coefficients
                active = matrix[:, np.any(matrix != 0, axis=0)]
                # A polar vector with A.T @ polar <= 0 gives a dual lower bound.
                for _ in range(3):
                    violation = active.T @ polar
                    if np.all(violation <= 0):
                        break
                    margin = 64 * np.finfo(float).eps * (np.abs(active).T @ np.abs(polar))
                    correction = np.linalg.lstsq(active.T, np.maximum(violation, 0) + margin, rcond=None)[0]
                    polar -= correction
                if np.any(active.T @ polar > 0):
                    raise RuntimeError("Positive-flux dual certificate is unresolved")
                certificate = max(0.0, 2 * vector @ polar - polar @ polar)
            squared_error += error**2
            dual_bound += certificate
            columns += 1
    size = np.asarray(targets).size
    if diagnostics is not None:
        diagnostics.update(
            columns=columns,
            primal_loss=squared_error / size,
            dual_loss_lower_bound=dual_bound / size,
            primal_dual_gap=(squared_error - dual_bound) / size,
            dual_feasibility="A.T @ polar <= 0 in every wet column",
        )
    return dual_bound / size


def build(ocean, carry, *, n_states, seed, refinement, projection, withheld_seed, stability_solver=None):
    import jax
    import jax.numpy as jnp
    from veros import veros_routine
    from veros.core import tke
    from veros.core.numerics import calc_initial_conditions_kernel
    from veros.core.utilities import enforce_boundaries
    from veros.routines import tracing_context
    from veros.integration import PreparedIntegrator
    from veros.ocean_response import canonical
    from .research_ocean import closure_law, YEAR
    from .provenance import file_hash
    from .grid_reference import refined_horizontal_origins, refined_vertical_widths, horizontal_alignment_residual

    if projection != "conservative" or refinement < 2 or n_states < 1:
        raise ValueError(
            "Independent references require conservative projection, refinement >= 2, and positive n_states"
        )
    settings, coarse = ocean.sim.state.settings, carry.variables
    parent_type = type(ocean.sim)
    fixed = dict(settings.items())
    fixed.update(
        nx=settings.nx * refinement,
        ny=settings.ny * refinement,
        nz=settings.nz * refinement,
        dt_tracer=settings.dt_tracer / refinement,
        dt_mom=settings.dt_mom / refinement,
        **refined_horizontal_origins(coarse, settings, refinement),
    )

    def prolong(value, dims, fine_shape):
        data = np.asarray(value)
        for axis, dimension in enumerate(dims or ()):
            if dimension in ("xt", "xu", "yt", "yu"):
                indices = np.arange(data.shape[axis])
                data = np.take(data, indices[2:-2], axis=axis)
                data = np.repeat(data, refinement, axis=axis)
                padding = [(0, 0)] * data.ndim
                padding[axis] = (2, 2)
                data = np.pad(data, padding, mode="edge")
            elif dimension in ("zt", "zw"):
                data = np.repeat(data, refinement, axis=axis)
        if data.shape != fine_shape:
            raise ValueError(f"Cannot prolong {dims}: {data.shape} differs from {fine_shape}")
        return jnp.asarray(data)

    class Refined(parent_type):
        @veros_routine(supports_compilation=True)
        def set_forcing(self, state):
            super().set_forcing(state)
            if ocean.config["setup"] == "global_4deg":
                # Retain the original physical salinity-restoring conductance.
                state.variables.forc_salt_surface *= coarse.dzt[-1] / state.variables.dzt[-1]

        @veros_routine
        def set_grid(self, state):
            vs = state.variables
            horizontal_units = state.settings.degtom if state.settings.coord_degree else 1
            vs.dxt = jnp.pad(
                jnp.repeat(coarse.dxt[2:-2] / (refinement * horizontal_units), refinement), (2, 2), mode="edge"
            )
            vs.dyt = jnp.pad(
                jnp.repeat(coarse.dyt[2:-2] / (refinement * horizontal_units), refinement), (2, 2), mode="edge"
            )
            vs.dzt = refined_vertical_widths(coarse, refinement)

        @veros_routine
        def set_topography(self, state):
            parent = jnp.repeat(jnp.repeat(coarse.kbot[2:-2, 2:-2], refinement, axis=0), refinement, axis=1)
            state.variables.kbot = state.variables.kbot.at[2:-2, 2:-2].set(
                jnp.where(parent > 0, (parent - 1) * refinement + 1, 0)
            )

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            if ocean.config["setup"] == "adjusted_basin":
                super().set_initial_conditions(state)
            prognostic = {"temp", "salt", "u", "v", "w", "tke", "eke", "E_iw"}
            prescribed = {
                "sss_clim",
                "sst_clim",
                "qnec",
                "qnet",
                "taux",
                "tauy",
                "t_star",
                "t_rest",
            }
            prescribed.update(name for name in coarse.fields() if name.startswith("research_"))
            for name in prognostic | prescribed:
                if name not in vs or name not in coarse:
                    continue
                values = prolong(getattr(coarse, name), coarse.__metadata__[name].dims, getattr(vs, name).shape)
                meta = vs.__metadata__[name]
                if meta.dims and len(meta.dims) >= 2 and meta.dims[0] in ("xt", "xu"):
                    values = enforce_boundaries(values, state.settings.enable_cyclic_x)
                setattr(vs, name, values)
            vs.time, vs.itt = coarse.time, coarse.itt
            vs.tau, vs.taum1, vs.taup1 = coarse.tau, coarse.taum1, coarse.taup1

    fine_simulation = Refined(override=fixed)
    fine_simulation.setup()
    fine = fine_simulation.prepare_integrator(record_diagnostics=(), checkpoint=True)
    fine_initial = fine.pack(fine_simulation.state)
    horizontal_error = float(
        horizontal_alignment_residual(
            coarse,
            fine_initial.variables,
            refinement=refinement,
            coord_degree=settings.coord_degree,
            metres_per_degree=settings.degtom,
        )
    )
    if not np.isfinite(horizontal_error) or horizontal_error > 1e-10:
        raise ValueError("Reference horizontal faces do not align with the coarse physical domain")
    bare = PreparedIntegrator(ocean.sim, record_diagnostics=(), checkpoint=True)
    coarse_volume = ocean.volume(carry)
    fine_volume = (
        fine_initial.variables.area_t[2:-2, 2:-2, None]
        * fine_initial.variables.dzt
        * fine_initial.variables.maskT[2:-2, 2:-2]
    )
    shape = coarse_volume.shape
    coarse_w_volume = (
        (coarse.area_t[2:-2, 2:-2, None] * coarse.dzw * coarse.maskW[2:-2, 2:-2]).at[..., -1].multiply(0.5)
    )
    fvs = fine_initial.variables
    fine_w_volume = (fvs.area_t[2:-2, 2:-2, None] * fvs.dzw * fvs.maskW[2:-2, 2:-2]).at[..., -1].multiply(0.5)

    def project(fine_carry):
        fine_carry = canonical(fine, fine_carry, fine_carry.variables.time)
        fv = fine_carry.variables
        reference = canonical(bare, carry, fv.time)
        changes = {}
        for name in ("temp", "salt"):
            values = getattr(reference.variables, name)
            for level in range(3):
                inventory = nested_inventory(getattr(fv, name)[2:-2, 2:-2, :, level], fine_volume, shape)
                values = values.at[2:-2, 2:-2, :, level].set(
                    jnp.where(coarse_volume > 0, inventory / jnp.maximum(coarse_volume, 1), 0)
                )
            changes[name] = enforce_boundaries(values, settings.enable_cyclic_x)
        for name in ("tke", "eke", "E_iw", "K_diss_v"):
            if name not in fv or name not in reference.variables:
                continue
            values = getattr(reference.variables, name)
            histories = range(3) if values.ndim == 4 else (None,)
            for level in histories:
                field = getattr(fv, name)[2:-2, 2:-2]
                if level is not None:
                    field = field[..., level]
                inventory = nested_inventory(field, fine_w_volume, shape)
                projected = jnp.where(coarse_w_volume > 0, inventory / jnp.maximum(coarse_w_volume, 1), 0)
                index = (slice(2, -2), slice(2, -2), slice(None)) + (() if level is None else (level,))
                values = values.at[index].set(projected)
            changes[name] = enforce_boundaries(values, settings.enable_cyclic_x)
        # Project face transports before converting back to velocities.
        for name, grid in (("u", "U"), ("v", "V")):
            values = getattr(reference.variables, name)
            wet = getattr(fv, "mask" + grid)[2:-2, 2:-2]
            if name == "u":
                area = fv.dyt[None, 2:-2, None] * fv.dzt * wet
                coarse_area = (
                    reference.variables.dyt[None, 2:-2, None]
                    * reference.variables.dzt
                    * reference.variables.maskU[2:-2, 2:-2]
                )
            else:
                area = fv.dxt[2:-2, None, None] * fv.cosu[None, 2:-2, None] * fv.dzt * wet
                coarse_area = (
                    reference.variables.dxt[2:-2, None, None]
                    * reference.variables.cosu[None, 2:-2, None]
                    * reference.variables.dzt
                    * reference.variables.maskV[2:-2, 2:-2]
                )
            for level in range(3):
                flux = getattr(fv, name)[2:-2, 2:-2, :, level] * area
                if name == "u":
                    flux = flux[refinement - 1 :: refinement]
                    flux = flux.reshape(shape[0], shape[1], refinement, shape[2], refinement).sum(axis=(2, 4))
                else:
                    flux = flux[:, refinement - 1 :: refinement]
                    flux = flux.reshape(shape[0], refinement, shape[1], shape[2], refinement).sum(axis=(1, 4))
                values = values.at[2:-2, 2:-2, :, level].set(
                    jnp.where(coarse_area > 0, flux / jnp.maximum(coarse_area, 1), 0)
                )
            changes[name] = enforce_boundaries(values, settings.enable_cyclic_x)
        values = reference.variables.w
        coarse_area = coarse.area_t[2:-2, 2:-2, None] * coarse.maskW[2:-2, 2:-2]
        for level in range(3):
            flux = fv.w[2:-2, 2:-2, refinement - 1 :: refinement, level] * fv.area_t[2:-2, 2:-2, None]
            flux = flux.reshape(shape[0], refinement, shape[1], refinement, shape[2]).sum(axis=(1, 3))
            values = values.at[2:-2, 2:-2, :, level].set(
                jnp.where(coarse_area > 0, flux / jnp.maximum(coarse_area, 1), 0)
            )
        changes["w"] = enforce_boundaries(values, settings.enable_cyclic_x)
        values = reference.variables.psi
        for level in range(3):
            if settings.enable_streamfunction:
                projected = fv.psi[2 + refinement - 1 : -2 : refinement, 2 + refinement - 1 : -2 : refinement, level]
            else:
                area = fv.area_t[2:-2, 2:-2] * fv.maskT[2:-2, 2:-2, -1]
                inventory = (
                    (fv.psi[2:-2, 2:-2, level] * area)
                    .reshape(shape[0], refinement, shape[1], refinement)
                    .sum(axis=(1, 3))
                )
                projected = inventory / jnp.maximum(coarse.area_t[2:-2, 2:-2], 1)
            values = values.at[2:-2, 2:-2, level].set(projected)
        changes["psi"] = enforce_boundaries(values, settings.enable_cyclic_x)
        reference = bare.replace_variables(reference, **changes)
        state = bare._make_state(reference.variables)
        with tracing_context(), state.variables.unlock():
            state.variables.update(calc_initial_conditions_kernel(state))
        return reference._replace(variables=state.variables)

    project = jax.jit(project)

    def native_process(prepared, state_carry):
        state = prepared._make_state(state_carry.variables)
        with tracing_context(), state.variables.unlock():
            tke.set_tke_diffusivities(state)
            tendency = vertical_flux_tendencies(state.variables, state.variables.kappaH)
        return tendency

    native_fine, native_coarse = jax.jit(lambda c: native_process(fine, c)), jax.jit(lambda c: native_process(bare, c))
    stride = max(refinement, round(YEAR / fine_simulation.state.settings.dt_tracer / 12))
    datasets = {}
    inventory_errors = []
    for label, start_seed, phase in (("train", seed, 0), ("heldout", withheld_seed, 90)):
        rng = np.random.default_rng(start_seed)
        noise = rng.normal(size=fine_volume.shape)
        parent_mean = nested_inventory(jnp.asarray(noise), fine_volume, shape) / jnp.maximum(
            nested_inventory(jnp.ones_like(fine_volume), fine_volume, shape), 1
        )
        parent_mean = jnp.repeat(
            jnp.repeat(jnp.repeat(parent_mean, refinement, axis=0), refinement, axis=1), refinement, axis=2
        )
        vs = fine_initial.variables
        temperature = vs.temp.at[2:-2, 2:-2].add(
            0.02 * (jnp.asarray(noise) - parent_mean)[..., None] * vs.maskT[2:-2, 2:-2, :, None]
        )
        current = fine.replace_variables(fine_initial, temp=temperature, time=vs.time + phase * 86400)
        shell = fine._make_state(current.variables)
        with tracing_context(), shell.variables.unlock():
            shell.variables.update(calc_initial_conditions_kernel(shell))
        current = current._replace(variables=shell.variables)
        states, corrections, fine_tendencies, coarse_tendencies = [], [], [], []
        for _ in range(n_states):
            current, _ = fine.advance(current, n_steps=stride, final_only=True)
            fine.check_status(current)
            matched = project(current)
            for name in ("temp", "salt"):
                original = float(
                    jnp.sum(getattr(current.variables, name)[2:-2, 2:-2, :, current.variables.tau] * fine_volume)
                )
                projected_inventory = float(
                    jnp.sum(getattr(matched.variables, name)[2:-2, 2:-2, :, matched.variables.tau] * coarse_volume)
                )
                inventory_errors.append(abs(projected_inventory - original) / max(abs(original), 1))
            tendency = native_fine(current)
            projected = jnp.stack(
                [
                    jnp.where(
                        coarse_volume > 0,
                        nested_inventory(value[2:-2, 2:-2], fine_volume, shape) / jnp.maximum(coarse_volume, 1),
                        0,
                    )
                    for value in tendency
                ]
            )
            target = jnp.zeros((2,) + matched.variables.maskT.shape).at[:, 2:-2, 2:-2].set(projected)
            baseline = native_coarse(matched)
            states.append(matched)
            corrections.append(target - baseline)
            fine_tendencies.append(target)
            coarse_tendencies.append(baseline)
        datasets[label] = dict(
            states=states, corrections=corrections, projected=fine_tendencies, baseline=coarse_tendencies
        )

    targets = np.stack(datasets["train"]["corrections"])
    scales = np.maximum(np.sqrt(np.mean(targets**2, axis=(0, 2, 3, 4))), 1e-12)
    flux_certificate = {}
    lower_bound = positive_flux_lower_bound(datasets["train"]["states"], targets, scales, diagnostics=flux_certificate)
    volume = np.asarray(coarse_volume)

    def budget_residual(weights):
        return jnp.stack(
            [
                jnp.sum(
                    vertical_flux_tendencies(state.variables, closure_law(state.variables, weights))[:, 2:-2, 2:-2]
                    * coarse_volume,
                    axis=(1, 2, 3),
                )
                / jnp.sum(coarse_volume)
                for state in datasets["train"]["states"]
            ]
        )

    reference_targets = {}
    adjusted_audits = []

    def response_target(kind, *, steps, heldout, statistics=None):
        if kind == "statistics" and statistics is None:
            raise ValueError("Statistical references require declared thresholds and sampling budgets")
        statistical_key = (
            (
                tuple(np.asarray(statistics["thresholds"]).reshape(-1)),
                statistics["sample_stride"],
                statistics["window_steps"],
                statistics["burnin_steps"],
                statistics["n_windows"],
            )
            if statistics is not None
            else None
        )
        key = kind, steps, heldout, statistical_key
        if key in reference_targets:
            return reference_targets[key]
        initial = fine.replace_variables(
            fine_initial, time=fine_initial.variables.time + (90 * 86400 if heldout and kind == "trajectory" else 0)
        )
        if heldout and kind in ("local", "equilibrium", "seasonal", "statistics"):
            shifted, _ = ocean.controls(jnp.zeros(16).at[0].set(0.2))
            difference = shifted.variables.temp - carry.variables.temp
            temperature = initial.variables.temp + prolong(
                difference, coarse.__metadata__["temp"].dims, initial.variables.temp.shape
            )
            initial = fine.replace_variables(initial, temp=temperature)
            shell = fine._make_state(initial.variables)
            with tracing_context(), shell.variables.unlock():
                shell.variables.update(calc_initial_conditions_kernel(shell))
            initial = initial._replace(variables=shell.variables)

        def observe(c):
            return ocean.observe(project(c))

        if kind == "local":
            values = []
            for phase in (45, 135, 225, 315) if heldout else (0, 90, 180, 270):
                current = fine.replace_variables(initial, time=fine_initial.variables.time + phase * 86400)
                current, _ = fine.advance(current, n_steps=refinement, final_only=True)
                fine.check_status(current)
                values.append(observe(current))
            result = jnp.stack(values)
        elif kind == "trajectory":
            current, previous, values = initial, 0, []
            horizon = steps * (2 if heldout else 1)
            for boundary in sorted({max(1, horizon // 3), max(1, 2 * horizon // 3), horizon}):
                current, _ = fine.advance(current, n_steps=(boundary - previous) * refinement, final_only=True)
                fine.check_status(current)
                values.append(observe(current))
                previous = boundary
            result = jnp.stack(values)
        elif kind in ("equilibrium", "seasonal"):
            if kind == "equilibrium":
                adjusted = fine.equilibrium_response(
                    initial, observe=observe, tolerance=1e-11, max_steps=20000, stability_solver=stability_solver
                )
            else:
                adjusted = fine.periodic_response(
                    initial,
                    observe=observe,
                    tolerance=1e-9,
                    period_steps=round(YEAR / fine_simulation.state.settings.dt_tracer),
                    phases=tuple(month / 12 for month in range(12)),
                    max_cycles=32,
                    stability_solver=stability_solver,
                )
            residuals = {
                name: float(adjusted[name])
                for name in ("residual", "spectral_radius", "stability_residual", "sensitivity_residual")
            }
            audit = dict(
                response=kind,
                heldout=heldout,
                phase_seconds=float(initial.variables.time),
                **residuals,
            )
            adjusted_audits.append(audit)
            if (
                adjusted["residual_kind"] != "full_state"
                or not np.isfinite(list(residuals.values())).all()
                or audit["residual"] >= 1e-7
                or audit["spectral_radius"] >= 1
                or audit["stability_residual"] >= min(1e-5, max(0, 1 - audit["spectral_radius"]))
                or audit["sensitivity_residual"] >= 1e-5
            ):
                raise RuntimeError("Independent refined adjusted reference has not resolved a stable branch")
            result = adjusted["observations"]
        elif kind == "statistics":
            from .statistical_reference import periodic_reference

            if ocean.config["setup"] != "adjusted_basin":
                raise ValueError(
                    "Refined statistical targets currently require the stable periodic basin configuration"
                )
            result, audit = periodic_reference(
                fine,
                initial,
                observe=observe,
                thresholds=statistics["thresholds"],
                period_steps=round(YEAR / fine_simulation.state.settings.dt_tracer),
                sample_stride=statistics["sample_stride"] * refinement,
                window_steps=statistics["window_steps"] * refinement,
                burnin_steps=statistics["burnin_steps"] * refinement,
                n_windows=statistics["n_windows"],
                seed=withheld_seed if heldout else seed,
                ram_checkpoints=ocean.config.get("ram_checkpoints", 64),
                stability_solver=stability_solver,
            )
            adjusted_audits.append(dict(response=kind, heldout=heldout, **audit))
        else:
            raise ValueError("Invariant statistical references require a separately validated estimator")
        reference_targets[key] = result
        return result

    path = Path(ocean.config.get("artifact_directory", tempfile.gettempdir())) / f"{ocean.case.name}-reference.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(coarse_volume=volume, fine_volume=np.asarray(fine_volume), training_normalization=scales)
    for prefix, variables in (("coarse", coarse), ("fine", fvs)):
        for name in ("xt", "yt", "xu", "yu", "dxt", "dyt"):
            arrays[prefix + "_" + name] = np.asarray(getattr(variables, name)[2:-2])
    if ocean.config["setup"] == "adjusted_basin":
        arrays.update(
            fine_basin_temperature_target=np.asarray(fvs.basin_temperature_target[2:-2, 2:-2]),
            fine_basin_salinity_target=np.asarray(fvs.basin_salinity_target[2:-2, 2:-2]),
            fine_xt=np.asarray(fvs.xt[2:-2]),
            fine_yt=np.asarray(fvs.yt[2:-2]),
            fine_zt=np.asarray(fvs.zt),
        )
    for label, data in datasets.items():
        arrays[label + "_projected_vertical_tendency"] = np.asarray(data["projected"])
        arrays[label + "_coarse_vertical_tendency"] = np.asarray(data["baseline"])
        arrays[label + "_vertical_correction"] = np.asarray(data["corrections"])
        arrays[label + "_tracers"] = np.stack(
            [
                np.stack(
                    [
                        np.asarray(getattr(c.variables, name)[2:-2, 2:-2, :, c.variables.tau])
                        for name in ("temp", "salt")
                    ]
                )
                for c in data["states"]
            ]
        )
    np.savez_compressed(path, **arrays)
    return dict(
        source="refined_veros",
        refinement=refinement,
        train_states=datasets["train"]["states"],
        train_tendencies=datasets["train"]["corrections"],
        heldout_states=datasets["heldout"]["states"],
        heldout_tendencies=datasets["heldout"]["corrections"],
        tendency_scale=jnp.asarray(scales)[:, None, None, None],
        flux_divergence=lambda c, k: vertical_flux_tendencies(c.variables, k),
        budget_residual=budget_residual,
        response_target=response_target,
        metadata=dict(
            dataset=str(path),
            sha256=file_hash(path),
            train_seed=seed,
            withheld_seed=withheld_seed,
            coarse_shape=list(shape),
            fine_shape=list(fine_volume.shape),
            process="instantaneous native vertical tracer diffusion; zero boundary flux",
            baseline="augment the native coarse vertical diffusion",
            normalization="training-only correction RMS, floored at 1e-12 tracer-units/s",
            positive_flux_loss_lower_bound=lower_bound,
            positive_flux_dual_certificate=flux_certificate,
            adjusted_responses=adjusted_audits,
            native_fine_settings=dict(fine_simulation.state.settings.items()),
            boundary_forcing=(
                "analytic basin restoring targets evaluated on refined coordinates"
                if ocean.config["setup"] == "adjusted_basin"
                else "coarse physical boundary data prolonged to the refined grid"
            ),
            projection="tracer inventories and normal face transports",
            projection_inventory_relative_error=max(inventory_errors),
            projection_horizontal_relative_error=horizontal_error,
            timestep_seconds=dict(coarse=settings.dt_tracer, refined=fine_simulation.state.settings.dt_tracer),
        ),
    )
