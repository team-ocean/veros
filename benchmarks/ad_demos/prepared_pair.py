"""Experimental paired vertical advance on prescribed, complete wet cells."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.ad_demos.prepared_column_events import (
    MOMENT_FIELDS,
    integrate_preceding_shear,
    integrate_preceding_shear_halves,
)
from benchmarks.ad_demos.prepared_column_native import _increments, _stability
from veros.core.operators import solve_tridiagonal_jax

L = np.longdouble
EPS = L(np.finfo(float).eps)


@dataclass(frozen=True)
class Column:
    high: np.ndarray
    low: np.ndarray
    energy: np.ndarray
    stratification: np.ndarray
    preceding_shear: np.ndarray
    eddy_energy: np.ndarray
    eddy_loss_rate: np.ndarray
    thickness: np.ndarray
    spacing: np.ndarray
    width: np.ndarray
    pressure: np.ndarray
    background: np.ndarray
    area: float


@dataclass(frozen=True)
class ColumnDirection:
    high: np.ndarray
    low: np.ndarray
    energy: np.ndarray
    stratification: np.ndarray
    preceding_shear: np.ndarray


@dataclass(frozen=True)
class Side:
    """One live column slice or a prescribed exterior viscosity profile."""

    column: int | None
    offset: int = 0
    exterior_viscosity: np.ndarray | None = None


@dataclass(frozen=True)
class WetFace:
    velocity: np.ndarray
    thickness: np.ndarray
    spacing: np.ndarray
    area: float
    sides: tuple[Side, Side]


def _viscosity(energy, stratification, thickness, settings):
    # This experiment is on the free-length branch of native choice 2.
    if np.any(energy <= 0) or np.any(6.6 * stratification <= 1e-11):
        raise ValueError("a positive, stably stratified column is required")
    sqrt_energy = np.sqrt(energy)
    length = np.sqrt(2.0) * sqrt_energy / np.sqrt(stratification)
    raw = settings["c_k"] * length * sqrt_energy
    if (
        np.any(length <= settings["mxl_min"])
        or np.max(length) >= np.min(thickness)
        or np.max(raw) >= settings["kappaM_max"]
    ):
        raise ValueError("the column must have free mixing length and uncapped viscosity")
    return np.maximum(settings["kappaM_min"], raw)


@jax.jit
def _face_increment(velocity, thickness, spacing, viscosity, duration):
    conductance = viscosity / spacing
    lower = -duration * conductance / thickness[1:]
    upper = -duration * conductance / thickness[:-1]
    diagonal = jnp.ones_like(velocity).at[:-1].add(-upper).at[1:].add(-lower)
    flux = duration * conductance * jnp.diff(velocity)
    rhs = jnp.zeros_like(velocity).at[:-1].add(flux / thickness[:-1]).at[1:].add(-flux / thickness[1:])
    zero = jnp.zeros(1, dtype=velocity.dtype)
    a = jnp.concatenate((zero, lower))[None, None, :]
    c = jnp.concatenate((upper, zero))[None, None, :]
    wet = jnp.ones_like(a, dtype=bool)
    edge = jnp.zeros_like(wet).at[..., 0].set(True)
    return solve_tridiagonal_jax(a, diagonal[None, None, :], c, rhs[None, None, :], wet, edge)[0, 0]


def _face_viscosity(face, profiles, exterior):
    n = len(face.spacing)
    sides = [
        exterior(side) if side.column is None else profiles[side.column][side.offset : side.offset + n]
        for side in face.sides
    ]
    return (sides[0].astype(L) + sides[1].astype(L)) / 2


def _face_work(columns, faces, viscosities, duration, rho):
    rates = [np.zeros(len(col.energy), dtype=L) for col in columns]
    outgoing_shear = [rate.copy() for rate in rates]
    assigned = np.zeros(len(columns), dtype=L)
    bounds = np.zeros(len(columns), dtype=L)
    union, exterior = L(0), L(0)
    endpoints = []
    for face in faces:
        n = len(face.spacing)
        viscosity = _face_viscosity(face, viscosities, lambda side: side.exterior_viscosity)
        incr = np.asarray(
            _face_increment(face.velocity, face.thickness, face.spacing, viscosity.astype(float), float(duration))
        )
        old = face.velocity.astype(L)
        new = old + incr.astype(L)
        spacing = face.spacing.astype(L)
        after = np.diff(new) / spacing
        midpoint = (np.diff(old) / spacing + after) / 2
        power = viscosity * after * midpoint
        measure = L(rho) * L(face.area) * face.thickness.astype(L)
        loss = -np.sum(measure * (new - old) * (new + old) / 2)
        scale = np.sum(measure * (new**2 + old**2) / 2)
        bound = 500 * EPS * scale
        resolved = L(duration) * L(rho) * L(face.area) * np.dot(spacing, power)
        if np.any(power < 0) or abs(resolved - loss) > bound:
            raise ValueError("resolved face kinetic work is inconsistent")
        union += loss
        exterior += loss * sum(side.column is None for side in face.sides) / 2
        for side in face.sides:
            if side.column is None:
                continue
            i, k = side.column, side.offset
            share = L(face.area) / (2 * L(columns[i].area))
            rates[i][k : k + n] += share * power
            outgoing_shear[i][k : k + n] += share * midpoint**2
            assigned[i] += loss / 2
            bounds[i] += bound / 2
        endpoints.append(np.asarray(new, dtype=float))
    for i, col in enumerate(columns):
        mapped = L(rho) * L(col.area) * L(duration) * np.dot(col.width.astype(L), rates[i])
        if abs(mapped - assigned[i]) > bounds[i]:
            raise ValueError("tracer friction rates do not account for their allocated face work")
    if abs(union - exterior - np.sum(assigned)) > np.sum(bounds):
        raise ValueError("shared and exterior work were not fully allocated")
    return dict(
        rates=tuple(rates),
        shear=tuple(np.asarray(x, dtype=float) for x in outgoing_shear),
        assigned=assigned,
        bounds=bounds,
        union=union,
        exterior=exterior,
        velocity=tuple(endpoints),
    )


@jax.jit
def _face_increment_tangent(velocity, thickness, spacing, viscosity, duration, velocity_change, viscosity_change):
    def increment(u, mu):
        return _face_increment(u, thickness, spacing, mu, duration)

    return jax.jvp(increment, (velocity, viscosity), (velocity_change, viscosity_change))


def solve_momentum_tangent(columns, faces, viscosities, duration, *, velocity, viscosity):
    """Differentiate resolved work and outgoing shear on the wet momentum faces."""
    rates = [np.zeros(len(c.energy), dtype=L) for c in columns]
    shear = [r.copy() for r in rates]
    endpoints = []
    for face, du0 in zip(faces, velocity, strict=True):
        nu = _face_viscosity(face, viscosities, lambda side: side.exterior_viscosity)
        dnu = _face_viscosity(face, viscosity, lambda side: np.zeros_like(side.exterior_viscosity))
        delta, delta_direction = _face_increment_tangent(
            face.velocity, face.thickness, face.spacing, nu.astype(float), float(duration), du0, dnu.astype(float)
        )
        old, old_direction = face.velocity.astype(L), np.asarray(du0, dtype=L)
        new = old + np.asarray(delta, dtype=L)
        new_direction = old_direction + np.asarray(delta_direction, dtype=L)
        spacing = face.spacing.astype(L)
        closing = np.diff(new) / spacing
        closing_direction = np.diff(new_direction) / spacing
        midpoint = (np.diff(old) / spacing + closing) / 2
        midpoint_direction = (np.diff(old_direction) / spacing + closing_direction) / 2
        power = dnu * closing * midpoint + nu * (closing_direction * midpoint + closing * midpoint_direction)
        for side in face.sides:
            if side.column is None:
                continue
            i, k = side.column, side.offset
            levels = slice(k, k + len(spacing))
            share = L(face.area) / (2 * L(columns[i].area))
            rates[i][levels] += share * power
            shear[i][levels] += share * 2 * midpoint * midpoint_direction
        endpoints.append(np.asarray(new_direction, dtype=float))
    return dict(rates=tuple(rates), shear=tuple(np.asarray(s, dtype=float) for s in shear), velocity=tuple(endpoints))


def _evaluate_column(q, fields, weights):
    ratio = fields["dt_tracer"] / fields["dt_mom"]
    phase_count = len(fields["source_schedule"])
    n = len(fields["mu"])

    applied_mu = fields["mu"] * jnp.exp(q[:n])
    applied_kh = jnp.concatenate((fields["kh"][:-1] * jnp.exp(q[n:]), jnp.zeros(1, dtype=q.dtype)))
    cn = _increments(
        fields["high"],
        fields["low"],
        jnp.stack((applied_kh, applied_kh)),
        fields["h"],
        fields["d"],
        jnp.repeat(fields["dt_tracer"], 2),
    )
    stability = _stability(
        fields["high"],
        fields["low"],
        cn["high"],
        cn["low"],
        fields["pressure"],
        fields["d"],
        fields["gravity"],
        fields["rho0"],
    )
    N = stability["Nmid"]
    fraction = jnp.minimum(
        1.0, jnp.min(fields["source_schedule"][:, :-1] / (ratio * fields["bg"][:-1] * N[:-1]), axis=1)
    )
    fraction = jnp.where(fraction < 1, fraction * (1 - 8 * jnp.finfo(q.dtype).eps), fraction)
    conductance = fields["alpha_tke"] * (applied_mu[1:] + applied_mu[:-1]) / 2 / fields["h"][1:]
    upper = conductance / fields["width"][:-1]
    lower = conductance / fields["width"][1:]
    diagonal = jnp.zeros_like(N).at[:-1].add(-upper).at[1:].add(-lower)
    Q = jnp.diag(diagonal) + jnp.diag(upper, 1) + jnp.diag(lower, -1)
    M = jnp.zeros_like(N)
    E = fields["energy"]
    means = {key: jnp.zeros_like(E) for key in MOMENT_FIELDS[1:]}
    traces = {}
    ok = jnp.bool_(True)
    for phase in range(phase_count):
        event = integrate_preceding_shear(
            N,
            Q,
            M,
            fields["bg"],
            fraction[phase],
            ratio,
            E,
            fields["preceding_shear"],
            fields["h"],
            fields["source_schedule"][phase],
            fields["dt_mom"] / phase_count,
            fields["ck"],
            fields["ce"],
            fields["mxl_min"],
            fields["mu_floor"],
            fields["mu_max"],
        )
        E = event["end"]
        means = {key: means[key] + event[key] for key in means}
        traces.update({f"phase{phase}_{key}": value for key, value in event.items()})
        ok = ok & (event["status"] == 0) & event["continuous_branch_certified"]
    if phase_count == 1:
        traces = event
    means = {key: value / phase_count for key, value in means.items()}
    return dict(
        end=E,
        **means,
        N=N,
        Nold=stability["Nold"],
        Nnew=stability["Nnew"],
        M=M,
        Q=Q,
        fraction=jnp.min(fraction),
        phase_fractions=fraction,
        source0=fields["source_rate"],
        residual=jnp.concatenate(
            (jnp.log(means["mean_mu"] / applied_mu), jnp.log(means["mean_kh"][:-1] / applied_kh[:-1]))
        ),
        work=jnp.vdot(jnp.asarray(weights), N * means["mean_kh"]),
        cn_delta=cn["delta"],
        cn_high=cn["high"],
        cn_low=cn["low"],
        input_mu=applied_mu,
        input_kh=applied_kh,
        incoming_energy=fields["energy"],
    ), (ok, traces)


def _solve_column(col, clocks, settings, friction, work, i, guess):
    dt, tr = map(L, clocks)
    if np.any(col.eddy_loss_rate < 0) or np.any(col.eddy_energy < 0):
        raise ValueError("eddy energy and its loss rate must be nonnegative")
    debit = np.minimum(col.eddy_energy.astype(L), dt * col.eddy_loss_rate.astype(L)) * (1 - 8 * EPS)
    eddy_rate = (debit / dt).astype(float)
    schedule = np.asarray(friction + eddy_rate.astype(L), dtype=float)
    average = np.mean(friction, axis=0).astype(float)
    if np.any(friction < 0) or np.any(schedule < eddy_rate):
        raise ValueError("scheduled friction work must be nonnegative in each phase")
    source = average + eddy_rate
    if np.any(abs(np.mean(schedule.astype(L), axis=0) - source) > 500 * EPS * source):
        raise ValueError("work schedule must deliver the original mean source")
    mu, kh = guess
    n = len(mu)
    weights = np.asarray(L(settings["rho_0"]) * L(col.area) * col.width.astype(L) * tr, dtype=float)
    host = dict(
        high=col.high,
        low=col.low,
        energy=col.energy,
        h=col.thickness,
        d=col.spacing,
        width=col.width,
        pressure=col.pressure,
        bg=col.background,
        preceding_shear=col.preceding_shear,
        source_rate=source,
        friction_rate=average,
        eke_rate=eddy_rate,
        source_schedule=schedule,
        outgoing_shear=work["shear"][i],
        resolved_face_loss_J=np.asarray(work["assigned"][i]),
        resolved_face_roundoff_J=np.asarray(work["bounds"][i]),
        dt_mom=np.asarray(float(dt)),
        dt_tracer=np.asarray(float(tr)),
        mu=mu,
        kh=kh,
    )
    for key, name in (
        ("gravity", "grav"),
        ("rho0", "rho_0"),
        ("ck", "c_k"),
        ("ce", "c_eps"),
        ("mxl_min", "mxl_min"),
        ("mu_floor", "kappaM_min"),
        ("mu_max", "kappaM_max"),
        ("alpha_tke", "alpha_tke"),
    ):
        host[key] = np.asarray(settings[name])
    fields = {
        key: jnp.asarray(v)
        for key, v in host.items()
        if key not in ("resolved_face_loss_J", "resolved_face_roundoff_J")
    }

    def evaluate(q):
        return _evaluate_column(q, fields, weights)

    @jax.jit
    def residual_product(q, seed):
        return jax.jvp(lambda x: evaluate(x)[0]["residual"], (q,), (seed,))[1]

    evaluate_jit = jax.jit(evaluate)
    q = np.zeros(2 * n - 1)
    matrix = None
    previous = None
    trials = []
    for request in range(1, 17):
        out, (ok, trace) = evaluate_jit(q)
        out = {key: np.asarray(value) for key, value in out.items()}
        if not bool(ok) or not np.all(out["end"] > 0):
            raise ValueError(f"the column left the certified event domain at iterate {request}")
        residual = np.max(abs(out["residual"]))
        mixing = tr * out["N"].astype(L) * out["input_kh"].astype(L)
        cnwork = np.dot(weights.astype(L), out["N"].astype(L) * out["input_kh"].astype(L))
        flow = dt * (out["Q"].astype(L) @ out["mean"].astype(L))
        credit = dt * average.astype(L)
        eddy = dt * eddy_rate.astype(L)
        bank = out["end"].astype(L) - col.energy.astype(L) - flow - credit - eddy + mixing + dt * out["heat"].astype(L)
        bound = 500 * EPS * np.max(col.energy.astype(L) + credit + eddy + np.maximum(flow, 0))
        change = None if previous is None else abs(L(out["work"]) - previous)
        trials.append(dict(residual=float(residual), max_bank=float(np.max(abs(bank))), bank_bound=float(bound)))
        eligible = residual <= 1e-13 and abs(L(out["work"]) - cnwork) <= L(".01") and np.max(abs(bank)) <= bound
        if eligible and change is not None and change <= L(".01"):
            return dict(
                inputs=host,
                output=out,
                event={key: np.asarray(value) for key, value in trace.items()},
                eddy_energy=col.eddy_energy.astype(L) - dt * eddy_rate.astype(L),
                clock=clocks,
                trials=trials,
            )
        previous = L(out["work"]) if eligible else None
        if matrix is None:
            matrix = np.column_stack([residual_product(q, seed) for seed in np.eye(len(q))])
        q = q - np.linalg.solve(matrix, out["residual"])
    raise ValueError("the physical column did not converge in sixteen iterations")


def solve_tracer_tangent(column, solved, *, high, low, energy, preceding_shear, source_schedule):
    """Differentiate an accepted tracer solve with fixed wet geometry and eddy supply.

    The inputs are changes in the two compensated tracer parts, incoming turbulent
    energy, squared preceding shear and the ordered total work rates. The
    derivative follows the accepted implicit viscosity and diffusivity root.
    """
    host, accepted = solved["inputs"], solved["output"]
    directions = dict(
        high=high, low=low, energy=energy, preceding_shear=preceding_shear, source_schedule=source_schedule
    )
    if any(np.shape(value) != np.shape(host[key]) for key, value in directions.items()):
        raise ValueError("column directions must match the accepted inputs")
    weights = np.asarray(L(host["rho0"]) * L(column.area) * host["width"].astype(L) * L(host["dt_tracer"]), dtype=float)
    fields = {
        key: jnp.asarray(value)
        for key, value in host.items()
        if key not in ("resolved_face_loss_J", "resolved_face_roundoff_J")
    }
    fields.update(mu=jnp.asarray(accepted["input_mu"]), kh=jnp.asarray(accepted["input_kh"]))
    directions = {key: jnp.asarray(np.asarray(value, dtype=host[key].dtype)) for key, value in directions.items()}
    directions["source_rate"] = jnp.mean(directions["source_schedule"], axis=0)
    size = 2 * len(fields["mu"]) - 1
    origin = jnp.zeros(size, dtype=fields["mu"].dtype)
    zero = jnp.zeros((), dtype=origin.dtype)
    one = jnp.ones_like(zero)

    def evaluate(q, alpha):
        inputs = dict(fields, **{name: fields[name] + alpha * slope for name, slope in directions.items()})
        state, (admitted, _) = _evaluate_column(q, inputs, weights)
        return state, admitted

    @jax.jit
    def product(q_direction, input_direction):
        return jax.jvp(evaluate, (origin, zero), (q_direction, input_direction), has_aux=True)

    state, partial, admitted = product(origin, one)
    residual = np.asarray(state["residual"])
    if not bool(admitted) or np.max(np.abs(residual)) > 1e-13:
        raise ValueError("the supplied tracer result is outside the certified root")
    columns = [product(axis, zero)[1] for axis in jnp.eye(size, dtype=origin.dtype)]
    jacobian = np.stack([np.asarray(value["residual"]) for value in columns], axis=-1)
    condition = np.linalg.cond(jacobian, np.inf)
    eps = np.finfo(residual.dtype).eps
    work_error = abs(
        np.dot(
            np.asarray([value["work"] for value in columns], dtype=L),
            np.linalg.solve(jacobian, -residual).astype(L),
        )
    )
    if not condition * (size * eps / (1 - size * eps)) < 0.002 or work_error > L(".01"):
        raise ValueError("the inferred column derivative is outside the certified accuracy bounds")
    root_direction = np.linalg.solve(jacobian, -np.asarray(partial["residual"]))
    tangent = {
        name: np.asarray(value)
        + np.tensordot(
            np.stack([np.asarray(column[name]) for column in columns], axis=-1), root_direction, axes=([-1], [0])
        )
        for name, value in partial.items()
    }
    if np.max(np.abs(tangent["residual"])) > 1e-13:
        raise ValueError("the differentiated tracer residual exceeds its original bound")
    return tangent


def _pair_clocks(columns, faces, dt_mom, dt_tracer):
    if len(columns) != 2 or sum(all(s.column is not None for s in f.sides) for f in faces) != 1:
        raise ValueError("two columns and one unique shared wet momentum face are required")
    if dt_mom <= 0 or dt_tracer <= 0:
        raise ValueError("both clocks must advance")
    return float(dt_mom), float(dt_tracer)


def _tracer_guesses(columns, viscosity, settings):
    guesses = []
    for c, mu in zip(columns, viscosity):
        raw = settings["c_k"] * c.energy * np.sqrt(2 / c.stratification)
        Pr = np.clip(6.6 * c.stratification / np.maximum(c.preceding_shear, 1e-12), 1, 10)
        kh = np.maximum(c.background, raw / Pr)
        kh[-1] = 0
        guesses.append((mu, kh))
    return guesses


def solve(columns, faces, settings, dt_mom, dt_tracer):
    """Advance an adjacent pair from two initial columns and their wet momentum faces.

    A face has two sides and gives half its work to each side. Exterior viscosity
    is prescribed. The caller supplies native geometry, incoming squared shear,
    eddy energy and its physical loss rate. Results carry both tracer profiles,
    outgoing face velocities and the work assigned to the pair and exterior.
    """
    full_clocks = _pair_clocks(columns, faces, dt_mom, dt_tracer)
    half_clocks = tuple(t / 2 for t in full_clocks)
    initial_mu = tuple(_viscosity(c.energy, c.stratification, c.thickness, settings) for c in columns)
    guesses = _tracer_guesses(columns, initial_mu, settings)
    early = _face_work(columns, faces, initial_mu, half_clocks[0], settings["rho_0"])
    predictor = tuple(
        _solve_column(c, half_clocks, settings, early["rates"][i][None, :], early, i, guesses[i])
        for i, c in enumerate(columns)
    )
    midpoint = tuple(
        _viscosity(p["output"]["end"], p["output"]["Nnew"], c.thickness, settings) for p, c in zip(predictor, columns)
    )
    average = tuple((a + b) / 2 for a, b in zip(initial_mu, midpoint))
    full = _face_work(columns, faces, average, full_clocks[0], settings["rho_0"])
    schedule = tuple(np.stack((a, 2 * b - a)) for a, b in zip(early["rates"], full["rates"]))
    corrected = tuple(
        _solve_column(
            c,
            full_clocks,
            settings,
            schedule[i],
            full,
            i,
            (predictor[i]["output"]["input_mu"], predictor[i]["output"]["input_kh"]),
        )
        for i, c in enumerate(columns)
    )
    return dict(predictor=predictor, midpoint_viscosity=midpoint, corrected=corrected, early_work=early, full_work=full)


def _carry_columns(columns, solved):
    return tuple(
        replace(
            column,
            high=step["output"]["cn_high"],
            low=step["output"]["cn_low"],
            energy=step["output"]["end"],
            stratification=step["output"]["Nnew"],
            preceding_shear=step["inputs"]["outgoing_shear"],
            eddy_energy=step["eddy_energy"],
        )
        for column, step in zip(columns, solved)
    )


def _carry_faces(faces, velocity):
    return tuple(replace(face, velocity=u) for face, u in zip(faces, velocity))


@jax.jit
def _split_viscosity(args):
    return tuple(
        (half["status"], half["continuous_branch_certified"], half["mean_mu"])
        for half in integrate_preceding_shear_halves(*args)
    )


def _forecast_phase_inputs(result):
    host, state, trace = (result[k] for k in ("inputs", "output", "event"))
    dt = float(host["dt_mom"]) / 4
    ratio = float(host["dt_tracer"]) / float(host["dt_mom"])
    for p, rate in enumerate(host["source_schedule"]):
        yield (
            state["N"],
            state["Q"],
            state["M"],
            host["bg"],
            state["phase_fractions"][p],
            ratio,
            trace[f"phase{p}_energy_path"][0],
            host["preceding_shear"],
            host["h"],
            rate,
            dt,
            host["ck"],
            host["ce"],
            host["mxl_min"],
            host["mu_floor"],
            host["mu_max"],
        )


def _forecast_viscosity(result):
    for args in _forecast_phase_inputs(result):
        for status, certified, viscosity in _split_viscosity(tuple(jnp.asarray(x) for x in args)):
            if int(status) or not bool(certified):
                raise ValueError("the viscosity left its certified branch within a work window")
            yield np.asarray(viscosity)


def _interval_shear(columns, opening, closing):
    shear = [np.zeros(len(c.energy), dtype=L) for c in columns]
    for face, u in zip(opening, closing):
        midpoint = np.diff((face.velocity.astype(L) + u.astype(L)) / 2) / face.spacing.astype(L)
        for side in face.sides:
            if side.column is None:
                continue
            i, k = side.column, side.offset
            share = L(face.area) / (2 * L(columns[i].area))
            shear[i][k : k + len(midpoint)] += share * midpoint**2
    return tuple(s.astype(float) for s in shear)


def solve_prospective(columns, faces, settings, dt_mom, dt_tracer):
    """Advance the prescribed pair using viscosity integrated within each work window.

    Short forecasts estimate when friction supplies energy. A provisional tracer
    step turns those estimates into eight viscosity means. The final momentum and
    tracer solves start at the original state, using those means and the resolved
    work from each momentum update. Pass the returned columns and faces to the
    next interval. The domain and boundary conditions are those of :func:`solve`.
    """
    clocks = _pair_clocks(columns, faces, dt_mom, dt_tracer)
    quarter = tuple(t / 4 for t in clocks)
    rho = settings["rho_0"]

    def viscosity(state):
        return tuple(_viscosity(c.energy, c.stratification, c.thickness, settings) for c in state)

    def tracers(state, duration, work, rates, guesses):
        return tuple(_solve_column(c, duration, settings, rates[i], work, i, guesses[i]) for i, c in enumerate(state))

    def accepted_guesses(results):
        return tuple((r["output"]["input_mu"], r["output"]["input_kh"]) for r in results)

    candidates, moving = columns, faces
    samples = [viscosity(columns)]
    leads = []
    lead_work = []
    for _ in range(3):
        work = _face_work(candidates, moving, samples[-1], quarter[0], rho)
        rates = tuple(rate[None, :] for rate in work["rates"])
        guess = _tracer_guesses(candidates, samples[-1], settings)
        lead = tracers(candidates, quarter, work, rates, guess)
        candidates = _carry_columns(candidates, lead)
        moving = _carry_faces(moving, work["velocity"])
        leads.append(lead)
        lead_work.append(work)
        samples.append(viscosity(candidates))

    mean = tuple(np.mean(np.stack(column), axis=0) for column in zip(*samples))
    provisional_work = _face_work(columns, faces, mean, clocks[0], rho)
    early = tuple(zip(*(tuple(r["inputs"]["friction_rate"].astype(L) for r in lead) for lead in leads)))
    rates = tuple(np.stack((*q, 4 * total - sum(q))) for q, total in zip(early, provisional_work["rates"]))
    forecast = tracers(columns, clocks, provisional_work, rates, accepted_guesses(leads[1]))

    moving = faces
    work_windows = []
    for means in zip(*map(_forecast_viscosity, forecast), strict=True):
        work = _face_work(columns, moving, means, clocks[0] / 8, rho)
        moving = _carry_faces(moving, work["velocity"])
        work_windows.append(work)
    work = {key: sum(stage[key] for stage in work_windows) for key in ("assigned", "bounds", "union", "exterior")}
    work["velocity"] = work_windows[-1]["velocity"]
    work["shear"] = _interval_shear(columns, faces, work["velocity"])
    schedule = tuple(np.stack(column) for column in zip(*(w["rates"] for w in work_windows)))
    corrected = tracers(columns, clocks, work, schedule, accepted_guesses(forecast))
    return dict(
        columns=_carry_columns(columns, corrected),
        faces=moving,
        leads=tuple(leads),
        forecast=forecast,
        corrected=corrected,
        full_work=work,
        schedule=schedule,
        lead_work=tuple(lead_work),
        work_windows=tuple(work_windows),
    )


def _viscosity_direction(column, viscosity, direction, settings):
    raw = (
        settings["c_k"]
        * np.sqrt(2.0)
        * np.sqrt(column.energy)
        / np.sqrt(column.stratification)
        * np.sqrt(column.energy)
    )
    relative = direction.energy / column.energy - direction.stratification / (2 * column.stratification)
    floor = settings["kappaM_min"]
    if np.any((raw == floor) & (relative != 0)):
        raise ValueError("the opening viscosity derivative meets the active floor")
    return np.where(raw > floor, viscosity * relative, 0)


@jax.jit
def _split_viscosity_tangent(args, direction):
    def moments(N, Q, fraction, energy, shear, rate):
        halves = integrate_preceding_shear_halves(
            N, Q, args[2], args[3], fraction, args[5], energy, shear, args[8], rate, *args[10:]
        )
        return halves[-1]["end"], tuple(half["mean_mu"] for half in halves)

    active = tuple(args[i] for i in (0, 1, 4, 6, 7, 9))
    return jax.jvp(moments, active, direction)[1]


def _forecast_viscosity_tangent(result, tangent, column, source):
    energy = column.energy
    for p, args in enumerate(_forecast_phase_inputs(result)):
        direction = (
            tangent["N"],
            tangent["Q"],
            tangent["phase_fractions"][p],
            energy,
            column.preceding_shear,
            source[p],
        )
        active = tuple(args[i] for i in (0, 1, 4, 6, 7, 9))
        slopes = tuple(
            jnp.asarray(np.asarray(value, dtype=np.asarray(base).dtype))
            for base, value in zip(active, direction, strict=True)
        )
        energy, means = _split_viscosity_tangent(tuple(map(jnp.asarray, args)), slopes)
        yield from map(np.asarray, means)


def _interval_shear_tangent(columns, faces, closing, opening_direction, closing_direction):
    shear = [np.zeros(len(c.energy), dtype=L) for c in columns]
    for face, u, dp, df in zip(faces, closing, opening_direction, closing_direction, strict=True):
        spacing = face.spacing.astype(L)
        mean = np.diff((face.velocity.astype(L) + u.astype(L)) / 2) / spacing
        variation = np.diff((np.asarray(dp, dtype=L) + np.asarray(df, dtype=L)) / 2) / spacing
        for side in face.sides:
            if side.column is None:
                continue
            i, k = side.column, side.offset
            share = L(face.area) / (2 * L(columns[i].area))
            shear[i][k : k + len(mean)] += 2 * share * mean * variation
    return tuple(np.asarray(s, dtype=float) for s in shear)


def solve_prospective_tangent(columns, faces, settings, dt_mom, dt_tracer, solved, *, column, velocity):
    """Differentiate one accepted prospective advance and return the carried directions."""
    clocks = _pair_clocks(columns, faces, dt_mom, dt_tracer)
    quarter = clocks[0] / 4

    def start_viscosity(state, directions):
        profiles = tuple(_viscosity(c.energy, c.stratification, c.thickness, settings) for c in state)
        slopes = tuple(
            _viscosity_direction(c, v, d, settings) for c, v, d in zip(state, profiles, directions, strict=True)
        )
        return profiles, slopes

    def tracer_tangents(state, roots, directions, source):
        return tuple(
            solve_tracer_tangent(
                c,
                root,
                high=d.high,
                low=d.low,
                energy=d.energy,
                preceding_shear=d.preceding_shear,
                source_schedule=rates,
            )
            for c, root, d, rates in zip(state, roots, directions, source, strict=True)
        )

    def carry_tracer(tangent, shear):
        return tuple(
            ColumnDirection(t["cn_high"], t["cn_low"], t["end"], t["Nnew"], s)
            for t, s in zip(tangent, shear, strict=True)
        )

    candidates, moving = columns, faces
    directions, speed = column, velocity
    moments, leads, profiles, slopes = [], [], [], []
    for roots, work in zip(solved["leads"], solved["lead_work"], strict=True):
        nu, dnu = start_viscosity(candidates, directions)
        profiles.append(nu)
        slopes.append(dnu)
        dh = solve_momentum_tangent(candidates, moving, nu, quarter, velocity=speed, viscosity=dnu)
        source = tuple(r[None, :] for r in dh["rates"])
        tracer = tracer_tangents(candidates, roots, directions, source)
        moments.append(dh)
        leads.append(tracer)
        candidates = _carry_columns(candidates, roots)
        moving = _carry_faces(moving, work["velocity"])
        directions, speed = carry_tracer(tracer, dh["shear"]), dh["velocity"]
    nu, dnu = start_viscosity(candidates, directions)
    profiles.append(nu)
    slopes.append(dnu)
    mean = tuple(np.mean(np.stack(v), axis=0) for v in zip(*profiles))
    dmean = tuple(np.mean(np.stack(v), axis=0) for v in zip(*slopes))
    provisional = solve_momentum_tangent(columns, faces, mean, clocks[0], velocity=velocity, viscosity=dmean)
    early = zip(*(item["rates"] for item in moments))
    source = tuple(np.stack((*q, 4 * total - sum(q))) for q, total in zip(early, provisional["rates"], strict=True))
    forecast = tracer_tangents(columns, solved["forecast"], column, source)
    means = zip(*map(_forecast_viscosity, solved["forecast"]), strict=True)
    forecast_directions = (
        _forecast_viscosity_tangent(root, tangent, old, rates)
        for root, tangent, old, rates in zip(solved["forecast"], forecast, column, source, strict=True)
    )

    moving, speed = faces, velocity
    mechanical = []
    for work, nu, dnu in zip(solved["work_windows"], means, zip(*forecast_directions, strict=True), strict=True):
        dh = solve_momentum_tangent(columns, moving, nu, clocks[0] / 8, velocity=speed, viscosity=dnu)
        mechanical.append(dh)
        moving = _carry_faces(moving, work["velocity"])
        speed = dh["velocity"]
    schedule = tuple(np.stack(r) for r in zip(*(work["rates"] for work in mechanical)))
    shear = _interval_shear_tangent(columns, faces, solved["full_work"]["velocity"], velocity, speed)
    corrected = tracer_tangents(columns, solved["corrected"], column, schedule)
    return dict(
        columns=carry_tracer(corrected, shear),
        velocity=speed,
        leads=tuple(leads),
        forecast=forecast,
        corrected=corrected,
        schedule=schedule,
    )
