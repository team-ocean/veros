"""Experimental paired vertical advance on prescribed, complete wet cells."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.ad_demos.prepared_column_events import integrate_preceding_shear, MOMENT_FIELDS
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


def _face_work(columns, faces, viscosities, duration, rho):
    rates = [np.zeros(len(col.energy), dtype=L) for col in columns]
    outgoing_shear = [rate.copy() for rate in rates]
    assigned = np.zeros(len(columns), dtype=L)
    bounds = np.zeros(len(columns), dtype=L)
    union, exterior = L(0), L(0)
    endpoints = []
    for face in faces:
        n = len(face.spacing)
        coeff = [
            side.exterior_viscosity if side.column is None else viscosities[side.column][side.offset : side.offset + n]
            for side in face.sides
        ]
        viscosity = (coeff[0].astype(L) + coeff[1].astype(L)) / 2
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
    ratio = fields["dt_tracer"] / fields["dt_mom"]
    phase_count = len(schedule)

    def evaluate(q):
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


def solve(columns, faces, settings, dt_mom, dt_tracer):
    """Advance an adjacent pair from two initial columns and their wet momentum faces.

    A face has two sides and gives half its work to each side. Exterior viscosity
    is prescribed. The caller supplies native geometry, incoming squared shear,
    eddy energy and its physical loss rate. Results carry both tracer profiles,
    outgoing face velocities and the work assigned to the pair and exterior.
    """
    if len(columns) != 2 or sum(all(s.column is not None for s in f.sides) for f in faces) != 1:
        raise ValueError("two columns and one unique shared wet momentum face are required")
    if dt_mom <= 0 or dt_tracer <= 0:
        raise ValueError("both clocks must advance")
    full_clocks = (float(dt_mom), float(dt_tracer))
    half_clocks = tuple(t / 2 for t in full_clocks)
    initial_mu = tuple(_viscosity(c.energy, c.stratification, c.thickness, settings) for c in columns)
    guesses = []
    for i, c in enumerate(columns):
        raw = settings["c_k"] * c.energy * np.sqrt(2 / c.stratification)
        Pr = np.clip(6.6 * c.stratification / np.maximum(c.preceding_shear, 1e-12), 1, 10)
        kh = np.maximum(c.background, raw / Pr)
        kh[-1] = 0
        guesses.append((initial_mu[i], kh))
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
