"""Experimental event integration for a prescribed-input mixing column.

Integration requires positive energy, free length, uncapped viscosity and
interval-certified branch transitions. The native history convention admits
downward crossings. The floored and preceding-shear conventions admit either
direction. Status zero denotes an accepted calculation. The caller owns transport conservation and source
accounting. See the acceptance-demo README for input meanings and units.

At continuous, transverse events, internal event-time variations cancel from
endpoints and moments. Reverse AD replays fixed internal times and branches,
retaining the final-time variation. Use integrate for forward AD and
differentiable_moments for reverse AD. Rejected columns yield nonfinite reverse
outputs. These experimental equations prescribe coefficients independently of
the coupled Veros calculation.
"""

import jax
import jax.numpy as jnp


def _coefficients(probe, N, previous, bg, fraction, a, mu_floor, *, history="raw"):
    """Return affine viscosity/thermal slopes and constants, and Prandtl number."""
    epsilon = jnp.asarray(1e-12, dtype=N.dtype)
    raw = a * probe
    if history == "preceding_shear":
        Pr = jnp.clip(6.6 * (N / jnp.maximum(previous, epsilon)), 1, 10)
        ts = a / Pr
        tc = jnp.zeros_like(N)
    else:
        history_floor = mu_floor if history == "floored" else epsilon
        normalized = previous / jnp.maximum(history_floor, raw)
        Ri = N / jnp.maximum(normalized, epsilon)
        Pr = jnp.clip(6.6 * Ri, 1, 10)
        intermediate = (Pr > 1) & (Pr < 10) & (raw > history_floor) & (normalized > epsilon)
        ts = jnp.where(intermediate, 0, a / Pr)
        tc = jnp.where(intermediate, previous / (6.6 * N), 0)
    below = ts * probe + tc < bg
    ks = jnp.where(below, (1 - fraction) * ts, ts)
    kc = jnp.where(below, fraction * bg + (1 - fraction) * tc, tc)
    ks = ks.at[-1].set(0)
    kc = kc.at[-1].set(0)
    active = raw > mu_floor
    return (jnp.where(active, a, 0), jnp.where(active, 0, mu_floor), ks, kc, Pr)


def _propagate(A, b, E, t):
    identity = jnp.eye(E.shape[0], dtype=A.dtype)
    norm = jnp.max(jnp.sum(jnp.abs(A), axis=1)) * jnp.abs(t)
    scale = jnp.maximum(0, jnp.ceil(jnp.log2(jnp.maximum(norm / 0.25, 1))).astype(jnp.int32))
    step = t / jnp.exp2(scale.astype(A.dtype))
    X = A * step

    def condition(c):
        return (c[0] < 39) & ~c[5]

    def body(c):
        k, term, P, C, D, done = c
        k = k + 1
        term = term @ X / k
        return (
            k,
            term,
            P + term,
            C + term / (k + 1),
            D + term / ((k + 1) * (k + 2)),
            jnp.max(jnp.abs(term)) < jnp.finfo(A.dtype).eps / 100,
        )

    k, term, P, C, D, done = jax.lax.while_loop(
        condition, body, (jnp.int32(0), identity, identity, identity, identity / 2, jnp.bool_(False))
    )
    C = C * step
    D = D * step * step

    def double(_, c):
        P, C, D = c
        return (P @ P, C + P @ C, 2 * D + C @ C)

    P, C, D = jax.lax.fori_loop(0, scale, double, (P, C, D))
    return (P @ E + C @ b, C @ E + D @ b, done)


def _integrate(
    N,
    Q,
    M,
    bg,
    fraction,
    ratio,
    energy,
    previous,
    h,
    eke_rate,
    total,
    ck,
    ce,
    mxl_min,
    mu_floor,
    mu_max,
    *,
    history="raw",
):
    if not jax.config.x64_enabled or N.dtype != jnp.float64:
        raise ValueError("prepared-column events require JAX float64")
    n = N.shape[0]
    directional = history != "raw"
    max_segments = 2 * n + 1 if history == "preceding_shear" else 4 * n + 1
    dtype = N.dtype
    epsilon = jnp.asarray(1e-12, dtype=dtype)
    a = ck * jnp.sqrt(2 / N)
    gamma = ce * jnp.sqrt(N / 2)
    if history == "preceding_shear":
        Pr = jnp.clip(6.6 * (N / jnp.maximum(previous, epsilon)), 1, 10)
        heatcross = (Pr * bg / a).at[-1].set(jnp.inf)
        bounds = jnp.stack((mu_floor / a, heatcross), axis=1)
    else:
        if history == "floored":
            pmin = jnp.clip(6.6 * N / jnp.maximum(previous / mu_floor, epsilon), 1, 10)
        else:
            pmin = jnp.where(
                previous > 0, jnp.clip(6.6 * N * epsilon / jnp.where(previous > 0, previous, 1), 1, 10), 10
            )
        middle = previous / (6.6 * N)
        lower = jnp.where(pmin < 10, pmin * middle / a, jnp.inf)
        upper = jnp.where(pmin < 10, 10 * middle / a, jnp.inf)
        heatcross = jnp.where(pmin < 10, jnp.where(bg <= middle, pmin, 10) * bg / a, 10 * bg / a)
        heatcross = heatcross.at[-1].set(jnp.inf)
        bounds = jnp.stack((mu_floor / a, lower, upper, heatcross), axis=1)
    pending = jnp.isfinite(bounds) & (bounds < energy[:, None])
    eligible = (
        jnp.all(energy > 0)
        & jnp.all(previous >= 0)
        & jnp.all(6.6 * N / epsilon >= 10)
        & (jnp.max(jnp.sqrt(2 * energy / N)) < jnp.min(h))
        & (jnp.max(a * energy) < mu_max)
        & jnp.all(N > 0)
        & jnp.all(M >= 0)
        & jnp.all(bg >= 0)
        & jnp.all(eke_rate >= 0)
        & jnp.all(h > 0)
        & (fraction >= 0)
        & (fraction <= 1)
        & (ratio > 0)
        & (total > 0)
        & (ck > 0)
        & (ce > 0)
        & (mxl_min > 0)
        & (mu_floor > 0)
        & (mu_max > mu_floor)
    )
    status = jnp.where(eligible, 0, 1).astype(jnp.int32)
    means = jnp.zeros((4, n), dtype=dtype)
    times = jnp.zeros(max_segments + 1, dtype=dtype)
    path = jnp.zeros((max_segments + 1, n), dtype=dtype).at[0].set(energy)
    events = jnp.zeros((max_segments, n), dtype=dtype)
    durations = jnp.zeros(max_segments, dtype=dtype)
    prandtl_trace = jnp.zeros((max_segments, n), dtype=dtype)

    def condition(c):
        return (c[0] < max_segments) & ~c[6] & (c[5] == 0)

    def body(c):
        (
            i,
            elapsed,
            E,
            means,
            pending,
            status,
            finished,
            times,
            path,
            events,
            durations,
            prandtl_trace,
            all_decreasing,
        ) = c
        low = jnp.max(jnp.where(pending, bounds, 0), axis=1)
        high = jnp.min(jnp.where(~pending, bounds, jnp.inf), axis=1)
        probe = jnp.where(jnp.isfinite(high), (low + high) / 2, jnp.maximum(2 * low, E))
        ms, mc, ks, kc, Pr = _coefficients(probe, N, previous, bg, fraction, a, mu_floor, history=history)
        A = Q + jnp.diag(M * ms - ratio * N * ks - gamma)
        b = M * mc + eke_rate - ratio * N * kc
        monotone = jnp.all(A @ E + b <= 0)
        remaining = total - elapsed
        end, integral, series_ok = _propagate(A, b, E, remaining)

        if directional:

            def gap(point):
                down = jnp.where(low > 0, (point - low) / jnp.where(low > 0, low, 1), jnp.inf)
                up = jnp.where(jnp.isfinite(high), (high - point) / jnp.where(jnp.isfinite(high), high, 1), jnp.inf)
                return jnp.min(jnp.minimum(down, up))

            crossing = gap(end) < 0
        else:

            def gap(point):
                safe = jnp.where(jnp.isfinite(bounds), bounds, 1)
                return jnp.min(jnp.where(pending, (point[:, None] - safe) / safe, jnp.inf))

            crossing = jnp.any(pending) & (gap(end) < 0)

        def locate(_):
            def bisect_condition(q):
                lo, hi, k, good = q
                mid = (lo + hi) / 2
                return (k < 80) & (mid != lo) & (mid != hi)

            def bisect_body(q):
                lo, hi, k, good = q
                mid = (lo + hi) / 2
                point, unused, ok = _propagate(A, b, E, mid)
                left = gap(point) >= 0
                return (jnp.where(left, mid, lo), jnp.where(left, hi, mid), k + 1, good & ok)

            lo, hi, k, good = jax.lax.while_loop(
                bisect_condition, bisect_body, (jnp.zeros((), dtype=dtype), remaining, jnp.int32(0), jnp.bool_(True))
            )
            duration = (lo + hi) / 2
            new, integ, ok = _propagate(A, b, E, duration)
            return (duration, new, integ, good & ok)

        duration, end, integral, event_ok = jax.lax.cond(
            crossing, locate, lambda _: (remaining, end, integral, jnp.bool_(True)), operand=None
        )
        unused, C, bound_ok = _propagate(A, jnp.zeros_like(A), jnp.eye(n, dtype=dtype), duration)
        eps = 500 * jnp.finfo(dtype).eps
        Cbound = jnp.abs(C) + eps * jnp.max(jnp.abs(C))
        d = A @ E + b
        derr = eps * (jnp.abs(A) @ jnp.abs(E) + jnp.abs(b))
        dl = d - derr
        du = d + derr
        lower_state = E + Cbound @ jnp.minimum(dl, 0) - eps * jnp.abs(E)
        upper_state = E + Cbound @ jnp.maximum(du, 0) + eps * jnp.abs(E)
        ad = A @ d
        aderr = jnp.abs(A) @ derr + eps * (jnp.abs(A) @ jnp.abs(d))
        lower_derivative = dl + Cbound @ jnp.minimum(ad - aderr, 0)
        upper_derivative = du + Cbound @ jnp.maximum(ad + aderr, 0)
        upper_resolved = (upper_state < high) | (upper_derivative <= 0)
        if directional:
            upper_resolved = upper_resolved | (lower_derivative > 0)
        lower_resolved = (low == 0) | (lower_state > low) | (upper_derivative < 0) | (lower_derivative >= 0)
        domain = (
            (lower_state > N * mxl_min**2 / 2) & (upper_state < N * jnp.min(h) ** 2 / 2) & (a * upper_state < mu_max)
        )
        positive_semigroup = jnp.all(A - jnp.diag(jnp.diag(A)) >= 0)
        branch_ok = positive_semigroup & jnp.all(upper_resolved & lower_resolved & domain)
        if directional:
            # Require motion across the boundary in the direction of the branch change.
            tolerance = 128 * jnp.finfo(dtype).eps
            remove = pending & (end[:, None] <= bounds * (1 + tolerance)) & (upper_derivative[:, None] < 0)
            add = (
                ~pending
                & jnp.isfinite(bounds)
                & (end[:, None] >= bounds * (1 - tolerance))
                & (lower_derivative[:, None] > 0)
            )
            new_pending = jnp.where(crossing, (pending & ~remove) | add, pending)
            event = jnp.min(jnp.where(remove | add, bounds, jnp.inf), axis=1)
            event = jnp.where(crossing & jnp.isfinite(event), event, 0)
            progress = ~crossing | jnp.any(remove | add)
        else:
            remove = pending & (end[:, None] <= bounds * (1 + 128 * jnp.finfo(dtype).eps))
            new_pending = jnp.where(crossing, pending & ~remove, pending)
            event = jnp.min(jnp.where(remove, bounds, jnp.inf), axis=1)
            event = jnp.where(crossing & jnp.isfinite(event), event, 0)
            progress = ~crossing | jnp.any(remove)
        status = jnp.where(
            ~branch_ok,
            2,
            jnp.where(
                ~(series_ok & event_ok & bound_ok), 3, jnp.where(jnp.any(end < 0), 4, jnp.where(~progress, 5, status))
            ),
        )
        means = means + jnp.stack(
            (integral, ms * integral + mc * duration, ks * integral + kc * duration, gamma * integral)
        )
        elapsed = elapsed + duration
        times = times.at[i + 1].set(elapsed)
        path = path.at[i + 1].set(end)
        events = events.at[i].set(event)
        durations = durations.at[i].set(duration)
        prandtl_trace = prandtl_trace.at[i].set(Pr)
        return (
            i + 1,
            elapsed,
            end,
            means,
            new_pending,
            status,
            ~crossing,
            times,
            path,
            events,
            durations,
            prandtl_trace,
            all_decreasing & monotone,
        )

    c = jax.lax.while_loop(
        condition,
        body,
        (
            jnp.int32(0),
            jnp.zeros((), dtype=dtype),
            energy,
            means,
            pending,
            status,
            jnp.bool_(False),
            times,
            path,
            events,
            durations,
            prandtl_trace,
            jnp.bool_(True),
        ),
    )
    (
        count,
        elapsed,
        end,
        means,
        pending,
        status,
        finished,
        times,
        path,
        events,
        durations,
        prandtl_trace,
        all_decreasing,
    ) = c
    status = jnp.where((status == 0) & ~finished, 6, status)
    status = jnp.where((status == 0) & ~jnp.all(jnp.sqrt(2 * end / N) > mxl_min), 7, status)
    minimum = jnp.min(jnp.where(jnp.arange(max_segments + 1)[:, None] <= count, path, jnp.inf))
    return dict(
        end=end,
        mean=means[0] / total,
        mean_mu=means[1] / total,
        mean_kh=means[2] / total,
        heat=means[3] / total,
        segment_count=count,
        segment_dt=durations,
        segment_event_threshold=events,
        segment_Pr=prandtl_trace,
        energy_path=path,
        times=times,
        status=status,
        continuous_branch_certified=status == 0,
        monotonicity_certified=(status == 0) & all_decreasing,
        stage_minimum=minimum,
    )


def integrate(N, Q, M, bg, fraction, ratio, energy, previous, h, eke_rate, total, ck, ce, mxl_min, mu_floor, mu_max):
    """Integrate using the native raw-viscosity friction-history convention."""
    return _integrate(
        N, Q, M, bg, fraction, ratio, energy, previous, h, eke_rate, total, ck, ce, mxl_min, mu_floor, mu_max
    )


def integrate_floored_history(
    N, Q, M, bg, fraction, ratio, energy, previous, h, eke_rate, total, ck, ce, mxl_min, mu_floor, mu_max
):
    """Integrate with max(mu_floor, raw viscosity) in friction-history normalization."""
    return _integrate(
        N,
        Q,
        M,
        bg,
        fraction,
        ratio,
        energy,
        previous,
        h,
        eke_rate,
        total,
        ck,
        ce,
        mxl_min,
        mu_floor,
        mu_max,
        history="floored",
    )


def integrate_preceding_shear(
    N, Q, M, bg, fraction, ratio, energy, previous_shear, h, source_rate, total, ck, ce, mxl_min, mu_floor, mu_max
):
    """Integrate one phase using supplied squared shear from the preceding momentum step."""
    return _integrate(
        N,
        Q,
        M,
        bg,
        fraction,
        ratio,
        energy,
        previous_shear,
        h,
        source_rate,
        total,
        ck,
        ce,
        mxl_min,
        mu_floor,
        mu_max,
        history="preceding_shear",
    )


MOMENT_FIELDS = ("end", "mean", "mean_mu", "mean_kh", "heat")
_MAX_SCALE = 16


def _propagate_fixed(A, b, energy, duration):
    """Scaled Taylor propagation with static loops for reverse differentiation."""
    eye = jnp.eye(energy.shape[0], dtype=A.dtype)
    norm = jnp.max(jnp.sum(jnp.abs(A), axis=1)) * jnp.abs(duration)
    scale = jnp.maximum(0, jnp.ceil(jnp.log2(jnp.maximum(norm / 0.25, 1))).astype(jnp.int32))
    step = duration / jnp.exp2(scale.astype(A.dtype))
    X = A * step

    def term(k, state):
        previous, P, C, D = state
        current = previous @ X / k
        return (current, P + current, C + current / (k + 1), D + current / ((k + 1) * (k + 2)))

    _, P, C, D = jax.lax.fori_loop(1, 40, term, (eye, eye, eye, eye / 2))
    C, D = (C * step, D * step * step)

    def double(k, state):
        def active(values):
            P, C, D = values
            return (P @ P, C + P @ C, 2 * D + C @ C)

        return jax.lax.cond(k < scale, active, lambda values: values, state)

    P, C, D = jax.lax.fori_loop(0, _MAX_SCALE, double, (P, C, D))
    end, integral = (P @ energy + C @ b, C @ energy + D @ b)
    return (jnp.where(scale <= _MAX_SCALE, end, jnp.nan), jnp.where(scale <= _MAX_SCALE, integral, jnp.nan))


def _replay(args, durations, count, probes, *, history="raw"):
    """Smooth replay with fixed internal event times and branch identities."""
    N, Q, M, bg, fraction, ratio, initial, previous, h, eke_rate, total, ck, ce, mxl_min, mu_floor, mu_max = args
    a = ck * jnp.sqrt(2 / N)
    gamma = ce * jnp.sqrt(N / 2)

    def body(i, state):
        def active(state):
            energy, moments, elapsed = state
            duration = jnp.where(i == count - 1, total - elapsed, durations[i])
            ms, mc, ks, kc, Pr = _coefficients(probes[i], N, previous, bg, fraction, a, mu_floor, history=history)
            A = Q + jnp.diag(M * ms - ratio * N * ks - gamma)
            b = M * mc + eke_rate - ratio * N * kc
            end, integral = _propagate_fixed(A, b, energy, duration)
            moments = moments + jnp.stack(
                (integral, ms * integral + mc * duration, ks * integral + kc * duration, gamma * integral)
            )
            return (end, moments, elapsed + duration)

        return jax.lax.cond(i < count, active, lambda state: state, state)

    end, moments, _ = jax.lax.fori_loop(
        0, durations.shape[0], body, (initial, jnp.zeros((4, N.shape[0]), dtype=N.dtype), jnp.zeros((), dtype=N.dtype))
    )
    return (end, *(moments[i] / total for i in range(4)))


def _capture(args, *, history="raw"):
    values = _integrate(*args, history=history)
    outputs = tuple((jnp.where(values["status"] == 0, values[key], jnp.nan) for key in MOMENT_FIELDS))
    probes = jax.lax.stop_gradient((values["energy_path"][:-1] + values["energy_path"][1:]) / 2)
    return (
        outputs,
        (args, jax.lax.stop_gradient(values["segment_dt"]), values["segment_count"], probes, values["status"]),
    )


@jax.custom_vjp
def differentiable_moments(*args):
    """Return five moment fields with a continuous-event reverse derivative."""
    return _capture(args)[0]


def _backward(residual, cotangent, *, history="raw"):
    args, durations, count, probes, status = residual
    _, pullback = jax.vjp(lambda values: _replay(values, durations, count, probes, history=history), args)
    gradient = pullback(cotangent)[0]
    return tuple((jnp.where(status == 0, value, jnp.nan) for value in gradient))


differentiable_moments.defvjp(lambda *args: _capture(args), _backward)


@jax.custom_vjp
def differentiable_floored_history_moments(*args):
    """Return moments with a reverse derivative for the floored-history convention."""
    return _capture(args, history="floored")[0]


differentiable_floored_history_moments.defvjp(
    lambda *args: _capture(args, history="floored"),
    lambda residual, cotangent: _backward(residual, cotangent, history="floored"),
)


@jax.custom_vjp
def differentiable_preceding_shear_moments(*args):
    """Return phase moments and their reverse derivative using preceding squared shear."""
    return _capture(args, history="preceding_shear")[0]


differentiable_preceding_shear_moments.defvjp(
    lambda *args: _capture(args, history="preceding_shear"),
    lambda residual, cotangent: _backward(residual, cotangent, history="preceding_shear"),
)
