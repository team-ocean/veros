"""Full-density and bounded-shadowing prototypes for an expanding circle map.

The family is F_a(x) = 2x + sum_h a_h sin(2*pi*h*x)/(2*pi), modulo one.
At zero parameters its invariant density is one. Differentiating its transfer
operator gives (I-L) rho'_h = -L(b'_h)/2, with integral rho'_h = 0.
These one-dimensional benchmarks do not implement an ocean response estimator.
"""

import numpy as np


def doubling_transfer(values):
    """Push a real trigonometric interpolant through the unperturbed map.

    The two inverse branches remove odd Fourier modes and halve even modes'
    frequencies. The Nyquist cosine has half the coefficient weight of an
    interior conjugate pair when it moves into the represented spectrum.
    """
    size = values.shape[0]
    if size < 16 or size & (size - 1):
        raise ValueError("Use a power-of-two grid with at least sixteen points")
    coefficients = np.fft.rfft(values, axis=0)
    coefficients[-1] *= 0.5
    moved = np.zeros_like(coefficients)
    even = coefficients[::2]
    moved[: even.shape[0]] = even
    return np.fft.irfft(moved, n=size, axis=0)


def expanding_density_response(size=128, harmonics=(1, 2, 4), observables=(1, 2, 4)):
    """Solve the normalized density response and its independent adjoint system."""
    harmonics, observables = np.asarray(harmonics), np.asarray(observables)
    if np.any(harmonics <= 0) or np.any(harmonics >= size // 2):
        raise ValueError("Forcing frequencies must lie inside the resolved spectrum")
    x = np.arange(size) / size
    inverse_branches = (x[:, None] + np.arange(2)) / 2
    derivative = harmonics * np.cos(2 * np.pi * inverse_branches[..., None] * harmonics)
    source = -derivative.mean(axis=1) / 2
    source_mean = source.mean(axis=0)
    if np.max(abs(source_mean)) > 1e-12:
        raise ValueError("The density forcing has unresolved total mass")
    source -= source_mean
    operator = np.eye(size) - doubling_transfer(np.eye(size)) + np.ones((size, size)) / size
    density = np.linalg.solve(operator, source)
    observations = np.cos(2 * np.pi * x[:, None] * observables)
    adjoint = np.linalg.solve(operator.T, observations / size)
    return dict(
        gradient=observations.T @ density / size,
        adjoint_gradient=adjoint.T @ source,
        density=density,
        source=source,
        residual=float(np.max(abs(operator @ density - source))),
        adjoint_residual=float(np.max(abs(operator.T @ adjoint - observations / size))),
        mass_error=float(np.max(abs(density.mean(axis=0)))),
        source_mass_error=float(np.max(abs(source_mean))),
        size=size,
    )


def expanding_stationary_observations(parameter, harmonic, size=128, observables=(1, 2, 4)):
    """Find a perturbed invariant density using both nonlinear inverse branches."""
    if 2 - abs(parameter) * harmonic <= 1:
        raise ValueError("This benchmark requires uniform expansion greater than one")
    grid = np.arange(size) / size
    target = grid[:, None] + np.arange(2)
    inverse = target / 2
    for _ in range(12):
        angle = 2 * np.pi * harmonic * inverse
        slope = 2 + parameter * harmonic * np.cos(angle)
        inverse -= (2 * inverse + parameter * np.sin(angle) / (2 * np.pi) - target) / slope
    slope = 2 + parameter * harmonic * np.cos(2 * np.pi * harmonic * inverse)
    frequency = np.fft.fftfreq(size) * size
    interpolant = np.exp(2j * np.pi * inverse[..., None] * frequency)
    basis = np.exp(-2j * np.pi * frequency[:, None] * grid) / size
    transfer = (np.sum(interpolant / slope[..., None], axis=1) @ basis).real
    density = np.linalg.solve(np.eye(size) - transfer + np.ones((size, size)) / size, np.ones(size))
    return dict(
        observations=np.cos(2 * np.pi * grid[:, None] * observables).T @ density / size,
        residual=float(np.max(abs(transfer @ density - density))),
        mass_error=float(abs(density.mean() - 1)),
        minimum_density=float(density.min()),
        inverse_residual=float(
            np.max(abs(2 * inverse + parameter * np.sin(2 * np.pi * harmonic * inverse) / (2 * np.pi) - target))
        ),
    )


def expanding_shadowing_response(length=16384, harmonics=(1, 2, 4), observables=(1, 2, 4), seed=12):
    """Compare full-subspace tangent and adjoint least-squares contributions."""
    from scipy.linalg import solve_banded
    from .feasibility import segmented_shadowing

    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length + 54)
    orbit = np.convolve(bits, 2.0 ** -np.arange(53, 0, -1), mode="valid")[: length + 1]
    forcing = np.sin(2 * np.pi * orbit[:-1, None] * harmonics) / (2 * np.pi)
    band = np.zeros((3, length))
    band[1] = 5
    band[0, 1:] = band[2, :-1] = -2
    dual = solve_banded((1, 1), band, forcing)
    tangent = np.zeros((length + 1, len(harmonics)))
    tangent[:-1] -= 2 * dual
    tangent[1:] += dual
    frequencies = np.asarray(observables)
    seeds = -2 * np.pi * frequencies * np.sin(2 * np.pi * orbit[:, None] * frequencies) / (length + 1)
    adjoint = solve_banded((1, 1), band, -2 * seeds[:-1] + seeds[1:])
    segmented = []
    for size in (4, 8, 16):
        values = np.column_stack([segmented_shadowing(column, size) for column in forcing.T])
        segmented.append(dict(segment_length=size, maximum_difference=float(np.max(abs(values - tangent)))))
    return dict(
        gradient=seeds.T @ tangent,
        adjoint_gradient=adjoint.T @ forcing,
        continuity_residual=float(np.max(abs(tangent[1:] - 2 * tangent[:-1] - forcing))),
        segmented=segmented,
        length=length,
    )
