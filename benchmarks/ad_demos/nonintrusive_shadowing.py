"""Segmented NILSS/NILSAS prototypes for discrete maps without neutral modes.

These research helpers compute the shadowing contribution. They do not supply
the unstable contribution to the derivative of a physical invariant measure.
The adjoint recurrence follows Ni and Talnikar (2019), Appendix B:
https://arxiv.org/html/1801.08674#A2
"""

import numpy as np


def segmented_minimum_norm(product, forcing, basis, segment_length):
    """Minimize a propagated affine solution over homogeneous segment bases.

    ``product(step, vectors)`` applies a tangent or backward-adjoint map to columns.
    The caller supplies a basis spanning its growing subspace. Stable components
    start at zero, retaining finite-window boundary transients. Maps with neutral
    directions and flows requiring time dilation are unsupported.
    Storage retains every local solution.
    """
    from scipy.sparse import block_diag, bmat, coo_matrix
    from scipy.sparse.linalg import spsolve

    forcing = np.asarray(forcing, dtype=float)
    basis = np.asarray(basis, dtype=float)
    length, dimension, directions = forcing.shape
    modes = basis.shape[1]
    if basis.shape[0] != dimension or not 0 < modes <= dimension:
        raise ValueError("Supply a nonempty growing-subspace basis")
    if segment_length <= 0 or length % segment_length:
        raise ValueError("Use complete positive-length segments")
    q, _ = np.linalg.qr(basis, mode="reduced")
    particular = np.zeros((dimension, directions))
    local, costs, linear, rotations, offsets = [], [], [], [], []
    for start in range(0, length, segment_length):
        w, v = [q], [particular]
        for step in range(start, start + segment_length):
            w.append(np.asarray(product(step, w[-1])))
            v.append(np.asarray(product(step, v[-1])) + forcing[step])
        w, v = np.asarray(w), np.asarray(v)
        # Count each time index once, adding the final endpoint separately.
        costs.append(np.einsum("tdm,tdn->mn", w[:-1], w[:-1]))
        linear.append(np.einsum("tdm,tdp->mp", w[:-1], v[:-1]))
        local.append((w[:-1], v[:-1]))
        q, r = np.linalg.qr(w[-1], mode="reduced")
        offset = q.T @ v[-1]
        particular = v[-1] - q @ offset
        rotations.append(r)
        offsets.append(offset)
    costs.append(q.T @ q)
    linear.append(q.T @ particular)
    segments = len(local)
    rows, columns, data = [], [], []
    for index, rotation in enumerate(rotations):
        for row in range(modes):
            rows.append(index * modes + row)
            columns.append((index + 1) * modes + row)
            data.append(1.0)
            for column in range(modes):
                rows.append(index * modes + row)
                columns.append(index * modes + column)
                data.append(-rotation[row, column])
    continuity = coo_matrix((data, (rows, columns)), shape=(segments * modes, (segments + 1) * modes)).tocsc()
    cost = block_diag([coo_matrix(item) for item in costs], format="csc")
    system = bmat([[cost, continuity.T], [continuity, None]], format="csc")
    right = np.concatenate((-np.concatenate(linear), np.concatenate(offsets)))
    solution = spsolve(system, right)
    if solution.ndim == 1:
        solution = solution[:, None]
    coefficients = solution[: (segments + 1) * modes].reshape(segments + 1, modes, directions)
    values = np.concatenate(
        [v + w @ coefficients[index] for index, (w, v) in enumerate(local)]
        + [(particular + q @ coefficients[-1])[None]]
    )
    residual = max(
        np.max(abs(values[step + 1] - product(step, values[step]) - forcing[step])) for step in range(length)
    )
    return dict(
        values=values,
        recurrence_residual=float(residual),
        reduced_system_residual=float(np.max(abs(system @ solution - right))),
        qr_log_growth=np.mean(np.log(abs(np.diagonal(rotations, axis1=1, axis2=2))), axis=0) / segment_length,
        homogeneous_modes=modes,
        state_dimension=dimension,
        reduced_unknowns=system.shape[0],
        stored_local_bytes=sum(w.nbytes + v.nbytes for w, v in local),
    )


def nilss(jvp, parameter_forcing, unstable_basis, segment_length):
    """Forward segmented nonintrusive least-squares shadowing for a map."""
    return segmented_minimum_norm(jvp, parameter_forcing, unstable_basis, segment_length)


def nilsas(vjp, observable_gradient, terminal_unstable_basis, segment_length):
    """Backward segmented adjoint shadowing, independently of parameter count.

    The input gradients are unnormalized instantaneous objective derivatives at
    indices 0 through N-1. Returned covectors satisfy w[n]=A[n].T w[n+1]+g[n].
    Contract w[1:] with the parameter forcing and divide by N, then add any
    direct objective-parameter derivative.
    """
    length = len(observable_gradient)
    result = segmented_minimum_norm(
        lambda step, vectors: vjp(length - 1 - step, vectors),
        np.asarray(observable_gradient)[::-1],
        terminal_unstable_basis,
        segment_length,
    )
    result["values"] = result["values"][::-1]
    return result
