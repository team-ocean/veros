import warnings

from veros import runtime_settings, runtime_state, veros_kernel


class Index:
    __slots__ = ()

    @staticmethod
    def __getitem__(key):
        return key


def update_numpy(arr, at, to):
    arr[at] = to
    return arr


def update_add_numpy(arr, at, to):
    arr[at] += to
    return arr


def update_multiply_numpy(arr, at, to):
    arr[at] *= to
    return arr


def solve_tridiagonal_numpy(a, b, c, d, water_mask, edge_mask):
    import numpy as np
    from scipy.linalg import lapack

    # remove couplings between slices
    a[edge_mask] = 0
    c[..., -1] = 0

    out = np.full(a.shape, np.nan, dtype=a.dtype)
    sol = lapack.dgtsv(a[water_mask][1:], b[water_mask], c[water_mask][:-1], d[water_mask])[3]
    out[water_mask] = sol
    return out


def scan_numpy(f, init, xs, length=None):
    import numpy as np
    if xs is None:
        xs = [None] * length

    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)

    return carry, np.stack(ys)


@veros_kernel
def solve_tridiagonal_jax(a, b, c, d, water_mask, edge_mask):
    import jax.lax
    import jax.numpy as jnp

    try:
        from veros.core.special.tdma import tdma
    except ImportError:
        has_tdma_special = False
    else:
        has_tdma_special = runtime_settings.device == 'cpu'

    if has_tdma_special:
        system_depths = jnp.sum(water_mask, axis=2).astype('int64')
        return tdma(a, b, c, d, system_depths)

    warnings.warn('Could not use custom TDMA implementation, falling back to pure JAX')
    # TODO: fix / test

    a = water_mask * a * jnp.logical_not(edge_mask)
    b = jnp.where(water_mask, b, 1.)
    c = water_mask * c
    d = water_mask * d

    def compute_primes(last_primes, x):
        last_cp, last_dp = last_primes
        a, b, c, d = x
        cp = c / (b - a * last_cp)
        dp = (d - a * last_dp) / (b - a * last_cp)
        new_primes = jnp.stack((cp, dp))
        return new_primes, new_primes

    diags_stacked = jnp.stack(
        [arr.transpose((2, 0, 1)) for arr in (a, b, c, d)],
        axis=1
    )
    _, primes = jax.lax.scan(compute_primes, jnp.zeros((2, *a.shape[:-1]), dtype=a.dtype), diags_stacked)

    def backsubstitution(last_x, x):
        cp, dp = x
        new_x = dp - cp * last_x
        return new_x, new_x

    _, sol = jax.lax.scan(backsubstitution, jnp.zeros(a.shape[:-1], dtype=a.dtype), primes, reverse=True)
    return sol.transpose((1, 2, 0))


def update_multiply_jax(arr, at, to):
    import jax
    return jax.ops.index_update(arr, at, arr[at] * to)


def tanh_jax(arr):
    import jax.numpy as jnp
    if runtime_settings.device != 'cpu':
        return jnp.tanh(arr)

    # https://math.stackexchange.com/questions/107292/rapid-approximation-of-tanhx
    # TODO: test this
    arr2 = arr * arr
    nom = arr * (135135. + arr2 * (17325. + arr2 * (378. + arr2)))
    denom = 135135. + arr2 * (62370. + arr2 * (3150. + arr2 * 28.))
    return jnp.clip(nom / denom, -1, 1)


numpy = runtime_state.backend_module

if runtime_settings.backend == 'numpy':
    update = update_numpy
    update_add = update_add_numpy
    update_multiply = update_multiply_numpy
    at = Index()
    solve_tridiagonal = solve_tridiagonal_numpy
    scan = scan_numpy
    tanh = numpy.tanh

elif runtime_settings.backend == 'jax':
    import jax.ops
    import jax.lax
    update = jax.ops.index_update
    update_add = jax.ops.index_add
    update_multiply = update_multiply_jax
    at = jax.ops.index
    solve_tridiagonal = solve_tridiagonal_jax
    scan = jax.lax.scan
    tanh = tanh_jax

else:
    raise ValueError('Unrecognized backend {}'.format(runtime_settings.backend))
