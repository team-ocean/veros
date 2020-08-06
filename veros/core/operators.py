from veros import runtime_settings, veros_kernel


class Index:
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


def solve_tridiagonal_numpy(a, b, c, d):
    from scipy.linalg import lapack
    # remove couplings between slices
    a[..., 0] = 0
    c[..., -1] = 0
    return lapack.dgtsv(a.flatten()[1:], b.flatten(), c.flatten()[:-1], d.flatten())[3].reshape(a.shape)


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
def solve_tridiagonal_jax(a, b, c, d):
    import jax.numpy as np
    import jax.lax

    def compute_primes(last_primes, x):
        last_cp, last_dp = last_primes
        a, b, c, d = x
        cp = c / (b - a * last_cp)
        dp = (d - a * last_dp) / (b - a * last_cp)
        new_primes = np.stack((cp, dp))
        return new_primes, new_primes

    diags_stacked = np.stack(
        [arr.transpose((2, 0, 1)) for arr in (a, b, c, d)],
        axis=1
    )
    _, primes = jax.lax.scan(compute_primes, np.zeros((2, *a.shape[:-1])), diags_stacked)

    def backsubstitution(last_x, x):
        cp, dp = x
        new_x = dp - cp * last_x
        return new_x, new_x

    _, sol = jax.lax.scan(backsubstitution, np.zeros(a.shape[:-1]), primes[::-1])
    return sol[::-1].transpose((1, 2, 0))


def update_multiply_jax(arr, at, to):
    import jax
    return jax.ops.index_update(arr, at, arr[at] * to)


if runtime_settings.backend == 'numpy':
    import numpy
    update = update_numpy
    update_add = update_add_numpy
    update_multiply = update_multiply_numpy
    at = Index()
    solve_tridiagonal = solve_tridiagonal_numpy
    scan = scan_numpy

elif runtime_settings.backend == 'jax':
    import jax
    import jax.numpy
    numpy = jax.numpy
    update = jax.ops.index_update
    update_add = jax.ops.index_add
    update_multiply = update_multiply_jax
    at = jax.ops.index
    solve_tridiagonal = solve_tridiagonal_jax
    scan = jax.lax.scan

else:
    raise ValueError('Unrecognized backend {}'.format(runtime_settings.backend))
