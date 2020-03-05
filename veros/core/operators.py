from veros import runtime_settings


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


def update_multiply_jax(arr, at, to):
    import jax
    return jax.ops.index_update(arr, at, arr[at] * to)


if runtime_settings.backend == 'numpy':
    update = update_numpy
    update_add = update_add_numpy
    update_multiply = update_multiply_numpy
    at = Index()

elif runtime_settings.backend == 'jax':
    import jax
    update = jax.ops.index_update
    update_add = jax.ops.index_add
    update_multiply = update_multiply_jax
    at = jax.ops.index

else:
    raise ValueError()
