from . import tdma_cython

import numpy as np
import jax.numpy as jnp

from jax import abstract_arrays
from jax.core import Primitive
from jax.lib import xla_client
from jax.interpreters import xla

_ops = xla_client.ops

for kernel_name in (b'tdma_cython_double', b'tdma_cython_float'):
    fn = tdma_cython.cpu_custom_call_targets[kernel_name]
    xla_client.register_cpu_custom_call_target(kernel_name, fn)


def _constant_s64_scalar(c, x):
    return _ops.Constant(c, np.int64(x))


def _unpack_builder(c):
    return getattr(c, '_builder', c)


def tdma(a, b, c, d):
    if not a.shape == b.shape == c.shape == d.shape:
        raise ValueError('all inputs must have identical shape')

    if not a.dtype == b.dtype == c.dtype == d.dtype:
        raise ValueError('all inputs must have the same dtype')

    # pre-allocate workspace for TDMA
    cp = jnp.empty(a.shape[-1], dtype=a.dtype)
    return tdma_p.bind(a, b, c, d, cp)


def tdma_impl(*args, **kwargs):
    return xla.apply_primitive(tdma_p, *args, **kwargs)


def tdma_xla_encode(builder, a, b, c, d, cp):
    builder = _unpack_builder(builder)
    x_shape = builder.GetShape(a)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    supported_dtypes = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )

    if dtype not in supported_dtypes:
        raise TypeError('TDMA only supports {} arrays, got: {}'.format(supported_dtypes, dtype))

    # compute system size and depth
    num_systems = 1
    for el in dims[:-1]:
        num_systems *= el

    system_depth = dims[-1]

    out_shape = xla_client.Shape.array_shape(
        dtype,
        dims,
    )

    if dtype is np.dtype(np.float32):
        kernel = b'tdma_cython_float'
    elif dtype is np.dtype(np.float64):
        kernel = b'tdma_cython_double'
    else:
        raise RuntimeError('got unrecognized dtype')

    out = _ops.CustomCall(
        builder,
        kernel,
        operands=(
            a, b, c, d,
            _constant_s64_scalar(builder, system_depth),
            _constant_s64_scalar(builder, num_systems),
            cp,
        ),
        shape=out_shape,
    )

    return out


def tdma_abstract_eval(a, b, c, d, cp):
    return abstract_arrays.ShapedArray(a.shape, a.dtype)


tdma_p = Primitive('tdma')
tdma_p.def_impl(tdma_impl)
tdma_p.def_abstract_eval(tdma_abstract_eval)

xla.backend_specific_translations['cpu'][tdma_p] = tdma_xla_encode
