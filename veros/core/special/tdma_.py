try:
    from . import tdma_cython_
except ImportError:
    HAS_CPU_TDMA = False
else:
    HAS_CPU_TDMA = True

try:
    from . import tdma_cuda_
except ImportError:
    HAS_GPU_TDMA = False
else:
    HAS_GPU_TDMA = True

import numpy as np
import jax.numpy as jnp

from jax import abstract_arrays
from jax.core import Primitive
from jax.lib import xla_client
from jax.interpreters import xla

_ops = xla_client.ops

if HAS_CPU_TDMA:
    for kernel_name in (b'tdma_cython_double', b'tdma_cython_float'):
        fn = tdma_cython_.cpu_custom_call_targets[kernel_name]
        xla_client.register_cpu_custom_call_target(kernel_name, fn)

if HAS_GPU_TDMA:
    for kernel_name in (b'tdma_cuda_double', b'tdma_cuda_float'):
        fn = tdma_cuda_.gpu_custom_call_targets[kernel_name]
        xla_client.register_gpu_custom_call_target(kernel_name, fn)


def _constant_s64_scalar(c, x):
    return _ops.Constant(c, np.int64(x))


def tdma(a, b, c, d, system_depths=None):
    if not a.shape == b.shape == c.shape == d.shape:
        raise ValueError('all inputs must have identical shape')

    if not a.dtype == b.dtype == c.dtype == d.dtype:
        raise ValueError('all inputs must have the same dtype')

    # pre-allocate workspace for TDMA
    cp = jnp.empty(a.shape[-1], dtype=a.dtype)

    if system_depths is None:
        system_depths = jnp.full(a.shape[:-1], a.shape[-1], dtype='int64')
    else:
        system_depths = system_depths.astype('int64')

    return tdma_p.bind(a, b, c, d, cp, system_depths)


def tdma_impl(*args, **kwargs):
    return xla.apply_primitive(tdma_p, *args, **kwargs)


def tdma_xla_encode_cpu(builder, a, b, c, d, cp, system_depths):
    x_shape = builder.GetShape(a)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    supported_dtypes = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )

    if dtype not in supported_dtypes:
        raise TypeError('TDMA only supports {} arrays, got: {}'.format(supported_dtypes, dtype))

    # compute number of elements to vectorize over
    num_systems = 1
    for el in dims[:-1]:
        num_systems *= el

    stride = dims[-1]

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

    return _ops.CustomCall(
        builder,
        kernel,
        operands=(
            a, b, c, d,
            system_depths,
            _constant_s64_scalar(builder, num_systems),
            _constant_s64_scalar(builder, stride),
            cp,
        ),
        shape=out_shape,
    )


def tdma_xla_encode_gpu(builder, a, b, c, d, cp, system_depths):
    if not HAS_GPU_TDMA:
        raise RuntimeError("GPU extensions could not be imported")

    a_shape = builder.get_shape(a)
    dtype = a_shape.element_type()
    dims = a_shape.dimensions()

    supported_dtypes = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )

    if dtype not in supported_dtypes:
        raise TypeError('TDMA only supports {} arrays, got: {}'.format(supported_dtypes, dtype))

    # compute number of elements to vectorize over
    num_systems = 1
    for el in dims[:-1]:
        num_systems *= el
    system_depth = dims[-1]

    total_size = num_systems * system_depth

    if dtype is np.dtype(np.float32):
        kernel = b'cuda_tridiag_float'
    elif dtype is np.dtype(np.float64):
        kernel = b'cuda_tridiag_double'
    else:
        raise RuntimeError('got unrecognized dtype')

    opaque = tdma_cuda_.build_tridiag_descriptor(total_size, num_systems, system_depth)

    shape = xla_client.Shape.array_shape(dtype, dims, (1, 0, 2)) # transpose here for coalesced access!

    return _ops.CustomCallWithLayout(
        builder,
        kernel,
        operands=(a, b, c, d),
        shape_with_layout=shape,
        operand_shapes_with_layout=(shape,) * 4,
        opaque=opaque
    )


def tdma_abstract_eval(a, b, c, d, cp, system_depths):
    return abstract_arrays.ShapedArray(a.shape, a.dtype)


tdma_p = Primitive('tdma')
tdma_p.def_impl(tdma_impl)
tdma_p.def_abstract_eval(tdma_abstract_eval)

xla.backend_specific_translations['cpu'][tdma_p] = tdma_xla_encode_cpu
xla.backend_specific_translations['gpu'][tdma_p] = tdma_xla_encode_gpu
