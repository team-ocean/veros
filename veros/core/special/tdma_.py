# defensive imports since extensions are optional
try:
    from veros.core.special import tdma_cython_
except ImportError:
    HAS_CPU_EXT = False
else:
    HAS_CPU_EXT = True

try:
    from veros.core.special import tdma_cuda_
except ImportError:
    HAS_GPU_EXT = False
else:
    HAS_GPU_EXT = True

import numpy as np

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.core import ShapedArray
from jax.interpreters import xla, mlir
import jaxlib.mlir.ir as ir
from jaxlib.mlir.dialects import mhlo

from jax.interpreters.mlir import custom_call


if HAS_CPU_EXT:
    for kernel_name in ("tdma_cython_double", "tdma_cython_float"):
        fn = tdma_cython_.cpu_custom_call_targets[kernel_name]
        jax.ffi.register_ffi_target(kernel_name, fn, platform="cpu", api_version=0)

if HAS_GPU_EXT:
    for kernel_name in ("tdma_cuda_double", "tdma_cuda_float"):
        fn = tdma_cuda_.gpu_custom_call_targets[kernel_name]
        jax.ffi.register_ffi_target(kernel_name, fn, platform="CUDA", api_version=0)


def as_mhlo_constant(val, dtype):
    if isinstance(val, mhlo.ConstantOp):
        return val

    return mhlo.ConstantOp(
        ir.DenseElementsAttr.get(np.array([val], dtype=dtype), type=mlir.dtype_to_ir_type(np.dtype(dtype)))
    ).result


def tdma(a, b, c, d, interior_mask, edge_mask, device=None):
    if device is None:
        device = jax.default_backend()

    if not a.shape == b.shape == c.shape == d.shape:
        raise ValueError("all inputs must have identical shape")

    if not a.dtype == b.dtype == c.dtype == d.dtype:
        raise ValueError("all inputs must have the same dtype")

    if device == "cpu":
        system_depths = jnp.sum(interior_mask, axis=-1, dtype="int32")
        return tdma_p.bind(a, b, c, d, system_depths)

    a = interior_mask * a * jnp.logical_not(edge_mask)
    b = jnp.where(interior_mask, b, 1.0)
    c = interior_mask * c
    d = interior_mask * d

    return tdma_p.bind(a, b, c, d, system_depths=None)


def tdma_impl(*args, **kwargs):
    return xla.apply_primitive(tdma_p, *args, **kwargs)


def tdma_xla_encode_cpu(ctx, a, b, c, d, system_depths):
    # try import again to trigger exception on ImportError
    from veros.core.special import tdma_cython_  # noqa: F401

    x_aval, *_ = ctx.avals_in
    np_dtype = x_aval.dtype

    x_type = ir.RankedTensorType(a.type)
    dtype = x_type.element_type
    dims = x_type.shape

    supported_dtypes = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )

    if np_dtype not in supported_dtypes:
        raise TypeError(f"TDMA only supports {supported_dtypes} arrays, got: {dtype}")

    # compute number of elements to vectorize over
    num_systems = 1
    for el in dims[:-1]:
        num_systems *= el

    stride = dims[-1]

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        ir.RankedTensorType.get((stride,), dtype),
    ]

    if np_dtype is np.dtype(np.float32):
        kernel = b"tdma_cython_float"
    elif np_dtype is np.dtype(np.float64):
        kernel = b"tdma_cython_double"
    else:
        raise RuntimeError("got unrecognized dtype")

    out = custom_call(
        kernel,
        operands=(
            a,
            b,
            c,
            d,
            system_depths,
            as_mhlo_constant(num_systems, np.int64),
            as_mhlo_constant(stride, np.int64),
        ),
        result_types=out_types,
    )
    return out.results[:-1]


def tdma_xla_encode_gpu(ctx, a, b, c, d, system_depths):
    # try import again to trigger exception on ImportError
    from veros.core.special import tdma_cuda_  # noqa: F401

    if system_depths is not None:
        raise ValueError("TDMA does not support system_depths argument on GPU")

    x_aval, *_ = ctx.avals_in
    np_dtype = x_aval.dtype

    x_type = ir.RankedTensorType(a.type)
    dtype = x_type.element_type
    dims = x_type.shape

    supported_dtypes = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )

    if np_dtype not in supported_dtypes:
        raise TypeError(f"TDMA only supports {supported_dtypes} arrays, got: {dtype}")

    # compute number of elements to vectorize over
    num_systems = 1
    for el in dims[:-1]:
        num_systems *= el

    system_depth = dims[-1]

    if np_dtype is np.dtype(np.float32):
        kernel = b"tdma_cuda_float"
    elif np_dtype is np.dtype(np.float64):
        kernel = b"tdma_cuda_double"
    else:
        raise RuntimeError("got unrecognized dtype")

    descriptor = tdma_cuda_.build_tridiag_descriptor(num_systems, system_depth)

    ndims = len(dims)
    arr_layout = tuple(range(ndims - 2, -1, -1)) + (ndims - 1,)

    out_types = [ir.RankedTensorType.get(dims, dtype), ir.RankedTensorType.get(dims, dtype)]
    out_layouts = (arr_layout, arr_layout)

    out = custom_call(
        kernel,
        operands=(a, b, c, d),
        result_types=out_types,
        result_layouts=out_layouts,
        operand_layouts=(arr_layout,) * 4,
        backend_config=descriptor,
    )
    return out.results[:-1]


def tdma_abstract_eval(a, b, c, d, system_depths):
    return ShapedArray(a.shape, a.dtype)


tdma_p = Primitive("tdma")
tdma_p.def_impl(tdma_impl)
tdma_p.def_abstract_eval(tdma_abstract_eval)

mlir.register_lowering(tdma_p, tdma_xla_encode_cpu, platform="cpu")
mlir.register_lowering(tdma_p, tdma_xla_encode_gpu, platform="cuda")
