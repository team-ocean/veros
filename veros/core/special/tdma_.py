# defensive imports since extensions are optional
try:
    from veros.core.special import tdma_cython_
except ImportError:
    HAS_CPU_EXT = False
else:
    HAS_CPU_EXT = hasattr(tdma_cython_, "cpu_ffi_targets")

try:
    from veros.core.special import tdma_cuda_
except ImportError:
    HAS_GPU_EXT = False
else:
    HAS_GPU_EXT = hasattr(tdma_cuda_, "gpu_ffi_targets")

import numpy as np

import jax
import jax.numpy as jnp


if HAS_CPU_EXT:
    for name, target in tdma_cython_.cpu_ffi_targets.items():
        jax.ffi.register_ffi_target(name, target, platform="cpu")

if HAS_GPU_EXT:
    for name, target in tdma_cuda_.gpu_ffi_targets.items():
        jax.ffi.register_ffi_target(name, target, platform="CUDA")


def tdma(a, b, c, d, interior_mask, edge_mask, device=None):
    if device is None:
        device = jax.default_backend()

    if not a.shape == b.shape == c.shape == d.shape:
        raise ValueError("all inputs must have identical shape")

    if not a.dtype == b.dtype == c.dtype == d.dtype:
        raise ValueError("all inputs must have the same dtype")

    if a.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"TDMA only supports float32 and float64 arrays, got: {a.dtype}")
    if not a.ndim or a.shape[-1] == 0:
        raise ValueError("TDMA requires a nonempty depth dimension")
    if interior_mask.shape != a.shape or edge_mask.shape != a.shape:
        raise ValueError("TDMA masks must have the same shape as the coefficients")

    result = jax.ShapeDtypeStruct(a.shape, a.dtype)
    precision = "double" if a.dtype == np.dtype("float64") else "float"
    if device == "cpu":
        if not HAS_CPU_EXT:
            raise ImportError("The CPU TDMA extension is unavailable; rebuild the native extensions")
        system_depths = jnp.sum(interior_mask, axis=-1, dtype="int32")
        workspace = jax.ShapeDtypeStruct((a.shape[-1],), a.dtype)
        return jax.ffi.ffi_call(f"tdma_cython_{precision}", (result, workspace), vmap_method="broadcast_all")(
            a, b, c, d, system_depths
        )[0]
    if device not in ("gpu", "cuda"):
        raise ValueError(f"TDMA does not support device {device!r}")
    if not HAS_GPU_EXT:
        raise ImportError("The CUDA TDMA extension is unavailable; rebuild the native extensions")

    a = interior_mask * a * jnp.logical_not(edge_mask)
    b = jnp.where(interior_mask, b, 1.0)
    c = interior_mask * c
    d = interior_mask * d

    # Depth is the leading storage dimension across all column systems.
    layout = (a.ndim - 1, *range(a.ndim - 1))
    return jax.ffi.ffi_call(
        f"tdma_cuda_{precision}",
        (result, result),
        input_layouts=(layout,) * 4,
        output_layouts=(layout,) * 2,
        vmap_method="sequential",
    )(a, b, c, d)[0]
