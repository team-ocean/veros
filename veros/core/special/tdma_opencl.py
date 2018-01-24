try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache
import logging
import functools
import operator

import pyopencl as cl


@lru_cache()
def compile_tdma(sys_depth, dtype):
    logging.debug("Re-compiling TDMA kernel")

    import bohrium as bh
    ctx = bh.interop_pyopencl.get_context()
    source = """
        kernel void tdma(
            global DTYPE *a,
            global DTYPE *b,
            global DTYPE *c,
            global DTYPE *d,
            global DTYPE *solution
        ){
            const int m = SYS_DEPTH;
            const int idx = get_global_id(0) * m;

            private DTYPE cp[SYS_DEPTH];
            private DTYPE dp[SYS_DEPTH];
            cp[0] = c[idx] / b[idx];
            dp[0] = d[idx] / b[idx];
            for (int j=1; j < m; ++j) {
                cp[j] = c[idx + j] / (b[idx + j] - a[idx + j] * cp[j-1]);
                dp[j] = (d[idx + j] - a[idx + j] * dp[j-1]) / (b[idx + j] - a[idx + j] * cp[j-1]);
            }

            solution[idx + m-1] = dp[m-1];

            for (int j=m-2; j >= 0; --j) {
                solution[idx + j] = dp[j] - cp[j] * solution[idx + j+1];
            }
        }
    """
    source = source.replace("SYS_DEPTH", "%d" % sys_depth)
    source = source.replace("DTYPE", dtype)
    prg = cl.Program(ctx, source).build()
    return prg


def tdma(a, b, c, d, workgrp_size=None):
    import bohrium as bh

    assert a.shape == b.shape == c.shape == d.shape
    assert a.dtype == b.dtype == c.dtype == d.dtype

    # Check that PyOpenCL is installed and that the Bohrium runtime uses the OpenCL backend
    if not bh.interop_pyopencl.available():
        raise NotImplementedError("OpenCL not available")

    # Get the OpenCL context from Bohrium
    ctx = bh.interop_pyopencl.get_context()
    queue = cl.CommandQueue(ctx)

    ret = bh.empty(a.shape, dtype=a.dtype)
    a_buf, b_buf, c_buf, d_buf, ret_buf = map(bh.interop_pyopencl.get_buffer, (a, b, c, d, ret))

    prg = compile_tdma(ret.shape[-1], bh.interop_pyopencl.type_np2opencl_str(a.dtype))
    global_size = functools.reduce(operator.mul, ret.shape[:-1])
    prg.tdma(queue, [global_size], workgrp_size, a_buf, b_buf, c_buf, d_buf, ret_buf)
    return ret
