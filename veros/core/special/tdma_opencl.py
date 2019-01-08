from string import Template
import functools
import operator


KERNEL = Template('''
kernel void execute(
    const global $DTYPE *a,
    const global $DTYPE *b,
    const global $DTYPE *c,
    const global $DTYPE *d,
    global $DTYPE *solution
){
    const size_t m = $SYS_DEPTH;
    const size_t idx = get_global_id(0) * m;

    private $DTYPE cp[$SYS_DEPTH];

    cp[0] = c[idx] / b[idx];
    solution[idx] = d[idx] / b[idx];

    for (size_t j = 1; j < m; ++j) {
        const $DTYPE norm_factor = b[idx+j] - a[idx+j] * cp[j-1];
        cp[j] = c[idx+j] / norm_factor;
        solution[idx+j] = (d[idx+j] - a[idx+j] * solution[idx+j-1]) / norm_factor;
    }

    for (size_t j = m-1; j > 0; --j) {
        solution[idx+j-1] -= cp[j-1] * solution[idx+j];
    }
}
'''.strip())


def tdma(a, b, c, d, workgrp_size=None):
    # import pyopencl as cl
    import bohrium as bh

    assert a.shape == b.shape == c.shape == d.shape
    assert a.dtype == b.dtype == c.dtype == d.dtype

    # Check that PyOpenCL is installed and that the Bohrium runtime uses the OpenCL backend
    # if not bh.interop_pyopencl.available():
    #     raise NotImplementedError("OpenCL not available")

    # Get the OpenCL context from Bohrium
    # ctx = bh.interop_pyopencl.get_context()
    # queue = cl.CommandQueue(ctx)

    res = bh.empty_like(a)
    # a_buf, b_buf, c_buf, d_buf, ret_buf = map(bh.interop_pyopencl.get_buffer, (a, b, c, d, res))

    cltype = bh.interop_pyopencl.type_np2opencl_str(a.dtype)
    kernel = KERNEL.substitute(SYS_DEPTH=res.shape[-1], DTYPE=cltype)
    # prg = cl.Program(ctx, kernel).build()

    global_size = res.size // res.shape[-1]
    # prg.execute(queue, [global_size], workgrp_size, a_buf, b_buf, c_buf, d_buf, ret_buf)
    bh.user_kernel.execute(kernel, [a, b, c, d, res],
                           tag="opencl",
                           param="global_work_size: %d; local_work_size: 1" % global_size)
    return res
