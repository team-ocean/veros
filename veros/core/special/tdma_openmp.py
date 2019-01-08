from string import Template


KERNEL = Template('''
#include <stddef.h>

void execute(
    const $DTYPE *a,
    const $DTYPE *b,
    const $DTYPE *c,
    const $DTYPE *d,
    $DTYPE *solution
){
    const size_t m = $SYS_DEPTH;
    const size_t total_size = $SIZE;

    #pragma omp parallel for
    for(size_t idx = 0; idx < total_size; idx += m) {
        $DTYPE cp[m];
        
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
}
'''.strip())


def tdma(a, b, c, d):
    import bohrium as bh

    assert a.shape == b.shape == c.shape == d.shape
    assert a.dtype == b.dtype == c.dtype == d.dtype

    if not a.flags.owndata or not a.flags.c_contiguous:
        a, b, c, d = (bh.array(k, order='C', copy=True) for k in (a, b, c, d))

    ctype = bh.user_kernel.dtype_to_c99(a.dtype)
    kernel = KERNEL.substitute(DTYPE=ctype, SYS_DEPTH=a.shape[-1], SIZE=a.size)
    res = bh.empty_like(a)
    bh.user_kernel.execute(kernel, [a, b, c, d, res])
    return res
