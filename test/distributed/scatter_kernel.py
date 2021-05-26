import numpy as np
from mpi4py import MPI

from veros import runtime_settings as rs, runtime_state as rst
from veros.distributed import scatter

global_arr = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
    ]
)

if rst.proc_num == 1:
    import sys

    comm = MPI.COMM_SELF.Spawn(sys.executable, args=["-m", "mpi4py", sys.argv[-1]], maxprocs=4)

    res = np.empty((6, 6))

    proc_slices = (
        (slice(None, -2), slice(None, -2)),
        (slice(2, None), slice(None, -2)),
        (slice(None, -2), slice(2, None)),
        (slice(2, None), slice(2, None)),
    )

    for proc, idx in enumerate(proc_slices):
        comm.Recv(res, proc)
        np.testing.assert_array_equal(res, global_arr[idx])

else:
    rs.num_proc = (2, 2)
    assert rst.proc_num == 4

    from veros.core.operators import numpy as npx

    dimensions = dict(xt=4, yt=4)

    if rst.proc_rank == 0:
        a = npx.array(global_arr)
    else:
        a = npx.empty((6, 6))

    b = scatter(a, dimensions, ("xt", "yt"))

    rs.mpi_comm.Get_parent().Send(np.array(b), 0)
