import numpy as np
from mpi4py import MPI

from veros import runtime_settings as rs, runtime_state as rst
from veros.distributed import gather

if rst.proc_num == 1:
    import sys

    comm = MPI.COMM_SELF.Spawn(sys.executable, args=["-m", "mpi4py", sys.argv[-1]], maxprocs=4)

    res = np.empty((8, 8))
    comm.Recv(res, 0)

    np.testing.assert_array_equal(
        res,
        np.array(
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
        ),
    )

else:
    rs.num_proc = (2, 2)
    assert rst.proc_num == 4

    from veros.core.operators import numpy as npx

    dimensions = dict(xt=4, yt=4)

    a = rst.proc_rank * npx.ones((6, 6))
    b = gather(a, dimensions, ("xt", "yt"))

    if rst.proc_rank == 0:
        rs.mpi_comm.Get_parent().Send(np.array(b), 0)
