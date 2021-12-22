import sys

import numpy as np
from mpi4py import MPI

from veros import runtime_settings as rs, runtime_state as rst
from veros.distributed import gather

rs.linear_solver = "scipy"
rs.diskless_mode = True

if rst.proc_num > 1:
    rs.num_proc = (2, 2)
    assert rst.proc_num == 4


from veros.setups.acc import ACCSetup  # noqa: E402

sim = ACCSetup(
    override=dict(
        runlen=86400 * 10,
    )
)

if rst.proc_num == 1:
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=["-m", "mpi4py", sys.argv[-1]], maxprocs=4)

    try:
        sim.setup()
        sim.run()
    except Exception as exc:
        print(str(exc))
        comm.Abort(1)
        raise

    other_psi = np.empty_like(sim.state.variables.psi)
    comm.Recv(other_psi, 0)

    np.testing.assert_allclose(sim.state.variables.psi, other_psi)
else:
    sim.setup()
    sim.run()

    psi_global = gather(sim.state.variables.psi, sim.state.dimensions, ("xt", "yt"))

    if rst.proc_rank == 0:
        rs.mpi_comm.Get_parent().Send(np.array(psi_global), 0)
