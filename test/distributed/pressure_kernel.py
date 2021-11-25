import sys

import numpy as onp
from mpi4py import MPI

from veros import runtime_settings as rs, runtime_state as rst

rs.diskless_mode = True

if rst.proc_num > 1:
    rs.num_proc = (2, 2)
    assert rst.proc_num == 4

from veros.state import get_default_state, resize_dimension  # noqa: E402
from veros.distributed import gather  # noqa: E402
from veros.core.operators import numpy as npx, update, at  # noqa: E402
from veros.core.external.solvers import get_linear_solver  # noqa: E402


def get_inputs():
    state = get_default_state()
    settings = state.settings

    with settings.unlock():
        settings.nx = 100
        settings.ny = 40
        settings.nz = 1

        settings.dt_tracer = 1800
        settings.dt_mom = 1800

        settings.enable_cyclic_x = True
        settings.enable_streamfunction = False
        settings.enable_free_surface = True

    state.initialize_variables()
    resize_dimension(state, "isle", 1)

    vs = state.variables

    nx_local, ny_local = settings.nx // rs.num_proc[0], settings.ny // rs.num_proc[1]
    idx_global = (
        slice(rst.proc_idx[0] * nx_local, (rst.proc_idx[0] + 1) * nx_local + 4),
        slice(rst.proc_idx[1] * ny_local, (rst.proc_idx[1] + 1) * ny_local + 4),
        Ellipsis,
    )

    with vs.unlock():
        vs.dxt = update(vs.dxt, at[...], 10e3)
        vs.dxu = update(vs.dxu, at[...], 10e3)

        vs.dyt = update(vs.dyt, at[...], 10e3)
        vs.dyu = update(vs.dyu, at[...], 10e3)

        h_global = npx.linspace(500, 2000, settings.nx + 4)[:, None] * npx.ones((settings.nx + 4, settings.ny + 4))
        vs.hu = h_global[idx_global]
        vs.hv = h_global[idx_global]

        vs.cosu = update(vs.cosu, at[...], 1)
        vs.cost = update(vs.cost, at[...], 1)

        boundary_mask = npx.zeros((settings.nx + 4, settings.ny + 4, settings.nz), dtype="bool")
        boundary_mask = update(boundary_mask, at[:50, :2], 1)
        boundary_mask = update(boundary_mask, at[20:30, 20:30], 1)
        vs.maskT = ~boundary_mask[idx_global]

    rhs = npx.ones_like(vs.hur)
    x0 = npx.ones_like(vs.hur)
    return state, rhs, x0


if rst.proc_num == 1:
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=["-m", "mpi4py", sys.argv[-1]], maxprocs=4)

    try:
        state, rhs, x0 = get_inputs()
        sol = get_linear_solver(state)
        psi = sol.solve(state, rhs, x0)
    except Exception as exc:
        print(str(exc))
        comm.Abort(1)
        raise

    other_psi = onp.empty_like(psi)
    comm.Recv(other_psi, 0)

    onp.testing.assert_allclose(psi, other_psi)
else:
    state, rhs, x0 = get_inputs()
    sol = get_linear_solver(state)
    psi = sol.solve(state, rhs, x0)

    psi_global = gather(psi, state.dimensions, ("xt", "yt"))

    if rst.proc_rank == 0:
        rs.mpi_comm.Get_parent().Send(onp.array(psi_global), 0)
