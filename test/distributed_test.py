import sys
import subprocess

from tempfile import NamedTemporaryFile
from textwrap import dedent


def run_dist_kernel(code):
    with NamedTemporaryFile(prefix='vs_test_', suffix='.py', mode='w') as f:
        f.write(code)
        f.flush()

        return subprocess.check_call(
            [sys.executable, '-m', 'mpi4py', f.name], stderr=subprocess.STDOUT, timeout=10
        )


def test_gather():
    test_kernel = dedent('''
    from mpi4py import MPI

    from veros import runtime_settings as rs, runtime_state as rst, VerosState
    from veros.distributed import gather

    if rst.proc_num == 1:
        import sys
        comm = MPI.COMM_SELF.Spawn(
            sys.executable,
            args=['-m', 'mpi4py', sys.argv[-1]],
            maxprocs=4
        )

        from veros.core.operators import numpy as npx
        res = npx.empty((8, 8))
        comm.Recv(res, 0)

        assert npx.array_equal(res, npx.array(
                [[0., 0., 0., 0., 2., 2., 2., 2.],
                 [0., 0., 0., 0., 2., 2., 2., 2.],
                 [0., 0., 0., 0., 2., 2., 2., 2.],
                 [0., 0., 0., 0., 2., 2., 2., 2.],
                 [1., 1., 1., 1., 3., 3., 3., 3.],
                 [1., 1., 1., 1., 3., 3., 3., 3.],
                 [1., 1., 1., 1., 3., 3., 3., 3.],
                 [1., 1., 1., 1., 3., 3., 3., 3.]]
            ))

    else:
        rs.num_proc = (2, 2)
        assert rst.proc_num == 4

        from veros.core.operators import numpy as npx
        dimensions = dict(xt=4, yt=4)

        a = rst.proc_rank * npx.ones((6, 6))
        b = gather(a, dimensions, ('xt', 'yt'))

        if rst.proc_rank == 0:
            rs.mpi_comm.Get_parent().Send(b, 0)

    ''')

    run_dist_kernel(test_kernel)


def test_scatter():
    test_kernel = dedent('''
    import numpy as np
    from mpi4py import MPI

    from veros import runtime_settings as rs, runtime_state as rst, VerosState
    from veros.distributed import scatter

    global_arr = np.array(
        [[0., 0., 0., 0., 2., 2., 2., 2.],
         [0., 0., 0., 0., 2., 2., 2., 2.],
         [0., 0., 0., 0., 2., 2., 2., 2.],
         [0., 0., 0., 0., 2., 2., 2., 2.],
         [1., 1., 1., 1., 3., 3., 3., 3.],
         [1., 1., 1., 1., 3., 3., 3., 3.],
         [1., 1., 1., 1., 3., 3., 3., 3.],
         [1., 1., 1., 1., 3., 3., 3., 3.]]
    )

    if rst.proc_num == 1:
        import sys
        comm = MPI.COMM_SELF.Spawn(
            sys.executable,
            args=['-m', 'mpi4py', sys.argv[-1]],
            maxprocs=4
        )

        from veros.core.operators import numpy as npx
        res = npx.empty((6, 6))

        proc_slices = (
            (slice(None, -2), slice(None, -2)),
            (slice(2, None), slice(None, -2)),
            (slice(None, -2), slice(2, None)),
            (slice(2, None), slice(2, None)),
        )

        for proc, idx in enumerate(proc_slices):
            comm.Recv(res, proc)
            assert npx.array_equal(res, global_arr[idx])

    else:
        rs.num_proc = (2, 2)
        assert rst.proc_num == 4

        from veros.core.operators import numpy as npx

        dimensions = dict(xt=4, yt=4)

        if rst.proc_rank == 0:
            a = npx.array(global_arr)
        else:
            a = npx.empty((6, 6))

        b = scatter(a, dimensions, ('xt', 'yt'))

        rs.mpi_comm.Get_parent().Send(b, 0)

    ''')

    run_dist_kernel(test_kernel)


def test_acc():
    test_kernel = dedent('''
    from mpi4py import MPI

    from veros import runtime_settings as rs, runtime_state as rst
    rs.diskless_mode = True
    rs.linear_solver = "scipy"

    if rst.proc_num != 1:
        rs.num_proc = (2, 2)

    from veros.distributed import gather
    from veros.setups.acc import ACCSetup
    from veros.core.operators import numpy as npx

    sim = ACCSetup(override=dict(
        runlen=86400 * 10,
    ))

    if rst.proc_num == 1:
        import sys

        comm = MPI.COMM_SELF.Spawn(
            sys.executable,
            args=['-m', 'mpi4py', sys.argv[-1]],
            maxprocs=4
        )

        try:
            sim.setup()
            sim.run()
        except Exception as exc:
            print(str(exc))
            comm.Abort(1)

        vs = sim.state.variables

        other_psi = npx.empty_like(vs.psi)
        comm.Recv(other_psi, 0)

        scale = max(
            npx.abs(vs.psi).max(),
            npx.abs(other_psi).max()
        )

        npx.testing.assert_allclose(vs.psi / scale, other_psi / scale, rtol=0, atol=1e-5)

    else:
        assert rst.proc_num == 4

        sim.setup()
        sim.run()

        vs = sim.state.variables
        psi_global = gather(vs.psi, sim.state.dimensions, ('xt', 'yt', 'isle'))

        if rst.proc_rank == 0:
            rs.mpi_comm.Get_parent().Send(psi_global, 0)

    ''')

    run_dist_kernel(test_kernel)


def test_acc_nompirun():
    from veros.setups.acc import acc
    import veros.cli

    subprocess.check_call([
        sys.executable,
        '-m', 'mpi4py',
        veros.cli.veros_run.__file__,
        acc.__file__,
        '-n', '2', '2',
        '--diskless_mode',
        '-s', 'runlen', '864000'
    ], stderr=subprocess.STDOUT)
