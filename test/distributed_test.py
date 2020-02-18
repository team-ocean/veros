
import os
import sys
import subprocess

from tempfile import NamedTemporaryFile
from textwrap import dedent

import pytest

ON_GPU = os.environ.get('BH_STACK', '').lower() in ('opencl', 'cuda')


def run_dist_kernel(code):
    with NamedTemporaryFile(prefix='vs_test_', suffix='.py', mode='w') as f:
        f.write(code)
        f.flush()

        return subprocess.check_call(
            [sys.executable, '-m', 'mpi4py', f.name], stderr=subprocess.STDOUT
        )


@pytest.mark.skipif(ON_GPU, reason='Cannot run MPI and OpenCL')
def test_gather(backend):
    test_kernel = dedent('''
    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    import {backend} as np
    from mpi4py import MPI

    from veros import runtime_settings as rs, runtime_state as rst, VerosState
    from veros.distributed import gather

    rs.backend = '{backend}'

    if rst.proc_num == 1:
        import sys
        comm = MPI.COMM_SELF.Spawn(
            sys.executable,
            args=['-m', 'mpi4py', sys.argv[-1]],
            maxprocs=4
        )

        res = np.empty((8, 8))
        comm.Recv(res, 0)

        assert np.array_equal(res, np.array(
                [[0., 0., 0., 0., 2., 2., 2., 2.],
                 [0., 0., 0., 0., 2., 2., 2., 2.],
                 [0., 0., 0., 0., 2., 2., 2., 2.],
                 [0., 0., 0., 0., 2., 2., 2., 2.],
                 [1., 1., 1., 1., 3., 3., 3., 3.],
                 [1., 1., 1., 1., 3., 3., 3., 3.],
                 [1., 1., 1., 1., 3., 3., 3., 3.],
                 [1., 1., 1., 1., 3., 3., 3., 3.]]
            ))

        comm.Disconnect()
    else:
        rs.num_proc = (2, 2)

        assert rst.proc_num == 4

        vs = VerosState()
        vs.nx = 4
        vs.ny = 4

        a = rst.proc_rank * np.ones((6, 6))
        b = gather(vs, a, ('xt', 'yt'))

        if rst.proc_rank == 0:
            try:
                b = b.copy2numpy()
            except AttributeError:
                pass

            rs.mpi_comm.Get_parent().Send(b, 0)

    '''.format(
        backend=backend
    ))

    run_dist_kernel(test_kernel)


@pytest.mark.skipif(ON_GPU, reason='Cannot run MPI and OpenCL')
def test_scatter(backend):
    test_kernel = dedent('''
    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    import numpy as np
    from mpi4py import MPI

    from veros import runtime_settings as rs, runtime_state as rst, VerosState
    from veros.distributed import scatter

    rs.backend = '{backend}'

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

        res = np.empty((6, 6))

        proc_slices = (
            (slice(None, -2), slice(None, -2)),
            (slice(2, None), slice(None, -2)),
            (slice(None, -2), slice(2, None)),
            (slice(2, None), slice(2, None)),
        )

        for proc, idx in enumerate(proc_slices):
            comm.Recv(res, proc)
            assert np.array_equal(res, global_arr[idx])

        comm.Disconnect()
    else:
        rs.num_proc = (2, 2)

        assert rst.proc_num == 4

        vs = VerosState()
        vs.nx = 4
        vs.ny = 4

        if rst.proc_rank == 0:
            a = global_arr.copy()
        else:
            a = np.empty((6, 6))

        b = scatter(vs, a, ('xt', 'yt'))

        try:
            b = b.copy2numpy()
        except AttributeError:
            pass

        rs.mpi_comm.Get_parent().Send(b, 0)

    '''.format(
        backend=backend
    ))

    run_dist_kernel(test_kernel)


@pytest.mark.skipif(ON_GPU, reason='Cannot run MPI and OpenCL')
def test_acc(backend):
    test_kernel = dedent('''
    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    import numpy as np
    from mpi4py import MPI

    from veros import runtime_settings as rs, runtime_state as rst
    from veros.distributed import gather
    from veros.setup.acc import ACCSetup

    rs.backend = '{backend}'
    rs.linear_solver = 'scipy'

    sim = ACCSetup(override=dict(
        diskless_mode=True,
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

        other_psi = np.empty_like(sim.state.psi)
        comm.Recv(other_psi, 0)

        scale = max(
            np.abs(sim.state.psi).max(),
            np.abs(other_psi).max()
        )

        np.testing.assert_allclose(sim.state.psi / scale, other_psi / scale, rtol=0, atol=1e-5)

        comm.Disconnect()
    else:
        rs.num_proc = (2, 2)

        assert rst.proc_num == 4

        sim.setup()
        sim.run()

        psi_global = gather(sim.state, sim.state.psi, ('xt', 'yt', None))

        if rst.proc_rank == 0:
            try:
                psi_global = psi_global.copy2numpy()
            except AttributeError:
                pass

            rs.mpi_comm.Get_parent().Send(psi_global, 0)

    '''.format(
        backend=backend
    ))

    run_dist_kernel(test_kernel)


@pytest.mark.skipif(ON_GPU, reason='Cannot run MPI and OpenCL')
def test_acc_nompirun(backend):
    from veros.setup.acc import acc

    subprocess.check_call([
        sys.executable,
        '-m', 'mpi4py',
        acc.__file__,
        '-n', '2', '2',
        '-b', backend,
        '-s' 'diskless_mode', '1',
        '-s', 'runlen', '864000'
    ], stderr=subprocess.STDOUT)
