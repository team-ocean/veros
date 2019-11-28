import inspect
import sys
import os
import time
import importlib


def run_via_mpi(func, nproc):
    if not hasattr(nproc, '__iter__') or len(nproc) != 2:
        raise TypeError('nproc argument must be in the form (nx, ny)')

    nproc = (int(nproc[0]), int(nproc[1]))

    if not callable(func):
        raise TypeError('func argument must be callable')

    source_file = inspect.getsourcefile(func)

    if not os.path.isfile(source_file):
        raise ValueError('func object must be importable')

    func_name = func.__qualname__

    from mpi4py import MPI
    total_proc = nproc[0] * nproc[1]

    comm = MPI.COMM_SELF.Spawn(
        sys.executable,
        args=['-m', 'mpi4py', __file__, source_file, func_name, str(nproc[0]), str(nproc[1])],
        maxprocs=total_proc
    )

    status_requests = [comm.irecv(source=p, tag=0) for p in range(total_proc)]

    while True:
        done, status = zip(*(f.test() for f in status_requests))

        if any(r is False for r in status):
            raise RuntimeError('Error while executing')

        if all(done):
            break

        time.sleep(0.1)


def _slave_entrypoint():
    from mpi4py import MPI
    comm = MPI.COMM_SELF.Get_parent()

    try:
        from veros import runtime_settings

        source_file, func_name, nproc_x, nproc_y = sys.argv[1:]
        module_name = os.path.splitext(os.path.basename(source_file))[0]

        spec = importlib.util.spec_from_file_location(module_name, source_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        runtime_settings.num_proc = (int(nproc_x), int(nproc_y))

        func = getattr(module, func_name)
        func()

    except:  # noqa: E722
        comm.send(False, dest=0, tag=0)
        raise

    else:
        comm.send(True, dest=0, tag=0)


if __name__ == '__main__':
    _slave_entrypoint()
