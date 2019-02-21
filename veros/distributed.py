import functools

from . import runtime_settings as rs
from .decorators import veros_method, CONTEXT

try:
    from mpi4py import MPI
except ImportError:
    HAS_MPI4PY = False
else:
    HAS_MPI4PY = True


if HAS_MPI4PY:
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
else:
    COMM = None
    RANK = 0
    SIZE = 1


SCATTERED_DIMENSIONS = (
    ("xt", "xu"),
    ("yt", "yu")
)


def distributed_veros_method(function):
    wrapped_function = veros_method(function)

    @functools.wraps(function)
    def distributed_veros_method_wrapper(vs, arr, *args, **kwargs):
        if not HAS_MPI4PY or SIZE == 1 or not CONTEXT.is_dist_safe:
            return arr

        try:
            return wrapped_function(vs, arr, *args, **kwargs)
        finally:
            barrier()

    return distributed_veros_method_wrapper


def ascontiguousarray(arr):
    if not arr.flags["C_CONTIGUOUS"] and not arr.flags["F_CONTIGUOUS"]:
        return arr.copy()
    if not arr.flags["OWNDATA"]:
        return arr.copy()
    return arr


@veros_method
def get_array_buffer(vs, arr):
    if rs.backend == "bohrium":
        return np.interop_numpy.get_array(arr)
    return arr


def validate_decomposition(vs):
    if not HAS_MPI4PY and (rs.num_proc[0] > 1 or rs.num_proc[1] > 1):
        raise RuntimeError("mpi4py is required for distributed execution")

    if rs.num_proc[0] * rs.num_proc[1] != SIZE:
        raise RuntimeError("number of processes does not match size of communicator")

    if vs.nx % rs.num_proc[0]:
        raise ValueError("processes do not divide domain evenly in x-direction")

    if vs.ny % rs.num_proc[1]:
        raise ValueError("processes do not divide domain evenly in y-direction")


def get_chunk_size(vs):
    return (vs.nx // rs.num_proc[0], vs.ny // rs.num_proc[1])


def proc_rank_to_index(vs, rank):
    return (rank % rs.num_proc[0], rank // rs.num_proc[0])


def proc_index_to_rank(vs, ix, iy):
    return ix + iy * rs.num_proc[0]


def get_global_idx(vs, rank):
    this_x, this_y = proc_rank_to_index(vs, rank)
    chunk_x, chunk_y = get_chunk_size(vs)
    return (
        slice(2 + this_x * chunk_x, 2 + (this_x + 1) * chunk_x),
        slice(2 + this_y * chunk_y, 2 + (this_y + 1) * chunk_y)
    )


def get_process_neighbors(vs):
    this_x, this_y = proc_rank_to_index(vs, RANK)

    west = this_x - 1 if this_x > 0 else None
    south = this_y - 1 if this_y > 0 else None
    east = this_x + 1 if (this_x + 1) < rs.num_proc[0] else None
    north = this_y + 1 if (this_y + 1) < rs.num_proc[1] else None

    neighbors = [
        (west, this_y),
        (this_x, south),
        (east, this_y),
        (this_x, north)
    ]

    global_neighbors = [
        proc_index_to_rank(vs, *i) if None not in i else None for i in neighbors
    ]
    return global_neighbors


@distributed_veros_method
def exchange_overlap(vs, arr, dim='xy'):
    arr = np.asarray(arr)

    if dim == 'xy':
        proc_neighbors = get_process_neighbors(vs)

        overlap_slices_from = (
            (slice(2, 4), slice(0, None), Ellipsis), # west
            (slice(0, None), slice(2, 4), Ellipsis), # south
            (slice(-4, -2), slice(0, None), Ellipsis), # east
            (slice(0, None), slice(-4, -2), Ellipsis), # north
        )

        overlap_slices_to = (
            (slice(0, 2), slice(0, None), Ellipsis), # west
            (slice(0, None), slice(0, 2), Ellipsis), # south
            (slice(-2, None), slice(0, None), Ellipsis), # east
            (slice(0, None), slice(-2, None), Ellipsis), # north
        )

        # flipped indices of overlap (n <-> s, w <-> e)
        send_to_recv = [2, 3, 0, 1]

    else:
        if dim == 'x':
            proc_neighbors = get_process_neighbors(vs)[0::2] # west and east
        elif dim == 'y':
            proc_neighbors = get_process_neighbors(vs)[1::2] # south and north
        else:
            raise NotImplementedError()

        overlap_slices_from = (
            (slice(2, 4), Ellipsis),
            (slice(-4, -2), Ellipsis),
        )

        overlap_slices_to = (
            (slice(0, 2), Ellipsis),
            (slice(-2, None), Ellipsis),
        )

        send_to_recv = [1, 0]

    receive_futures = []
    for i_s in range(len(proc_neighbors)):
        i_r = send_to_recv[i_s]
        other_proc = proc_neighbors[i_s]

        if other_proc is None:
            continue

        send_idx = overlap_slices_from[i_s]
        recv_idx = overlap_slices_to[i_s]

        send_arr = ascontiguousarray(arr[send_idx])
        recv_arr = np.empty_like(arr[recv_idx])

        COMM.Isend(get_array_buffer(vs, send_arr), dest=other_proc, tag=i_s)
        future = COMM.Irecv(get_array_buffer(vs, recv_arr), source=other_proc, tag=i_r)
        receive_futures.append((future, recv_idx, recv_arr))

    for future, recv_idx, recv_arr in receive_futures:
        future.wait()
        arr[recv_idx] = recv_arr


@distributed_veros_method
def exchange_cyclic_boundaries(vs, arr):
    ix, iy = proc_rank_to_index(vs, RANK)

    if 0 < ix < (rs.num_proc[0] - 1):
        return

    if ix == 0:
        other_proc = proc_index_to_rank(vs, rs.num_proc[0] - 1, iy)
        send_idx = (slice(2, 4), Ellipsis)
        recv_idx = (slice(0, 2), Ellipsis)
    else:
        other_proc = proc_index_to_rank(vs, 0, iy)
        send_idx = (slice(-4, -2), Ellipsis)
        recv_idx = (slice(-2, None), Ellipsis)

    recv_arr = np.empty_like(arr[recv_idx])
    send_arr = ascontiguousarray(arr[send_idx])

    send_future = COMM.Isend(get_array_buffer(vs, send_arr), dest=other_proc)
    recv_future = COMM.Irecv(get_array_buffer(vs, recv_arr), source=other_proc)

    send_future.wait()
    recv_future.wait()

    arr[recv_idx] = recv_arr


@distributed_veros_method
def _reduce(vs, arr, op):
    if np.isscalar(arr):
        arr = np.array([arr])
    arr = ascontiguousarray(arr)
    res = np.empty(1, dtype=arr.dtype)
    COMM.Allreduce(
        get_array_buffer(vs, arr),
        get_array_buffer(vs, res),
        op=op
    )
    return res[0]


@distributed_veros_method
def global_max(vs, arr):
    return _reduce(vs, arr, MPI.MAX)


@distributed_veros_method
def global_min(vs, arr):
    return _reduce(vs, arr, MPI.MIN)


@distributed_veros_method
def global_sum(vs, arr):
    return _reduce(vs, arr, MPI.SUM)


@distributed_veros_method
def zonal_sum(vs, arr):
    pass


@distributed_veros_method
def _gather_1d(vs, arr, dim):
    assert dim in (0, 1)

    nx = get_chunk_size(vs)[dim]
    nproc = rs.num_proc[dim]

    def get_buffer(proc):
        px = proc_rank_to_index(vs, proc)[dim]
        sl = 0 if px == 0 else 2
        su = nx + 4 if (px + 1) == nproc else nx + 2
        local_slice = slice(sl, su)
        global_slice = slice(sl + px * nx, su + px * nx)
        buffer = np.empty((su - sl, *arr.shape[1:]), dtype=arr.dtype)
        return local_slice, global_slice, buffer

    otherdim = int(not dim)
    pi = proc_rank_to_index(vs, RANK)
    if pi[otherdim] != 0:
        return arr

    idx, gidx, sendbuf = get_buffer(RANK)
    sendbuf[...] = arr[idx]

    if RANK == 0:
        buffer_list = []
        futures = []
        for proc in range(1, SIZE):
            pi = proc_rank_to_index(vs, proc)
            if pi[otherdim] != 0:
                continue
            sidx, tidx, recvbuf = get_buffer(proc)
            buffer_list.append((sidx, tidx, recvbuf))
            futures.append(
                COMM.Irecv(get_array_buffer(vs, recvbuf), source=proc)
            )

        for future in futures:
            future.wait()

        out_shape = ((vs.nx + 4, vs.ny + 4)[dim], *arr.shape[1:])
        out = np.empty(out_shape, dtype=arr.dtype)
        out[gidx] = sendbuf
        for _, idx, val in buffer_list:
            out[idx] = val

        return out
    else:
        COMM.Send(get_array_buffer(vs, sendbuf), dest=0)
        return arr


@distributed_veros_method
def _gather_xy(vs, arr):
    nxi, nyi = get_chunk_size(vs)

    def get_buffer(proc):
        px, py = proc_rank_to_index(vs, proc)
        sxl = 0 if px == 0 else 2
        sxu = nxi + 4 if (px + 1) == rs.num_proc[0] else nxi + 2
        syl = 0 if py == 0 else 2
        syu = nyi + 4 if (py + 1) == rs.num_proc[1] else nyi + 2
        local_slice = (slice(sxl, sxu), slice(syl, syu))
        global_slice = (
            slice(sxl + px * nxi, sxu + px * nxi),
            slice(syl + py * nyi, syu + py * nyi)
        )
        buffer = np.empty((sxu - sxl, syu - syl, *arr.shape[2:]), dtype=arr.dtype)
        return local_slice, global_slice, buffer

    assert arr.shape[:2] == (nxi + 4, nyi + 4)

    idx, gidx, sendbuf = get_buffer(RANK)
    sendbuf[...] = arr[idx]

    if RANK == 0:
        buffer_list = []
        futures = []
        for proc in range(1, SIZE):
            sidx, tidx, recvbuf = get_buffer(proc)
            buffer_list.append((sidx, tidx, recvbuf))
            futures.append(
                COMM.Irecv(get_array_buffer(vs, recvbuf), source=proc)
            )
        MPI.Request.Waitall(futures)

        out_shape = (vs.nx + 4, vs.ny + 4, *arr.shape[2:])
        out = np.empty(out_shape, dtype=arr.dtype)
        out[gidx] = sendbuf
        for _, idx, val in buffer_list:
            out[idx] = val

        return out
    else:
        COMM.Send(get_array_buffer(vs, sendbuf), dest=0)

    return arr


@distributed_veros_method
def gather(vs, arr, var_grid):
    if len(var_grid) < 2:
        d1, d2 = var_grid[0], None
    else:
        d1, d2 = var_grid[:2]

    if d1 not in SCATTERED_DIMENSIONS[0] and d1 not in SCATTERED_DIMENSIONS[1] and d2 not in SCATTERED_DIMENSIONS[1]:
        # neither x nor y dependent, nothing to do
        return arr

    if d1 in SCATTERED_DIMENSIONS[0] and d2 not in SCATTERED_DIMENSIONS[1]:
        # only x dependent
        return _gather_1d(vs, arr, 0)

    elif d1 in SCATTERED_DIMENSIONS[1]:
        # only y dependent
        return _gather_1d(vs, arr, 1)

    elif d1 in SCATTERED_DIMENSIONS[0] and d2 in SCATTERED_DIMENSIONS[1]:
        # x and y dependent
        return _gather_xy(vs, arr)

    else:
        raise NotImplementedError()


@distributed_veros_method
def broadcast(vs, obj):
    if isinstance(obj, np.ndarray):
        obj = ascontiguousarray(obj)
        return COMM.Bcast(get_array_buffer(vs, obj), root=0)
    else:
        return COMM.bcast(obj, root=0)


@distributed_veros_method
def _scatter_constant(vs, arr):
    arr = ascontiguousarray(arr)
    COMM.Bcast(get_array_buffer(vs, arr), root=0)
    return arr


@distributed_veros_method
def _scatter_1d(vs, arr, dim):
    assert dim in (0, 1)

    nx = get_chunk_size(vs)[dim]

    def get_buffer(proc):
        px = proc_rank_to_index(vs, proc)[dim]
        sl = 0 if px == 0 else 2
        su = nx + 4 if (px + 1) == rs.num_proc[dim] else nx + 2
        local_slice = slice(sl, su)
        global_slice = slice(sl + px * nx, su + px * nx)
        buffer = np.empty((su - sl, *arr.shape[1:]), dtype=arr.dtype)
        return local_slice, global_slice, buffer

    local_slice, _, recvbuf = get_buffer(RANK)
    recv_future = COMM.Irecv(get_array_buffer(vs, recvbuf), source=0)

    if RANK == 0:
        futures = []
        for proc in range(0, SIZE):
            _, global_slice, sendbuf = get_buffer(proc)
            sendbuf = ascontiguousarray(arr[global_slice])
            sendbuf = get_array_buffer(vs, sendbuf)
            futures.append(
                COMM.Isend(sendbuf, dest=proc)
            )

        for future in futures:
            future.wait()

    recv_future.wait()

    if RANK == 0:
        # arr changes shape in main process
        arr = np.zeros((nx + 4, *arr.shape[2:]), dtype=arr.dtype)

    arr[local_slice] = recvbuf
    exchange_overlap(vs, arr, dim='x' if dim == 0 else 'y')

    if vs.enable_cyclic_x:
        exchange_cyclic_boundaries(vs, arr)

    return arr


@distributed_veros_method
def _scatter_xy(vs, arr):
    nxi, nyi = get_chunk_size(vs)

    def get_buffer(proc):
        px, py = proc_rank_to_index(vs, proc)
        sxl = 0 if px == 0 else 2
        sxu = nxi + 4 if (px + 1) == rs.num_proc[0] else nxi + 2
        syl = 0 if py == 0 else 2
        syu = nyi + 4 if (py + 1) == rs.num_proc[1] else nyi + 2
        local_slice = (slice(sxl, sxu), slice(syl, syu))
        global_slice = (
            slice(sxl + px * nxi, sxu + px * nxi),
            slice(syl + py * nyi, syu + py * nyi)
        )
        buffer = np.empty((sxu - sxl, syu - syl, *arr.shape[2:]), dtype=arr.dtype)
        return local_slice, global_slice, buffer

    local_slice, _, recvbuf = get_buffer(RANK)
    recv_future = COMM.Irecv(get_array_buffer(vs, recvbuf), source=0)

    if RANK == 0:
        futures = []
        for proc in range(0, SIZE):
            _, global_slice, _ = get_buffer(proc)
            sendbuf = ascontiguousarray(arr[global_slice])
            futures.append(
                COMM.Isend(get_array_buffer(vs, sendbuf), dest=proc)
            )

        for future in futures:
            future.wait()

    recv_future.wait()

    if RANK == 0:
        # arr changes shape in main process
        arr = np.empty((nxi + 4, nyi + 4, *arr.shape[2:]), dtype=arr.dtype)

    arr[local_slice] = recvbuf

    exchange_overlap(vs, arr)

    if vs.enable_cyclic_x:
        exchange_cyclic_boundaries(vs, arr)

    return arr


@distributed_veros_method
def scatter(vs, arr, var_grid):
    if len(var_grid) < 2:
        d1, d2 = var_grid[0], None
    else:
        d1, d2 = var_grid[:2]

    arr = np.asarray(arr)

    if d1 not in SCATTERED_DIMENSIONS[0] and d1 not in SCATTERED_DIMENSIONS[1] and d2 not in SCATTERED_DIMENSIONS[1]:
        # neither x nor y dependent
        return _scatter_constant(vs, arr)

    if d1 in SCATTERED_DIMENSIONS[0] and d2 not in SCATTERED_DIMENSIONS[1]:
        # only x dependent
        return _scatter_1d(vs, arr, 0)

    elif d1 in SCATTERED_DIMENSIONS[1]:
        # only y dependent
        return _scatter_1d(vs, arr, 1)

    elif d1 in SCATTERED_DIMENSIONS[0] and d2 in SCATTERED_DIMENSIONS[1]:
        # x and y dependent
        return _scatter_xy(vs, arr)

    else:
        raise NotImplementedError()


def barrier():
    COMM.barrier()


def abort():
    COMM.Abort()
