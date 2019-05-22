from . import runtime_settings as rs, runtime_state as rst
from .decorators import veros_method, dist_context_only


SCATTERED_DIMENSIONS = (
    ('xt', 'xu'),
    ('yt', 'yu')
)


def ascontiguousarray(arr):
    if not arr.flags['C_CONTIGUOUS'] and not arr.flags['F_CONTIGUOUS']:
        return arr.copy()
    if not arr.flags['OWNDATA']:
        return arr.copy()
    return arr


@veros_method(inline=True)
def get_array_buffer(vs, arr):
    from mpi4py import MPI

    MPI_TYPE_MAP = {
        'int8': MPI.CHAR,
        'int16': MPI.SHORT,
        'int32': MPI.INT,
        'int64': MPI.LONG,
        'int128': MPI.LONG_LONG,
        'float32': MPI.FLOAT,
        'float64': MPI.DOUBLE,
        'bool': MPI.BOOL,
    }

    if rs.backend == 'bohrium':
        if np.check(arr):
            buf = np.interop_numpy.get_array(arr)
        else:
            buf = arr
    else:
        buf = arr

    return [buf, arr.size, MPI_TYPE_MAP[str(arr.dtype)]]


@veros_method
def validate_decomposition(vs):
    if rs.mpi_comm is None:
        if (rs.num_proc[0] > 1 or rs.num_proc[1] > 1):
            raise RuntimeError('mpi4py is required for distributed execution')
        return

    comm_size = rs.mpi_comm.Get_size()
    proc_num = rs.num_proc[0] * rs.num_proc[1]
    if proc_num != comm_size:
        raise RuntimeError('number of processes ({}) does not match size of communicator ({})'
                           .format(proc_num, comm_size))

    if vs.nx % rs.num_proc[0]:
        raise ValueError('processes do not divide domain evenly in x-direction')

    if vs.ny % rs.num_proc[1]:
        raise ValueError('processes do not divide domain evenly in y-direction')


def get_chunk_size(vs):
    return (vs.nx // rs.num_proc[0], vs.ny // rs.num_proc[1])


def get_global_size(vs, arr_shp, dim_grid, include_overlap=False):
    ovl = 4 if include_overlap else 0
    shape = []
    for s, dim in zip(arr_shp, dim_grid):
        if dim in SCATTERED_DIMENSIONS[0]:
            shape.append(vs.nx + ovl)
        elif dim in SCATTERED_DIMENSIONS[1]:
            shape.append(vs.ny + ovl)
        else:
            shape.append(s)
    return shape


def get_local_size(vs, arr_shp, dim_grid, include_overlap=False):
    ovl = 4 if include_overlap else 0
    shape = []
    for s, dim in zip(arr_shp, dim_grid):
        if dim in SCATTERED_DIMENSIONS[0]:
            shape.append(vs.nx // rs.num_proc[0] + ovl)
        elif dim in SCATTERED_DIMENSIONS[1]:
            shape.append(vs.ny // rs.num_proc[1] + ovl)
        else:
            shape.append(s)
    return shape


def proc_rank_to_index(rank):
    return (rank % rs.num_proc[0], rank // rs.num_proc[0])


def proc_index_to_rank(ix, iy):
    return ix + iy * rs.num_proc[0]


def get_chunk_slices(vs, dim_grid, proc_idx=None, include_overlap=False):
    if proc_idx is None:
        proc_idx = proc_rank_to_index(rst.proc_rank)

    px, py = proc_idx
    nx, ny = get_chunk_size(vs)

    if include_overlap:
        sxl = 0 if px == 0 else 2
        sxu = nx + 4 if (px + 1) == rs.num_proc[0] else nx + 2
        syl = 0 if py == 0 else 2
        syu = ny + 4 if (py + 1) == rs.num_proc[1] else ny + 2
    else:
        sxl = syl = 0
        sxu = nx
        syu = ny

    global_slice, local_slice = [], []

    for dim in dim_grid:
        if dim in SCATTERED_DIMENSIONS[0]:
            global_slice.append(slice(sxl + px * nx, sxu + px * nx))
            local_slice.append(slice(sxl, sxu))
        elif dim in SCATTERED_DIMENSIONS[1]:
            global_slice.append(slice(syl + py * ny, syu + py * ny))
            local_slice.append(slice(syl, syu))
        else:
            global_slice.append(slice(None))
            local_slice.append(slice(None))

    return tuple(global_slice), tuple(local_slice)


def get_process_neighbors(vs):
    this_x, this_y = proc_rank_to_index(rst.proc_rank)

    west = this_x - 1 if this_x > 0 else None
    south = this_y - 1 if this_y > 0 else None
    east = this_x + 1 if (this_x + 1) < rs.num_proc[0] else None
    north = this_y + 1 if (this_y + 1) < rs.num_proc[1] else None

    neighbors = [
        # direct neighbors
        (west, this_y),
        (this_x, south),
        (east, this_y),
        (this_x, north),
        # corners
        (west, south),
        (east, south),
        (east, north),
        (west, north),
    ]

    global_neighbors = [
        proc_index_to_rank(*i) if None not in i else None for i in neighbors
    ]
    return global_neighbors


@dist_context_only
@veros_method
def exchange_overlap(vs, arr, var_grid):
    if len(var_grid) < 2:
        d1, d2 = var_grid[0], None
    else:
        d1, d2 = var_grid[:2]

    if d1 not in SCATTERED_DIMENSIONS[0] and d1 not in SCATTERED_DIMENSIONS[1] and d2 not in SCATTERED_DIMENSIONS[1]:
        # neither x nor y dependent, nothing to do
        return arr

    if d1 in SCATTERED_DIMENSIONS[0] and d2 in SCATTERED_DIMENSIONS[1]:
        proc_neighbors = get_process_neighbors(vs)

        overlap_slices_from = (
            (slice(2, 4), slice(0, None), Ellipsis), # west
            (slice(0, None), slice(2, 4), Ellipsis), # south
            (slice(-4, -2), slice(0, None), Ellipsis), # east
            (slice(0, None), slice(-4, -2), Ellipsis), # north
            (slice(2, 4), slice(2, 4), Ellipsis), # south-west
            (slice(-4, -2), slice(2, 4), Ellipsis), # south-east
            (slice(-4, -2), slice(-4, -2), Ellipsis), # north-east
            (slice(2, 4), slice(-4, -2), Ellipsis), # north-west
        )

        overlap_slices_to = (
            (slice(0, 2), slice(0, None), Ellipsis), # west
            (slice(0, None), slice(0, 2), Ellipsis), # south
            (slice(-2, None), slice(0, None), Ellipsis), # east
            (slice(0, None), slice(-2, None), Ellipsis), # north
            (slice(0, 2), slice(0, 2), Ellipsis), # south-west
            (slice(-2, None), slice(0, 2), Ellipsis), # south-east
            (slice(-2, None), slice(-2, None), Ellipsis), # north-east
            (slice(0, 2), slice(-2, None), Ellipsis), # north-west
        )

        # flipped indices of overlap (n <-> s, w <-> e)
        send_to_recv = [2, 3, 0, 1, 6, 7, 4, 5]

    else:
        if d1 in SCATTERED_DIMENSIONS[0]:
            proc_neighbors = get_process_neighbors(vs)[0:4:2] # west and east
        elif d1 in SCATTERED_DIMENSIONS[1]:
            proc_neighbors = get_process_neighbors(vs)[1:4:2] # south and north
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
    for i_s, other_proc in enumerate(proc_neighbors):
        if other_proc is None:
            continue

        i_r = send_to_recv[i_s]
        recv_idx = overlap_slices_to[i_s]
        recv_arr = np.empty_like(arr[recv_idx])

        future = rs.mpi_comm.Irecv(get_array_buffer(vs, recv_arr), source=other_proc, tag=i_r)
        receive_futures.append((future, recv_idx, recv_arr))

    for i_s, other_proc in enumerate(proc_neighbors):
        if other_proc is None:
            continue

        send_idx = overlap_slices_from[i_s]
        send_arr = ascontiguousarray(arr[send_idx])

        rs.mpi_comm.Send(get_array_buffer(vs, send_arr), dest=other_proc, tag=i_s)

    for future, recv_idx, recv_arr in receive_futures:
        future.wait()
        arr[recv_idx] = recv_arr


@dist_context_only
@veros_method
def exchange_cyclic_boundaries(vs, arr):
    if rs.num_proc[0] == 1:
        arr[-2:, ...] = arr[2:4, ...]
        arr[:2, ...] = arr[-4:-2, ...]
        return

    ix, iy = proc_rank_to_index(rst.proc_rank)

    if 0 < ix < (rs.num_proc[0] - 1):
        return

    if ix == 0:
        other_proc = proc_index_to_rank(rs.num_proc[0] - 1, iy)
        send_idx = (slice(2, 4), Ellipsis)
        recv_idx = (slice(0, 2), Ellipsis)
    else:
        other_proc = proc_index_to_rank(0, iy)
        send_idx = (slice(-4, -2), Ellipsis)
        recv_idx = (slice(-2, None), Ellipsis)

    recv_arr = np.empty_like(arr[recv_idx])
    send_arr = ascontiguousarray(arr[send_idx])

    rs.mpi_comm.Sendrecv(
        sendbuf=get_array_buffer(vs, send_arr), dest=other_proc, sendtag=10,
        recvbuf=get_array_buffer(vs, recv_arr), source=other_proc, recvtag=10
    )

    arr[recv_idx] = recv_arr


@dist_context_only
@veros_method(inline=True)
def _reduce(vs, arr, op):
    if np.isscalar(arr):
        squeeze = True
        arr = np.array([arr])
    else:
        squeeze = False

    arr = ascontiguousarray(arr)
    res = np.empty_like(arr)

    rs.mpi_comm.Allreduce(
        get_array_buffer(vs, arr),
        get_array_buffer(vs, res),
        op=op
    )

    if squeeze:
        res = res[0]

    return res


@dist_context_only
@veros_method
def global_and(vs, arr):
    from mpi4py import MPI
    return _reduce(vs, arr, MPI.LAND)


@dist_context_only
@veros_method
def global_or(vs, arr):
    from mpi4py import MPI
    return _reduce(vs, arr, MPI.LOR)


@dist_context_only
@veros_method
def global_max(vs, arr):
    from mpi4py import MPI
    return _reduce(vs, arr, MPI.MAX)


@dist_context_only
@veros_method
def global_min(vs, arr):
    from mpi4py import MPI
    return _reduce(vs, arr, MPI.MIN)


@dist_context_only
@veros_method
def global_sum(vs, arr):
    from mpi4py import MPI
    return _reduce(vs, arr, MPI.SUM)


@dist_context_only
@veros_method(inline=True)
def _gather_1d(vs, arr, dim):
    assert dim in (0, 1)

    otherdim = 1 - dim
    pi = proc_rank_to_index(rst.proc_rank)
    if pi[otherdim] != 0:
        return arr

    dim_grid = ['xt' if dim == 0 else 'yt'] + [None] * (arr.ndim - 1)
    gidx, idx = get_chunk_slices(vs, dim_grid, include_overlap=True)
    sendbuf = ascontiguousarray(arr[idx])

    if rst.proc_rank == 0:
        buffer_list = []
        for proc in range(1, rst.proc_num):
            pi = proc_rank_to_index(proc)
            if pi[otherdim] != 0:
                continue
            idx_g, idx_l = get_chunk_slices(vs, dim_grid, include_overlap=True, proc_idx=pi)
            recvbuf = np.empty_like(arr[idx_l])
            future = rs.mpi_comm.Irecv(get_array_buffer(vs, recvbuf), source=proc, tag=20)
            buffer_list.append((future, idx_g, recvbuf))

        out_shape = ((vs.nx + 4, vs.ny + 4)[dim],) + arr.shape[1:]
        out = np.empty(out_shape, dtype=arr.dtype)
        out[gidx] = sendbuf

        for future, idx, val in buffer_list:
            future.wait()
            out[idx] = val

        return out

    else:
        rs.mpi_comm.Send(get_array_buffer(vs, sendbuf), dest=0, tag=20)
        return arr


@dist_context_only
@veros_method(inline=True)
def _gather_xy(vs, arr):
    nxi, nyi = get_chunk_size(vs)
    assert arr.shape[:2] == (nxi + 4, nyi + 4), arr.shape

    dim_grid = ['xt', 'yt'] + [None] * (arr.ndim - 2)
    gidx, idx = get_chunk_slices(vs, dim_grid, include_overlap=True)
    sendbuf = ascontiguousarray(arr[idx])

    if rst.proc_rank == 0:
        buffer_list = []
        for proc in range(1, rst.proc_num):
            idx_g, idx_l = get_chunk_slices(
                vs, dim_grid, include_overlap=True,
                proc_idx=proc_rank_to_index(proc)
            )
            recvbuf = np.empty_like(arr[idx_l])
            future = rs.mpi_comm.Irecv(get_array_buffer(vs, recvbuf), source=proc, tag=30)
            buffer_list.append((future, idx_g, recvbuf))

        out_shape = (vs.nx + 4, vs.ny + 4) + arr.shape[2:]
        out = np.empty(out_shape, dtype=arr.dtype)
        out[gidx] = sendbuf

        for future, idx, val in buffer_list:
            future.wait()
            out[idx] = val

        return out
    else:
        rs.mpi_comm.Send(get_array_buffer(vs, sendbuf), dest=0, tag=30)

    return arr


@dist_context_only
@veros_method
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


@dist_context_only
@veros_method
def broadcast(vs, obj):
    return rs.mpi_comm.bcast(obj, root=0)


@dist_context_only
@veros_method(inline=True)
def _scatter_constant(vs, arr):
    arr = ascontiguousarray(arr)
    rs.mpi_comm.Bcast(get_array_buffer(vs, arr), root=0)
    return arr


@dist_context_only
@veros_method(inline=True)
def _scatter_1d(vs, arr, dim):
    assert dim in (0, 1)

    nx = get_chunk_size(vs)[dim]
    dim_grid = ['xt' if dim == 0 else 'yt'] + [None] * (arr.ndim - 1)
    _, local_slice = get_chunk_slices(vs, dim_grid, include_overlap=True)

    if rst.proc_rank == 0:
        recvbuf = arr[local_slice]

        for proc in range(1, rst.proc_num):
            global_slice, _ = get_chunk_slices(vs, dim_grid, include_overlap=True, proc_idx=proc_rank_to_index(proc))
            sendbuf = ascontiguousarray(arr[global_slice])
            rs.mpi_comm.Send(get_array_buffer(vs, sendbuf), dest=proc, tag=40)

        # arr changes shape in main process
        arr = np.zeros((nx + 4,) + arr.shape[1:], dtype=arr.dtype)
    else:
        recvbuf = np.empty_like(arr[local_slice])
        rs.mpi_comm.Recv(get_array_buffer(vs, recvbuf), source=0, tag=40)

    arr[local_slice] = recvbuf

    exchange_overlap(vs, arr, ['xt' if dim == 0 else 'yt'])

    return arr


@dist_context_only
@veros_method(inline=True)
def _scatter_xy(vs, arr):
    nxi, nyi = get_chunk_size(vs)

    dim_grid = ['xt', 'yt'] + [None] * (arr.ndim - 2)
    _, local_slice = get_chunk_slices(vs, dim_grid, include_overlap=True)

    if rst.proc_rank == 0:
        recvbuf = arr[local_slice]

        for proc in range(1, rst.proc_num):
            global_slice, _ = get_chunk_slices(vs, dim_grid, include_overlap=True, proc_idx=proc_rank_to_index(proc))
            sendbuf = ascontiguousarray(arr[global_slice])
            rs.mpi_comm.Send(get_array_buffer(vs, sendbuf), dest=proc, tag=50)

        # arr changes shape in main process
        arr = np.empty((nxi + 4, nyi + 4) + arr.shape[2:], dtype=arr.dtype)
    else:
        recvbuf = np.empty_like(arr[local_slice])
        rs.mpi_comm.Recv(get_array_buffer(vs, recvbuf), source=0, tag=50)

    arr[local_slice] = recvbuf

    exchange_overlap(vs, arr, ['xt', 'yt'])

    return arr


@dist_context_only
@veros_method
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
    rs.mpi_comm.barrier()


def abort():
    rs.mpi_comm.Abort()
