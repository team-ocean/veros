import functools

from . import runtime_settings as rs
from .decorators import CONTEXT, veros_method
from .core import utilities

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
    RANK = 1
    SIZE = 1

SCATTERED_DIMENSIONS = (
    ("xt", "xu"),
    ("yt", "yu")
)


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


@veros_method(dist_only=True)
def exchange_overlap(vs, arr):
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

    receive_futures = []
    for i_s in range(4):
        i_r = send_to_recv[i_s]
        other_proc = proc_neighbors[i_s]

        if other_proc is None:
            continue

        send_idx = overlap_slices_from[i_s]
        recv_idx = overlap_slices_to[i_s]

        send_arr = np.ascontiguousarray(arr[send_idx])
        recv_arr = np.empty_like(arr[recv_idx])

        COMM.Isend(send_arr, dest=other_proc, tag=i_s)
        future = COMM.Irecv(recv_arr, source=other_proc, tag=i_r)
        receive_futures.append((future, recv_idx, recv_arr))

    for future, recv_idx, recv_arr in receive_futures:
        future.wait()
        arr[recv_idx] = recv_arr


@veros_method(dist_only=True)
def exchange_cyclic_boundaries(vs, arr):
    import numpy as np
    ix, iy = proc_rank_to_index(vs, RANK)

    if 0 < ix < (rs.num_proc[0] - 1):
        return

    if ix == 0:
        other_proc = proc_index_to_rank(vs, rs.num_proc[0] - 1, iy)
        send_idx = (slice(2, 4), Ellipsis)
        recv_idx = (slice(0, 2), Ellipsis)
        tag_s, tag_r = 0, 1
    else:
        other_proc = proc_index_to_rank(vs, 0, iy)
        send_idx = (slice(-4, -2), Ellipsis)
        recv_idx = (slice(-2, None), Ellipsis)
        tag_s, tag_r = 1, 0

    recv_arr = np.empty_like(arr[recv_idx])

    send_future = COMM.Isend(np.ascontiguousarray(arr[send_idx]), dest=other_proc, tag=tag_s)
    recv_future = COMM.Irecv(recv_arr, source=other_proc, tag=tag_r)

    send_future.wait()
    recv_future.wait()

    arr[recv_idx] = recv_arr


@veros_method(dist_only=True)
def global_max(vs, arr):
    pass


@veros_method(dist_only=True)
def global_min(vs, arr):
    pass


@veros_method(dist_only=True)
def global_sum(vs, arr):
    pass


@veros_method(dist_only=True)
def zonal_sum(vs, arr):
    pass


def _process_grid(vs, arr, var_grid):
    """yields (has_ghosts, chunk_size, num_blocks)"""
    nxi, nyi = get_chunk_size(vs)

    for shp, dim in zip(arr.shape, var_grid):
        if dim in SCATTERED_DIMENSIONS[0]:
            yield (True, nxi, rs.num_proc[0])
        elif dim in SCATTERED_DIMENSIONS[1]:
            yield (True, nyi, rs.num_proc[1])
        else:
            yield (False, shp, 1)


@veros_method(dist_only=True)
def gather(vs, arr, var_grid):
    ghost_slices = []
    block_nums = []
    chunk_sizes = []
    out_shape = []

    for has_ghost, chunk_size, num_blocks in _process_grid(vs, arr, var_grid):
        ghost_slices.append(slice(2, -2) if has_ghost else slice(None))
        block_nums.append(num_blocks)
        chunk_sizes.append(chunk_size)
        global_size = chunk_size * num_blocks
        out_shape.append(global_size + 4 if has_ghost else global_size)

    if not any(block > 1 for block in block_nums):
        # nothing to do
        return arr

    recv_shape = block_nums + chunk_sizes
    ghost_slices = tuple(ghost_slices)
    blocks_last = list(range(0, len(recv_shape), 2)) + list(range(1, len(recv_shape), 2))

    recvbuf = None
    if RANK == 0:
        recvbuf = np.empty(recv_shape, dtype=arr.dtype)

    send_arr = np.ascontiguousarray(
        arr[ghost_slices]
    )
    COMM.Gather(send_arr, recvbuf, root=0)

    if RANK != 0:
        # no-op for other processes
        return arr

    out = np.zeros(out_shape, dtype=arr.dtype)
    out[ghost_slices] = recvbuf.transpose(*blocks_last).reshape(out[ghost_slices].shape)
    return out


@veros_method(dist_only=True)
def scatter(vs, arr, var_grid):
    ghost_slices = []
    block_nums = []
    chunk_sizes = []
    block_shape = []
    out_shape = []

    for has_ghost, chunk_size, num_blocks in _process_grid(vs, arr, var_grid):
        ghost_slices.append(slice(2, -2) if has_ghost else slice(None))
        block_nums.append(num_blocks)
        chunk_sizes.append(chunk_size)
        block_shape.extend([num_blocks, chunk_size])
        out_shape.append(chunk_size + 4 if has_ghost else chunk_size)

    if not any(block > 1 for block in block_nums):
        # nothing to do
        return arr

    ghost_slices = tuple(ghost_slices)
    blocks_last = list(range(1, len(block_shape), 2)) + list(range(0, len(block_shape), 2))

    sendbuf = None
    if RANK == 0:
        sendbuf = arr[ghost_slices].reshape(block_shape).transpose(*blocks_last)
        sendbuf = np.ascontiguousarray(sendbuf)

    recvbuf = np.empty(chunk_sizes, dtype=arr.dtype)
    COMM.Scatter(sendbuf, recvbuf, root=0)

    if RANK == 0:
        # arr changes shape in main process
        arr = np.zeros(out_shape, dtype=arr.dtype)

    arr[ghost_slices] = recvbuf
    #utilities.enforce_boundaries(vs, arr)
    return arr


def barrier():
    COMM.barrier()


def abort():
    COMM.Abort()
