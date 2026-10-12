"""Differentiable linear MPI transfers used by compiled integrations."""

import functools

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, mlir
import mpi4jax
from mpi4py import MPI

from veros import runtime_settings as rs

_communicators = {}


def prepare_communicator(comm):
    """Clone a host communicator before tracing numerical traffic."""
    from jax._src import effects
    from mpi4jax._src.utils import OrderedMPIEffect

    # These numerical transfers can be replayed during checkpoint recomputation.
    # All ranks use the same schedule and a communicator separate from host I/O.
    effects.remat_allowed_effects.add_type(OrderedMPIEffect)
    key = MPI._handleof(comm)
    if key not in _communicators:
        _communicators[key] = comm.Dup()
    return _communicators[key]


def numerical_communicator(comm):
    try:
        return _communicators[MPI._handleof(comm)]
    except KeyError:
        raise RuntimeError("MPI communicator must be prepared before tracing an integration") from None


@functools.lru_cache(None)
def _sum_transfer(comm_handle):
    comm = _communicators[comm_handle]

    def transfer(x):
        return mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)

    primitive = Primitive("veros_sum_transfer")
    effects = jax.make_jaxpr(transfer)(jnp.zeros(())).jaxpr.effects
    primitive.def_effectful_abstract_eval(lambda aval: (aval, effects))
    primitive.def_impl(jax.jit(transfer))
    mlir.register_lowering(primitive, mlir.lower_fun(transfer, multiple_results=False))

    def jvp(primals, tangents):
        return primitive.bind(primals[0]), primitive.bind(ad.instantiate_zeros(tangents[0]))

    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = lambda ct, _: (primitive.bind(ad.instantiate_zeros(ct)),)
    return primitive.bind


def sum_transfer(value, comm=None):
    """Sum distributed contributions with a summing linear transpose."""
    return _sum_transfer(MPI._handleof(rs.mpi_comm if comm is None else comm))(value)


def exchange(sendbuf, recvbuf, source, dest):
    source = MPI.PROC_NULL if source is None else source
    dest = MPI.PROC_NULL if dest is None else dest
    received = mpi4jax.sendrecv(sendbuf, recvbuf, source=source, dest=dest, comm=numerical_communicator(rs.mpi_comm))
    return jnp.zeros_like(recvbuf) if source == MPI.PROC_NULL else received


def _halo_adjoint(value, var_grid, cyclic):
    from veros.distributed import _overlap_slices, get_process_neighbors

    slices = _overlap_slices(var_grid)
    if slices is None:
        return value
    send_order, recv_order, send_slices, recv_slices = slices
    neighbors = get_process_neighbors(cyclic)
    for send_dir, recv_dir in reversed(tuple(zip(send_order, recv_order))):
        send_proc, recv_proc = neighbors[send_dir], neighbors[recv_dir]
        if send_proc is None and recv_proc is None:
            continue
        send_idx, recv_idx = send_slices[send_dir], recv_slices[recv_dir]
        outgoing = value[recv_idx] if recv_proc is not None else jnp.zeros_like(value[recv_idx])
        if recv_proc is not None:
            value = value.at[recv_idx].set(0)
        incoming = exchange(outgoing, jnp.zeros_like(value[send_idx]), send_proc, recv_proc)
        if send_proc is not None:
            value = value.at[send_idx].add(incoming)
    return value


@functools.lru_cache(None)
def _halo_operator(comm_handle, var_grid, cyclic, shape, dtype):
    from veros.distributed import _exchange_overlap
    from veros.routines import tracing_context

    def forward(value):
        with tracing_context():
            return _exchange_overlap(value, var_grid, cyclic)

    def reverse(value):
        return _halo_adjoint(value, var_grid, cyclic)

    effects = jax.make_jaxpr(forward)(jnp.zeros(shape, dtype=dtype)).jaxpr.effects
    primitives = (Primitive("veros_halo_exchange"), Primitive("veros_halo_adjoint"))
    for primitive, implementation in zip(primitives, (forward, reverse)):
        primitive.def_effectful_abstract_eval(lambda aval: (aval, effects))
        primitive.def_impl(jax.jit(implementation))
        mlir.register_lowering(primitive, mlir.lower_fun(implementation, multiple_results=False))

        def jvp(primals, tangents, primitive=primitive):
            return primitive.bind(primals[0]), primitive.bind(ad.instantiate_zeros(tangents[0]))

        ad.primitive_jvps[primitive] = jvp
    for primitive, transposed in zip(primitives, reversed(primitives)):
        ad.primitive_transposes[primitive] = lambda ct, _, transposed=transposed: (
            transposed.bind(ad.instantiate_zeros(ct)),
        )
    return primitives[0].bind


def halo_exchange(value, var_grid, cyclic):
    """Keep each field's complete exchange ordered under JVP and transpose.

    Boundary ranks have different point-to-point data dependencies. Treating
    the complete exchange as one linear operation keeps their transformed
    programs from interleaving messages for different fields.
    """
    return _halo_operator(MPI._handleof(rs.mpi_comm), tuple(var_grid), cyclic, value.shape, value.dtype)(value)


def gather(arr, dimensions, var_grid):
    """Replicate unique local cells through a differentiable linear collective."""
    from veros import runtime_state as rst
    from veros.distributed import get_chunk_slices, proc_rank_to_index, SCATTERED_DIMENSIONS

    if not var_grid:
        return arr
    shape = list(arr.shape)
    axes = []
    for axis, dim in enumerate(var_grid):
        for direction, names in enumerate(SCATTERED_DIMENSIONS):
            if dim in names:
                axes.append(direction)
                shape[axis] = dimensions[dim] + 4
    if not axes:
        return arr
    global_slice, local_slice = get_chunk_slices(dimensions["xt"], dimensions["yt"], var_grid, include_overlap=True)
    rank_index = proc_rank_to_index(rst.proc_rank)
    representative = all(rank_index[direction] == 0 for direction in (0, 1) if direction not in axes)
    padded = jnp.zeros(shape, dtype=arr.dtype)
    if representative:
        padded = padded.at[global_slice].set(arr[local_slice])
    result = sum_transfer(padded)
    return result


def local_slice(arr, dimensions, var_grid):
    """Select a local domain, including halos, from a replicated global array."""
    from veros import runtime_state as rst
    from veros.distributed import proc_rank_to_index, SCATTERED_DIMENSIONS

    indices = [slice(None)] * arr.ndim
    rank_index = proc_rank_to_index(rst.proc_rank)
    for axis, dim in enumerate(var_grid or ()):
        for direction, names in enumerate(SCATTERED_DIMENSIONS):
            if dim in names:
                size = dimensions[dim] // rs.num_proc[direction]
                start = size * rank_index[direction]
                indices[axis] = slice(start, start + size + 4)
    return arr[tuple(indices)]
