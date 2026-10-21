"""Distributed PETSc solves with ordered callbacks and derivatives for fixed operators."""

from functools import partial
from inspect import signature
from threading import Lock

import jax
import numpy as np
from jax.extend.core import Primitive
from jax.interpreters import ad, mlir
from jax._src import core, dispatch, effects
from jax._src.callback import emit_python_callback
from mpi4py import MPI
from petsc4py import PETSc

from veros import runtime_settings as rs, runtime_state as rst
from veros.core.operators import at, update
from veros.core.external.solvers.petsc_ import PETScSolver, prepare_solver_inputs

_callback_lock = Lock()


class PETScEffect(effects.Effect):
    """Order PETSc callbacks on a single rank."""


_single_rank_effect = PETScEffect()


def _solver_effect():
    if rst.proc_num > 1:
        from mpi4jax._src.utils import ordered_effect

        effect = ordered_effect
    else:
        effect = _single_rank_effect
        for registry in (
            effects.lowerable_effects,
            effects.control_flow_allowed_effects,
            effects.ordered_effects,
            effects.shardable_ordered_effects,
        ):
            registry.add_type(PETScEffect)
    effects.remat_allowed_effects.add_type(type(effect))
    return effect


_solve_p = Primitive("veros_petsc_solve")
_solve_p.multiple_results = True


@_solve_p.def_effectful_abstract_eval
def _abstract_solve(rhs, *, solver, transpose):
    return (rhs, core.ShapedArray((), np.dtype("bool"))), {solver._effect}


_solve_p.def_impl(partial(dispatch.apply_primitive, _solve_p))


def _lower_solve(ctx, rhs, *, solver, transpose):
    # Share mpi4jax's ordering token to sequence PETSc and numerical
    # transfers on the same communicator.
    callback = solver._transpose_callback if transpose else solver._callback
    result, token, _ = emit_python_callback(
        ctx,
        callback,
        ctx.tokens_in.get(solver._effect),
        [rhs],
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=True,
    )
    ctx.set_tokens_out(mlir.TokenSet({solver._effect: token}))
    return result


_lowering_options = {}
if "cacheable" in signature(mlir.register_lowering).parameters:
    _lowering_options["cacheable"] = False
mlir.register_lowering(_solve_p, _lower_solve, **_lowering_options)


def _solve_jvp(primals, tangents, *, solver, transpose):
    primal = _solve_p.bind(primals[0], solver=solver, transpose=transpose)
    tangent, _ = _solve_p.bind(ad.instantiate_zeros(tangents[0]), solver=solver, transpose=transpose)
    return primal, (tangent, ad.Zero(core.typeof(primal[1]).to_tangent_aval()))


def _solve_transpose(ct, rhs, *, solver, transpose):
    return (_solve_p.bind(ad.instantiate_zeros(ct[0]), solver=solver, transpose=not transpose)[0],)


ad.primitive_jvps[_solve_p] = _solve_jvp
ad.primitive_transposes[_solve_p] = _solve_transpose


class CompiledPETScSolver(PETScSolver):
    def __init__(self, state):
        if MPI.Query_thread() < MPI.THREAD_MULTIPLE:
            raise RuntimeError("Compiled PETSc requires MPI_THREAD_MULTIPLE for host callback execution")
        jax.effects_barrier()
        comm = rs.mpi_comm
        if rst.proc_num > 1:
            from veros._distributed_jax import numerical_communicator

            comm = numerical_communicator(comm)
        self._callback_comm = comm
        # Host callbacks use CPU vectors even when the model runs on a GPU.
        super().__init__(state, comm=comm, use_gpu=False)
        self._transpose_matrix = self._matrix.transpose(PETSc.Mat())
        self._transpose_ksp = self._create_ksp(self._transpose_matrix, self._options)
        for ksp in (self._ksp, self._transpose_ksp):
            ksp.setInitialGuessNonzero(False)
            ksp.setErrorIfNotConverged(False)
            ksp.setUp()
        self._residual_petsc = self._rhs_petsc.duplicate()
        self._magnitude = np.empty((), dtype=np.float64)
        self._effect = _solver_effect()

    def _compiled_solve(self, rhs):
        return _solve_p.bind(rhs, solver=self, transpose=False)

    def _callback(self, rhs):
        return self._solve_callback(rhs, self._matrix, self._ksp)

    def _transpose_callback(self, rhs):
        return self._solve_callback(rhs, self._transpose_matrix, self._transpose_ksp)

    def _solve_callback(self, rhs, matrix, ksp):
        with _callback_lock:
            # One native reduction checks finiteness and normalizes all ranks.
            magnitude = np.max(np.abs(rhs))
            self._magnitude[()] = magnitude if np.isfinite(magnitude) else np.inf
            if rst.proc_num > 1:
                self._callback_comm.Allreduce(MPI.IN_PLACE, self._magnitude, op=MPI.MAX)
            scale = self._magnitude.item()
            if not np.isfinite(scale):
                return np.full(rhs.shape, np.nan, dtype=rhs.dtype), np.asarray(False)
            if scale == 0:
                return np.zeros(rhs.shape, dtype=rhs.dtype), np.asarray(True)
            self._da.getVecArray(self._rhs_petsc)[...] = np.asarray(rhs)
            # Normalize globally so tiny tangent/adjoint RHSs retain accuracy.
            self._rhs_petsc.scale(1 / scale)
            ksp.solve(self._rhs_petsc, self._sol_petsc)
            matrix.mult(self._sol_petsc, self._residual_petsc)
            self._residual_petsc.axpy(-1, self._rhs_petsc)
            # PETSc combines these split-phase norms into one reduction.
            self._rhs_petsc.normBegin()
            self._residual_petsc.normBegin()
            rhs_norm = self._rhs_petsc.normEnd()
            residual = self._residual_petsc.normEnd()
            tolerance = max(1e-8, 100 * np.finfo(rhs.dtype).eps) * rhs_norm
            converged = ksp.getConvergedReason() > 0 and residual <= tolerance
            solution = np.empty(rhs.shape, dtype=rhs.dtype)
            np.multiply(self._da.getVecArray(self._sol_petsc)[...], scale, out=solution, casting="unsafe")
            if not converged:
                solution.fill(np.nan)
            return solution, np.asarray(converged)

    def solve(self, state, rhs, x0, boundary_val=None):
        rhs, _ = prepare_solver_inputs(state, rhs, x0, boundary_val, self._boundary_mask, self._boundary_fac)
        solution, converged = self._compiled_solve(rhs[2:-2, 2:-2])
        state._solver_residual_ok = state._solver_residual_ok & converged
        return update(rhs, at[2:-2, 2:-2], solution)
