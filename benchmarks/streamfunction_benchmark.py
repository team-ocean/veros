from benchmark_base import benchmark_cli

from time import perf_counter

import numpy as onp

from veros import logger
from veros.pyom_compat import get_random_state


@benchmark_cli
def main(pyom2_lib, timesteps, size):
    from veros.distributed import barrier
    from veros.variables import allocate
    from veros.core.streamfunction.solvers import get_linear_solver
    from veros.core.operators import flush, update, at, numpy as npx

    states = get_random_state(
        pyom2_lib,
        extra_settings=dict(
            nx=size[0],
            ny=size[1],
            nz=size[2],
            dt_tracer=3600,
            dt_mom=3600,
            enable_cyclic_x=True,
        ),
    )

    if pyom2_lib:
        state, pyom_obj = states
    else:
        state = states

    rhs = allocate(state.dimensions, ("xt", "yt"))
    rhs = update(rhs, at[2:-2, 2:-2], onp.random.randn(*rhs[2:-2, 2:-2].shape))

    if not pyom2_lib:
        solver = get_linear_solver(state)

        def run():
            x0 = npx.zeros_like(rhs)
            return solver.solve(state, rhs, x0)

    else:
        m = pyom_obj.main_module

        def run():
            sol = npx.zeros_like(rhs)
            pyom_obj.congrad_streamfunction(
                is_=m.is_pe - m.onx,
                ie_=m.ie_pe + m.onx,
                js_=m.js_pe - m.onx,
                je_=m.je_pe + m.onx,
                forc=rhs,
                iterations=m.congr_itts,
                sol=sol,
                converged=False,
            )

    for _ in range(timesteps):
        start = perf_counter()

        run()
        flush()
        barrier()

        end = perf_counter()

        logger.debug(f"Time step took {end-start}s")


if __name__ == "__main__":
    main()
