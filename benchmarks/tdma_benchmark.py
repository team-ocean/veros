from benchmark_base import benchmark_cli

from time import perf_counter

import numpy as np
from veros import logger
from veros.pyom_compat import load_pyom, pyom_from_state


@benchmark_cli
def main(pyom2_lib, timesteps, size):
    from veros.state import get_default_state
    from veros.distributed import barrier
    from veros.core.utilities import create_water_masks
    from veros.core.operators import flush, solve_tridiagonal

    state = get_default_state()

    with state.settings.unlock():
        state.settings.update(
            nx=size[0],
            ny=size[1],
            nz=size[2],
            enable_neutral_diffusion=True,
        )

    state.initialize_variables()
    state.variables.__locked__ = False

    nx, ny, nz = 70, 60, 50
    a, b, c, d = (np.random.randn(nx, ny, nz) for _ in range(4))
    kbot = np.random.randint(0, nz, size=(nx, ny))

    if not pyom2_lib:
        _, water_mask, edge_mask = create_water_masks(kbot, nz)

        def run():
            out_vs = solve_tridiagonal(a, b, c, d, water_mask, edge_mask)
            return out_vs

    else:
        pyom_obj = load_pyom(pyom2_lib)
        pyom_obj = pyom_from_state(state, pyom_obj, init_streamfunction=False)

        def run():
            out_pyom = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    ks = kbot[i, j] - 1
                    ke = nz

                    if ks < 0:
                        continue

                    out_pyom[i, j, ks:ke] = pyom_obj.solve_tridiag(
                        a=a[i, j, ks:ke], b=b[i, j, ks:ke], c=c[i, j, ks:ke], d=d[i, j, ks:ke], n=ke - ks
                    )
            return out_pyom

    for _ in range(timesteps):
        start = perf_counter()

        run()
        flush()
        barrier()

        end = perf_counter()

        logger.debug(f"Time step took {end-start}s")


if __name__ == "__main__":
    main()
