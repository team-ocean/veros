from benchmark_base import benchmark_cli

from time import perf_counter

from veros import logger
from veros.pyom_compat import load_pyom, pyom_from_state


@benchmark_cli
def main(pyom2_lib, timesteps, size):
    from veros.state import get_default_state
    from veros.distributed import barrier
    from veros.core.isoneutral import isoneutral_diffusion_pre
    from veros.core.operators import flush

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

    if not pyom2_lib:

        def run():
            return isoneutral_diffusion_pre(state)

    else:
        pyom_obj = load_pyom(pyom2_lib)
        pyom_obj = pyom_from_state(state, pyom_obj, init_streamfunction=False)

        def run():
            return pyom_obj.isoneutral_diffusion_pre()

    for _ in range(timesteps):
        start = perf_counter()

        run()
        flush()
        barrier()

        end = perf_counter()

        logger.debug(f"Time step took {end-start}s")


if __name__ == "__main__":
    main()
