from time import perf_counter
from benchmark_base import benchmark_cli
from veros import logger
from veros.pyom_compat import load_pyom, pyom_from_state


@benchmark_cli
def main(pyom2_lib, timesteps, size):

    from copy import deepcopy
    import h5py
    from veros.state import VerosState
    from veros.tools import get_assets
    from veros.core.streamfunction.solve_stream import get_linear_solver
    from veros.variables import allocate
    from veros.state import resize_dimension
    from veros.core.operators import flush
    from veros.distributed import barrier

    if size[0] > 3000 and size[0] < 4000:
        assets = get_assets("bench-external", "bench-external-assets.json")["01deg"]
    elif size[0] > 300 and size[0] < 500:
        assets = get_assets("bench-external", "bench-external-assets.json")["1deg"]
    elif size[0] > 50 and size[0] < 100:
        assets = get_assets("bench-external", "bench-external-assets.json")["4deg"]
    else:
        raise ValueError(
            "External mode benchmark only works for 4deg, 1deg and 01deg setups. with nx = 94, 364, 3604 respectively."
        )

    f = h5py.File(assets)

    def solver_state():
        from veros import variables as var_mod, settings as settings_mod

        default_settings = deepcopy(settings_mod.SETTINGS)
        default_dimensions = deepcopy(var_mod.DIM_TO_SHAPE_VAR)
        var_meta = deepcopy(var_mod.VARIABLES)
        keys_to_extract = [
            "boundary_mask",
            "cost",
            "cosu",
            "dxt",
            "dxu",
            "dyt",
            "dyu",
            "hu",
            "hur",
            "hv",
            "hvr",
            "maskT",
        ]
        variables_subset = {key: var_meta[key] for key in keys_to_extract}

        state = VerosState(var_meta=variables_subset, setting_meta=default_settings, dimensions=default_dimensions)
        with state.settings.unlock():
            state.settings.update(
                nx=f["maskT"].shape[0] - 4,
                ny=f["maskT"].shape[1] - 4,
                nz=f["maskT"].shape[2],
                dt_tracer=86400,
                dt_mom=1800,
                enable_free_surface=True,
                enable_streamfunction=False,
                enable_cyclic_x=True,
            )

        state.initialize_variables()
        boundary_mask = f["boundary_mask"]
        nisle = boundary_mask.shape[2]
        resize_dimension(state, "isle", nisle)
        with state.variables.unlock():
            for key in keys_to_extract:
                setattr(state.variables, key, f[key])

        return state

    state = solver_state()
    x0 = allocate(state.dimensions, ("xt", "yt"), fill=0)
    solver = get_linear_solver(state)

    if not pyom2_lib:

        def run():
            solver.solve(state, f["rhs_stream"][:], x0)

    else:
        pyom_obj = load_pyom(pyom2_lib)
        pyom_obj = pyom_from_state(state, pyom_obj, init_streamfunction=False)
        m = pyom_obj.main_module
        m.forc = f["rhs_stream"][:]

        def run():
            pyom_obj.congrad_streamfunction(
                m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx, m.je_pe + m.onx, m.forc, 100
            )

    start = perf_counter()
    for _ in range(timesteps):
        run()
        flush()
        barrier()
    end = perf_counter()

    logger.debug(f"Streamfunction solver took {(end-start)/timesteps}s")


if __name__ == "__main__":
    main()
