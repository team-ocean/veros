from time import perf_counter

from benchmark_base import benchmark_cli

from veros import logger
from veros.pyom_compat import load_pyom, pyom_from_state


@benchmark_cli
def main(pyom2_lib, timesteps, size):
    import h5py

    from veros.state import VerosState
    from veros.tools import get_assets
    from veros.core.external.solve_pressure import get_linear_solver
    from veros.core.operators import flush, numpy as npx
    from veros.distributed import barrier

    assets = get_assets("bench-external", "bench-external-assets.json")

    if size[0] > 3000 and size[0] < 4000:
        infile = assets["01deg-press"]
    elif size[0] > 300 and size[0] < 500:
        infile = assets["1deg-press"]
    elif size[0] > 50 and size[0] < 100:
        infile = assets["4deg-press"]
    else:
        raise ValueError(
            "Pressure solver benchmark only works for 4deg, 1deg and 01deg setups. with nx = 94, 364, 3604 respectively."
        )

    with h5py.File(infile, "r") as f:
        input_data = {v: npx.array(f[v]) for v in f.keys()}
        input_settings = dict(f["settings"].attrs.items())

    def get_solver_state():
        from veros import variables as var_mod, settings as settings_mod

        default_settings = settings_mod.SETTINGS.copy()
        default_dimensions = var_mod.DIM_TO_SHAPE_VAR.copy()
        var_meta = var_mod.VARIABLES.copy()

        keys_to_extract = ["cost", "cosu", "dxt", "dxu", "dyt", "dyu", "hu", "hv", "maskT"]
        variables_subset = {key: var_meta[key] for key in keys_to_extract}

        settings_used = [
            "nx",
            "ny",
            "nz",
            "dt_tracer",
            "dt_mom",
            "enable_free_surface",
            "enable_streamfunction",
            "enable_cyclic_x",
        ]

        state = VerosState(var_meta=variables_subset, setting_meta=default_settings, dimensions=default_dimensions)
        with state.settings.unlock():
            state.settings.update({sett: input_settings[sett] for sett in settings_used})

        state.initialize_variables()

        with state.variables.unlock():
            for key in keys_to_extract:
                setattr(state.variables, key, input_data[key])

        return state

    state = get_solver_state()
    solver = get_linear_solver(state)

    if not pyom2_lib:

        def reset():
            pass

        def run():
            return solver.solve(state, input_data["rhs"], input_data["x0"])

    else:
        pyom_obj = load_pyom(pyom2_lib)
        pyom_obj = pyom_from_state(state, pyom_obj, init_streamfunction=False)
        m = pyom_obj.main_module
        m.taum1, m.tau, m.taup1 = 1, 2, 3

        def reset():
            m.psi[..., m.taup1 - 1] = input_data["x0"]

        def run():
            pyom_obj.congrad_surf_press(
                m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx, m.je_pe + m.onx, input_data["rhs"], 0
            )
            return m.psi[..., m.taup1 - 1]

    for _ in range(timesteps):
        reset()
        flush()
        barrier()

        start = perf_counter()
        _ = run()
        flush()
        barrier()
        end = perf_counter()

        logger.debug(f"Pressure solver took {end - start}s")


if __name__ == "__main__":
    main()
