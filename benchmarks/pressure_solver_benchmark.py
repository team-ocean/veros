import os
from time import perf_counter

from benchmark_base import benchmark_cli

from veros import logger
from veros.pyom_compat import load_pyom, pyom_from_state

VARIABLES_USED = ["cost", "cosu", "dxt", "dxu", "dyt", "dyu", "hu", "hv", "maskT"]

VARIABLES_ALLOCATED = VARIABLES_USED + ["hur", "hvr", "boundary_mask"]

SETTINGS_USED = [
    "nx",
    "ny",
    "nz",
    "dt_tracer",
    "dt_mom",
    "enable_free_surface",
    "enable_streamfunction",
    "enable_cyclic_x",
]


def get_dummy_state(infile):
    import h5py
    from veros.state import VerosState
    from veros.distributed import get_chunk_slices, exchange_overlap
    from veros.variables import VARIABLES, DIM_TO_SHAPE_VAR, get_shape
    from veros.settings import SETTINGS
    from veros.core.operators import numpy as npx, update, at

    variables_subset = {key: VARIABLES[key] for key in VARIABLES_ALLOCATED}

    state = VerosState(var_meta=variables_subset, setting_meta=SETTINGS, dimensions=DIM_TO_SHAPE_VAR)

    with h5py.File(infile, "r") as f:
        input_settings = dict(f["settings"].attrs.items())

        with state.settings.unlock():
            state.settings.update({sett: input_settings[sett] for sett in SETTINGS_USED})

        state.initialize_variables()

        dimensions = state.dimensions
        var_meta = state.var_meta

        input_data = {}
        for v in f.keys():
            if v == "settings":
                continue

            if v in var_meta:
                dims = var_meta[v].dims
            else:
                dims = ("xt", "yt")

            local_shape = get_shape(dimensions, dims, local=True, include_ghosts=True)
            gidx, lidx = get_chunk_slices(dimensions["xt"], dimensions["yt"], dims, include_overlap=True)

            var = npx.empty(local_shape, dtype=str(f[v].dtype))
            var = update(var, at[lidx], f[v][gidx])
            var = exchange_overlap(var, dims, state.settings.enable_cyclic_x)
            input_data[v] = var

    with state.variables.unlock():
        for key in VARIABLES_USED:
            setattr(state.variables, key, input_data[key])

    return state, input_data


@benchmark_cli
def main(pyom2_lib, timesteps, size):
    from veros.tools import get_assets
    from veros.distributed import barrier
    from veros.core.operators import flush, numpy as npx
    from veros.core.external.solve_pressure import get_linear_solver

    here = os.path.dirname(__file__)
    assets = get_assets("bench-external", os.path.join(here, "bench-external-assets.json"))

    total_size = size[0] * size[1] * size[2]

    if 1e8 <= total_size <= 1e9:
        infile = assets["01deg-press"]
    elif 1e6 <= total_size <= 1e7:
        infile = assets["1deg-press"]
    elif 1e4 <= total_size <= 1e5:
        infile = assets["4deg-press"]
    else:
        raise ValueError(
            "Pressure solver benchmark only support 4deg, 1deg, and 0.1deg resolution"
            " (n = 5e4, 5e6, 5e8, respectively)."
        )

    state, input_data = get_dummy_state(infile)
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
        res = run()
        flush()
        barrier()
        end = perf_counter()

        logger.debug(f"Time step took {end - start}s")

    if not pyom2_lib:
        # monitor residual to expected solution, with generous margin
        def rms(arr):
            return npx.sqrt(npx.mean(arr ** 2))

        rms_err = rms(res[2:-2, 2:-2] - input_data["res"][2:-2, 2:-2]) / rms(input_data["res"][2:-2, 2:-2])
        assert rms_err < 1e-3, rms_err


if __name__ == "__main__":
    main()
