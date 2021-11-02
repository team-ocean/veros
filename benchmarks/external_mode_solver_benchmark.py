
from benchmark_base import benchmark_cli



@benchmark_cli
def main(pyom2_lib, timesteps, size):

    from copy import deepcopy
    import h5py
    from veros.state import VerosState
    from veros.tools import get_assets
    from veros.core.streamfunction.pressure_solvers import get_linear_solver
    from veros.variables import allocate
    from veros.state import resize_dimension

    assets = get_assets("bench-external", "bench-external-assets.json")["4deg"]

    f = h5py.File(assets)

    
    def solver_state():
        from veros import(variables as var_mod,settings as settings_mod)
        default_settings = deepcopy(settings_mod.SETTINGS)
        default_dimensions = deepcopy(var_mod.DIM_TO_SHAPE_VAR)
        var_meta = deepcopy(var_mod.VARIABLES)
        keys_to_extract = ["boundary_mask","cost","cosu","dxt","dxu","dyt","dyu","hu","hur","hv","hvr","maskT"]
        variables_subset = {key: var_meta[key] for key in keys_to_extract}

        state = VerosState(var_meta=variables_subset, setting_meta=default_settings, dimensions=default_dimensions)
        with state.settings.unlock():
            state.settings.update(
            nx=f["maskT"].shape[0]-4,
            ny=f["maskT"].shape[1]-4,
            nz=f["maskT"].shape[2],
            dt_tracer = 86400,
            dt_mom = 1800,
            enable_free_surface = True,
            enable_streamfunction = False,
            enable_cyclic_x= True)
        
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
    
    def run():
        solver = get_linear_solver(state)
        solver.solve(state,f["rhs_press"][:],x0)


    for i in range(timesteps):
        run()




if __name__ == "__main__":
    main()