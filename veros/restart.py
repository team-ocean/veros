import os

from veros import logger, runtime_settings, runtime_state
from veros.io_tools import hdf5 as h5tools
from veros.signals import do_not_disturb
from veros.distributed import get_chunk_slices, exchange_overlap
from veros.variables import get_shape


def read_from_h5(dimensions, var_meta, infile, groupname, enable_cyclic_x):
    from veros.core.operators import numpy as npx, update, at

    variables = {}

    for key, var in infile[groupname].items():
        if not var_meta[key].dims:
            variables[key] = npx.array(var)
            continue

        local_shape = get_shape(dimensions, var_meta[key].dims, local=True, include_ghosts=True)
        gidx, lidx = get_chunk_slices(dimensions["xt"], dimensions["yt"], var_meta[key].dims, include_overlap=True)

        # pass dtype as str to prevent endianness from leaking into array
        variables[key] = npx.empty(local_shape, dtype=str(var.dtype))
        variables[key] = update(variables[key], at[lidx], var[gidx])
        variables[key] = exchange_overlap(variables[key], var_meta[key].dims, enable_cyclic_x)

    attributes = {key: var.item() for key, var in infile[groupname].attrs.items()}

    return attributes, variables


def write_to_h5(dimensions, var_meta, var_data, outfile, groupname, attributes=None):
    if attributes is None:
        attributes = {}

    group = outfile.require_group(groupname)

    for key, var in var_data.items():
        var_dims = var_meta[key].dims
        if var_dims is None:
            var_dims = []

        global_shape = get_shape(dimensions, var_dims, local=False)
        gidx, lidx = get_chunk_slices(dimensions["xt"], dimensions["yt"], var_dims, include_overlap=True)

        kwargs = dict(
            exact=True,
        )

        if var_dims:
            chunksize = []
            for d in var_dims:
                if d in dimensions:
                    chunksize.append(get_shape(dimensions, (d,), local=True, include_ghosts=False)[0])
                else:
                    chunksize.append(1)

            kwargs.update(chunks=tuple(chunksize))

            if runtime_settings.hdf5_gzip_compression and runtime_state.proc_num == 1:
                kwargs.update(compression="gzip", compression_opts=1)

        group.require_dataset(key, global_shape, var.dtype, **kwargs)
        group[key][gidx] = var[lidx]

    for key, val in attributes.items():
        group.attrs[key] = val


def read_restart(state):
    settings = state.settings

    if not settings.restart_input_filename:
        return

    if runtime_settings.force_overwrite:
        raise RuntimeError("To prevent data loss, force_overwrite cannot be used in restart runs")

    statedict = dict(state.variables.items())
    statedict.update(state.settings.items())
    restart_filename = settings.restart_input_filename.format(**statedict)

    if not os.path.isfile(restart_filename):
        raise IOError(f"restart file {restart_filename} not found")

    logger.info(f"Reading restart data from {restart_filename}")

    with h5tools.threaded_io(restart_filename, "r") as infile, state.variables.unlock():
        # core restart
        restart_vars = {var: meta for var, meta in state.var_meta.items() if meta.write_to_restart and meta.active}
        _, restart_data = read_from_h5(state.dimensions, restart_vars, infile, "core", settings.enable_cyclic_x)

        for key in restart_vars.keys():
            try:
                var_data = restart_data[key]
            except KeyError:
                raise RuntimeError(f"No restart data found for variable {key} in {restart_filename}") from None

            setattr(state.variables, key, var_data)

        # diagnostic restarts
        for diag_name, diagnostic in state.diagnostics.items():
            if not diagnostic.var_meta:
                # nothing to do
                continue

            dimensions = dict(state.dimensions)
            if diagnostic.extra_dimensions:
                dimensions.update(diagnostic.extra_dimensions)

            restart_vars = {
                var: meta for var, meta in diagnostic.var_meta.items() if meta.write_to_restart and meta.active
            }
            _, restart_data = read_from_h5(dimensions, restart_vars, infile, diag_name, settings.enable_cyclic_x)

            for key in restart_vars.keys():
                try:
                    var_data = restart_data[key]
                except KeyError:
                    raise RuntimeError(
                        f'No restart data found for variable {key} in {restart_filename} (from diagnostic "{diag_name}")'
                    ) from None

                setattr(diagnostic.variables, key, var_data)

    return state


@do_not_disturb
def write_restart(state, force=False):
    vs = state.variables
    settings = state.settings

    if runtime_settings.diskless_mode:
        return

    if not settings.restart_output_filename:
        return

    write_now = force or (
        settings.restart_frequency and vs.itt > 0 and vs.time % settings.restart_frequency < settings.dt_tracer
    )

    if not write_now:
        return

    statedict = dict(state.variables.items())
    statedict.update(state.settings.items())
    restart_filename = settings.restart_output_filename.format(**statedict)

    logger.info(f"Writing restart file {restart_filename}")

    with h5tools.threaded_io(restart_filename, "w") as outfile:
        # core restart
        vs = state.variables
        restart_vars = {var: meta for var, meta in state.var_meta.items() if meta.write_to_restart and meta.active}
        restart_data = {var: getattr(vs, var) for var in restart_vars}
        write_to_h5(state.dimensions, restart_vars, restart_data, outfile, "core")

        # diagnostic restarts
        for diag_name, diagnostic in state.diagnostics.items():
            if not diagnostic.var_meta:
                # nothing to do
                continue

            dimensions = dict(state.dimensions)
            if diagnostic.extra_dimensions:
                dimensions.update(diagnostic.extra_dimensions)

            restart_vars = {
                var: meta for var, meta in diagnostic.var_meta.items() if meta.write_to_restart and meta.active
            }
            restart_data = {var: getattr(diagnostic.variables, var) for var in restart_vars}
            write_to_h5(dimensions, restart_vars, restart_data, outfile, diag_name)
