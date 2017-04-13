import logging
import h5py
import os

from . import veros_method

RESTART_ATTRIBUTES = {"itt", "tau", "taum1", "taup1"}

@veros_method
def read_restart(veros):
    filename = veros.restart_filename
    if not os.path.isfile(filename):
        logging.warning("Not reading restart data: no restart file {} present"
                        .format(filename))
        return
    logging.debug("Reading restart file {}".format(filename))
    with h5py.File(filename, "r") as restart_file:
        for var_name, var in veros.variables.items():
            if var.write_to_restart:
                veros_var = getattr(veros, var_name)
                try:
                    restart_var = restart_file[var_name][...]
                except KeyError:
                    logging.warning(" not reading restart data for variable {}:"
                                    " no matching data found in restart file"
                                    .format(var_name))
                    continue
                if not veros_var.shape == restart_var.shape:
                    logging.warning(" not reading restart data for variable {}:"
                                    " restart data dimensions do not match model"
                                    " grid".format(var_name))
                    continue
                veros_var[...] = restart_var
        for attr in RESTART_ATTRIBUTES:
            setattr(veros, attr, restart_file.attrs[attr])

@veros_method
def write_restart(veros):
    filename = veros.restart_filename
    with h5py.File(filename, "w") as restart_file:
        for var_name, var in veros.variables.items():
            if var.write_to_restart:
                var_data = getattr(veros, var_name)
                if veros.backend_name == "bohrium":
                    var_data = var_data.copy2numpy()
                restart_file.create_dataset(var_name, data=var_data, compression="gzip")
        for attr in RESTART_ATTRIBUTES:
            restart_file.attrs[attr] = getattr(veros, attr)
