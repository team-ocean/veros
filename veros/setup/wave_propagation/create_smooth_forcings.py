import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import shutil
from skimage.restoration import inpaint

def make_cyclic_1d(arr):
    return np.hstack((arr, arr, arr))

def make_cyclic(arr):
    return np.vstack((arr, arr, arr))

variables = ("q_net", "temperature", "sst", "tidal_energy", "wind_energy", "swf",
             "sss", "tau_x", "tau_y", "salinity", "dqdt")
data = {}
with Dataset("forcing_1deg_global.nc", "r") as forcing_file:
    x, y, z = (forcing_file.variables[k][...].T for k in ("xt","yt","zt"))
    x_c = make_cyclic_1d(x)
    x_c[:len(x)] -= 360.
    x_c[2*len(x):] += 360.
    for k in variables:
        data[k] = make_cyclic(forcing_file.variables[k][...].T)
        data[k][np.abs(data[k]) < 1e-10] = np.nan

xx, yy = np.meshgrid(x, y)
xxc, yyc = np.meshgrid(x_c, y)

def interpolate(arr):
    every = (slice(None, None, 3), slice(None, None, 10))
    print(arr.shape, xxc.shape, yyc.shape, xx.shape, yy.shape)
    invalid_mask = np.isfinite(arr[every])
    interpolator = scipy.interpolate.Rbf(xxc[every][invalid_mask], yyc[every][invalid_mask], arr[every][invalid_mask])
    return interpolator(xx, yy)
    print("yo")
    return inpaint.inpaint_biharmonic(arr, (~invalid_mask).astype(np.int))



shutil.copy("forcing_1deg_global.nc", "forcing_1deg_global_smooth_rbf.nc")
with Dataset("forcing_1deg_global_smooth_rbf.nc", "a") as out_file:
    for var in ("temperature", "salinity"):
        print(var)
        for k in range(data[var].shape[2]):
            out_file[var][k, :, :] = interpolate(data[var][:, :, k].T)

    for var in ("tau_x", "tau_y", "q_net", "dqdt", "swf", "sss", "sst"):
        print(var)
        for k in range(data[var].shape[2]):
            out_file[var][k, :, :] = interpolate(data[var][:, :, k].T)

    for var in ("tidal_energy",):
        out_file[var][:, :] = interpolate(data[var][:, :].T)

    for var in ("wind_energy",):
        print(var)
        for l in range(data[var].shape[3]):
            for k in range(data[var].shape[2]):
                out_file[var][l, k, :, :] = interpolate(data[var][:, :, k, l].T)
