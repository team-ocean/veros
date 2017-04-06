import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import shutil

variables = ("q_net", "temperature", "sst", "tidal_energy", "wind_energy", "swf",
             "sss", "tau_x", "tau_y", "salinity", "dqdt")
data = {}
with Dataset("forcing_1deg_global.nc", "r") as forcing_file:
    x, y, z = (forcing_file.variables[k][...].T for k in ("xt","yt","zt"))
    for k in variables:
        data[k] = forcing_file.variables[k][...].T
        data[k][np.abs(data[k]) < 1e-10] = np.nan

xx, yy = np.meshgrid(x, y)

def interpolate(arr):
    invalid_mask = np.isfinite(arr)
    valid_points = np.vstack((xx[invalid_mask], yy[invalid_mask])).T
    interpolator = scipy.interpolate.CloughTocher2DInterpolator(valid_points, arr[invalid_mask])
    return interpolator(np.vstack((xx.flatten(), yy.flatten())).T).reshape(arr.shape)
    interpolator = scipy.interpolate.interp2d(xx[invalid_mask][::5], yy[invalid_mask][::5], arr[invalid_mask][::5], kind="linear")
    return interpolator(x, y)
    print(yy.min(), yy.max())
    theta, phi = (yy + 90.) * np.pi / 180., (xx - 90.) * np.pi / 180.
    print(theta.min(), theta.max(), phi.min(), phi.max())
    interpolator = scipy.interpolate.SmoothSphereBivariateSpline(theta[invalid_mask], phi[invalid_mask], arr[invalid_mask])
    return interpolator(theta[0,:], phi[:,0])



shutil.copy("forcing_1deg_global.nc", "forcing_1deg_global_smooth.nc")
with Dataset("forcing_1deg_global_smooth.nc", "a") as out_file:
    for var in ("temperature", "salinity"):
        print(var)
        for k in range(data[var].shape[2]):
            out_file[var][k, :, :] = interpolate(data[var][:, :, k].T)

    for var in ("tau_x", "tau_y", "q_net", "dqdt", "swf", "sss", "sst", "tidal_energy"):
        print(var)
        for k in range(data[var].shape[2]):
            out_file[var][k, :, :] = interpolate(data[var][:, :, k].T)

    for var in ("wind_energy",):
        print(var)
        for l in range(data[var].shape[3]):
            for k in range(data[var].shape[2]):
                out_file[var][l, k, :, :] = interpolate(data[var][:, :, k, l].T)
