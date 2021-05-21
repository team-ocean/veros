#!/usr/bin/env python

import functools

import click


def get_mask_data(depth):
    import numpy as np

    return np.where(depth > 0, 255, 0).astype(np.uint8)


def smooth_image(data, sigma):
    from scipy import ndimage

    return ndimage.gaussian_filter(data, sigma=sigma)


def save_image(data, path):
    import numpy as np
    from PIL import Image

    Image.fromarray(np.flipud(data)).convert("1").save(path)


def create_mask(infile, outfile, variable="z", scale=None):
    """Creates a mask image from a given netCDF file"""
    import numpy as np
    import h5netcdf

    with h5netcdf.File(infile, "r") as topo:
        z = np.array(topo.variables[variable])
    if scale is not None:
        z = smooth_image(z, scale)
    data = get_mask_data(z)
    save_image(data, outfile)


@click.command("veros-create-mask")
@click.argument("infile", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--variable", default="z", help="Variable holding topography data (default: z)")
@click.option("-o", "--outfile", default="topography.png", help="Output filename (default: topography.png)")
@click.option(
    "-s",
    "--scale",
    nargs=2,
    type=click.INT,
    default=None,
    help="Standard deviation in grid cells for Gaussian smoother (default: disable smoother)",
)
@functools.wraps(create_mask)
def cli(*args, **kwargs):
    create_mask(**kwargs)
