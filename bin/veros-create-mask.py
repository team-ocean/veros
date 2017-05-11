#!/usr/bin/env python

"""Creates a mask image from a given netCDF file"""

import argparse


def get_mask_data(depth):
    import numpy as np
    return np.where(depth > 0, 255, 0).astype(np.uint8)


def smooth_image(data, sigma):
    from scipy import ndimage
    return ndimage.gaussian_filter(data, sigma=sigma)


def save_image(data, path):
    from PIL import Image
    Image.fromarray(np.flipud(data)).convert("1").save(path + ".png")


if __name__ == "__main__":
    from netCDF4 import Dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--file", help="Input file holding topography information",
                        required=True)
    parser.add_argument("-v,", help="Variable holding topography data (defaults to 'z')",
                        default="z", required=False)
    parser.add_argument("-o", "--out", help="", default="topography", required=False)
    parser.add_argument("-s", "--scale", nargs=2, type=int, required=False, default=None)
    args = parser.parse_args()

    with Dataset(args.file, "r") as topo:
        z = topo.variables[args.v][...]
    if args.scale is not None:
        z = smooth_image(z, args.scale)
    data = get_mask_data(z)
    save_image(data, args.out)
