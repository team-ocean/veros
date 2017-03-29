#!/usr/bin/env python

from netCDF4 import Dataset
from PIL import Image
import numpy as np
import argparse
from scipy import ndimage

def get_mask_data(depth):
    return np.where(depth > 0, 255, 0).astype(np.uint8)

def smooth_image(data, sigma):
    return ndimage.gaussian_filter(data, sigma=sigma)

def save_image(data, path):
    Image.fromarray(np.flipud(data)).convert("1").save(path + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Input file holding topography information", required=True)
    parser.add_argument("-o", "--out", help="", default="topography", required=False)
    parser.add_argument("-s", "--scale", nargs=2, type=int, required=False, default=None)
    args = parser.parse_args()

    with Dataset(args.file, "r") as topo:
        z = topo.variables["z"][...]
    if not args.scale is None:
        z = smooth_image(z, args.scale)
    data = get_mask_data(z)
    save_image(data, args.out)
