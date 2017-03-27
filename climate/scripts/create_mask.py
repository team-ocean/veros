#!/usr/bin/env python

from netCDF4 import Dataset
from PIL import Image
import numpy as np
import argparse

def get_mask_data(depth):
    return np.where(depth > 0, 255, 0).astype(np.uint8)

def save_image(data, path):
    Image.fromarray(np.flipud(data)).convert("1").save(path + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Input file holding topography information", required=True)
    parser.add_argument("-o", "--out", help="", default="topography", required=False)
    args = parser.parse_args()

    with Dataset(args.file, "r") as topo:
        x, y, z = (topo.variables[k][...] for k in ("x","y","z"))
    data = get_mask_data(z)
    save_image(data, args.out)
