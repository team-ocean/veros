#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
import os


def find_scripts(scriptdir):
    return [os.path.join(scriptdir, s) for s in os.listdir(scriptdir)]

setup(
    name = "veros",
    version = "0.0.1b0",
    packages = find_packages(),
    install_requires = [
        "numpy<1.12",
        "scipy",
        "pyamg",
        "netCDF4",
        "h5py",
        ],
    author = "NBI Copenhagen",
    author_email = "dion.haefner@nbi.ku.dk",
    scripts = find_scripts("bin")
)
