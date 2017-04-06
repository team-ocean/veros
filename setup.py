#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

setup(
    name = "veros",
    version = "0.0.1b0",
    packages = find_packages(),
    install_requires = [
        "numpy<1.12",
        "scipy",
        "pyamg",
        "netCDF4",
        ],
    author = "NBI Copenhagen",
    author_email = "dion.haefner@nbi.ku.dk",
    scripts = [
	   "veros/scripts/create_mask.py",
	],
)
