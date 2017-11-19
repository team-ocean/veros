#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
import os


def find_scripts(scriptdir):
    """scrape all available scripts from 'bin' folder"""
    return [os.path.join(scriptdir, s) for s in os.listdir(scriptdir) if not s.endswith(".pyc")]

setup(
    name = "veros",
    version = "0.0.1b0",
    packages = find_packages(),
    install_requires = [
        "numpy",
        "scipy",
        "netCDF4",
        "h5py",
        "pillow",
        "backports.lru_cache"
        ],
    author = "Dion Häfner (NBI Copenhagen)",
    author_email = "dion.haefner@nbi.ku.dk",
    scripts = find_scripts("bin")
)
