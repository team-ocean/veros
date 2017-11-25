#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
import os


def find_scripts(scriptdir):
    """scrape all available scripts from 'bin' folder"""
    items_in_scriptdir = map(lambda x: os.path.join(scriptdir, x), os.listdir(scriptdir))
    return [s for s in items_in_scriptdir if os.path.isfile(s) and not s.endswith(".pyc")]

setup(
    name = "veros",
    version = "0.0.1b0",
    packages = find_packages(),
    install_requires = [
        "numpy>=1.13",
        "scipy",
        "netCDF4",
        "h5py",
        "pillow",
        "backports.functools_lru_cache",
        'future'
        ],
    author = "Dion Häfner (NBI Copenhagen)",
    author_email = "mail@dionhaefner.de",
    scripts = find_scripts("bin")
)
