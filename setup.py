#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
import versioneer


CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

EXTRAS_REQUIRE = {
    "fast": ["bohrium", "pyamg"],
    "bohrium": ["bohrium"],
    "gpu": ["bohrium", "pyopencl"]
}
EXTRAS_REQUIRE["all"] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name="veros",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    install_requires=[
        "click",
        "numpy>=1.13",
        "scipy",
        "netCDF4",
        "h5py",
        "pillow",
        "backports.functools_lru_cache",
        "future"
    ],
    extras_require=EXTRAS_REQUIRE,
    author="Dion Häfner (NBI Copenhagen)",
    author_email="mail@dionhaefner.de",
    entry_points={
        "console_scripts": ["veros = veros.cli:veros"]
    },
    license="MIT",
    classifiers=[c for c in CLASSIFIERS.split("\n") if c],
    keywords="oceanography python parallel numpy multi-core "
             "geophysics ocean-model bohrium",
    long_description=open("README.rst").read(),
)
