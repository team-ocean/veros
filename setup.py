#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
from setuptools.extension import Extension

from codecs import open
import os
import re
import sys

from Cython.Build import cythonize

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)
import versioneer  # noqa: E402
import cuda_ext  # noqa: E402


CLASSIFIERS = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

MINIMUM_VERSIONS = {
    "numpy": "1.13",
    "requests": "2.18",
    "jax": "0.2.10",
}


CONSOLE_SCRIPTS = [
    "veros = veros.cli.veros:cli",
    "veros-run = veros.cli.veros_run:cli",
    "veros-copy-setup = veros.cli.veros_copy_setup:cli",
    "veros-resubmit = veros.cli.veros_resubmit:cli",
    "veros-create-mask = veros.cli.veros_create_mask:cli",
]

PACKAGE_DATA = ["setups/*/assets.json", "setups/*/*.npy", "setups/*/*.png"]

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def parse_requirements(reqfile):
    requirements = []

    with open(os.path.join(here, reqfile), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            pkg = re.match(r"(\w+)\b.*", line).group(1)
            if pkg in MINIMUM_VERSIONS:
                line = "".join([line, ",>=", MINIMUM_VERSIONS[pkg]])
            line = line.replace("==", "<=")
            requirements.append(line)

    return requirements


INSTALL_REQUIRES = parse_requirements("requirements.txt")

EXTRAS_REQUIRE = {
    "test": ["pytest", "pytest-cov", "pytest-forked", "codecov", "xarray"],
}
EXTRAS_REQUIRE["jax"] = parse_requirements("requirements_jax.txt")


def get_extensions():
    cuda_info = cuda_ext.cuda_info

    extension_modules = {
        "veros.core.special.tdma_cython_": ["tdma_cython_.pyx"],
        "veros.core.special.tdma_cuda_": ["tdma_cuda_.pyx", "cuda_tdma_kernels.cu"],
    }

    extensions = []
    for module, sources in extension_modules.items():
        extension_dir = os.path.join(*module.split(".")[:-1])

        kwargs = dict()
        if any(source.endswith(".cu") for source in sources):
            # skip GPU extension if CUDA is not available
            if cuda_info["cuda_root"] is None:
                continue

            kwargs.update(
                library_dirs=cuda_info["lib64"],
                libraries=["cudart"],
                runtime_library_dirs=cuda_info["lib64"],
                include_dirs=cuda_info["include"],
            )

        ext = Extension(
            name=module,
            sources=[os.path.join(extension_dir, f) for f in sources],
            extra_compile_args={
                "gcc": [],
                "nvcc": cuda_info["cflags"],
            },
            **kwargs,
        )

        extensions.append(ext)

    return cythonize(extensions, language_level=3, exclude_failures=True)


cmdclass = versioneer.get_cmdclass()
cmdclass.update(build_ext=cuda_ext.custom_build_ext)


setup(
    name="veros",
    license="MIT",
    author="Dion Häfner (NBI Copenhagen)",
    author_email="dion.haefner@nbi.ku.dk",
    keywords="oceanography python parallel numpy multi-core geophysics ocean-model mpi4py jax",
    description="The versatile ocean simulator, in pure Python, powered by JAX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://veros.readthedocs.io",
    python_requires=">=3.6",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    ext_modules=get_extensions(),
    entry_points={"console_scripts": CONSOLE_SCRIPTS, "veros.setup_dirs": ["base = veros.setups"]},
    package_data={"veros": PACKAGE_DATA},
    classifiers=[c for c in CLASSIFIERS.split("\n") if c],
    zip_safe=False,
)
