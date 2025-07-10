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
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3.12
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
            parsed = re.match(r"(\w+)(.*?)(;.*)?$", line)
            pkg, deps, extra = parsed.groups()
            if extra is None:
                extra = ""
            deps = deps.replace("==", "<=")
            if pkg in MINIMUM_VERSIONS:
                deps = f"{deps},>={MINIMUM_VERSIONS[pkg]}"
            line = "".join([pkg, deps, extra])
            requirements.append(line)

    return requirements


INSTALL_REQUIRES = parse_requirements("requirements.txt")

jax_req = parse_requirements("requirements_jax.txt")
for line in jax_req:  # inject jaxlib requirement
    if line.startswith("jax"):
        jax_req.append(line.replace("jax", "jaxlib"))
        break

EXTRAS_REQUIRE = {
    "test": ["pytest", "pytest-cov", "pytest-forked", "xarray"],
    "jax": jax_req,
}


def get_extensions(require_cython_ext, require_cuda_ext):
    cuda_info = cuda_ext.cuda_info

    extension_modules = {
        "veros.core.special.tdma_cython_": ["tdma_cython_.pyx"],
        "veros.core.special.tdma_cuda_": ["tdma_cuda_.pyx", "cuda_tdma_kernels.cu"],
    }

    def is_cuda_ext(sources):
        return any(source.endswith(".cu") for source in sources)

    extensions = []
    for module, sources in extension_modules.items():
        extension_dir = os.path.join(*module.split(".")[:-1])

        kwargs = dict()
        if is_cuda_ext(sources):
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

    extensions = cythonize(extensions, language_level=3, exclude_failures=True)

    for ext in extensions:
        is_required = (not is_cuda_ext(ext.sources) and require_cython_ext) or (
            is_cuda_ext(ext.sources) and require_cuda_ext
        )

        if not is_required:
            ext.optional = True

    return extensions


cmdclass = versioneer.get_cmdclass()
build_ext = type("custom_build_ext", (cuda_ext.custom_build_ext, cmdclass["build_ext"]), {})
cmdclass.update(build_ext=build_ext)


def _env_to_bool(envvar):
    return os.environ.get(envvar, "").lower() in ("1", "true", "on")


extensions = get_extensions(
    require_cython_ext=_env_to_bool("VEROS_REQUIRE_CYTHON_EXT"),
    require_cuda_ext=_env_to_bool("VEROS_REQUIRE_CUDA_EXT"),
)

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
    python_requires=">=3.10",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    ext_modules=extensions,
    entry_points={"console_scripts": CONSOLE_SCRIPTS, "veros.setup_dirs": ["base = veros.setups"]},
    package_data={"veros": PACKAGE_DATA},
    classifiers=[c for c in CLASSIFIERS.split("\n") if c],
    zip_safe=False,
)
