#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
from setuptools.extension import Extension

from codecs import open
import os
import re
import sys

try:
    from Cython.Build import cythonize
except ImportError:
    HAS_CYTHON = False
else:
    HAS_CYTHON = True

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)
import versioneer  # noqa: E402
import cuda_ext  # noqa: E402


CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

MINIMUM_VERSIONS = {
    'numpy': '1.13',
    'requests': '2.18',
}

EXTRAS_REQUIRE = {
    'test': [
        'pytest',
        'pytest-cov',
        'pytest-xdist',
        'codecov',
        'petsc4py',
        'mpi4py'
    ]
}

CONSOLE_SCRIPTS = [
    'veros = veros.cli.veros:cli',
    'veros-run = veros.cli.veros_run:cli',
    'veros-copy-setup = veros.cli.veros_copy_setup:cli',
    'veros-resubmit = veros.cli.veros_resubmit:cli',
    'veros-create-mask = veros.cli.veros_create_mask:cli'
]

PACKAGE_DATA = ['setup/*/assets.yml', 'setup/*/*.npy', 'setup/*/*.png']

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

INSTALL_REQUIRES = []
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        pkg = re.match(r'(\w+)\b.*', line).group(1)
        if pkg in MINIMUM_VERSIONS:
            line = ''.join([line, ',>=', MINIMUM_VERSIONS[pkg]])
        line = line.replace('==', '<=')
        INSTALL_REQUIRES.append(line)


def get_extensions():
    if not HAS_CYTHON:
        return []

    ext_ext = ('pyx', 'cu')
    extensions = []
    cuda_extensions = []

    cuda_paths = cuda_ext.cuda_paths

    for f in os.listdir(os.path.join(here, 'veros', 'core', 'special')):
        modname, file_ext = os.path.splitext(f)
        if file_ext not in ext_ext:
            continue

        ext = Extension(
            name='veros.core.special.{}'.format(modname),
            sources=['veros/core/special/{}.{}'.format(modname, file_ext)],
            language='c',
            optional=True,
            ibrary_dirs=[cuda_paths['lib64']],
            libraries=['cudart'],
            runtime_library_dirs=[cuda_paths['lib64']],
            # This syntax is specific to this build system
            # we're only going to use certain compiler args with nvcc
            # and not with gcc the implementation of this trick is in
            # customize_compiler()
            extra_compile_args={
                'gcc': [],
                'nvcc': [
                    '-gencode=arch=compute_75,code=compute_75', '--ptxas-options=-v', '-c',
                    '--compiler-options', "'-fPIC'"
                ]
            },
            include_dirs=[cuda_paths['include']]
        )

        extensions.append(ext)

    return cythonize(extensions, exclude_failures=True)


cmdclass = versioneer.get_cmdclass()
cmdclass.update(build_ext=cuda_ext.custom_build_ext)

setup(
    name='veros',
    license='MIT',
    author='Dion Häfner (NBI Copenhagen)',
    author_email='dion.haefner@nbi.ku.dk',
    keywords='oceanography python parallel numpy multi-core '
             'geophysics ocean-model mpi4py jax',
    description='The versatile ocean simulator, in pure Python, powered by JAX.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://veros.readthedocs.io',
    python_requires='>=3.6',
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    ext_modules=get_extensions(),
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
        'veros.setup_dirs': [
            'base = veros.setup'
        ]
    },
    package_data={
        'veros': PACKAGE_DATA
    },
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
    zip_safe=False,
)
