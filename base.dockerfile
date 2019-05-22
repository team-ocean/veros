FROM ubuntu:18.04

MAINTAINER Dion Häfner <mail@dionhaefner.de>

# Install dependencies
RUN apt-get update && apt-get install -y \
      'python-pip' \
      'python3-pip' \
      'python-virtualenv' \
      'python3-virtualenv' \
      'locales' \
      'git' \
      'curl' \
      'gcc' \
      'gfortran' \
      'cmake' \
      'libopenmpi-dev' \
      'libsigsegv-dev' \
      'libboost-serialization-dev' \
      'libboost-system-dev' \
      'libboost-filesystem-dev' \
      'libboost-thread-dev' \
      'libboost-regex-dev' \
      'libopenblas-dev' \
      'libgl1-mesa-dev' \
      'libffi-dev' \
      'swig' \
      'cython' \
      'cython3' \
      'unzip' \
      'lsb-release' \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy -U && \
    mv /usr/local/bin/f2py /usr/local/bin/f2py2.7 && \
    python -c "import numpy; print(numpy.__version__)" && \
    pip3 install numpy -U && \
    mv /usr/local/bin/f2py /usr/local/bin/f2py3.6 && \
    python3 -c "import numpy; print(numpy.__version__)"

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install mpi4py
RUN pip install mpi4py --no-binary mpi4py -U && \
    python -c "from mpi4py import MPI" && \
    pip3 install mpi4py --no-binary mpi4py -U && \
    python3 -c "from mpi4py import MPI"

# Build PETSc
ENV PETSC_ARCH=arch-linux2-c-opt
ENV PETSC_DIR="/tmp/petsc-3.11.1"

WORKDIR /tmp
RUN curl -L http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.11.1.tar.gz > /tmp/petsc.tar.gz && \
    tar xzf petsc.tar.gz && \
    rm petsc.tar.gz
WORKDIR /tmp/petsc-3.11.1
RUN ./configure \
      --prefix=/usr/local \
      --download-hypre && \
    make -j 4 && \
    make install

ENV PETSC_DIR="/usr/local"

RUN pip install "petsc4py>=3.11.0,<3.12.0" -U && \
    pip3 install "petsc4py>=3.11.0,<3.12.0" -U

# Install OpenCL
RUN apt-get update && apt-get install -y \
    'alien' \
    'opencl-dev' \
    'opencl-headers' \
    'clinfo' && \
  rm -rf /var/lib/apt/lists/*

ARG INTEL_DRIVER=opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz
ARG INTEL_DRIVER_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/9019

WORKDIR /tmp/opencl-driver-intel
RUN echo INTEL_DRIVER is $INTEL_DRIVER && \
    curl -O $INTEL_DRIVER_URL/$INTEL_DRIVER && \
    tar -xzf $INTEL_DRIVER && \
    for i in $(basename $INTEL_DRIVER .tgz)/rpm/*.rpm; do alien --to-deb $i; done && \
    dpkg -i *.deb && \
    rm -rf $INTEL_DRIVER $(basename $INTEL_DRIVER .tgz) *.deb && \
    mkdir -p /etc/OpenCL/vendors && \
    echo /opt/intel/*/lib64/libintelocl.so > /etc/OpenCL/vendors/intel.icd && \
    rm -rf /tmp/opencl-driver-intel && \
    clinfo

# Build bohrium
WORKDIR /tmp
ADD https://github.com/bh107/bohrium/archive/master.zip bohrium-master.zip
RUN unzip bohrium-master.zip && \
    mkdir -p /tmp/bohrium-master/build && \
    cd /tmp/bohrium-master/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DEXT_VISUALIZER=OFF -DPY_EXE_LIST="python2.7;python3.6" && \
    make -j 4 > /dev/null && \
    make install > /dev/null && \
    rm -rf /tmp/bohrium-master /tmp/bohrium-master.zip

ENV BH_CONFIG=/usr/etc/bohrium/config.ini

RUN ln -s /usr/lib/python2.7/site-packages/bohrium /usr/lib/python2.7/dist-packages/ && \
    ln -s /usr/lib/python2.7/site-packages/bohrium_api /usr/lib/python2.7/dist-packages/ && \
    python2.7 -m bohrium_api --info && \
    BH_STACK=opencl python2.7 -m bohrium_api --info

RUN ln -s /usr/lib/python3.6/site-packages/bohrium /usr/lib/python3/dist-packages/ && \
    ln -s /usr/lib/python3.6/site-packages/bohrium_api /usr/lib/python3/dist-packages/ && \
    python3.6 -m bohrium_api --info && \
    BH_STACK=opencl python3.6 -m bohrium_api --info && \
    BH_STACK=opencl python3.6 -c "import bohrium as bh; print(bh.random.rand(100, 100).sum())"

# Build pyOM2 with Python 2 and Python 3 support
RUN mkdir -p /tmp/pyOM2
COPY vendor/pyom2/pyOM2.1.0.tar.gz /tmp/pyOM2
COPY vendor/pyom2/pyOM2_site_specific /tmp/pyOM2/site_specific.mk_

WORKDIR /tmp/pyOM2
RUN tar xzf pyOM2.1.0.tar.gz
WORKDIR /tmp/pyOM2/py_src
RUN mv Makefile Makefile.template

RUN sed s/f2py/f2py2.7/g Makefile.template > Makefile && \
    make -j 4 && \
    ls -l && \
    mv pyOM_code.so /usr/local/lib/pyOM_code_py2.so && \
    mv pyOM_code_MPI.so /usr/local/lib/pyOM_code_MPI_py2.so && \
    make clean

RUN sed s/f2py/f2py3.6/g Makefile.template > Makefile && \
    make -j 4 > /dev/null && \
    ls -l && \
    mv pyOM_code.cpython-36m-x86_64-linux-gnu.so /usr/local/lib/pyOM_code_py3.so && \
    mv pyOM_code_MPI.cpython-36m-x86_64-linux-gnu.so /usr/local/lib/pyOM_code_MPI_py3.so && \
    rm -rf /tmp/pyOM2

WORKDIR /veros
