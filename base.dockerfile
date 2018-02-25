FROM ubuntu:17.10

MAINTAINER Dion Häfner <mail@dionhaefner.de>

# Install dependencies
RUN apt-get update && apt-get install -y \
      'python-pip' \
      'python3-pip' \
      'python-mpi4py' \
      'python3-mpi4py' \
      'python-pyopencl' \
      'python3-pyopencl' \
      'locales' \
      'git' \
      'curl' \
      'gcc' \
      'gfortran' \
      'cmake' \
      'libnetcdf-dev' \
      'libopenmpi-dev' \
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
    python -c "import numpy; print(numpy.__version__)"

RUN pip3 install numpy -U && \
    mv /usr/local/bin/f2py /usr/local/bin/f2py3.6 && \
    python3 -c "import numpy; print(numpy.__version__)"

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install OpenCL
WORKDIR /tmp
ADD AMD-APP-SDK-linux-v2.9-1.599.381-GA-x64.tar.bz2 amd_src
ENV OPENCL_HOME "/opt/AMDAPPSDK-2.9-1"
ENV OPENCL_LIBPATH "/opt/AMDAPPSDK-2.9-1/lib/x86_64"
RUN sh amd_src/AMD-APP-SDK-v2.9-1.599.381-GA-linux64.sh -- -s -a yes && \
    rm -rf /tmp/amd_src
ENV OpenCL_LIBPATH "/opt/AMDAPPSDK-2.9-1/lib/x86_64/"
ENV OpenCL_INCPATH "/opt/AMDAPPSDK-2.9-1/include"
ENV LD_LIBRARY_PATH "$OpenCL_LIBPATH:$LD_LIBRARY_PATH"

# Build bohrium
WORKDIR /tmp
ADD https://github.com/bh107/bohrium/archive/master.zip bohrium-master.zip
RUN unzip bohrium-master.zip && \
    mkdir -p /tmp/bohrium-master/build && \
    cd /tmp/bohrium-master/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DEXT_VISUALIZER=OFF && \
    make -j 4 > /dev/null && \
    make install > /dev/null && \
    rm -rf /tmp/bohrium-master /tmp/bohrium-master.zip

RUN ln -s /usr/lib/python2.7/site-packages/bohrium /usr/lib/python2.7/dist-packages/ && \
    python2.7 -c "import bohrium"

RUN ln -s /usr/lib/python3.6/site-packages/bohrium /usr/lib/python3/dist-packages/ && \
    python3.6 -c "import bohrium"

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
