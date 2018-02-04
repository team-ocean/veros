FROM bohrium/ubuntu:16.04

MAINTAINER Dion Häfner <mail@dionhaefner.de>

# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install dependencies
RUN apt-get update > /dev/null && apt-get install -y \
  python-pip python3-pip \
  git curl libopenmpi-dev > /dev/null

# Build bohrium
RUN git clone https://github.com/bh107/bohrium.git /tmp/bohrium-master && \
    mkdir -p /tmp/bohrium-master/build && \
    cd /tmp/bohrium-master/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DEXT_VISUALIZER=OFF && \
    make > /dev/null && make install > /dev/null && \
    rm -rf /tmp/bohrium-master

RUN ln -s /usr/lib/python2.7/site-packages/bohrium /usr/lib/python2.7/dist-packages/ && \
    python2.7 -c "import bohrium"

RUN ln -s /usr/lib/python3.5/site-packages/bohrium /usr/lib/python3/dist-packages/ && \
    python3.5 -c "import bohrium"

# Build pyOM2 with Python 2 and Python 3 support
RUN mkdir -p /tmp/pyOM2
COPY vendor/pyom2/pyOM2.1.0.tar.gz /tmp/pyOM2
COPY vendor/pyom2/pyOM2_site_specific /tmp/pyOM2/site_specific.mk_

WORKDIR /pyOM2
RUN tar xzf pyOM2.1.0.tar.gz && \
    cd /tmp/pyOM2/py_src && \
    make > /dev/null && \
    mv pyOM_code.so /usr/local/lib/pyOM_code_py2.so && \
    mv pyOM_code_MPI.so /usr/local/lib/pyOM_code_MPI_py2.so && \
    make clean && \
    cd /tmp/pyOM2/py_src && \
    sed -i.py2 s/f2py/f2py3/g Makefile && make > /dev/null && \
    mv pyOM_code.cpython-35m-x86_64-linux-gnu.so /usr/local/lib/pyOM_code_py3.so && \
    mv pyOM_code_MPI.cpython-35m-x86_64-linux-gnu.so /usr/local/lib/pyOM_code_MPI_py3.so && \
    rm -rf /tmp/pyOM2

# Install optional dependencies
RUN pip install pyopencl && \
    pip3 install pyopencl

RUN mkdir -p /veros
WORKDIR /veros
