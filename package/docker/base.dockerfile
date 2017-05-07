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
RUN git clone https://github.com/bh107/bohrium.git /bohrium/bohrium-master
RUN mkdir -p /bohrium/bohrium-master/build
WORKDIR /bohrium/bohrium-master/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DEXT_VISUALIZER=OFF
RUN make > /dev/null && make install > /dev/null
RUN ln -s /usr/lib/python2.7/site-packages/bohrium /usr/lib/python2.7/dist-packages/ && \
    python2.7 -c "import bohrium"
RUN ln -s /usr/lib/python3.5/site-packages/bohrium /usr/lib/python3/dist-packages/ && \
    python3.5 -c "import bohrium"

# Build pyOM2 with Python 2 and Python 3 support
RUN mkdir /pyOM2
WORKDIR /pyOM2
RUN curl -L -o pyOM2.1.0.tar.gz "https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2?action=AttachFile&do=get&target=pyOM2.1.0.tar.gz" > /dev/null
RUN tar xzf pyOM2.1.0.tar.gz
ADD pyOM2_site_specific site_specific.mk_
WORKDIR /pyOM2/py_src

RUN make > /dev/null
RUN mv pyOM_code.so pyOM_code_py2.so && \
    mv pyOM_code_MPI.so pyOM_code_MPI_py2.so

RUN sed -i.py2 s/f2py/f2py3/g Makefile && make > /dev/null
RUN mv pyOM_code.cpython-35m-x86_64-linux-gnu.so pyOM_code_py3.so && \
    mv pyOM_code_MPI.cpython-35m-x86_64-linux-gnu.so pyOM_code_MPI_py3.so

RUN ls -l /pyOM2/py_src

RUN mkdir -p /veros
WORKDIR /veros
