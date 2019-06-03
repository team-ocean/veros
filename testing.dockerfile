FROM veros/ubuntu:18.04
MAINTAINER Dion Häfner <mail@dionhaefner.de>

COPY . /veros
WORKDIR /veros
RUN ls -la /veros

# avoid precision differences because of Intel's use of 80-bit floats internally
ENV BH_OPENMP_VOLATILE=true
ENV BH_OPENCL_VOLATILE=true
ENV BH_CUDA_VOLATILE=true

RUN echo "petsc4py>=3.11.0,<3.12.0" > /tmp/constraints && \
    pip3 install -U -e .[test] -c /tmp/constraints && \
    rm /tmp/constraints
