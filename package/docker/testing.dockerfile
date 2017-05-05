FROM veros/ubuntu:16.04
MAINTAINER Dion Häfner <mail@dionhaefner.de>

# Install testing dependencies
RUN apt-get update > /dev/null && apt-get install -y python-mpi4py python3-mpi4py

# Install Veros
RUN mkdir -p /veros
WORKDIR /veros
ADD . .
RUN ls /veros
RUN pip install -e .
RUN pip3 install -e .
