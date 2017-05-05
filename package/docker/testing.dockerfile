FROM veros/ubuntu:16.04
MAINTAINER Dion Häfner <mail@dionhaefner.de>

# Install Veros
RUN mkdir -p /veros
WORKDIR /veros
ADD . .
RUN ls /veros
RUN pip install -e .
RUN pip3 install -e .
