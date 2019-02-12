FROM veros/ubuntu:18.04
MAINTAINER Dion Häfner <mail@dionhaefner.de>

COPY . /veros
WORKDIR /veros
RUN ls -la /veros

RUN pip install -e .[test] -U
RUN pip3 install -e .[test] -U
