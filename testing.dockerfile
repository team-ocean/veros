FROM veros/ubuntu:17.10
MAINTAINER Dion Häfner <mail@dionhaefner.de>

COPY . /veros
WORKDIR /veros
RUN ls -la /veros

RUN pip install -e .[test]
RUN pip3 install -e .[test]
