FROM veros/ubuntu:18.04
MAINTAINER Dion Häfner <mail@dionhaefner.de>

COPY . /veros
WORKDIR /veros
RUN ls -la /veros

# RUN apt-get update && apt-get purge -y python-petsc4py python3-petsc4py
RUN pip install "petsc4py>=3.11.0,<3.12.0" -U && \
    pip3 install "petsc4py>=3.11.0,<3.12.0" -U

RUN pip install -e .[test]
RUN pip3 install -e .[test]
