FROM python:3.10

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /dataclay/storage; \
    mkdir -p /dataclay/metadata

WORKDIR /usr/src