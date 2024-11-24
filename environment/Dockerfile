# hash:sha256:a8db9e2833598fc3dcc49e402975cfd470d267af7269f8488167ce106803c6a8
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    h5py==3.2.1 \
    numba==0.57.1 \
    numpy==1.24.4 \
    pytreegrav==1.1 \
    scipy==1.6.1
