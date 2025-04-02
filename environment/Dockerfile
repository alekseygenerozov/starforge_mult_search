# hash:sha256:15bd3fc5a8f8ffdefbdad805ffd2edf8743487c18fb9b441c96584fd6b01fee5
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    astropy==6.0.1 \
    h5py==3.2.1 \
    hydra-core==1.3.2 \
    meshoid==1.46.1 \
    numba==0.57.1 \
    numpy==1.24.4 \
    pandas==2.2.3 \
    pyarrow==19.0.1 \
    pytreegrav==1.1 \
    scipy==1.6.1 \
    seaborn==0.13.2
