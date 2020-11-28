FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# because 'docker build' cannot call GPU, so you need to run like this:
# $ docker build -t superscaler -f Dockerfile.CUDA .
# run docker in interactive mode:
# $ docker run --it --runtime=nvidia superscaler bash
# or if you want to specify the GPUs to use:
# $ docker run --it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=2,3 superscaler bash


# python version is related to tensorflow version
ARG python=3.7
ENV PYTHON_VERSION=${python}

# install the dependencies
RUN apt-get update && apt-get install --no-install-recommends --allow-downgrades -y \
    build-essential \
    git \
    wget \
    openssh-client \
    openssh-server \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip

RUN wget -qO- "https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz" | \
  tar --strip-components=1 -xz -C /usr/local

# set softlink of python3 and python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# install openmpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
RUN tar xzvf openmpi-4.0.2.tar.gz

WORKDIR openmpi-4.0.2
RUN ./configure
RUN make -j10
RUN make install
RUN ldconfig
WORKDIR /

# check openmpi
RUN which mpicc
RUN mpicc -show
RUN which mpiexec
RUN mpiexec --version

# install nccl
RUN git clone https://github.com/NVIDIA/nccl.git
RUN cd nccl && make -j10 src.build

# Install python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install setuptools && \
    python3 -m pip install flake8 && \
    python3 -m pip install --no-cache-dir \
        tensorflow==1.15 \
        pytest==5.3.2 \
        protobuf==3.8 \
        setuptools==41.0.0 \
        bitmath==1.3.3.1 \
        humanreadable==0.1.0 \
        PyYAML==5.1.2

# Install horovod
RUN HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_NCCL_HOME=/nccl/build HOROVOD_CUDA_HOME=/usr/local/cuda-10.0/ python -m pip install --no-cache-dir horovod

# RUN git clong msrasrg@vs-ssh.visualstudio.com:v3/msrasrg/SuperScaler/SuperScaler
# RUN export PYTHONPATH=SuperScaler/src/

RUN git clone https://github.com/wjfjfm/horovod_profiling.git
WORKDIR /horovod_profiling

# # Install SuperScaler package
# RUN python3 -m pip install .