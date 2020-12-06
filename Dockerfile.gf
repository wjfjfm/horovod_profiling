FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Python 3.7 is supported by Ubuntu Bionic out of the box
ARG python=3.7
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install future typing packaging
RUN pip install tensorflow==1.15 \
                keras \
                h5py

# Install MPI.
RUN wget --progress=dot:mega -O /tmp/openmpi-3.0.0-bin.tar.gz https://github.com/horovod/horovod/files/1596799/openmpi-3.0.0-bin.tar.gz && \
            cd /usr/local && tar -zxf /tmp/openmpi-3.0.0-bin.tar.gz && ldconfig && \
            echo "mpirun -allow-run-as-root -np 2 -H localhost:2 -bind-to none -map-by slot -mca mpi_abort_print_stack 1" > /mpirun_command;

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Install nccl

RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j8 src.build

# Install Horovod, temporarily using CUDA stubs
RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_NCCL_HOME=/nccl/build HOROVOD_CUDA_HOME=/usr/local/cuda/ \
    python -m pip install --no-cache-dir horovod && \
    ldconfig

# Install infinibands
RUN apt-get install -y --no-install-recommends rdma-core ibverbs-utils libtool m4 automake libibverbs-dev librdmacm-dev libibumad-dev net-tools

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# download perttest
RUN git clone https://github.com/linux-rdma/perftest.git && \
    cd perftest && \
    ./autogen.sh && ./configure CUDA_H_PATH=/usr/local/cuda/include/cuda.h && make -j 

# download nccl-tests
RUN git clone https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make MPI=1 -j8

# download experiment repo
RUN git clone https://github.com/wjfjfm/horovod_profiling.git