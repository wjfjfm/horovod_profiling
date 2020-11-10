FROM ubuntu:18.04

# Install build environment
RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y software-properties-common

RUN \
    apt-get update && \
    apt-get install -y python3-dev python3-pip wget && \
    python3 -m pip install --upgrade pip 

# RUN \
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     bash ~/miniconda.sh -b -p $HOME/miniconda && \
#     bash ~/miniconda/bin/activate
# RUN \
#     conda init &&\
#     conda install tensorflow-gpu=1.15


# Create workspace
WORKDIR /Profiling
COPY . /Profiling

# install OpenMPI
RUN \
    apt-get install -y openmpi-bin openmpi-doc libopenmpi-dev openssh-client


RUN \
    python3 -m pip install tensorflow==1.15 &&\ 
    python3 -m pip install -r requirements.txt
    

 # Run pytest
RUN \
    python3 -m pytest -s
