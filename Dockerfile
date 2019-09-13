FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
RUN apt-get update
RUN apt install -y wget unzip
RUN apt install -y build-essential cmake
RUN apt install -y vim
RUN apt install -y gdb
RUN apt install -y python3
RUN apt install -y git
RUN mkdir /pytorch
WORKDIR /pytorch
RUN git clone --recursive https://github.com/pytorch/pytorch /pytorch
RUN apt install -y libomp-dev
RUN apt install -y python3-setuptools python3-yaml
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
RUN apt-get update
RUN apt install -y intel-mkl-2018.2-046
ENV TORCH_CUDA_ARCH_LIST=3.5;5.0+PTX;6.0;6.1;7.0
RUN apt install -y python-dev python3-dev python python-setuptools python-yaml
RUN apt install -y valgrind
RUN python3 setup.py build
RUN apt install -y python3-pip
RUN pip3 install mysql-connector-python requests
RUN apt install -y scons
RUN apt install -y zlib1g-dev
RUN apt-get update
RUN apt install -y libsdl2-dev
RUN apt install -y libsdl-dev
RUN apt install -y libgtk2.0-dev
