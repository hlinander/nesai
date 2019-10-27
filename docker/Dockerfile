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
RUN apt install -y libsfml-dev
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y liblua5.3-dev
RUN apt install -y liblua5.3
RUN apt install -y lua5.3
RUN apt install -y libsdl2-dev
RUN apt install -y libsdl2-ttf-dev
RUN apt install -y curl
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt install -y nodejs
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN echo 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' > /etc/apt/sources.list.d/rstuff.list
RUN apt update && apt install -y r-base
RUN echo "install.packages('rjson')" | R --no-save
RUN echo "install.packages('ggplot2')" | R --no-save
RUN echo "install.packages('gridExtra')" | R --no-save
RUN echo "install.packages('reshape2')" | R --no-save
RUN echo "install.packages('ggridges')" | R --no-save
RUN echo "install.packages('qpcR')" | R --no-save
ENV CPATH=/pytorch/torch/include/:/pytorch/torch/include/torch/csrc/api/include/
ENV LIBRARY_PATH=/pytorch/torch/lib:/usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/intel64_lin/:/pytorch/torch/lib
