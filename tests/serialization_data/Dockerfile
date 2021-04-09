FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y \
    apt-utils \
    sudo \
    build-essential \
    gcc \
    g++ \
    gfortran \
    gdb \
    wget \
    curl \
    tar \
    git \
    vim \
    make \
    cmake \
    cmake-curses-gui \
    python3-pip \
    python3-dev \
    libssl-dev \
    libboost-all-dev \
    libnetcdf-dev \
    libnetcdff-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    update-alternatives  --set python /usr/bin/python3 && \
    update-alternatives  --set pip /usr/bin/pip3

# set TZ
ENV TZ=US/Pacific
RUN echo $TZ > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

# install some python packages
RUN pip install numpy
RUN pip install matplotlib

# install serialbox from source
RUN git clone -b v2.6.1 --depth 1 https://github.com/GridTools/serialbox.git /serialbox  && \
    cd /serialbox && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/serialbox -DCMAKE_BUILD_TYPE=Debug \
          -DSERIALBOX_USE_NETCDF=ON -DSERIALBOX_ENABLE_FORTRAN=ON \
          -DSERIALBOX_TESTING=ON  ../ && \
    make -j8 && \
    make test && \
    make install && \
    /bin/rm -rf /serialbox
COPY . /microph
RUN cd /microph && make
CMD ["/bin/bash"]
