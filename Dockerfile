ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.6.3

# Builder stage: installs needed dev packages and compiles dependencies
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

ARG CUDA_ARCHITECTURES=native
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages & dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        cmake \
        ninja-build build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libgflags-dev \
        libatlas-base-dev \
        libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install Ceres, Pyceres, Colmap, Pycolmap
RUN git clone --branch 2.2.0 --depth 1 https://ceres-solver.googlesource.com/ceres-solver /ceres-solver && \
    mkdir /ceres-solver/build && cd /ceres-solver/build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja install
RUN git clone --depth 1 https://github.com/cvg/pyceres.git /pyceres && \
    python3 -m pip install /pyceres
RUN python3 -m pip install ruff
ARG COLMAP_GIT_COMMIT=main
RUN git clone --depth 1 https://github.com/Zador-Pataki/colmap.git /colmap && \
    cd /colmap && git checkout ${COLMAP_GIT_COMMIT} && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja install && \
    python3 -m pip install /colmap

# Clean up dev files to reduce image size
RUN rm -rf \
    /colmap \
    /ceres-solver \
    /pyceres \
    /root/.cache \
    /usr/local/include \
    /usr/local/lib/cmake \
    /usr/local/share \
    /usr/local/lib/*.a \
    /usr/local/lib/*.la

# Runtime stage: minimal runtime dependencies
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime
ENV DEBIAN_FRONTEND=noninteractive

# Install only what's needed at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        python-is-python3 \
        git \
        wget \
        libboost-program-options-dev \
        libatlas-base-dev \
        libceres-dev \
        libfreeimage-dev \
        libglew-dev \
        libgoogle-glog-dev \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        libcurl4 \
        ninja-build \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled artifacts from builder
COPY --from=builder /usr/local/ /usr/local/
ENV PATH=/usr/local/bin:$PATH

WORKDIR /workspace/PriMo
# Install Python requirements & finalize
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    grep -v 'ml-depth-pro' requirements.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt && \
    rm -rf /root/.cache

# Final entrypoint
ENTRYPOINT ["bash"]