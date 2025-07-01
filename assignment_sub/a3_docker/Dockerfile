FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and system deps
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3.8-venv python3-pip \
    libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev \
    libgl1-mesa-dev libglfw3-dev patchelf unzip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Use Python 3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip + install packages
RUN pip install --upgrade pip setuptools wheel

# Install Python packages
RUN pip install \
    "cython<3" \
    "gym==0.17.2" \
    "stable-baselines3==2.3.0" \
    "shimmy>=0.2.1" \
    "mujoco-py<2.2,>=2.1" \
    moviepy numpy torch matplotlib scipy tqdm

# Make mujoco path (will be bind-mounted)
RUN mkdir -p /root/.mujoco

# Set proper MuJoCo version
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV OMP_NUM_THREADS=1

CMD ["/bin/bash"]
