# =========================================================================================
# FINAL & DEFINITIVE DOCKERFILE FOR ROBOVERSE (SERVER & LOCAL)
# This version resolves the dependency conflict by installing dependencies step by step
# to avoid torch version conflicts between different extras.
# Updated to include Xvfb and OpenGL setup for headless operation.
# =========================================================================================

# 1. Base Image: Start with the official, pre-configured Isaac Sim image.
FROM nvcr.io/nvidia/isaac-sim:4.2.0

# 2. Build Arguments: For passing secrets and configurations securely.
ARG NVIDIA_API_KEY

# 3. User and Permissions Setup
ARG DOCKER_UID=1000
ARG DOCKER_GID=1000
ARG DOCKER_USER=user
ARG HOME=/home/${DOCKER_USER}

USER root
# Create/modify user and group, then change ownership of /isaac-sim for write access.
RUN EXISTING_GROUP=$(getent group $DOCKER_GID | cut -d: -f1) && \
    EXISTING_USER=$(getent passwd $DOCKER_UID | cut -d: -f1) && \
    (if [ "$EXISTING_GROUP" != "" ] && [ "$EXISTING_GROUP" != "$DOCKER_USER" ]; then groupmod -n $DOCKER_USER $EXISTING_GROUP; else groupadd -g $DOCKER_GID $DOCKER_USER; fi) && \
    (if [ "$EXISTING_USER" != "" ] && [ "$EXISTING_USER" != "$DOCKER_USER" ]; then usermod -l $DOCKER_USER -d $HOME -m $EXISTING_USER; else useradd --uid $DOCKER_UID --gid $DOCKER_GID -m $DOCKER_USER; fi) && \
    echo "$DOCKER_USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    chown -R $DOCKER_USER:$DOCKER_USER /isaac-sim

ENV DEBIAN_FRONTEND=noninteractive

# 4. System Dependencies: Install all necessary tools including CUDA development tools and Xvfb.
RUN apt-get update && apt-get install -y -o Dpkg::Options::="--force-confold" --no-install-recommends \
    build-essential cmake git wget ssh x11-apps mesa-utils ninja-build vulkan-tools libglu1 libglib2.0-0 libxrandr2 sudo \
    xvfb x11vnc fluxbox wmctrl

# Install CUDA 11.8 toolkit for curobo compilation
RUN cd /tmp \
    && wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run \
    && sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override \
    && rm cuda_11.8.0_520.61.05_linux.run

# Set CUDA environment variables
ENV PATH=/usr/local/cuda-11.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda-11.8

# Switch to the non-root user for the rest of the build.
USER ${DOCKER_USER}
WORKDIR ${HOME}

SHELL ["/bin/bash", "-c"]

# 5. Install Conda
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p ${HOME}/conda \
    && rm "Miniforge3-$(uname)-$(uname -m).sh"
ENV PATH=${HOME}/conda/bin:${PATH}
RUN conda init bash && mamba shell init --shell bash

# 6. Install uv for faster package management
RUN pip install uv

# 7. Copy RoboVerse Source Code
COPY --chown=${DOCKER_USER} . ${HOME}/RoboVerse

WORKDIR ${HOME}/RoboVerse

# 7. Create Conda Env and Install Dependencies
RUN mamba create -n metasim python=3.10 -y && mamba clean -a -y
RUN echo "mamba activate metasim" >> ${HOME}/.bashrc

# 7.1. Install RoboVerse base dependencies first (without extras)
RUN eval "$(mamba shell hook --shell bash)" && mamba activate metasim \
    && pip install --extra-index-url "https://$oauthtoken:${NVIDIA_API_KEY}@pypi.nvidia.com" -e .

# 7.2. Install PyTorch with CUDA support first from PyTorch index
RUN eval "$(mamba shell hook --shell bash)" && mamba activate metasim \
    && pip install --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.4.0+cu118 torchvision==0.19.0+cu118

# 7.3. Install isaaclab extra (PyTorch will be skipped as already installed)
RUN eval "$(mamba shell hook --shell bash)" && mamba activate metasim \
    && pip install --extra-index-url "https://$oauthtoken:${NVIDIA_API_KEY}@pypi.nvidia.com" -e ".[isaaclab]"

# 7.4. Install MuJoCo support
RUN eval "$(mamba shell hook --shell bash)" && mamba activate metasim \
    && pip install --extra-index-url "https://$oauthtoken:${NVIDIA_API_KEY}@pypi.nvidia.com" -e ".[mujoco]"

# 7.5. Install Genesis support
RUN eval "$(mamba shell hook --shell bash)" && mamba activate metasim \
    && pip install --extra-index-url "https://$oauthtoken:${NVIDIA_API_KEY}@pypi.nvidia.com" -e ".[genesis]"

# 7.6. Install Sapien3 support
RUN eval "$(mamba shell hook --shell bash)" && mamba activate metasim \
    && pip install --extra-index-url "https://$oauthtoken:${NVIDIA_API_KEY}@pypi.nvidia.com" -e ".[sapien3]"

# 7.7. Install PyBullet support
RUN eval "$(mamba shell hook --shell bash)" && mamba activate metasim \
    && pip install --extra-index-url "https://$oauthtoken:${NVIDIA_API_KEY}@pypi.nvidia.com" -e ".[pybullet]"

# 8. Install IsaacLab Extensions
# This script will now run in an environment that already has the correct torch version.
RUN mkdir -p ${HOME}/packages \
    && cd ${HOME}/packages \
    && eval "$(mamba shell hook --shell bash)" \
    && mamba activate metasim \
    && git clone --depth 1 --branch v1.4.1 https://github.com/isaac-sim/IsaacLab.git IsaacLab \
    && ln -s /isaac-sim ./IsaacLab/_isaac_sim \
    && cd IsaacLab \
    && sed -i '/^EXTRAS_REQUIRE = {$/,/^}$/c\EXTRAS_REQUIRE = {\n    "sb3": [],\n    "skrl": [],\n    "rl-games": [],\n    "rsl-rl": [],\n    "robomimic": [],\n}' source/extensions/omni.isaac.lab_tasks/setup.py \
    && ./isaaclab.sh -i \
    && pip cache purge

# 8.1. Fix warp-lang version for Isaac Lab compatibility
RUN eval "$(mamba shell hook --shell bash)" && mamba activate metasim \
    && pip uninstall warp-lang -y \
    && pip install warp-lang==1.1.0

# =========================================================================================
# 9. Setup Xvfb and OpenGL Environment
# =========================================================================================

# Create startup script for Xvfb and environment setup
RUN echo '#!/bin/bash' > ${HOME}/setup_display.sh && \
    echo '' >> ${HOME}/setup_display.sh && \
    echo '# Set display environment' >> ${HOME}/setup_display.sh && \
    echo 'export DISPLAY=:99' >> ${HOME}/setup_display.sh && \
    echo '' >> ${HOME}/setup_display.sh && \
    echo '# Start Xvfb if not already running' >> ${HOME}/setup_display.sh && \
    echo 'if ! pgrep -x "Xvfb" > /dev/null; then' >> ${HOME}/setup_display.sh && \
    echo '    echo "Starting Xvfb..."' >> ${HOME}/setup_display.sh && \
    echo '    Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &' >> ${HOME}/setup_display.sh && \
    echo '    sleep 2' >> ${HOME}/setup_display.sh && \
    echo 'else' >> ${HOME}/setup_display.sh && \
    echo '    echo "Xvfb is already running"' >> ${HOME}/setup_display.sh && \
    echo 'fi' >> ${HOME}/setup_display.sh && \
    echo '' >> ${HOME}/setup_display.sh && \
    echo '# Set NVIDIA and OpenGL environment variables' >> ${HOME}/setup_display.sh && \
    echo 'export NVIDIA_VISIBLE_DEVICES=all' >> ${HOME}/setup_display.sh && \
    echo 'export NVIDIA_DRIVER_CAPABILITIES=all' >> ${HOME}/setup_display.sh && \
    echo 'export __GLX_VENDOR_LIBRARY_NAME=nvidia' >> ${HOME}/setup_display.sh && \
    echo 'export LIBGL_ALWAYS_INDIRECT=0' >> ${HOME}/setup_display.sh && \
    echo '' >> ${HOME}/setup_display.sh && \
    echo '# Verify setup' >> ${HOME}/setup_display.sh && \
    echo 'echo "Display: $DISPLAY"' >> ${HOME}/setup_display.sh && \
    echo 'echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"' >> ${HOME}/setup_display.sh && \
    echo 'echo "NVIDIA_DRIVER_CAPABILITIES: $NVIDIA_DRIVER_CAPABILITIES"' >> ${HOME}/setup_display.sh && \
    echo 'echo "Display and OpenGL environment setup complete!"' >> ${HOME}/setup_display.sh

# Make the script executable
RUN chmod +x ${HOME}/setup_display.sh

# Add CUDA environment variables to bashrc
RUN echo '' >> ${HOME}/.bashrc && \
    echo '# CUDA 11.8 environment variables' >> ${HOME}/.bashrc && \
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ${HOME}/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ${HOME}/.bashrc && \
    echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ${HOME}/.bashrc && \
    echo '' >> ${HOME}/.bashrc && \
    echo '# Xvfb and OpenGL setup for headless operation' >> ${HOME}/.bashrc && \
    echo 'export DISPLAY=:99' >> ${HOME}/.bashrc && \
    echo 'export NVIDIA_VISIBLE_DEVICES=all' >> ${HOME}/.bashrc && \
    echo 'export NVIDIA_DRIVER_CAPABILITIES=all' >> ${HOME}/.bashrc && \
    echo 'export __GLX_VENDOR_LIBRARY_NAME=nvidia' >> ${HOME}/.bashrc && \
    echo 'export LIBGL_ALWAYS_INDIRECT=0' >> ${HOME}/.bashrc && \
    echo '' >> ${HOME}/.bashrc && \
    echo '# Check if Xvfb is running, start only if not running' >> ${HOME}/.bashrc && \
    echo 'if [ -z "$XVFB_STARTED" ] && [ "$TERM" != "dumb" ]; then' >> ${HOME}/.bashrc && \
    echo '    if ! pgrep -x Xvfb > /dev/null 2>&1; then' >> ${HOME}/.bashrc && \
    echo '        echo "Starting Xvfb for user session..."' >> ${HOME}/.bashrc && \
    echo '        Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset > /dev/null 2>&1 &' >> ${HOME}/.bashrc && \
    echo '        sleep 1' >> ${HOME}/.bashrc && \
    echo '    fi' >> ${HOME}/.bashrc && \
    echo '    export XVFB_STARTED=1' >> ${HOME}/.bashrc && \
    echo 'fi' >> ${HOME}/.bashrc

# =========================================================================================
# 10. Curobo installation skipped - can be added later if needed
# =========================================================================================
# Note: Curobo requires CUDA development tools which are not included in the runtime image
# Users can install it manually later if needed:
# 1. Install CUDA dev tools: apt install cuda-nvcc-11-8 cuda-toolkit-11-8
# 2. Clone and build curobo from https://github.com/NVlabs/curobo.git

# =========================================================================================
# 11. Install IsaacGym (Optional, kept from original file)
# =========================================================================================
RUN mamba create -n metasim_isaacgym python=3.8 -y \
    && mamba clean -a -y
RUN mkdir -p ${HOME}/packages \
    && cd ${HOME}/packages \
    && wget https://developer.nvidia.com/isaac-gym-preview-4 \
    && tar -xf isaac-gym-preview-4 \
    && rm isaac-gym-preview-4
RUN find ${HOME}/packages/isaacgym/python -type f -name "*.py" -exec sed -i 's/np\.float/np.float32/g' {} +
RUN cd ${HOME}/RoboVerse \
    && eval "$(mamba shell hook --shell bash)" \
    && mamba activate metasim_isaacgym \
    && uv pip install -e ".[isaacgym]" "isaacgym @ ${HOME}/packages/isaacgym/python" \
    && uv cache clean
## Fix error: libpython3.8.so.1.0: cannot open shared object file
## Refer to https://stackoverflow.com/a/75872751
RUN export CONDA_PREFIX=${HOME}/conda/envs/metasim_isaacgym \
    && mkdir -p $CONDA_PREFIX/etc/conda/activate.d \
    && echo "export OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:\$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh \
    && mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d \
    && echo "export LD_LIBRARY_PATH=\$OLD_LD_LIBRARY_PATH && unset OLD_LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
## Fix error: No such file or directory: '.../lib/python3.8/site-packages/isaacgym/_bindings/src/gymtorch/gymtorch.cpp'
RUN mkdir -p ${HOME}/conda/envs/metasim_isaacgym/lib/python3.8/site-packages/isaacgym/_bindings/src \
    && cp -r ${HOME}/packages/isaacgym/python/isaacgym/_bindings/src/gymtorch ${HOME}/conda/envs/metasim_isaacgym/lib/python3.8/site-packages/isaacgym/_bindings/src/gymtorch


# =========================================================================================
# 12. Final Touches
# =========================================================================================
RUN echo 'echo "Welcome to the RoboVerse Development Environment!"' >> ${HOME}/.bashrc && \
    echo 'echo "CUDA 11.8 toolkit with nvcc is available for Curobo compilation."' >> ${HOME}/.bashrc && \
    echo 'echo "Display environment automatically configured for headless operation."' >> ${HOME}/.bashrc && \
    echo 'echo "To run IsaacLab examples, run: mamba activate metasim && source /isaac-sim/setup_conda_env.sh && python <your_script.py>"' >> ${HOME}/.bashrc && \
    echo 'echo "To run MuJoCo/Genesis/Sapien3/PyBullet examples, run: mamba activate metasim && python <your_script.py>"' >> ${HOME}/.bashrc && \
    echo 'echo "For GUI apps from host, remember to run: xhost +local:docker on the host."' >> ${HOME}/.bashrc && \
    echo 'echo "Isaac Gym can be installed manually if needed. See documentation for details."' >> ${HOME}/.bashrc
