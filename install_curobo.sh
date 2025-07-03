#!/bin/bash

set -e

eval "$(mamba shell hook --shell bash)"
mamba activate metasim

source /isaac-sim/setup_conda_env.sh
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc version:"
nvcc --version
echo "PyTorch CUDA version:"
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

cd third_party

sudo apt update && sudo apt install -y git-lfs

# Install setuptools_scm first
pip install setuptools_scm

# Install compatible versions to resolve conflicts
pip install "scipy>=0.13.2,<1.12.0" "plotly==5.24.1" "numpy==1.26.4" "pydantic==2.7.1" "mujoco>=3.3.3"
pip install open3d

if [ -d "curobo" ]; then
    echo "Removing existing curobo directory..."
    rm -rf curobo
fi
git clone https://github.com/NVlabs/curobo.git

cd curobo
CUDA_HOME=/usr/local/cuda-11.8 pip install -e . --no-build-isolation

python -c "import curobo; print('cuRobo installed successfully')"
