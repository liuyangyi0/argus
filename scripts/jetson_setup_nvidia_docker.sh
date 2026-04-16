#!/bin/bash
set -e

# Add NVIDIA container toolkit repo
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
echo "Distribution: $distribution"

wget -qO - https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
wget -qO /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list

apt-get update -qq
apt-get install -y nvidia-container-toolkit

# Configure Docker to use nvidia runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "NVIDIA_CONTAINER_TOOLKIT_DONE"
