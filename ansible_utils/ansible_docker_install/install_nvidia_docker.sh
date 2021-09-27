# visiongpu 16.04, docker 19.03
# from https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/#install-using-the-repository
# nvidia is fully supported with 19.03
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
# if the previous gets stuck (problem with Ipv6), do it manually:
echo "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
# then edit apt-get repositories manually with the previous line:
sudo vim /etc/apt/sources.list

sudo apt-get install docker-ce
# up to here installs normal docker
# now we install nvidia stuff
# https://github.com/NVIDIA/nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# add yourself as docker user, to avoid having to use sudo
#https://gist.github.com/Brainiarc7/a8ab5f89494d053003454efc3be2d2ef
sudo groupadd docker
sudo gpasswd -a $USER docker
# reactivate group, so user gets acitve in the group
newgrp docker

# finally test, should output nvidia-smi:
docker run --gpus all nvidia/cuda:9.0-base nvidia-smi