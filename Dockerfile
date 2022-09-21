FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
    && apt-get -y install build-essential wget nano git \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video

COPY Video_Codec_SDK_11.1.5 /tmp/Video_Codec_SDK_11.1.5

RUN git clone https://github.com/pytorch/vision.git ~/torchvision

#RUN conda install cuda -c nvidia
#RUN conda install cuda -c nvidia/label/cuda-11.3.0
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda

RUN cd /root/torchvision && \
    TORCHVISION_INCLUDE=/tmp/Video_Codec_SDK_11.1.5/Interface \
    TORCHVISION_LIBRARY=/tmp/Video_Codec_SDK_11.1.5/Lib/linux/stubs/x86_64 \
    TORCH_CUDA_ARCH_LIST=8.6+PTX \
#    CUDA_HOME=/opt/conda/pkgs/pytorch-1.11.0-py3.8_cuda11.3_cudnn8.2.0_0/lib/python3.8/site-packages/torch/cuda\
    FORCE_CUDA=1  \
    /opt/conda/bin/python setup.py install
## this was shadowing the source installed torchvision, uninstalling made the source version visible
#RUN pip uninstall -y torchvision
#/opt/conda/pkgs/pytorch-1.11.0-py3.8_cuda11.3_cudnn8.2.0_0/lib/python3.8/site-packages/torch/backends/cuda
#/opt/conda/pkgs/pytorch-1.11.0-py3.8_cuda11.3_cudnn8.2.0_0/lib/python3.8/site-packages/torch/cuda
