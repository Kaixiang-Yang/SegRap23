# Pull the docker image | 拉取镜像
# FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

FROM python:3.10-slim
# FROM nvcr.io/nvidia/pytorch:23.03-py3
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
RUN python -m piptools sync requirements.txt

# RUN python -m pip install --user -r requirements.txt
# RUN python -m pip install --user torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# RUN python -m pip install --no-deps --user nnunet==1.7.1
RUN python -m pip install --user torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user setup.py /opt/app/
COPY --chown=user:user post_processing.py /opt/app/
COPY --chown=user:user Task001_postprocess.py /opt/app/
COPY --chown=user:user inference_code.py /opt/app/
# COPY --chown=user:user images/ /opt/app/images/
COPY --chown=user:user weight/ /opt/app/weight/
# COPY --chown=user:user output/ /opt/app/

# COPY --chown=user:user input/ /input/
COPY --chown=user:user nnunet/ /opt/app/nnunet/

#only for testing postpro alg
# COPY --chown=user:user result/ /opt/app/result/

RUN python -m pip install --user -e .

## For nnUNetV2
# ENV nnUNet_raw="/opt/app/data/nnUNet_raw"
# ENV nnUNet_preprocessedv2="/opt/app/data/nnUNet_preprocessed"
# ENV nnUNet_results="/opt/app/data/nnUNet_results"

ENV nnUNet_raw_data_base="/opt/app/data/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/app/data/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/app/data/nnUNet_results"

ENTRYPOINT [ "python", "-m", "process" ]
# docker container run --ipc=host -it --rm --name nnunetv2 --gpus all segrap2023_oar_segmentationcontainer /bin/bash