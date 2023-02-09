FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
  
ENV TZ Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get -y install \
    python3 python3-pip python3-dev \
    libgl1 unzip wget \
    ffmpeg libsm6 libxext6 \
    git ssh vim

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN echo "root:root" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

RUN pip3 install --upgrade pip
RUN pip3 install setuptools

# create required folder
RUN mkdir /app
RUN mkdir /app/deepface
# -----------------------------------
# Copy required files from repo into image
COPY ./deepface /app/deepface
COPY ./api/app.py /app/
COPY ./api/routes.py /app/
COPY ./api/service.py /app/
COPY ./setup.py /app/
COPY ./README.md /app/

RUN apt-get update && apt-get install -y locales
RUN locale-gen ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

WORKDIR /workspace
ADD . .

ENV PYTHONPATH $PYTHONPATH:/workspace

RUN chmod -R a+w /workspace

