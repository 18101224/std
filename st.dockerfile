FROM nvidia/cuda:12.0-base-ubuntu20.04give me the code that check the ubuntu vers

RUN apt-get update && apt-get install -y python3-pip=3.9.13

RUN pip3 install torch==2.0.0+cu12.0 torchvision==0.12.0+cu12.0

RUN pip3 install numpy scipy opencv-python pycocotools matplotlib onnxruntime onnx

COPY .

