FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    transformers==4.40.0 \
    pillow numpy onnx onnxruntime opencv-python \
    tqdm datasets huggingface-hub num2words onnxscript

COPY . /workspace/

CMD ["python", "quantize_model.py"]
