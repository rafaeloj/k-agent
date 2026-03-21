#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_DEVICE_IDS>"
    exit 1
fi

docker stop ollama
docker rm ollama
docker run -d --gpus='"device=0"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
sleep 1
docker exec -it ollama ollama run "$2"