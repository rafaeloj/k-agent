#!/bin/bash

# Estimated runtime: ~140h
# You may need to grant execute permission: chmod +x k_selections.sh
# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_DEVICE_IDS>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1

export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0

DATASETS=("mnist" "cifar10")
K_SAMPLE_SIZE=(15 10 5 -2)
SELECTION_METHODS=("oort" "poc" "rrobin" "random")

cd "../../selection-agent/"
for DS in "${DATASETS[@]}"; do
    for method in "${SELECTION_METHODS[@]}"; do
        for K in "${K_SAMPLE_SIZE[@]}"; do
            echo "Starting runs for method: $method"
            for i in {1..3}; do
                echo "  Run #$i for $method"
                flwr run . local-simulation --run-config "selection-method='$method' sample-size='$K' dataset-name='$DS'"
            done
        done
    done
done
echo "Finished"
