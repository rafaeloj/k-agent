#!/bin/bash

# Estimated runtime: ~240h
# You may need to grant execute permission: chmod +x k_agent_prompt_type.sh
# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_DEVICE_IDS>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
DATASETS=("mnist" "cifar10")
LLM_MODEL_NAMES=("llama3.1:8b" "llama3.2:3b" "qwen3:8b")
SELECTION_METHOD="k-agent"
K_AGENT_SELECTION_METHOD=("random_selection" "oort_selection" "poc_selection")
PROMPTS_TYPE=("chain-of-thought" "few-shot" "description-only")
for DS in "${DATASETS[@]}"; do
    for PROMPT_TYPE in "${PROMPTS_TYPE[@]}"; do
        for MODEL_NAME in "${LLM_MODEL_NAMES[@]}"; do
            ../utils/k_start_ollama.sh "$1" "$MODEL_NAME"
            cd "../../selection-agent/"
            for m in "${K_AGENT_SELECTION_METHOD[@]}"; do
                echo "Starting runs for method: $SELECTION_METHOD"
                # for i in {1..3}; do
                echo "  Running with MODEL_NAME=$MODEL_NAME, SELECTION_METHOD=$SELECTION_METHOD, METHOD=$m, PROMPT_TYPE=$PROMPT_TYPE, Run=$i"
                echo "  Running with ... Run=$i"
                flwr run . local-simulation --run-config "selection-method='$SELECTION_METHOD' sample-size=-1 llm-model-name='$MODEL_NAME' k-agent-selection-method='$m' prompt-type='$PROMPT_TYPE' dataset-name='$DS'" &
                pid1=$!
                flwr run . local-simulation --run-config "selection-method='$SELECTION_METHOD' sample-size=-1 llm-model-name='$MODEL_NAME' k-agent-selection-method='$m' prompt-type='$PROMPT_TYPE' dataset-name='$DS'" &
                pid2=$!
                flwr run . local-simulation --run-config "selection-method='$SELECTION_METHOD' sample-size=-1 llm-model-name='$MODEL_NAME' k-agent-selection-method='$m' prompt-type='$PROMPT_TYPE' dataset-name='$DS'" &
                pid3=$!
                # done
                wait $pid1 $pid2 $pid3
                done
            done
            cd ../scripts/k_agent/
        done
    done
done
echo "Finished"
