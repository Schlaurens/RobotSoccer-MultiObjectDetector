#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2

# CPN_EVAL=True
# LOG_DIR_BASE="logs/fit/cpn-yuyv"
# MODEL_DIR_BASE="models/evaluation/cpn-yuyv"
# DISTANCE=9

NUM_RUNS=3

for run in $(seq 1 $NUM_RUNS); do
    LOG_DIR="${LOG_DIR_BASE}/run_${run}"
    MODEL_DIR="${MODEL_DIR_BASE}/run_${run}"

    for timestamp in "$LOG_DIR"/*/*; do
        timestamp=$(basename "$timestamp")
        if [[ "$timestamp" =~ ^[0-9]{8}-[0-9]{6}$ ]]; then
            uv run src/evaluation/thresholdless_metrics.py --model_timestamp "$timestamp" --log_dir "$LOG_DIR" --save_dir "$SAVE_DIR" --distance "$DISTANCE" --model_dir "$MODEL_DIR" --cpn "$CPN_EVAL" --classifier "$CLASSIFIER_EVAL"
        fi
    done
done
