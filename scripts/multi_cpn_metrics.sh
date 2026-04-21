#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2

# CPN_EVAL=True
# LOG_DIR="logs/fit/cpn-yuyv"
# MODEL_DIR="models/evaluation/cpn-yuyv"

CLASSIFIER_EVAL=True
LOG_DIR="logs/fit/classifier-basic"
MODEL_DIR="models/evaluation/classifier-basic"

for timestamp in "$LOG_DIR"/* "$LOG_DIR"/*/*; do
    timestamp=$(basename "$timestamp")
    if [[ "$timestamp" =~ ^[0-9]{8}-[0-9]{6}$ ]]; then
        uv run src/evaluation/cpn_metrics.py --model_timestamp "$timestamp" --log_dir "$LOG_DIR" --model_dir "$MODEL_DIR" --cpn "$CPN_EVAL" --classifier "$CLASSIFIER_EVAL"
    fi
done