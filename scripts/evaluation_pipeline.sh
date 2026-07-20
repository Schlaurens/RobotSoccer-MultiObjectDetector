#!/bin/bash

# Define variables
MODEL_TIMESTAMP="20260711-000000"

LOG_DIR="logs/fit/final"
DATA_DIR="data/evaluation/final"
MODEL_DIR="models/evaluation/final"

DISTANCE=3
BETA=0.3
DISTANCE_THRESHOLD_WORLD=0.05
DISTANCE_THRESHOLD_IMAGE=5.0
NMS_IOU=0.35
N_CANDIDATES="5,4,11"

# Derived unique identifier
UNIQUE_ID="d_${DISTANCE}-K_$(echo $N_CANDIDATES | tr ',' '-')"

# Execute the scripts in order
echo "Running thresholdless_metrics.py..."
uv run src/evaluation/thresholdless_metrics.py \
    --model_timestamp "$MODEL_TIMESTAMP" \
    --log_dir "$LOG_DIR" \
    --save_dir "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --classifier True \
    --distance "$DISTANCE" \
    --nms_iou "$NMS_IOU" \
    --n_candidates "$N_CANDIDATES" \
    --beta "$BETA"

echo "Running compare_predictions.py..."
uv run src/evaluation/compare_predictions.py \
    --model_timestamp "$MODEL_TIMESTAMP" \
    --directory_predictions "${LOG_DIR}/${MODEL_TIMESTAMP}/predictions/${UNIQUE_ID}" \
    --threshold_world "$DISTANCE_THRESHOLD_WORLD" \
    --threshold_image "$DISTANCE_THRESHOLD_IMAGE" \

echo "Running compare_coord_errors.py..."
uv run src/evaluation/compare_coord_errors.py \
    --directory_matches "${LOG_DIR}/${MODEL_TIMESTAMP}/matches/${UNIQUE_ID}"

echo "All scripts executed successfully."
