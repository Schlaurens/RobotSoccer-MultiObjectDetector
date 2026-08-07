#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=1

# ====================
# == CPN-Evaluation ==
# ====================

SETTINGSFILES=(
    # "classifier/v0.yaml"
    "classifier/v1.yaml"
)

for F in "${SETTINGSFILES[@]}"; do
    echo "Running with settings file: $F"
    uv run src/training/train.py "$F"
done