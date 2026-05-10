#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2

# ====================
# == CPN-Evaluation ==
# ====================

# SETTINGSFILES=(
#     "240x320_v1.yaml"
#     "240x320_v2.yaml"
#     "240x320_v3.yaml"
# )

# SETTINGSFILES=(
#     "288x384_v1.yaml"
#     "288x384_v2.yaml"
#     "288x384_v3.yaml"
# )

# SETTINGSFILES=(
#     "360x480_v1.yaml"
#     "360x480_v2.yaml"
#     "360x480_v3.yaml"
# )

# SETTINGSFILES=(
#     "432x576_v1.yaml"
#     "432x576_v2.yaml"
#     "432x576_v3.yaml"
# )

# SETTINGSFILES=(
#     "480x640_v1.yaml"
#     "480x640_v2.yaml"
#     "480x640_v3.yaml"
# )

# SETTINGSFILES=(
#     "240x320_v1.yaml"
#     "240x320_v2.yaml"
#     "240x320_v3.yaml"
#     "288x384_v1.yaml"
#     "288x384_v2.yaml"
#     "288x384_v3.yaml"
#     "360x480_v1.yaml"
#     "360x480_v2.yaml"
#     "360x480_v3.yaml"
#     "432x576_v1.yaml"
#     "432x576_v2.yaml"
#     "432x576_v3.yaml"
#     "480x640_v1.yaml"
#     "480x640_v2.yaml"
#     "480x640_v3.yaml"
# )


GRAYSCALE=false
YUYV=true

# ===========================
# == Classifier Evaluation ==
# ===========================

SETTINGSFILES=(
    # "classifier-evaluation/288x384_v0.yaml"
    # "classifier-evaluation/288x384_v1.yaml"
    # "classifier-evaluation/288x384_v2.yaml"
    # "classifier-evaluation/288x384_v3.yaml"
    # "classifier-evaluation/288x384_v4.yaml"
    # "classifier-evaluation/288x384_v5.yaml"
    # "classifier-evaluation/288x384_v6.yaml"
)

# ==============================
# == Contextvector Saturation ==
# ==============================
CPN_GRAYSCALE=false
CPN_YUYV=false

for F in "${SETTINGSFILES[@]}"; do
    if [ "$CPN_GRAYSCALE" = true ]; then
        F="cpn-evaluation-grayscale/$F"
    fi
    if [ "$CPN_YUYV" = true ]; then
        F="cpn-evaluation-yuyv/$F"
    fi
    echo "Running with settings file: $F"
    uv run src/training/train.py "$F"
done