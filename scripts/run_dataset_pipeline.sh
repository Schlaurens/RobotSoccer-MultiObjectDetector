#!/bin/bash

# Dataset Pipeline Script

# This script automates the dataset processing pipeline. It performs the following steps in sequence:

# 1. Saves datasets using `save_datasets.sh` into .tfrecords files.
# 2. Splits the dataset into training, validation, and test sets using the specified split ratios.
# 3. Generates JSON files from selected data (used for evaluations purposes)
# 4. Computes and displays dataset statistics.

# Usage:
#     ./run_dataset_pipeline.sh

# Configuration:
#     Adjust the following variables at the top of the script:
#     - DATASET_PATH: Base path to the dataset directory.
#     - SAVE_DIR_TFRECORDS: Directory to save the split datasets in TFRecord format.
#     - GROUNDTRUTH_SOURCE: Path to the directory containing the groundtruth data.
#     - SAVE_DIR_EVALUATION: Directory to save the generated JSON files generated in the 3rd step.
#     - BHUMAN_PREDICTION_SOURCE: Path to the directory containing prediction data from the current B-Human detectors.
#     - VAL_SPLIT: Fraction of the dataset to use for validation (default: 0.2).
#     - TEST_SPLIT: Fraction of the dataset to use for testing (default: 0.15).
#     - TEST_DATASET: Name of the test dataset .tfrecords file (default: 'test_ds_v3_1840(0.15).tfrecords').
#     - CALCULATE_DISTANCES: Flag to calculate distances (default: true).
#     - PRINT_OUTPUT: Flag to print statistics to the console (default: true).

# Environment Variables:
#     - TF_CPP_MIN_LOG_LEVEL: Set to suppress TensorFlow logs (default: 3).

# Notes:
#     - Ensure all paths are correctly set before running the script.
#     - Make sure `save_datasets.sh` and other required scripts are in the specified directories and executable.

# Suppress TensorFlow warnings and messages
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow logs (0 = all, 1 = info, 2 = warnings, 3 = errors)

# ===== Configuration =====
# Paths
DATASET_PATH="data/"
SAVE_DIR_TFRECORDS="${DATASET_PATH}tfrecords/"
GROUNDTRUTH_SOURCE="${DATASET_PATH}groundtruth_raw/"
SPLIT_GROUNDTRUTH_DESTINATION="${DATASET_PATH}groundtruth_tfrecords/"
SAVE_DIR_EVALUATION="${DATASET_PATH}evaluation/"

# Prediction source
BHUMAN_PREDICTION_SOURCE="${DATASET_PATH}b-human_predictions/"

IMAGE_RESOLUTION="240 320"
CELL_DIMENSIONS="16 16" 
# IMAGE_RESOLUTION="288 384"
# CELL_DIMENSIONS="16 16" 
# IMAGE_RESOLUTION="360 480"
# CELL_DIMENSIONS="24 24" 
# IMAGE_RESOLUTION="432 576" 
# CELL_DIMENSIONS="24 24" 
# IMAGE_RESOLUTION="480 640"
# CELL_DIMENSIONS="32 32" 

read IM_HEIGHT IM_WIDTH <<< "$IMAGE_RESOLUTION"
read CELL_HEIGHT CELL_WIDTH <<< "$CELL_DIMENSIONS"

# Split ration
VAL_SPLIT=0.2
TEST_SPLIT=0.15

# Path to save_datasets.sh (adjust if it's in a different directory)
SAVE_DATASETS_SH="./scripts/save_datasets.sh"

# Test dataset file (adjust version and sample count as needed)
TEST_DATASET="test_ds_1840(0.15).tfrecords"

# Flags for statistics calculation
CALCULATE_DISTANCES=true
PRINT_OUTPUT=true

# ===== Script =====
# Step 1: Save all datasets into .tfrecords files
echo "Saving all datasets..."
if [ -f "$SAVE_DATASETS_SH" ]; then
    "$SAVE_DATASETS_SH" "$GROUNDTRUTH_SOURCE" "$SPLIT_GROUNDTRUTH_DESTINATION" "$IM_HEIGHT" "$IM_WIDTH" "$CELL_HEIGHT" "$CELL_WIDTH"
else
    echo "Error: $SAVE_DATASETS_SH not found. Please check the path."
    exit 1
fi

# Step 2: Split the dataset
echo "Splitting dataset..."
uv run src/dataset/split_dataset.py \
    --src_dir "${SPLIT_GROUNDTRUTH_DESTINATION}/${IM_WIDTH}x${IM_HEIGHT}/" \
    --save_dir "$SAVE_DIR_TFRECORDS" \
    --val_split "$VAL_SPLIT" \
    --test_split "$TEST_SPLIT" \
    --image_res "$IM_HEIGHT" "$IM_WIDTH" \
    --cell_dims "$CELL_HEIGHT" "$CELL_WIDTH" \

# Step 3: Generate JSON files from selection
# echo "Generating JSON files from selection..."
# uv run src/dataset/from_selection.py \
#     --test_dataset "${SAVE_DIR_TFRECORDS}${IM_WIDTH}x${IM_HEIGHT}/${TEST_DATASET}" \
#     --groundtruth_source "$GROUNDTRUTH_SOURCE" \
#     --prediction_source "$BHUMAN_PREDICTION_SOURCE" \
#     --destination "$SAVE_DIR_EVALUATION" \
#     --image_res "$IM_HEIGHT" "$IM_WIDTH" \

# # Step 4: Compute statistics
# echo "Computing statistics..."
# if [ "$CALCULATE_DISTANCES" = true ]; then
#     CALCULATE_DISTANCES_FLAG="--calculate_distances"
# fi
# if [ "$PRINT_OUTPUT" = true ]; then
#     PRINT_OUTPUT_FLAG="--print_output"
# fi

# uv run src/dataset/statistics.py \
#     "$GROUNDTRUTH_SOURCE" \
#     $CALCULATE_DISTANCES_FLAG \
#     $PRINT_OUTPUT_FLAG
