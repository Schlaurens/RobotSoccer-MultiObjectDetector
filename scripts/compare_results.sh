DIRECTORY="data/evaluation"
DISTANCE_THRESHOLD="30"

# Check if the parent directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_timestamp>"
    exit 1
fi

timestamp="$1"

echo "Comparing Predictions..."
uv run src/evaluation/compare_predictions.py \
    --directory "$DIRECTORY" \
    --model_timestamp "$timestamp" \
    --distance_threshold "$DISTANCE_THRESHOLD" \

echo "Comparing Coordinate Errors..."
uv run src/evaluation/compare_coord_errors.py \
    --model_timestamp "$timestamp" \
    --directory "$DIRECTORY" \
