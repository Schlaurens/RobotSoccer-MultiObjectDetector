DIRECTORY="data/evaluation"
TIMESTAMP="20260316-145655"
DISTANCE_THRESHOLD="30"

echo "Comparing Predictions..."
uv run src/evaluation/compare_predictions.py \
    --directory "$DIRECTORY" \
    --model_timestamp "$TIMESTAMP" \
    --distance_threshold "$DISTANCE_THRESHOLD" \

echo "Comparing Coordinate Errors..."
uv run src/evaluation/compare_coord_errors.py \
    --model_timestamp "$TIMESTAMP" \
    --directory "$DIRECTORY" \
