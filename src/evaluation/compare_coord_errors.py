import argparse
import glob
import os
import sys

import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np


def load_data(path_to_data: Path) -> dict:
    data = {k: {l: [] for l in ["matches", "distances"]} for k in ["bhuman", "model"]}
    for prefix in data:
        for metric in data[prefix]:
            data[prefix][metric] = np.load(path_to_data / f"{prefix}_{metric}.npy")
    return data


def main(args):
    results = {}
    for dir in Path(args.directory, args.model_timestamp, "matches").glob("**/"):
        # Only leaf node directories
        if any(p.is_dir() for p in dir.iterdir()):
            continue

        data = load_data(Path(dir))

        print(f"======= {Path(dir).parts[-1]} =======")

        # === Calculate MAE for Image ===
        bhuman_image_error = np.linalg.norm(
            np.subtract(data["bhuman"]["matches"][:, 0, :], data["bhuman"]["matches"][:, 1, :]),
            axis=-1,
        )
        model_image_error = np.linalg.norm(
            np.subtract(data["model"]["matches"][:, 0, :], data["model"]["matches"][:, 1, :]),
            axis=-1,
        )

        bhuman_image_mae = np.mean(bhuman_image_error)
        model_image_mae = np.mean(model_image_error)

        bhuman_image_var = np.var(bhuman_image_error)
        model_image_var = np.var(model_image_error)
        print("======")
        print("B-Human Image MAE:", bhuman_image_mae)
        print("Model Image MAE:", model_image_mae)

        print("B-Human Image Variance:", bhuman_image_var)
        print("Model Image Variance:", model_image_var)

        # === Calculate MAE for World ===
        bhuman_world_error = np.linalg.norm(
            np.expand_dims(
                np.subtract(data["bhuman"]["distances"][:, 0], data["bhuman"]["distances"][:, 1]),
                axis=-1,
            ),
            axis=-1,
        )
        model_world_error = np.linalg.norm(
            np.expand_dims(
                np.subtract(data["model"]["distances"][:, 0], data["model"]["distances"][:, 1]),
                axis=-1,
            ),
            axis=-1,
        )

        bhuman_world_mae = np.mean(bhuman_world_error)
        model_world_mae = np.mean(model_world_error)

        bhuman_world_var = np.var(bhuman_world_error)
        model_world_var = np.var(model_world_error)
        print("======")
        print("B-Human World MAE:", bhuman_world_mae)
        print("Model World MAE:", model_world_mae)

        print("B-Human World Variance:", bhuman_world_var)
        print("Model World Variance:", model_world_var)
        print("======")

        results[Path(dir).parts[-1]] = {
            "bhuman_image_mae": float(bhuman_image_mae),
            "model_image_mae": float(model_image_mae),
            "bhuman_image_var": float(bhuman_image_var),
            "model_image_var": float(model_image_var),
            "bhuman_world_mae": float(bhuman_world_mae),
            "model_world_mae": float(model_world_mae),
            "bhuman_world_var": float(bhuman_world_var),
            "model_world_var": float(model_world_var),
        }

    with open(Path(args.directory) / args.model_timestamp / "regression_error.yaml", "w") as file:
        yaml.dump(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the detection errors of the model with the groundtruth and the current B-Human detectors."
    )
    parser.add_argument("--model_timestamp")
    parser.add_argument("--directory", default="data/evaluation")
    args = parser.parse_args()

    main(args)
