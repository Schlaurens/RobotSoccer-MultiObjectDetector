import argparse
import os
import re
import shutil
from pathlib import Path


def find_timestamp_paths(root_dir="models", exclude_folders=None):
    if exclude_folders is None:
        exclude_folders = ["runtime"]

    pattern = re.compile(r"^\d{8}-\d{6}$")
    timestamp_folders = []

    for root, dirs, _ in os.walk(root_dir):
        for dir_name in dirs[:]:  # Iterate over a copy to allow modification
            if dir_name in exclude_folders:
                dirs.remove(dir_name)  # Skip this folder and its subfolders
            elif pattern.match(dir_name):
                full_path = os.path.join(root, dir_name)
                timestamp_folders.append(full_path)
                dirs.remove(dir_name)  # Prevent descending into this folder

    return timestamp_folders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", default=False, type=bool)
    args = parser.parse_args()

    log_dir = Path("logs", "fit")
    model_dir = Path("models")

    log_timestamps = [p.split("/")[-1] for p in find_timestamp_paths(log_dir)]
    model_paths = find_timestamp_paths(model_dir, exclude_folders=["runtime_test"])

    print("Removing unused models...")
    counter = 0
    for p in model_paths:
        model_timestamps = p.split("/")[-1]
        if model_timestamps not in log_timestamps:
            counter += 1
            if not args.dry_run:
                shutil.rmtree(p)

    print(f"Done! Removed {counter} models.")
