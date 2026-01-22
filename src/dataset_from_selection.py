"""
Generates a single .json file from multiple json files.
The raw groundtruth data consists of multiple .json files. This script takes an assortment of unique identifier found in the already split dataset (e. g. the test dataset) and selects the corresponding samples from the groundtruth samples or predicted samples.
"""

import argparse
import json
import os

from util import dataset_io as u_dataset_io


def load_json_files(dir: str) -> dict:
    """Merges all .json files in the given directory into a single .json file.
    The resulting dict contains all the .json files so that they are accessible with the key of their corresponding log.
    So the content of the .json file `"data/log_a/labels.json"` can be accessed with `jsons["log_a"]`. And the content of a .json file called `"log_b.json"` can be accessed with `jsons["log_b"]`.

    Args:
        dir: The directory which contains the .json files or the directories to the .json files.

    Returns:
        A dictionary which contains all the .json files from `dir`.
    """
    jsons = {}
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".json"):
                # Get the relative path from "dir"
                rel_path = os.path.relpath(os.path.join(root, file), start=dir)
                # Normalize path separators (optional, for consistency)
                rel_path = rel_path.replace(os.sep, "/")
                filepath = os.path.join(dir, rel_path)

                key = rel_path[:-5].split("/")[0]
                with open(filepath) as file:
                    jsons[key] = json.load(file)

    return jsons


def select_samples_by_identifier(identifiers: list[dict], jsons: dict[dict]) -> list[dict]:
    """Looks up the samples in `jsons` by their unique identifiers found in `identifiers` and returns them.

    Args:
        identifiers: A list of unique identifiers. Each identifier consists of a `log_name: str` and a `frame_time: int`.
        jsons: The dictionary of jsons. The keys are the `log_names` and each element in the jsons has a key `frame_time`.

    Returns:
        A list of samples that have the same unique identifiers as the elements in `identifiers`.
    """
    selected_samples = []
    for i in identifiers:
        found = False
        for sample in jsons[i["log_name"]]:
            if sample["frame_time"] == i["frame_time"]:
                found = True
                selected_samples.append(sample)
                break

        if not found:
            print("FAIL: ", i["frame_time"], " - ", i["log_name"])

    return selected_samples


def main(args):
    test_ds = u_dataset_io.get_dataset(args.test_dataset)
    prediction_jsons = load_json_files(args.prediction_source)
    groundtruth_jsons = load_json_files(args.groundtruth_source)

    # Get the identifier of the samples in the test dataset.
    print("Extracting Identifiers...")
    test_sample_identifiers = []
    for s in test_ds:
        identifier = {
            "log_name": u_dataset_io.log_name_from_label(s),
            "frame_time": s["frame_time"].numpy(),
        }
        test_sample_identifiers.append(identifier)

    print("Selecting Predictions....")
    selected_predictions = select_samples_by_identifier(test_sample_identifiers, prediction_jsons)
    print("Selecting Groundtruth...")
    selected_groundtruth = select_samples_by_identifier(test_sample_identifiers, groundtruth_jsons)

    assert len(selected_predictions) == len(selected_groundtruth)
    assert len(test_sample_identifiers) == len(selected_predictions)

    print("Writing JSONs...")
    with open(args.destination + args.prediction_source.split("/")[-1] + ".json", "w") as f:
        json.dump(selected_predictions, f, indent=4)

    with open(args.destination + "groundtruth" + ".json", "w") as f:
        json.dump(selected_groundtruth, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script shows statistics about the dataset.")
    parser.add_argument("--test_dataset", default="data/tfrecords/test_ds_v3_1840(0.15).tfrecords")
    parser.add_argument("--groundtruth_source", default="data/groundtruth/")
    parser.add_argument("--prediction_source")
    parser.add_argument("--destination", default="data/selected/")

    args = parser.parse_args()

    main(args)
