import itertools
import os

from util import dataset as u_dataset
from util import labels as u_labels


def load_dataset(data_path):
    label_dirs = [dir[0] for dir in os.walk(data_path)][1:]

    labels = [u_dataset.load_labels(dir) for dir in label_dirs]

    return labels


def main(data_path: str):
    labels = load_dataset(data_path)
    labels_concat = list(itertools.chain.from_iterable(labels))

    number_of_samples = len(labels_concat)
    number_of_non_empty_samples = len(
        [
            _
            for _ in labels_concat
            if u_labels.has_ball(_) or u_labels.has_penalty_mark(_) or u_labels.has_intersections(_)
        ]
    )
    number_of_ball_samples = len([_ for _ in labels_concat if u_labels.has_ball(_)])
    number_of_penalty_mark_samples = len([_ for _ in labels_concat if u_labels.has_penalty_mark(_)])

    number_of_l_intersection_samples = sum(
        [
            len(x["intersections"][u_labels.IntersectionType.L.value])
            for x in labels_concat
            if u_labels.has_intersections(x)
        ]
    )
    number_of_t_intersection_samples = sum(
        [
            len(x["intersections"][u_labels.IntersectionType.T.value])
            for x in labels_concat
            if u_labels.has_intersections(x)
        ]
    )
    number_of_x_intersection_samples = sum(
        [
            len(x["intersections"][u_labels.IntersectionType.X.value])
            for x in labels_concat
            if u_labels.has_intersections(x)
        ]
    )

    print("Number of logs: ", len(labels))
    print("Number of samples: ", number_of_samples)
    print("Number of non empty samples: ", number_of_non_empty_samples)
    print("Number of ball samples:", number_of_ball_samples)
    print("Number of penaltyMark samples:", number_of_penalty_mark_samples)
    print("Number of L intersection samples:", number_of_l_intersection_samples)
    print("Number of T intersection samples:", number_of_t_intersection_samples)
    print("Number of X intersection samples:", number_of_x_intersection_samples)
    print("=========")
    print("% of ball samples: ", round((number_of_ball_samples / number_of_samples) * 100, 2))
    print(
        f"% of penaltyMark samples: {((number_of_penalty_mark_samples / number_of_samples) * 100):.2f}"
    )
    print(
        f"% of L intersection samples: {((number_of_l_intersection_samples / number_of_samples) * 100):.2f}"
    )
    print(
        f"% of T intersection samples: {((number_of_t_intersection_samples / number_of_samples) * 100):.2f}"
    )
    print(
        f"% of X intersection samples: {((number_of_x_intersection_samples / number_of_samples) * 100):.2f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="This script shows statistics about the dataset.")
    parser.add_argument("data_path")
    args = parser.parse_args()

    main(args.data_path)
