if __name__ == "__main__":
    import argparse
    import math
    import os

    import pybh.logs as bhlogs

    from util import dataset as u_dataset
    from util import labels as u_labels

    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-filter", default=["Upper"], nargs="+")
    parser.add_argument("--import-labels", default=True, action="store_true")
    parser.add_argument(
        "--destination",
        default=os.path.join(os.path.dirname(__file__), "data"),
    )
    parser.add_argument("--step", default=30, type=int)
    parser.add_argument("path")
    args = parser.parse_args()

    log = bhlogs.Log(args.path)
    destination = os.path.join(args.destination, os.path.basename(args.path)[:-4])
    os.makedirs(destination)
    labels = []

    cooldown = 0
    for index, frame in enumerate(log):
        if args.thread_filter and frame.thread not in args.thread_filter:
            continue

        if "CameraInfo" not in frame or "CameraMatrix" not in frame:
            continue

        # Also decrease counter if images are not used.
        if cooldown > 0:
            cooldown -= 1

        camera_info = frame["CameraInfo"]
        camera_matrix = frame["CameraMatrix"]

        if not camera_matrix.isValid:
            continue

        jpeg_image = frame["JPEGImage"]
        # if jpeg_image.width != 160 or jpeg_image.height != 120:  # TODO: 160, 120
        #     continue

        ball_percept = frame.get("BallPercept", None)
        obstacles_image_percept = (
            frame.get("ObstaclesImagePercept", None)
        )
        penalty_mark_percept = (
            frame.get("PenaltyMarkPercept", None)
        )

        interesting = ball_percept.status != 0

        # Skip image or reset counter.
        if cooldown > (0 if not interesting else cooldown // 2):
            continue
        cooldown = args.step

        name = f"{frame.thread}{index}"
        assert len(jpeg_image._data) == jpeg_image.size + 16
        with open(u_dataset.get_image_path(destination, name), "wb") as f:
            f.write(jpeg_image._data[16:])

        label = u_labels.create_empty_label(name)
        u_labels.set_camera_pose(
            label, camera_matrix.translation.z, [_.elems[2] for _ in camera_matrix.rotation.cols]
        )  # the latter is the z-axis ("up") in camera coordinates, which fixes the attitude, but not the yaw relative to the ground. this could also be expressed with two DoF as theta=inclination=arccos(z), phi=azimuth=sgn(y)*arccos(x/sqrt(x*x+y*y))
        u_labels.set_camera_intrinsics(
            label,
            camera_info.opticalCenter.x,
            camera_info.opticalCenter.y,
            0.5 * camera_info.width / math.tan(0.5 * camera_info.openingAngleWidth),
            0.5 * camera_info.height / math.tan(0.5 * camera_info.openingAngleHeight),
        )
        if args.import_labels:
            if ball_percept and ball_percept.status != 0:
                u_labels.set_ball(
                    label,
                    ball_percept.positionInImage.x,
                    ball_percept.positionInImage.y,
                    ball_percept.radiusInImage,
                )
            """
            if obstacles_image_percept and len(obstacles_image_percept.obstacles) > 0:
                for obstacle in obstacles_image_percept.obstacles:
                    # TODO: 16
                    u_labels.set_obstacles(
                        label,
                        obstacle.left // 16,
                        obstacle.top // 16,
                        obstacle.right // 16,
                        obstacle.bottom // 16,
                        op=u_labels.ObstaclesOp.SET,
                    )
            if penalty_mark_percept and penalty_mark_percept.wasSeen:
                u_labels.set_penalty_mark(
                    label,
                    penalty_mark_percept.positionInImage.x,
                    penalty_mark_percept.positionInImage.y,
                )
            """
        labels.append(label)

    u_dataset.save_labels(destination, labels)
