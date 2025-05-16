from enum import Enum


class ObstaclesOp(Enum):
    INVERT = 0
    SET = 1
    UNSET = 2


OBSTACLES_WIDTH = 20
OBSTACLES_HEIGHT = 15


def create_empty_label(name):
    return {"name": name}


def set_camera_pose(label, h, z):
    label["cpose"] = {}
    label["cpose"]["h"] = h
    label["cpose"]["z"] = [z[0], z[1], z[2]]


def set_camera_intrinsics(label, cx, cy, fx, fy):
    label["cintr"] = {}
    label["cintr"]["cx"] = cx
    label["cintr"]["cy"] = cy
    label["cintr"]["fx"] = fx
    label["cintr"]["fy"] = fy


def has_ball(label):
    return "ball" in label


def get_ball(label):
    ll = label["ball"]
    return ll["x"], ll["y"], ll["radius"]


def set_ball(label, x, y, radius):
    label["ball"] = {}
    label["ball"]["x"] = x
    label["ball"]["y"] = y
    label["ball"]["radius"] = radius


def unset_ball(label):
    del label["ball"]


def has_obstacles(label):
    return "obstacles" in label


def get_obstacles(label):
    return label["obstacles"]["mask"]


def set_obstacles(label, x1, y1, x2, y2, op=ObstaclesOp.INVERT):
    if "obstacles" not in label:
        if op == ObstaclesOp.UNSET:
            return
        label["obstacles"] = {}
        label["obstacles"]["mask"] = []
        for _ in range(OBSTACLES_HEIGHT):
            label["obstacles"]["mask"].append([0] * OBSTACLES_WIDTH)
    for y in range(max(0, y1), min(y2 + 1, OBSTACLES_HEIGHT)):
        for x in range(max(0, x1), min(x2 + 1, OBSTACLES_WIDTH)):
            label["obstacles"]["mask"][y][x] = (
                (1 - label["obstacles"]["mask"][y][x])
                if op == ObstaclesOp.INVERT
                else (1 if op == ObstaclesOp.SET else 0)
            )
    if all(all(_ <= 0 for _ in row) for row in label["obstacles"]["mask"]):
        unset_obstacles(label)


def set_obstacles_direct(label, mask):
    if all(all(_ <= 0 for _ in row) for row in mask):
        unset_obstacles(label)
    else:
        label["obstacles"] = {}
        label["obstacles"]["mask"] = mask


def unset_obstacles(label):
    del label["obstacles"]


def has_penalty_mark(label):
    return "penaltyMark" in label


def get_penalty_mark(label):
    ll = label["penaltyMark"]
    return ll["x"], ll["y"]


def set_penalty_mark(label, x, y):
    label["penaltyMark"] = {}
    label["penaltyMark"]["x"] = x
    label["penaltyMark"]["y"] = y


def unset_penalty_mark(label):
    del label["penaltyMark"]
