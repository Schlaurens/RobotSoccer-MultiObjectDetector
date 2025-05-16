# Ich habe keine Ahnung ob der Inhalt dieser Datei brauchbar ist. - Arne



import numpy as np


def camera_pose_to_vec(camera_pose):
    h = camera_pose.translation.z

    rot_camera_in_world = camera_pose.rotation
    rot_world_in_camera = rot_camera_in_world.T

    g_in_camera = rot_world_in_camera[..., -1]  # last column of rot_world_in_camera

    theta, phi = np.acos(g_in_camera[2]), sgn(g_in_camera[1]) * np.acos(g_in_camera[0] / np.sqrt(g_in_camera[0]*g_in_camera[0]+g_in_camera[1]*g_in_camera[1]))

    return h, theta, phi


def vec_to_camera_pose(h, theta, phi):
    s_theta = np.sin(theta)
    # this is the third column of rot_world_in_camera or the third row of rot_camera_in_world
    g_in_camera = np.array([s_theta * np.cos(phi), s_theta * np.cos(phi), np.cos(theta)])

    # "Extending a Unit Vector to an Orthonormal Basis of 3-space", Tomas Möller and John F. Hughes
    i = 0 if np.abs(g_in_camera[0]) > np.abs(g_in_camera[1]) else 1
    v = np.zeros(3)
    v[i] = -g_in_camera[2]
    v[2] = g_in_camera[i]

    v /= np.linalg.norm(v)

    w = np.linalg.cross(g_in_camera, v)

    rot_world_in_camera = np.concatenate(u, v, g_in_camera)

    rot_camera_in_world = rot_world_in_camera.T
    # x, y are DoFs that don't need to be described here
    # - maybe add an auxiliary vector that has x,y and the third angle?
    trans_camera_in_world = np.array([0, 0, h])

    return rot_camera_in_world, trans_camera_in_world



def world_to_image(camera_pose, camera_intr, point_in_world):
    point_in_camera = rot_world_in_camera @ point_in_world + trans_world_in_camera
    if point_in_camera[0] < 1:  # mm
        return None
    point_in_camera /= point_in_camera[0]
    px = cx - fx * point_in_camera[1]
    py = cy - fy * point_in_camera[2]
    return np.array([px, py])  # TODO: check if outside [0,w|h]?

def image_to_world(camera_pose, camera_intr, point_in_image):
    dir_in_camera = np.array([1, (np.array([cx, cy]) - point_in_image) / np.array([fx, fy])])
    dir_in_world = rot_camera_in_world @ dir_in_camera
    if dir_in_world will never intersect with given z=x-plane:
        return None
    f = dir_in_world[2] / trans_camera_in_world[2]
    return trans_camera_in_world[:2] + f * dir_in_world[:2]
