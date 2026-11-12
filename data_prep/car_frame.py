"""The target-car frame of a sample, and the way back to the world.

Car frame: origin at the car's position at time t, +x where its nose points at t, +y to its left.
Yaw in the car frame is the world yaw minus the car's yaw at t.
The anchor pose is the car's world pose at t: x, y, z, yaw, as anchor_pose in the samples file.
"""
import math

import numpy as np


def world_to_car_frame(world_xy: np.ndarray, world_yaws: np.ndarray,
                       anchor_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World x, y (N, 2) and yaws (N,) into the car frame of anchor_pose."""
    anchor_x = anchor_pose[0]
    anchor_y = anchor_pose[1]
    anchor_yaw = anchor_pose[3]
    cos_yaw = math.cos(anchor_yaw)
    sin_yaw = math.sin(anchor_yaw)
    dx = world_xy[:, 0] - anchor_x
    dy = world_xy[:, 1] - anchor_y
    car_x = cos_yaw * dx + sin_yaw * dy
    car_y = -sin_yaw * dx + cos_yaw * dy
    car_yaws = world_yaws - anchor_yaw
    return np.column_stack([car_x, car_y]), car_yaws


def car_frame_to_world(car_xy: np.ndarray, car_yaws: np.ndarray,
                       anchor_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Car frame x, y (N, 2) and yaws (N,) back into the world. For z, use the anchor's z."""
    anchor_x = anchor_pose[0]
    anchor_y = anchor_pose[1]
    anchor_yaw = anchor_pose[3]
    cos_yaw = math.cos(anchor_yaw)
    sin_yaw = math.sin(anchor_yaw)
    world_x = anchor_x + cos_yaw * car_xy[:, 0] - sin_yaw * car_xy[:, 1]
    world_y = anchor_y + sin_yaw * car_xy[:, 0] + cos_yaw * car_xy[:, 1]
    world_yaws = car_yaws + anchor_yaw
    return np.column_stack([world_x, world_y]), world_yaws
