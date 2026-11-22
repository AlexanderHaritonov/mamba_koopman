"""Data classes of the files that data_prep/ writes and later steps read.

samples/ego_poses.csv                   EgoPose, one per drive and frame
samples/vehicle_poses_world.csv         VehicleTrack, one per vehicle (file contract 1)
samples/samples_<split>.npz             Sample when writing, Samples when reading (file contract 2)
predictions/<predictor>_<split>.npz     Prediction per sample when predicting, Predictions for the file (file contract 3)
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class EgoPose:
    """One row of samples/ego_poses.csv."""
    time: float  # seconds since the drive's first frame
    T_world_velo: np.ndarray  # 4x4, p_world = T_world_velo · p_velo
    T_world_cam2: np.ndarray  # 4x4, p_world = T_world_cam2 · p_cam2


EgoPoses = dict[tuple[str, int], EgoPose]  # (drive, frame) -> ego pose


@dataclass
class VehicleTrack:
    """All frames of one vehicle, in frame order. Arrays have one entry per frame."""
    drive: str
    track_id: int
    object_type: str  # Car, Van, Truck or Ego
    frames: np.ndarray  # (N,) int, video frames, consecutive
    times: np.ndarray  # (N,) seconds since the drive's first frame
    positions: np.ndarray  # (N, 3) x, y, z in the world, metres
    yaws: np.ndarray  # (N,) radians, counter-clockwise from East, unwrapped
    truncations: np.ndarray  # (N,) int, 0 and 1 mean in the image


@dataclass
class Sample:
    """One sample: one car at one time t, as the cutting script makes it."""
    past: np.ndarray  # (input_steps, 3) x, y, yaw in the car frame at t
    future: np.ndarray  # (output_steps, 3) x, y, yaw in the car frame at t
    anchor_pose: np.ndarray  # (4,) x, y, z, yaw of the car at t, in the world
    drive: str
    track_id: int
    object_type: str  # Car, Van, Truck or Ego
    frame: int  # video frame of t
    displacement: float  # metres from the first input to the last output position, >= 0
    yaw_change: float  # radians from the first input to the last output frame, + = left turn


@dataclass
class Samples:
    """All samples of one samples file: the fields of Sample, one row per sample."""
    past: np.ndarray  # (N, input_steps, 3)
    future: np.ndarray  # (N, output_steps, 3)
    anchor_pose: np.ndarray  # (N, 4)
    drive: np.ndarray  # (N,) str
    track_id: np.ndarray  # (N,) int
    object_type: np.ndarray  # (N,) str
    frame: np.ndarray  # (N,) int
    displacement: np.ndarray  # (N,)
    yaw_change: np.ndarray  # (N,)
    seconds_per_step: float  # median time between frames, the same for all samples


@dataclass
class Prediction:
    """One sample, as a predictor makes it."""
    mu: np.ndarray  # (output_steps, 2) predicted x, y in the car frame at t
    sigma: np.ndarray | None  # same shape as mu, spread of x and y, > 0. None: drawn as a line


@dataclass
class Predictions:
    """One predictions file. Row i belongs to sample i of the samples file of the same split."""
    mu: np.ndarray  # (N, output_steps, 2) predicted x, y in the car frame at t
    sigma: np.ndarray | None  # same shape as mu, spread of x and y, > 0. None: drawn as a line
