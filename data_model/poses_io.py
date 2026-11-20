"""Readers of the pose files that later steps use. data_prep/ writes them.

samples/ego_poses.csv              written by compute_ego_poses.py, read as EgoPoses
samples/vehicle_poses_world.csv    written by transform_vehicle_poses_to_world.py (file contract 1),
                                   read as one VehicleTrack per vehicle

Vehicle tracks have no smoothing and no velocity: later steps get the raw positions.
A track must have no missing frames: tracklets have none, annotators interpolate through occlusions.
A gap is an error.
"""
import csv
import os

import numpy as np

from data_model.data_classes import EgoPose, EgoPoses, VehicleTrack

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EGO_POSES_PATH = os.path.join(PROJECT_DIR, "samples", "ego_poses.csv")
VEHICLE_POSES_WORLD_PATH = os.path.join(PROJECT_DIR, "samples", "vehicle_poses_world.csv")


def read_ego_poses(filepath: str = EGO_POSES_PATH) -> EgoPoses:
    """Load samples/ego_poses.csv."""
    ego_poses = {}
    with open(filepath, newline="") as f:
        for row in csv.DictReader(f):
            T_world_velo = np.eye(4)
            T_world_velo[0] = [float(row["velo_r11"]), float(row["velo_r12"]), float(row["velo_r13"]), float(row["velo_tx"])]
            T_world_velo[1] = [float(row["velo_r21"]), float(row["velo_r22"]), float(row["velo_r23"]), float(row["velo_ty"])]
            T_world_velo[2] = [float(row["velo_r31"]), float(row["velo_r32"]), float(row["velo_r33"]), float(row["velo_tz"])]
            T_world_cam2 = np.eye(4)
            T_world_cam2[0] = [float(row["cam2_r11"]), float(row["cam2_r12"]), float(row["cam2_r13"]), float(row["cam2_tx"])]
            T_world_cam2[1] = [float(row["cam2_r21"]), float(row["cam2_r22"]), float(row["cam2_r23"]), float(row["cam2_ty"])]
            T_world_cam2[2] = [float(row["cam2_r31"]), float(row["cam2_r32"]), float(row["cam2_r33"]), float(row["cam2_tz"])]
            ego_poses[(row["drive"], int(row["frame"]))] = EgoPose(time=float(row["time"]),
                                                                   T_world_velo=T_world_velo,
                                                                   T_world_cam2=T_world_cam2)
    return ego_poses


def read_vehicle_tracks(filepath: str = VEHICLE_POSES_WORLD_PATH) -> list[VehicleTrack]:
    """All tracks of vehicle_poses_world.csv, sorted by drive and track_id.

    Reads only the contract columns: drive, track_id, object_type, frame, time, x, y, z, yaw, truncation.
    """
    rows_by_track = {}  # (drive, track_id) -> rows of the track
    with open(filepath, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["drive"], int(row["track_id"]))
            if key not in rows_by_track:
                rows_by_track[key] = []
            rows_by_track[key].append(row)

    tracks = []
    for key in sorted(rows_by_track):
        drive, track_id = key
        rows = sorted(rows_by_track[key], key=lambda row: int(row["frame"]))

        frames = []
        times = []
        positions = []
        yaws = []
        truncations = []
        for row in rows:
            if row["object_type"] != rows[0]["object_type"]:
                raise ValueError(f"{drive} track {track_id}: object_type changes inside the track")
            frames.append(int(row["frame"]))
            times.append(float(row["time"]))
            positions.append([float(row["x"]), float(row["y"]), float(row["z"])])
            yaws.append(float(row["yaw"]))
            truncations.append(int(row["truncation"]))

        for frame_index in range(1, len(frames)):
            if frames[frame_index] != frames[frame_index - 1] + 1:
                raise ValueError(f"{drive} track {track_id}: frame {frames[frame_index - 1]} "
                                 f"is followed by frame {frames[frame_index]}")

        tracks.append(VehicleTrack(drive=drive,
                                   track_id=track_id,
                                   object_type=rows[0]["object_type"],
                                   frames=np.array(frames),
                                   times=np.array(times),
                                   positions=np.array(positions),
                                   yaws=np.array(yaws),
                                   truncations=np.array(truncations)))
    return tracks
