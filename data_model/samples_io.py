"""Samples file (file contract 2 of 3): samples/samples_<split>.npz, one per split, with the fields of Samples.

Written by data_prep/cut_tracks_into_samples.py.
"""
import os

import numpy as np

from data_model.data_classes import Sample, Samples

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")


def samples_path(split: str, samples_dir: str = SAMPLES_DIR) -> str:
    """<samples_dir>/samples_<split>.npz, by default samples/samples_<split>.npz"""
    return os.path.join(samples_dir, f"samples_{split}.npz")


def save_samples(split: str, samples: list[Sample], seconds_per_step: float, samples_dir: str = SAMPLES_DIR):
    """Write <samples_dir>/samples_<split>.npz, compressed."""
    filepath = samples_path(split, samples_dir)
    if len(samples) == 0:
        raise ValueError(f"no samples for {filepath}")
    os.makedirs(samples_dir, exist_ok=True)
    pasts = []
    futures = []
    anchor_poses = []
    drives = []
    track_ids = []
    object_types = []
    frames = []
    displacements = []
    yaw_changes = []
    for sample in samples:
        pasts.append(sample.past)
        futures.append(sample.future)
        anchor_poses.append(sample.anchor_pose)
        drives.append(sample.drive)
        track_ids.append(sample.track_id)
        object_types.append(sample.object_type)
        frames.append(sample.frame)
        displacements.append(sample.displacement)
        yaw_changes.append(sample.yaw_change)
    np.savez_compressed(filepath,
                        past=np.array(pasts, dtype=np.float32),
                        future=np.array(futures, dtype=np.float32),
                        anchor_pose=np.array(anchor_poses, dtype=np.float64),
                        drive=np.array(drives),
                        track_id=np.array(track_ids, dtype=np.int64),
                        object_type=np.array(object_types),
                        frame=np.array(frames, dtype=np.int64),
                        displacement=np.array(displacements, dtype=np.float32),
                        yaw_change=np.array(yaw_changes, dtype=np.float32),
                        seconds_per_step=np.array(seconds_per_step, dtype=np.float64))


def check_row_counts(samples: Samples, name: str):
    """Every per-sample array has as many rows as past."""
    sample_count = samples.past.shape[0]
    row_counts = {"future": samples.future.shape[0],
                  "anchor_pose": samples.anchor_pose.shape[0],
                  "drive": samples.drive.shape[0],
                  "track_id": samples.track_id.shape[0],
                  "object_type": samples.object_type.shape[0],
                  "frame": samples.frame.shape[0],
                  "displacement": samples.displacement.shape[0],
                  "yaw_change": samples.yaw_change.shape[0]}
    for array_name in row_counts:
        if row_counts[array_name] != sample_count:
            raise ValueError(f"{name}: {array_name} has {row_counts[array_name]} rows, past has {sample_count}")


def read_samples(split: str, samples_dir: str = SAMPLES_DIR) -> Samples:
    """All arrays of <samples_dir>/samples_<split>.npz."""
    filepath = samples_path(split, samples_dir)
    with np.load(filepath) as data:
        samples = Samples(past=data["past"],
                          future=data["future"],
                          anchor_pose=data["anchor_pose"],
                          drive=data["drive"],
                          track_id=data["track_id"],
                          object_type=data["object_type"],
                          frame=data["frame"],
                          displacement=data["displacement"],
                          yaw_change=data["yaw_change"],
                          seconds_per_step=float(data["seconds_per_step"]))
    check_row_counts(samples, filepath)
    return samples
