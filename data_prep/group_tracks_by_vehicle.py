"""Group the rows of vehicle_poses_world.csv into one track per vehicle, frames in order.

No smoothing and no velocity: later steps get the raw positions.
Reads only the contract columns of prepared/vehicle_poses_world.csv: drive, track_id, object_type, frame, time, x, y, z, yaw, truncation

Writes no file.
Run as a script, it prints how many tracks and frames are kept.

A track must have no missing frames: tracklets have none, annotators interpolate through occlusions.
A gap stops the script with an error.
"""
import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEHICLE_POSES_WORLD_PATH = os.path.join(PROJECT_DIR, "prepared", "vehicle_poses_world.csv")


@dataclass
class VehicleTrack:
    """All frames of one vehicle, in frame order. Arrays have one entry per frame."""
    drive: str
    track_id: int
    object_type: str  # Car, Van or Truck
    frames: np.ndarray  # (N,) int, video frames, consecutive
    times: np.ndarray  # (N,) seconds since the drive's first frame
    positions: np.ndarray  # (N, 3) x, y, z in the world, metres
    yaws: np.ndarray  # (N,) radians, counter-clockwise from East, unwrapped
    truncations: np.ndarray  # (N,) int, 0 and 1 mean in the image


def read_vehicle_tracks(filepath) -> list[VehicleTrack]:
    """All tracks of vehicle_poses_world.csv, sorted by drive and track_id."""
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


def keep_long_tracks(tracks: list[VehicleTrack], min_frame_count: int) -> list[VehicleTrack]:
    """Tracks with at least min_frame_count frames"""
    long_tracks = []
    for track in tracks:
        if len(track.frames) >= min_frame_count:
            long_tracks.append(track)
    return long_tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vehicle-poses-world", default=VEHICLE_POSES_WORLD_PATH,
                        help="step 3 CSV (default: prepared/vehicle_poses_world.csv)")
    parser.add_argument("--min-frames", type=int, default=25,
                        help="smallest track length to keep: input steps + output steps of step 5 (default: 25)")
    args = parser.parse_args()

    all_tracks = read_vehicle_tracks(args.vehicle_poses_world)
    long_tracks = keep_long_tracks(all_tracks, args.min_frames)

    drives = []
    for track in all_tracks:
        if track.drive not in drives:
            drives.append(track.drive)
    for drive in drives:
        track_count = 0
        long_track_count = 0
        long_frame_count = 0
        for track in all_tracks:
            if track.drive == drive:
                track_count += 1
        for track in long_tracks:
            if track.drive == drive:
                long_track_count += 1
                long_frame_count += len(track.frames)
        print(f"{drive}: kept {long_track_count} of {track_count} tracks, {long_frame_count} frames")

    all_frame_count = 0
    for track in all_tracks:
        all_frame_count += len(track.frames)
    long_frame_count = 0
    track_count_by_type = {}
    for track in long_tracks:
        long_frame_count += len(track.frames)
        if track.object_type not in track_count_by_type:
            track_count_by_type[track.object_type] = 0
        track_count_by_type[track.object_type] += 1
    print(f"all drives: kept {len(long_tracks)} of {len(all_tracks)} tracks with >= {args.min_frames} frames, "
          f"{long_frame_count} of {all_frame_count} frames")
    print(f"kept tracks by type: {track_count_by_type}")


if __name__ == "__main__":
    main()
