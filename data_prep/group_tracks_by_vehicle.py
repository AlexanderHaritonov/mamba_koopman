"""Cut label errors out of the vehicle tracks, then keep the tracks that are long enough for one sample.

The tracks come from read_vehicle_tracks() in data_model/poses_io.py, one per vehicle, frames in order.

Label errors: the position of a labeled car relative to the ego car jumps (ID switch, one wrong frame, frozen labels).
A frame is cut out when the car's acceleration relative to the ego car is over JUMP_ACCELERATION_THRESHOLD, or when it is in LABEL_ERROR_FRAMES. The track is split there; the parts keep the track_id.
Ego tracks are not checked.

Output: no file. Run as a script, it prints the cut frames and how many tracks and frames are kept.
"""
import argparse

import numpy as np

from data_model.data_classes import VehicleTrack
from data_model.poses_io import VEHICLE_POSES_WORLD_PATH, read_vehicle_tracks

JUMP_ACCELERATION_THRESHOLD = 60.0  # m/s², of a labeled car relative to the ego car, from frame to frame

# first and last frame to cut, in every labeled track of the drive
LABEL_ERROR_FRAMES = {
    "2011_09_26_drive_0009": (177, 181),  # labels frozen in the Velodyne frame for 177-180, then a jump at 181
}


def ego_positions_by_frame(tracks: list[VehicleTrack]) -> dict[tuple[str, int], np.ndarray]:
    """(drive, frame) -> x, y of the ego car, from the ego tracks in the list."""
    ego_positions = {}
    for track in tracks:
        if track.object_type != "Ego":
            continue
        for frame_index in range(len(track.frames)):
            ego_positions[(track.drive, int(track.frames[frame_index]))] = track.positions[frame_index, :2]
    return ego_positions


def relative_accelerations(track: VehicleTrack, ego_positions: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
    """Acceleration of the car relative to the ego car, from x and y, in m/s², one per frame.
    0 at the first and last frame.
    """
    frame_count = len(track.frames)
    relative_positions = np.zeros((frame_count, 2))
    for frame_index in range(frame_count):
        key = (track.drive, int(track.frames[frame_index]))
        if key not in ego_positions:
            raise ValueError(f"{track.drive} track {track.track_id}: no ego position at frame {key[1]}")
        relative_positions[frame_index] = track.positions[frame_index, :2] - ego_positions[key]

    # velocities[i]: from frame i-1 to frame i
    velocities = np.zeros((frame_count, 2))
    for frame_index in range(1, frame_count):
        time_step = track.times[frame_index] - track.times[frame_index - 1]
        velocities[frame_index] = (relative_positions[frame_index] - relative_positions[frame_index - 1]) / time_step

    accelerations = np.zeros(frame_count)
    for frame_index in range(1, frame_count - 1):
        time_step = (track.times[frame_index + 1] - track.times[frame_index - 1]) / 2
        velocity_change = velocities[frame_index + 1] - velocities[frame_index]
        accelerations[frame_index] = np.hypot(velocity_change[0], velocity_change[1]) / time_step
    return accelerations


def cut_frame_indices(track: VehicleTrack, ego_positions: dict[tuple[str, int], np.ndarray]) -> list[int]:
    """Indices of the frames to cut out of a labeled track, in order."""
    accelerations = relative_accelerations(track, ego_positions)
    cut_indices = []
    for frame_index in range(len(track.frames)):
        frame = int(track.frames[frame_index])
        in_error_frames = False
        if track.drive in LABEL_ERROR_FRAMES:
            first_frame, last_frame = LABEL_ERROR_FRAMES[track.drive]
            in_error_frames = first_frame <= frame <= last_frame
        if accelerations[frame_index] > JUMP_ACCELERATION_THRESHOLD or in_error_frames:
            cut_indices.append(frame_index)
    return cut_indices


def track_part(track: VehicleTrack, start: int, end: int) -> VehicleTrack:
    """Frames start .. end - 1 of the track, as a track with the same drive, track_id and type."""
    return VehicleTrack(drive=track.drive,
                        track_id=track.track_id,
                        object_type=track.object_type,
                        frames=track.frames[start:end],
                        times=track.times[start:end],
                        positions=track.positions[start:end],
                        yaws=track.yaws[start:end],
                        truncations=track.truncations[start:end])


def split_at_label_errors(tracks: list[VehicleTrack]) -> list[VehicleTrack]:
    """The tracks with their label error frames cut out, split there into parts. Ego tracks unchanged.
    Parts can be short: keep_long_tracks() comes after.
    """
    ego_positions = ego_positions_by_frame(tracks)
    parts = []
    for track in tracks:
        if track.object_type == "Ego":
            parts.append(track)
            continue
        cut_indices = cut_frame_indices(track, ego_positions)
        if not cut_indices:
            parts.append(track)
            continue
        part_start = 0
        for cut_index in cut_indices:
            if cut_index > part_start:
                parts.append(track_part(track, part_start, cut_index))
            part_start = cut_index + 1
        if part_start < len(track.frames):
            parts.append(track_part(track, part_start, len(track.frames)))
    return parts


def keep_long_tracks(tracks: list[VehicleTrack], min_frame_count: int) -> list[VehicleTrack]:
    """Tracks with at least min_frame_count frames"""
    long_tracks = []
    for track in tracks:
        if len(track.frames) >= min_frame_count:
            long_tracks.append(track)
    return long_tracks


def print_cut_frames(tracks: list[VehicleTrack]):
    """The cut frames of each labeled track, then the totals."""
    ego_positions = ego_positions_by_frame(tracks)
    cut_frame_count = 0
    cut_track_count = 0
    for track in tracks:
        if track.object_type == "Ego":
            continue
        cut_indices = cut_frame_indices(track, ego_positions)
        if not cut_indices:
            continue
        cut_frames = []
        for cut_index in cut_indices:
            cut_frames.append(int(track.frames[cut_index]))
        print(f"{track.drive} track {track.track_id} ({track.object_type}): cut frames {cut_frames}")
        cut_frame_count += len(cut_indices)
        cut_track_count += 1
    print(f"cut {cut_frame_count} frames in {cut_track_count} tracks "
          f"(relative acceleration over {JUMP_ACCELERATION_THRESHOLD:.0f} m/s², or in {LABEL_ERROR_FRAMES})\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vehicle-poses-world", default=VEHICLE_POSES_WORLD_PATH,
                        help="step 3 CSV (default: samples/vehicle_poses_world.csv)")
    parser.add_argument("--min-frames", type=int, default=25,
                        help="smallest track length to keep: input steps + output steps of step 5 (default: 25)")
    args = parser.parse_args()

    all_tracks = read_vehicle_tracks(args.vehicle_poses_world)
    print_cut_frames(all_tracks)
    split_tracks = split_at_label_errors(all_tracks)
    long_tracks = keep_long_tracks(split_tracks, args.min_frames)

    drives = []
    for track in all_tracks:
        if track.drive not in drives:
            drives.append(track.drive)
    for drive in drives:
        track_count = 0
        split_track_count = 0
        long_track_count = 0
        long_frame_count = 0
        for track in all_tracks:
            if track.drive == drive:
                track_count += 1
        for track in split_tracks:
            if track.drive == drive:
                split_track_count += 1
        for track in long_tracks:
            if track.drive == drive:
                long_track_count += 1
                long_frame_count += len(track.frames)
        print(f"{drive}: {track_count} tracks, {split_track_count} after the split, "
              f"kept {long_track_count}, {long_frame_count} frames")

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
    print(f"all drives: {len(all_tracks)} tracks, {len(split_tracks)} after the split, "
          f"kept {len(long_tracks)} with >= {args.min_frames} frames, {long_frame_count} of {all_frame_count} frames")
    print(f"kept tracks by type: {track_count_by_type}")


if __name__ == "__main__":
    main()
