"""Keep the vehicle tracks that are long enough for one sample.

The tracks come from read_vehicle_tracks() in data_model/poses_io.py, one per vehicle, frames in order.

Output: no file. Run as a script, it prints how many tracks and frames are kept.
"""
import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)  # for data_model/
from data_model.data_classes import VehicleTrack
from data_model.poses_io import VEHICLE_POSES_WORLD_PATH, read_vehicle_tracks


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
                        help="step 3 CSV (default: samples/vehicle_poses_world.csv)")
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
