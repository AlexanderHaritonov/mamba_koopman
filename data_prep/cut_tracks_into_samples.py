"""Cut the vehicle tracks into samples: one car at one time t, with its past and future positions.

t is the last observed frame:
    input   frames t - input_steps + 1 .. t     (10 positions by default, the last one is the car at t)
    output  frames t + 1 .. t + output_steps    (15 positions by default)
So t runs from the track's first frame + input_steps - 1 to its last frame - output_steps.
"""
import argparse
import os
from dataclasses import dataclass

import numpy as np

from group_tracks_by_vehicle import VehicleTrack, keep_long_tracks, read_vehicle_tracks

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEHICLE_POSES_WORLD_PATH = os.path.join(PROJECT_DIR, "prepared", "vehicle_poses_world.csv")


@dataclass
class Window:
    """One sample in the world frame, before re-centring."""
    track: VehicleTrack
    anchor_index: int  # index of time t in the track arrays
    past_positions: np.ndarray  # (input_steps, 3) x, y, z in the world, frames up to t
    past_yaws: np.ndarray  # (input_steps,)
    future_positions: np.ndarray  # (output_steps, 3) x, y, z in the world, frames after t
    future_yaws: np.ndarray  # (output_steps,)


def cut_windows(track: VehicleTrack, input_steps: int, output_steps: int) -> list[Window]:
    """All windows of one track, one per time t."""
    windows = []
    first_anchor_index = input_steps - 1
    last_anchor_index = len(track.frames) - output_steps - 1
    for anchor_index in range(first_anchor_index, last_anchor_index + 1):
        past_start = anchor_index - input_steps + 1
        future_end = anchor_index + output_steps + 1
        windows.append(Window(track=track,
                              anchor_index=anchor_index,
                              past_positions=track.positions[past_start:anchor_index + 1],
                              past_yaws=track.yaws[past_start:anchor_index + 1],
                              future_positions=track.positions[anchor_index + 1:future_end],
                              future_yaws=track.yaws[anchor_index + 1:future_end]))
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vehicle-poses-world", default=VEHICLE_POSES_WORLD_PATH,
                        help="step 3 CSV (default: prepared/vehicle_poses_world.csv)")
    parser.add_argument("--input-steps", type=int, default=10, help="positions in, up to and with t (default: 10)")
    parser.add_argument("--output-steps", type=int, default=15, help="positions out, after t (default: 15)")
    args = parser.parse_args()

    tracks = keep_long_tracks(read_vehicle_tracks(args.vehicle_poses_world), args.input_steps + args.output_steps)
    windows = []
    for track in tracks:
        windows.extend(cut_windows(track, args.input_steps, args.output_steps))

    expected_window_count = 0
    for track in tracks:
        expected_window_count += len(track.frames) - (args.input_steps + args.output_steps) + 1
    print(f"{len(windows)} windows from {len(tracks)} tracks "
          f"({args.input_steps} in, {args.output_steps} out; expected {expected_window_count})")

    example = windows[0]
    frames = example.track.frames
    past_start = example.anchor_index - args.input_steps + 1
    future_end = example.anchor_index + args.output_steps
    print(f"first window: {example.track.drive} track {example.track.track_id}, t = frame {frames[example.anchor_index]}, "
          f"input frames {frames[past_start]}..{frames[example.anchor_index]}, "
          f"output frames {frames[example.anchor_index + 1]}..{frames[future_end]}")


if __name__ == "__main__":
    main()
