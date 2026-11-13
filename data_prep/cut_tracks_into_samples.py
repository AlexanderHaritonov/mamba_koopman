"""Cut the vehicle tracks into samples: one car at one time t, with its past and future positions.

t is the last observed frame:
    input   frames t - input_steps + 1 .. t     (10 positions by default, the last one is the car at t)
    output  frames t + 1 .. t + output_steps    (15 positions by default)
So t runs from the track's first frame + input_steps - 1 to its last frame - output_steps.

Past and future positions are re-centred on the car's pose at t (car_frame.py): x, y, yaw, facing +x.
Each sample also keeps where it comes from (drive, track, type, frame of t, world pose at t) and how far the car moves and turns over the whole window.
"""
import argparse
import math
import os
from dataclasses import dataclass

import numpy as np

from car_frame import car_frame_to_world, world_to_car_frame
from group_tracks_by_vehicle import VehicleTrack, keep_long_tracks, read_vehicle_tracks

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEHICLE_POSES_WORLD_PATH = os.path.join(PROJECT_DIR, "prepared", "vehicle_poses_world.csv")
CHECK_MOVING_SPEED = 0.6  # m/s, average speed from first to last frame of a window, only for the printed check
CHECK_TURN_YAW_CHANGE = math.radians(15)  # smallest yaw change of a turn, only for the printed check


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


def anchor_pose(window: Window) -> np.ndarray:
    """The car's world pose at t: x, y, z, yaw."""
    position = window.track.positions[window.anchor_index]
    yaw = window.track.yaws[window.anchor_index]
    return np.array([position[0], position[1], position[2], yaw])


def window_in_car_frame(window: Window) -> tuple[np.ndarray, np.ndarray]:
    """Past (input_steps, 3) and future (output_steps, 3) as x, y, yaw in the car frame at t."""
    pose = anchor_pose(window)
    past_xy, past_yaws = world_to_car_frame(window.past_positions[:, :2], window.past_yaws, pose)
    future_xy, future_yaws = world_to_car_frame(window.future_positions[:, :2], window.future_yaws, pose)
    past = np.column_stack([past_xy, past_yaws])
    future = np.column_stack([future_xy, future_yaws])
    return past, future


@dataclass
class Sample:
    """One sample as it goes into the samples file."""
    past: np.ndarray  # (input_steps, 3) x, y, yaw in the car frame at t
    future: np.ndarray  # (output_steps, 3) x, y, yaw in the car frame at t
    anchor_pose: np.ndarray  # (4,) x, y, z, yaw of the car at t, in the world
    drive: str
    track_id: int
    object_type: str  # Car, Van or Truck
    frame: int  # video frame of t
    displacement: float  # metres from the first input to the last output position, >= 0
    yaw_change: float  # radians from the first input to the last output frame, + = left turn


def make_sample(window: Window) -> Sample:
    """The window in the car frame, with its metadata."""
    past, future = window_in_car_frame(window)
    first_position = window.past_positions[0]
    last_position = window.future_positions[-1]
    return Sample(past=past,
                  future=future,
                  anchor_pose=anchor_pose(window),
                  drive=window.track.drive,
                  track_id=window.track.track_id,
                  object_type=window.track.object_type,
                  frame=int(window.track.frames[window.anchor_index]),
                  displacement=float(np.hypot(last_position[0] - first_position[0],
                                              last_position[1] - first_position[1])),
                  yaw_change=float(window.future_yaws[-1] - window.past_yaws[0]))


def window_duration(window: Window) -> float:
    """Seconds from the first input to the last output frame."""
    first_index = window.anchor_index - len(window.past_yaws) + 1
    last_index = window.anchor_index + len(window.future_yaws)
    return float(window.track.times[last_index] - window.track.times[first_index])


def print_window_checks(tracks: list[VehicleTrack], windows: list[Window]):
    """Number of windows against the expected number, and the frames of the first window."""
    example = windows[0]
    input_steps = len(example.past_yaws)
    output_steps = len(example.future_yaws)

    expected_window_count = 0
    for track in tracks:
        expected_window_count += len(track.frames) - (input_steps + output_steps) + 1
    print(f"{len(windows)} windows from {len(tracks)} tracks "
          f"({input_steps} in, {output_steps} out; expected {expected_window_count})")

    frames = example.track.frames
    past_start = example.anchor_index - input_steps + 1
    future_end = example.anchor_index + output_steps
    print(f"first window: {example.track.drive} track {example.track.track_id}, t = frame {frames[example.anchor_index]}, "
          f"input frames {frames[past_start]}..{frames[example.anchor_index]}, "
          f"output frames {frames[example.anchor_index + 1]}..{frames[future_end]}")


def print_car_frame_checks(windows: list[Window]):
    """The car at t is at 0; going back to the world gives the world poses; moving cars mostly drive towards +x."""
    largest_offset_at_t = 0.0
    largest_round_trip_error = 0.0
    moving_count = 0
    ahead_count = 0
    for window in windows:
        past, future = window_in_car_frame(window)
        largest_offset_at_t = max(largest_offset_at_t, float(np.abs(past[-1]).max()))

        back_xy, back_yaws = car_frame_to_world(future[:, :2], future[:, 2], anchor_pose(window))
        position_error = float(np.abs(back_xy - window.future_positions[:, :2]).max())
        yaw_error = float(np.abs(back_yaws - window.future_yaws).max())
        largest_round_trip_error = max(largest_round_trip_error, position_error, yaw_error)

        displacement = float(np.hypot(future[-1, 0] - past[0, 0], future[-1, 1] - past[0, 1]))
        if displacement / window_duration(window) > CHECK_MOVING_SPEED:
            moving_count += 1
            if future[-1, 0] > 0:
                ahead_count += 1
    print(f"car frame: largest |x, y, yaw| at t {largest_offset_at_t:.1e} (should be 0), "
          f"largest error after going back to the world {largest_round_trip_error:.1e}")
    print(f"moving windows (> {CHECK_MOVING_SPEED} m/s): {moving_count}, "
          f"last future position ahead of the car (x > 0): {ahead_count} ({100 * ahead_count / moving_count:.1f}%)")


def print_sample_checks(windows: list[Window], samples: list[Sample]):
    """Samples per vehicle type; in moving turns, the sign of yaw_change matches the side the path bends to."""
    sample_count_by_type = {}
    turn_count = 0
    matching_side_count = 0
    for sample_index in range(len(samples)):
        sample = samples[sample_index]
        if sample.object_type not in sample_count_by_type:
            sample_count_by_type[sample.object_type] = 0
        sample_count_by_type[sample.object_type] += 1

        speed = sample.displacement / window_duration(windows[sample_index])
        if speed <= CHECK_MOVING_SPEED or abs(sample.yaw_change) <= CHECK_TURN_YAW_CHANGE:
            continue
        turn_count += 1
        # Side of the last position from the line through the first position, along the first heading: + = left
        start_x, start_y, start_yaw = sample.past[0]
        end_x, end_y = sample.future[-1, :2]
        side = math.cos(start_yaw) * (end_y - start_y) - math.sin(start_yaw) * (end_x - start_x)
        if side * sample.yaw_change > 0:
            matching_side_count += 1
    print(f"samples by type: {sample_count_by_type}")
    print(f"moving turns (> {math.degrees(CHECK_TURN_YAW_CHANGE):.0f} deg): {turn_count}, "
          f"path bends to the side of yaw_change: {matching_side_count} ({100 * matching_side_count / turn_count:.1f}%)")


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
    samples = []
    for window in windows:
        samples.append(make_sample(window))

    print_window_checks(tracks, windows)
    print_car_frame_checks(windows)
    print_sample_checks(windows, samples)


if __name__ == "__main__":
    main()
