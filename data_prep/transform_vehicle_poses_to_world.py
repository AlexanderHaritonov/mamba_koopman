"""Car, Van, Truck and ego car poses in the fixed world frame of their drive, one row per vehicle per frame.

The bird's-eye view is (x, y, yaw). z is kept as well, for projecting back into the camera and for later use.

    samples/vehicle_poses_world.csv
        drive, track_id, object_type, frame    as in vehicle_poses_velo.csv
        time         seconds since the drive's first frame
        x, y, z      centre of the box bottom, in metres, East-North-Up world of ego_poses.csv
        yaw          where the car's nose points, seen from above: radians, counter-clockwise from East
                     (0 = East, pi/2 = North). No jump between +pi and -pi inside a track.
        h, w, l, occlusion, truncation, state, finished    as in vehicle_poses_velo.csv

The ego car is one more track per drive: track_id -1, object_type Ego, truncation 2, position at the middle of
the car's bottom, every frame of the drive. h, w, l, occlusion, state and finished are empty.

Drives without ego poses are skipped.

Printed checks:
    static      parked cars the ego car drove past: how many stay within 0.2 m
    heading     yaw minus direction of motion, for vehicles faster than 3 m/s: should be near 0 degrees
    height      car z minus road height under the ego car, for cars within 20 m: should be near 0 m
    ego heading the heading check for the ego car
"""
import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from compute_ego_poses import EgoPoses, read_ego_poses

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEHICLE_POSES_VELO_PATH = os.path.join(PROJECT_DIR, "samples", "vehicle_poses_velo.csv")
EGO_POSES_PATH = os.path.join(PROJECT_DIR, "samples", "ego_poses.csv")
OUT_PATH = os.path.join(PROJECT_DIR, "samples", "vehicle_poses_world.csv")
COLUMNS = ["drive", "track_id", "object_type", "frame", "time", "x", "y", "z", "yaw", "h", "w", "l",
           "occlusion", "truncation", "state", "finished"]

VeloRow = dict[str, str]  # one row of vehicle_poses_velo.csv: column name -> value as text
VeloTracks = dict[int, list[VeloRow]]  # track_id -> rows of the track, frames in order
VeloTracksByDrive = dict[str, VeloTracks]  # drive -> tracks of the drive


class WorldRow(TypedDict):
    """One row of vehicle_poses_world.csv. Columns copied from vehicle_poses_velo.csv stay text."""
    drive: str
    track_id: str
    object_type: str
    frame: str
    time: float
    x: float
    y: float
    z: float
    yaw: float
    h: str
    w: str
    l: str
    occlusion: str
    truncation: str
    state: str
    finished: str


VELODYNE_HEIGHT = 1.73  # metres above the road
EGO_TRACK_ID = -1
EGO_OBJECT_TYPE = "Ego"
EGO_TRUNCATION = 2  # outside the image
EGO_POINT_VELO = np.array([0.53, 0.0, -VELODYNE_HEIGHT])  # middle of the ego car's bottom, in the Velodyne frame
DOT_SPREAD = 0.2  # metres, largest world spread of a parked car
STATIC_SPREAD = 1.0  # metres, largest world spread of a standing track
PASSED_SPREAD = 2.0  # metres, smallest Velodyne spread of a track the ego car drove past
MOVING_SPEED = 3.0  # m/s, smallest speed for the heading check
HEADING_GAP = 5  # frames between the two positions that give the direction of motion
NEAR_DISTANCE = 20.0  # metres, largest distance from the ego car for the height check


def read_vehicle_poses_velo(filepath) -> VeloTracksByDrive:
    """Load vehicle_poses_velo.csv."""
    tracks_by_drive = {}
    with open(filepath, newline="") as f:
        for row in csv.DictReader(f):
            drive = row["drive"]
            track_id = int(row["track_id"])
            if drive not in tracks_by_drive:
                tracks_by_drive[drive] = {}
            if track_id not in tracks_by_drive[drive]:
                tracks_by_drive[drive][track_id] = []
            tracks_by_drive[drive][track_id].append(row)
    return tracks_by_drive


def world_track_rows(velo_rows: list[VeloRow], ego_poses: EgoPoses) -> list[WorldRow]:
    """The rows of one track in the world frame."""
    times = []
    positions_world = []
    wrapped_yaws = []
    for velo_row in velo_rows:
        ego_pose = ego_poses[(velo_row["drive"], int(velo_row["frame"]))]
        rotation = ego_pose.T_world_velo[:3, :3]
        translation = ego_pose.T_world_velo[:3, 3]
        times.append(ego_pose.time)

        position_velo = np.array([float(velo_row["x"]), float(velo_row["y"]), float(velo_row["z"])])
        positions_world.append(rotation @ position_velo + translation)

        # Box length axis in the world, seen from above
        yaw_velo = float(velo_row["yaw"])
        forward_velo = np.array([math.cos(yaw_velo), math.sin(yaw_velo), 0.0])
        forward_world = rotation @ forward_velo
        wrapped_yaws.append(math.atan2(forward_world[1], forward_world[0]))
    unwrapped_yaws = np.unwrap(wrapped_yaws)

    rows = []
    for row_index in range(len(velo_rows)):
        velo_row = velo_rows[row_index]
        position_world = positions_world[row_index]
        rows.append(WorldRow(drive=velo_row["drive"],
                             track_id=velo_row["track_id"],
                             object_type=velo_row["object_type"],
                             frame=velo_row["frame"],
                             time=times[row_index],
                             x=float(position_world[0]),
                             y=float(position_world[1]),
                             z=float(position_world[2]),
                             yaw=float(unwrapped_yaws[row_index]),
                             h=velo_row["h"],
                             w=velo_row["w"],
                             l=velo_row["l"],
                             occlusion=velo_row["occlusion"],
                             truncation=velo_row["truncation"],
                             state=velo_row["state"],
                             finished=velo_row["finished"]))
    return rows


def ego_track_rows(drive: str, ego_poses: EgoPoses) -> list[WorldRow]:
    """The ego car as one more track of the drive, every frame."""
    frames = []
    for pose_drive, frame in ego_poses:
        if pose_drive == drive:
            frames.append(frame)
    frames.sort()

    times = []
    positions_world = []
    wrapped_yaws = []
    for frame in frames:
        ego_pose = ego_poses[(drive, frame)]
        rotation = ego_pose.T_world_velo[:3, :3]
        translation = ego_pose.T_world_velo[:3, 3]
        times.append(ego_pose.time)
        positions_world.append(rotation @ EGO_POINT_VELO + translation)
        forward_world = rotation @ np.array([1.0, 0.0, 0.0])
        wrapped_yaws.append(math.atan2(forward_world[1], forward_world[0]))
    unwrapped_yaws = np.unwrap(wrapped_yaws)

    rows = []
    for row_index in range(len(frames)):
        position_world = positions_world[row_index]
        rows.append(WorldRow(drive=drive,
                             track_id=str(EGO_TRACK_ID),
                             object_type=EGO_OBJECT_TYPE,
                             frame=str(frames[row_index]),
                             time=times[row_index],
                             x=float(position_world[0]),
                             y=float(position_world[1]),
                             z=float(position_world[2]),
                             yaw=float(unwrapped_yaws[row_index]),
                             h="",
                             w="",
                             l="",
                             occlusion="",
                             truncation=str(EGO_TRUNCATION),
                             state="",
                             finished=""))
    return rows


def xy_spread(rows: list[VeloRow] | list[WorldRow]) -> float:
    """Largest xy distance of a track's positions from their mean, in metres."""
    xs = []
    ys = []
    for row in rows:
        xs.append(float(row["x"]))
        ys.append(float(row["y"]))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    largest_distance = 0.0
    for point_index in range(len(xs)):
        distance = math.hypot(xs[point_index] - mean_x, ys[point_index] - mean_y)
        largest_distance = max(largest_distance, distance)
    return largest_distance


def wrap_angle(angle: float) -> float:
    """Angle in radians wrapped to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def heading_errors(world_rows: list[WorldRow]) -> list[float]:
    """|yaw - direction of motion| in degrees, for each frame faster than MOVING_SPEED."""
    errors = []
    for row_index in range(len(world_rows) - HEADING_GAP):
        start = world_rows[row_index]
        end = world_rows[row_index + HEADING_GAP]
        dx = end["x"] - start["x"]
        dy = end["y"] - start["y"]
        dt = end["time"] - start["time"]
        if math.hypot(dx, dy) / dt < MOVING_SPEED:
            continue
        motion_direction = math.atan2(dy, dx)
        errors.append(abs(math.degrees(wrap_angle(start["yaw"] - motion_direction))))
    return errors


def height_errors(world_rows: list[WorldRow], ego_poses: EgoPoses) -> list[float]:
    """Car z minus road height under the ego car, in metres, for each frame within NEAR_DISTANCE."""
    errors = []
    for row in world_rows:
        ego_pose = ego_poses[(row["drive"], int(row["frame"]))]
        velo_position = ego_pose.T_world_velo[:3, 3]
        if math.hypot(row["x"] - velo_position[0], row["y"] - velo_position[1]) > NEAR_DISTANCE:
            continue
        errors.append(row["z"] - (velo_position[2] - VELODYNE_HEIGHT))
    return errors


@dataclass
class CheckValues:
    """Values of the printed checks."""
    static_spreads: list[float]  # metres
    heading_errors_degrees: list[float]
    height_errors_metres: list[float]
    ego_heading_errors_degrees: list[float]


def transform_drive_to_world(drive: str, velo_tracks: VeloTracks,
                             ego_poses: EgoPoses) -> tuple[list[WorldRow], CheckValues]:
    """World rows of all tracks of one drive and of its ego car, and their check values."""
    drive_rows = []
    check_values = CheckValues(static_spreads=[], heading_errors_degrees=[], height_errors_metres=[],
                               ego_heading_errors_degrees=[])
    for velo_rows in velo_tracks.values():
        world_rows = world_track_rows(velo_rows, ego_poses)
        drive_rows.extend(world_rows)

        world_spread = xy_spread(world_rows)
        if xy_spread(velo_rows) > PASSED_SPREAD and world_spread < STATIC_SPREAD:
            check_values.static_spreads.append(world_spread)
        check_values.heading_errors_degrees.extend(heading_errors(world_rows))
        check_values.height_errors_metres.extend(height_errors(world_rows, ego_poses))

    ego_rows = ego_track_rows(drive, ego_poses)
    drive_rows.extend(ego_rows)
    check_values.ego_heading_errors_degrees.extend(heading_errors(ego_rows))
    return drive_rows, check_values


def check_summary(check_values: CheckValues) -> str:
    """One line with the three checks."""
    static_spreads = check_values.static_spreads
    heading_errors_degrees = check_values.heading_errors_degrees
    height_errors_metres = check_values.height_errors_metres
    dot_count = 0
    for spread in static_spreads:
        if spread < DOT_SPREAD:
            dot_count += 1
    parts = []

    if static_spreads:
        parts.append(f"static {dot_count}/{len(static_spreads)} under {DOT_SPREAD} m (median {np.median(static_spreads):.2f} m)")
    else:
        parts.append("static none")

    if heading_errors_degrees:
        backwards_count = 0
        for error in heading_errors_degrees:
            if error > 90:
                backwards_count += 1
        parts.append(f"heading median {np.median(heading_errors_degrees):.1f} deg, {backwards_count}/{len(heading_errors_degrees)} over 90 deg")
    else:
        parts.append("heading none moving")

    if height_errors_metres:
        parts.append(f"height median {np.median(height_errors_metres):+.2f} m, "
                     f"5-95% {np.percentile(height_errors_metres, 5):+.2f}..{np.percentile(height_errors_metres, 95):+.2f} m")
    else:
        parts.append("height none near")

    ego_heading_errors_degrees = check_values.ego_heading_errors_degrees
    if ego_heading_errors_degrees:
        parts.append(f"ego heading median {np.median(ego_heading_errors_degrees):.1f} deg, "
                     f"max {np.max(ego_heading_errors_degrees):.1f} deg")
    else:
        parts.append("ego heading none moving")
        
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vehicle-poses-velo", default=VEHICLE_POSES_VELO_PATH,
                        help="step 1 CSV (default: samples/vehicle_poses_velo.csv)")
    parser.add_argument("--ego-poses", default=EGO_POSES_PATH, help="step 2 CSV (default: samples/ego_poses.csv)")
    parser.add_argument("--out", default=OUT_PATH, help="CSV to write (default: samples/vehicle_poses_world.csv)")
    args = parser.parse_args()

    tracks_by_drive = read_vehicle_poses_velo(args.vehicle_poses_velo)
    ego_poses = read_ego_poses(args.ego_poses)
    drives_with_ego_poses = set()
    for drive, frame in ego_poses:
        drives_with_ego_poses.add(drive)

    all_check_values = CheckValues(static_spreads=[], heading_errors_degrees=[], height_errors_metres=[],
                                   ego_heading_errors_degrees=[])
    total_rows = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for drive in tracks_by_drive:
            velo_tracks = tracks_by_drive[drive]
            if drive not in drives_with_ego_poses:
                skipped_rows = 0
                for velo_rows in velo_tracks.values():
                    skipped_rows += len(velo_rows)
                print(f"{drive}: skipped {skipped_rows} rows, no ego poses")
                continue

            drive_rows, check_values = transform_drive_to_world(drive, velo_tracks, ego_poses)
            writer.writerows(drive_rows)
            print(f"{drive}: {len(drive_rows)} rows, {len(velo_tracks)} tracks + ego | {check_summary(check_values)}")
            total_rows += len(drive_rows)
            all_check_values.static_spreads.extend(check_values.static_spreads)
            all_check_values.heading_errors_degrees.extend(check_values.heading_errors_degrees)
            all_check_values.height_errors_metres.extend(check_values.height_errors_metres)
            all_check_values.ego_heading_errors_degrees.extend(check_values.ego_heading_errors_degrees)

    print(f"all drives: {check_summary(all_check_values)}")
    print(f"wrote {total_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
