"""Ego pose (Velodyne) and camera 2 pose in the world, per video frame, from OXTS and the calibration.

The world is East-North-Up in metres, with its origin at the drive's first GPS position: one fixed frame per drive.
Positions use the Mercator projection of the devkit's convertOxtsToPose.m,
but like pykitti only the first position is subtracted, not the whole first pose, so z stays vertical.

The OXTS packet of a frame is recorded at oxts_time, up to ~35 ms before or after the camera time of the frame.
The labels belong to the camera time, so the pose is moved to it with the packet's own velocity (ve, vn, vu) and yaw rate (wu), over time - oxts_time. Roll and pitch are kept.

Output: samples/ego_poses.csv
        drive                    2011_09_26_drive_0001
        frame                    video frame, as in vehicle_poses_velo.csv
        time                     seconds since the drive's first frame, from image_02/timestamps.txt
                                 (frames are ~0.1035 s apart)
        oxts_time                time of the OXTS packet, from oxts/timestamps.txt, on the same clock as time
        velo_r11 .. velo_r33,    T_world_velo at time, row by row, then its translation:
        velo_tx, velo_ty, velo_tz    p_world = R · p_velo + t
        cam2_r11 .. cam2_r33,    T_world_cam2 likewise. Camera 2 is the rectified image_02 camera:
        cam2_tx, cam2_ty, cam2_tz    x right, y down, z forward

Read back with read_ego_poses() in data_model/poses_io.py.

Projecting a world point into image_02: p_cam2 = R^T · (p_world - t), then pixel = K · p_cam2 / z_cam2
with K = P_rect_02[:, :3].
Not the full P_rect_02: its last column is the offset from camera 0 to camera 2, which T_world_cam2 already contains.

Naming: T_a_b is a 4x4 transform taking points from frame b into frame a, p_a = T_a_b · p_b.
"""
import argparse
import csv
import math
import os

import numpy as np

from data_model.poses_io import EGO_POSES_PATH
from data_prep.calibration import CamToCamCalibration, RigidCalibration, parse_cam_to_cam_calibration, parse_rigid_calibration
from data_prep.oxts import OxtsPacket, parse_oxts

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "raw_data")
EARTH_RADIUS = 6378137.0  # metres, as in the devkit's latlonToMercator.m

# Drives that get no ego poses, and so drop out of step 3 onwards. Their raw data and step 1 rows are kept.
EXCLUDED_DRIVES = {
    "2011_09_26_drive_0093": "GPS position jumps up to 9.8 m between frames (accuracy ~21 m): needs odometry (2b)",
}

COLUMNS = ["drive", "frame", "time", "oxts_time",
           "velo_r11", "velo_r12", "velo_r13", "velo_r21", "velo_r22", "velo_r23", "velo_r31", "velo_r32", "velo_r33",
           "velo_tx", "velo_ty", "velo_tz",
           "cam2_r11", "cam2_r12", "cam2_r13", "cam2_r21", "cam2_r22", "cam2_r23", "cam2_r31", "cam2_r32", "cam2_r33",
           "cam2_tx", "cam2_ty", "cam2_tz"]


def rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def make_4_4_homogenous_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform from a 3x3 rotation and a translation of 3."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def imu_poses_in_world(packets: list[OxtsPacket], time_shifts: list[float]) -> list[np.ndarray]:
    """T_world_imu per frame, at the camera time: a list of 4x4 homogeneous transforms, one per packet,
    p_world = T_world_imu · p_imu.
    The IMU frame is x forward, y left, z up
    (the comment in  convertOxtsToPose.m says y right, z down, but its math gives y left, z up).

    time_shifts: camera time minus OXTS time, per packet, in seconds.
    """
    scale = math.cos(packets[0].lat * math.pi / 180)  # latToScale.m, from the first frame, its 1 on equator, for Karlsruhe ~ 0.656
    origin = None
    poses = []
    for frame in range(len(packets)):
        packet = packets[frame]
        # latlonToMercator.m: x east, y north, in metres
        imu_position = np.array([scale * packet.lon * math.pi * EARTH_RADIUS / 180,
                                 scale * EARTH_RADIUS * math.log(math.tan((90 + packet.lat) * math.pi / 360)),
                                 packet.alt])
        if origin is None:
            origin = imu_position  # the first GPS position, not moved

        # move to the camera time with the packet's own velocity and yaw rate
        time_shift = time_shifts[frame]
        imu_position = imu_position + np.array([packet.ve, packet.vn, packet.vu]) * time_shift
        yaw = packet.yaw + packet.wu * time_shift

        imu_rotation = rotation_z(yaw) @ rotation_y(packet.pitch) @ rotation_x(packet.roll)
        poses.append(make_4_4_homogenous_transform(imu_rotation, imu_position - origin))
    return poses


def cam2_from_velo(velo_to_cam: RigidCalibration, cam_to_cam: CamToCamCalibration) -> np.ndarray:
    """T_cam2_velo: Velodyne -> camera 0 -> rectified camera 0 -> rectified camera 2.

    The same chain as P_rect_02 · R_rect_00 · [R|T]_velo_to_cam, with P_rect_02 = K · [I | t_cam2_rect0]
    split into the intrinsics K and the offset of camera 2 from camera 0.
    """
    T_cam0_velo = make_4_4_homogenous_transform(velo_to_cam.R, velo_to_cam.T)
    T_rect0_cam0 = make_4_4_homogenous_transform(cam_to_cam.R_rect_00, np.zeros(3))
    K = cam_to_cam.P_rect_02[:, :3]
    t_cam2_rect0 = np.linalg.solve(K, cam_to_cam.P_rect_02[:, 3])
    T_cam2_rect0 = make_4_4_homogenous_transform(np.eye(3), t_cam2_rect0)
    return T_cam2_rect0 @ T_rect0_cam0 @ T_cam0_velo


def seconds_of_day(timestamps_path: str) -> list[float]:
    """Clock time in seconds, one per line like 2011-09-26 13:02:25.964389445.
    Only the clock time is used: no drive crosses midnight."""
    seconds = []
    with open(timestamps_path) as f:
        for line in f:
            if not line.strip():
                continue
            clock = line.split()[1]
            hours, minutes, secs = clock.split(":")
            seconds.append(int(hours) * 3600 + int(minutes) * 60 + float(secs))
    return seconds


def print_timing_check(drive: str, rows: list[dict[str, str | int | float]], packets: list[OxtsPacket]):
    """Ego speed from the moved Velodyne positions over camera time, against the OXTS velocity.
    Prints the range of time - oxts_time, and the RMS and largest difference in m/s."""
    time_shifts = []
    for row in rows:
        time_shifts.append(row["time"] - row["oxts_time"])
    speed_differences = []
    for frame in range(1, len(rows)):
        previous = rows[frame - 1]
        current = rows[frame]
        distance = math.hypot(current["velo_tx"] - previous["velo_tx"], current["velo_ty"] - previous["velo_ty"])
        speed = distance / (current["time"] - previous["time"])
        # mean of the two packets
        oxts_speed = (math.hypot(packets[frame].ve, packets[frame].vn)
                      + math.hypot(packets[frame - 1].ve, packets[frame - 1].vn)) / 2
        speed_differences.append(speed - oxts_speed)
    speed_differences = np.array(speed_differences)
    print(f"{drive}: time - oxts_time {min(time_shifts) * 1000:+.0f} to {max(time_shifts) * 1000:+.0f} ms, "
          f"speed minus OXTS velocity RMS {np.sqrt(np.mean(speed_differences ** 2)):.2f}, "
          f"largest {np.abs(speed_differences).max():.2f} m/s")


def ego_pose_rows(drive: str, drive_dir: str) -> list[dict[str, str | int | float]]:
    """One row per frame of the drive, as a dict keyed by column name."""
    sync_dir = os.path.join(drive_dir, drive + "_sync")
    calib_dir = os.path.join(drive_dir, "calib")
    packets: list[OxtsPacket] = parse_oxts(os.path.join(sync_dir, "oxts"))
    camera_seconds = seconds_of_day(os.path.join(sync_dir, "image_02", "timestamps.txt"))
    oxts_seconds = seconds_of_day(os.path.join(sync_dir, "oxts", "timestamps.txt"))
    if len(camera_seconds) != len(packets) or len(oxts_seconds) != len(packets):
        raise ValueError(f"{drive}: {len(packets)} OXTS packets, {len(camera_seconds)} camera times, "
                         f"{len(oxts_seconds)} OXTS times")

    # both from the drive's first camera frame
    times = []
    oxts_times = []
    time_shifts = []
    for frame in range(len(packets)):
        times.append(camera_seconds[frame] - camera_seconds[0])
        oxts_times.append(oxts_seconds[frame] - camera_seconds[0])
        time_shifts.append(camera_seconds[frame] - oxts_seconds[frame])

    imu_to_velo = parse_rigid_calibration(os.path.join(calib_dir, "calib_imu_to_velo.txt"))
    velo_to_cam = parse_rigid_calibration(os.path.join(calib_dir, "calib_velo_to_cam.txt"))
    cam_to_cam = parse_cam_to_cam_calibration(os.path.join(calib_dir, "calib_cam_to_cam.txt"))

    T_velo_imu = make_4_4_homogenous_transform(imu_to_velo.R, imu_to_velo.T)
    T_imu_velo = np.linalg.inv(T_velo_imu)
    T_velo_cam2 = np.linalg.inv(cam2_from_velo(velo_to_cam, cam_to_cam))

    rows = []
    for frame, T_world_imu in enumerate(imu_poses_in_world(packets, time_shifts)):
        T_world_velo = T_world_imu @ T_imu_velo
        T_world_cam2 = T_world_velo @ T_velo_cam2
        row = {}
        row["drive"] = drive
        row["frame"] = frame
        row["time"] = times[frame]
        row["oxts_time"] = oxts_times[frame]
        row["velo_r11"] = T_world_velo[0, 0]
        row["velo_r12"] = T_world_velo[0, 1]
        row["velo_r13"] = T_world_velo[0, 2]
        row["velo_r21"] = T_world_velo[1, 0]
        row["velo_r22"] = T_world_velo[1, 1]
        row["velo_r23"] = T_world_velo[1, 2]
        row["velo_r31"] = T_world_velo[2, 0]
        row["velo_r32"] = T_world_velo[2, 1]
        row["velo_r33"] = T_world_velo[2, 2]
        row["velo_tx"] = T_world_velo[0, 3]
        row["velo_ty"] = T_world_velo[1, 3]
        row["velo_tz"] = T_world_velo[2, 3]
        row["cam2_r11"] = T_world_cam2[0, 0]
        row["cam2_r12"] = T_world_cam2[0, 1]
        row["cam2_r13"] = T_world_cam2[0, 2]
        row["cam2_r21"] = T_world_cam2[1, 0]
        row["cam2_r22"] = T_world_cam2[1, 1]
        row["cam2_r23"] = T_world_cam2[1, 2]
        row["cam2_r31"] = T_world_cam2[2, 0]
        row["cam2_r32"] = T_world_cam2[2, 1]
        row["cam2_r33"] = T_world_cam2[2, 2]
        row["cam2_tx"] = T_world_cam2[0, 3]
        row["cam2_ty"] = T_world_cam2[1, 3]
        row["cam2_tz"] = T_world_cam2[2, 3]
        rows.append(row)

    print_timing_check(drive, rows, packets)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", default=RAW_DIR, help="drive folders from fetch_kitti_raw.py (default: raw_data)")
    parser.add_argument("--out", default=EGO_POSES_PATH, help="CSV to write (default: samples/ego_poses.csv)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total_rows = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for drive in sorted(os.listdir(args.raw)):
            if drive in EXCLUDED_DRIVES:
                print(f"{drive}: skipped, {EXCLUDED_DRIVES[drive]}")
                continue
            rows = ego_pose_rows(drive, os.path.join(args.raw, drive))
            writer.writerows(rows)
            total_rows += len(rows)
            print(f"{drive}: {len(rows)} rows")
    print(f"wrote {total_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
