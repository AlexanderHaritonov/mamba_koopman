"""Flatten the Car, Van and Truck tracklets of all drives into one CSV table, one row per vehicle per frame.

Positions are in the Velodyne frame of that video frame (x forward, y left, z up), which moves with the ego car.

Output: samples/vehicle_poses_velo.csv
        drive        2011_09_26_drive_0001
        track_id     position of the tracklet in tracklet_labels.xml (all classes counted), unique per drive
        object_type  Car, Van or Truck
        frame        video frame: oxts/data/{frame:010d}.txt and image_02/data/{frame:010d}.png
        x, y, z      centre of the box bottom, in metres
        yaw          rz in radians, as in the XML: not wrapped to +-pi
        h, w, l      box size in metres, the same in every row of a track
        occlusion, truncation, state, finished   as in tracklets.Pose and tracklets.Tracklet
"""
import argparse
import csv
import os

from data_prep.tracklets import parse_tracklets

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "raw_data")
OUT_PATH = os.path.join(PROJECT_DIR, "samples", "vehicle_poses_velo.csv")
VEHICLE_TYPES = ("Car", "Van", "Truck")
COLUMNS = ["drive", "track_id", "object_type", "frame", "x", "y", "z", "yaw", "h", "w", "l",
           "occlusion", "truncation", "state", "finished"]


def vehicle_rows(drive, tracklet_path):
    """One row per pose of every vehicle tracklet in the file, as a dict keyed by column name."""
    rows = []
    for track_id, tracklet in enumerate(parse_tracklets(tracklet_path)):
        if tracklet.objectType not in VEHICLE_TYPES:
            continue
        for pose_index, pose in enumerate(tracklet.poses):
            row = {}
            row["drive"] = drive
            row["track_id"] = track_id
            row["object_type"] = tracklet.objectType
            row["frame"] = tracklet.first_frame + pose_index
            row["x"] = pose.tx
            row["y"] = pose.ty
            row["z"] = pose.tz
            row["yaw"] = pose.rz
            row["h"] = tracklet.h
            row["w"] = tracklet.w
            row["l"] = tracklet.l
            row["occlusion"] = pose.occlusion
            row["truncation"] = pose.truncation
            row["state"] = pose.state
            row["finished"] = tracklet.finished
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", default=RAW_DIR, help="drive folders from fetch_kitti_raw.py (default: raw_data)")
    parser.add_argument("--out", default=OUT_PATH, help="CSV to write (default: samples/vehicle_poses_velo.csv)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total_rows = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for drive in sorted(os.listdir(args.raw)):
            rows = vehicle_rows(drive, os.path.join(args.raw, drive, "tracklet_labels.xml"))
            writer.writerows(rows)
            total_rows += len(rows)
            print(f"{drive}: {len(rows)} rows")
    print(f"wrote {total_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
