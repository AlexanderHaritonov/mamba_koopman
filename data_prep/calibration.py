"""Parse KITTI raw calibration files: the rigid calib_imu_to_velo.txt and calib_velo_to_cam.txt, and the camera 2 part of calib_cam_to_cam.txt."""
from dataclasses import dataclass

import numpy as np


@dataclass
class RigidCalibration:
    calib_time: str  # when the calibration was made, e.g. 25-May-2012 16:47:16
    R: np.ndarray  # 3x3 rotation: rotates points from the first frame into the second
    T: np.ndarray  # 3 translation in metres, applied after R


@dataclass
class CamToCamCalibration:
    """What projecting into image_02 needs
    pixel = P_rect_02 · R_rect_00 · [R|T]_velo_to_cam · point_velo, in homogeneous coordinates."""
    calib_time: str  # when the calibration was made, e.g. 09-Jan-2012 13:57:47
    R_rect_00: np.ndarray  # 3x3 rectifying rotation: camera 0 frame into the rectified camera 0 frame
    P_rect_02: np.ndarray  # 3x4 projection: rectified camera 0 frame into image_02 pixels
    S_rect_02: np.ndarray  # 2 image_02 size in pixels after rectification: width, height


def parse_rigid_calibration(filepath) -> RigidCalibration:
    """Load calib_imu_to_velo.txt or calib_velo_to_cam.txt.

    calib_velo_to_cam.txt also has delta_f and delta_c, which are zero in every drive and not read.
    """
    lines = {}
    with open(filepath) as f:
        for line in f:
            name, _, value = line.partition(":")
            lines[name] = value.strip()
    return RigidCalibration(calib_time=lines["calib_time"],
                            R=np.array(lines["R"].split(), dtype=float).reshape(3, 3),
                            T=np.array(lines["T"].split(), dtype=float).reshape(3))


def parse_cam_to_cam_calibration(filepath) -> CamToCamCalibration:
    """Load the camera 2 entries of calib_cam_to_cam.txt. The other cameras and the raw (unrectified) K_, D_, R_, T_ entries are not read: image_02/data is already rectified."""
    lines = {}
    with open(filepath) as f:
        for line in f:
            name, _, value = line.partition(":")
            lines[name] = value.strip()
    return CamToCamCalibration(calib_time=lines["calib_time"],
                               R_rect_00=np.array(lines["R_rect_00"].split(), dtype=float).reshape(3, 3),
                               P_rect_02=np.array(lines["P_rect_02"].split(), dtype=float).reshape(3, 4),
                               S_rect_02=np.array(lines["S_rect_02"].split(), dtype=float).reshape(2))
