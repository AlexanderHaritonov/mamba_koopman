"""Parse KITTI raw tracklet_labels.xml: the annotated object tracks of one drive, in the Velodyne frame.

The classes mirror the XML: same names and types as in the devkit's cpp/tracklets.h.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class Pose:
    tx: float  # centre of the box bottom, in metres
    ty: float
    tz: float
    rx: float  # always 0
    ry: float  # always 0
    rz: float  # yaw in radians, not wrapped to +-pi
    state: int  # 0 unset, 1 interpolated, 2 labeled
    occlusion: int  # -1 unset, 0 visible, 1 partly, 2 fully
    occlusion_kf: int  # 1 on occlusion keyframes
    truncation: int  # 0 in image, 1 truncated, 2 outside image, 99 unclear ("behind image" in the devkit)
    amt_occlusion: float  # amt_*: Mechanical Turk labels
    amt_occlusion_kf: int
    amt_border_l: float
    amt_border_r: float
    amt_border_kf: int


@dataclass
class Tracklet:
    objectType: str  # Car, Van, Truck, Pedestrian, Person (sitting), Cyclist, Tram or Misc
    h: float  # box size in metres, fixed for the whole track
    w: float
    l: float
    first_frame: int  # video frame of the first pose, the others follow without gaps
    poses: list[Pose]
    finished: int  # 0 if not fully labeled


def parse_pose(element) -> Pose:
    return Pose(tx=float(element.findtext("tx")),
                ty=float(element.findtext("ty")),
                tz=float(element.findtext("tz")),
                rx=float(element.findtext("rx")),
                ry=float(element.findtext("ry")),
                rz=float(element.findtext("rz")),
                state=int(element.findtext("state")),
                occlusion=int(element.findtext("occlusion")),
                occlusion_kf=int(element.findtext("occlusion_kf")),
                truncation=int(element.findtext("truncation")),
                amt_occlusion=float(element.findtext("amt_occlusion")),
                amt_occlusion_kf=int(element.findtext("amt_occlusion_kf")),
                amt_border_l=float(element.findtext("amt_border_l")),
                amt_border_r=float(element.findtext("amt_border_r")),
                amt_border_kf=int(element.findtext("amt_border_kf")))


def parse_tracklet(element) -> Tracklet:
    poses = []
    for item in element.find("poses").findall("item"):
        poses.append(parse_pose(item))
    return Tracklet(objectType=element.findtext("objectType"),
                    h=float(element.findtext("h")),
                    w=float(element.findtext("w")),
                    l=float(element.findtext("l")),
                    first_frame=int(element.findtext("first_frame")),
                    poses=poses,
                    finished=int(element.findtext("finished")))


def parse_tracklets(filepath) -> list[Tracklet]:
    """Load all tracklets of a tracklet_labels.xml file, in file order."""
    tracklets = []
    for item in ET.parse(filepath).getroot().find("tracklets").findall("item"):
        tracklets.append(parse_tracklet(item))
    return tracklets
