"""Parse KITTI raw OXTS GPS/IMU packets (oxts/data/*.txt).

The OXTS fields mirror oxts/dataformat.txt, in file order. Like the devkit's loadOxtsliteData.m, this only reads the numbers,
"""
import os
from dataclasses import dataclass


@dataclass
class OxtsPacket:
    lat: float  # latitude of the OXTS unit, in degrees
    lon: float  # longitude, in degrees
    alt: float  # altitude, in metres
    roll: float  # radians, 0 = level, positive = left side up, range -pi .. +pi
    pitch: float  # radians, 0 = level, positive = front down, range -pi/2 .. +pi/2
    yaw: float  # heading in radians, 0 = east, positive = counter clockwise, range -pi .. +pi
    vn: float  # velocity towards north, in m/s
    ve: float  # velocity towards east
    vf: float  # forward velocity, parallel to the earth surface
    vl: float  # leftward velocity, parallel to the earth surface
    vu: float  # upward velocity, perpendicular to the earth surface
    ax: float  # acceleration in m/s^2, along the vehicle front
    ay: float  # along the vehicle left
    az: float  # along the vehicle top (dataformat.txt calls it "ay" a second time)
    af: float  # forward acceleration
    al: float  # leftward acceleration
    au: float  # upward acceleration
    wx: float  # angular rate in rad/s, around the vehicle x axis
    wy: float  # around y
    wz: float  # around z
    wf: float  # around the forward axis
    wl: float  # around the leftward axis
    wu: float  # around the upward axis
    pos_accuracy: float  # position accuracy north/east, in metres (dataformat.txt says "velocity accuracy")
    vel_accuracy: float  # velocity accuracy north/east, in m/s
    navstat: int  # navigation status
    numsats: int  # number of satellites tracked by the primary GPS receiver
    posmode: int  # position mode of the primary GPS receiver, -1 on packets KITTI filled in by interpolation
    velmode: int  # velocity mode, -1 likewise
    orimode: int  # orientation mode, -1 likewise


def parse_oxts_packet(filepath) -> OxtsPacket:
    """Load one OXTS packet: a single line of 30 numbers.

    The five integer fields are usually written as integers, but as 4.00000000000000 or -1.00000000000000 in interpolated packets, hence int(float(...)).
    """
    with open(filepath) as f:
        values = f.read().split()
    return OxtsPacket(lat=float(values[0]),
                      lon=float(values[1]),
                      alt=float(values[2]),
                      roll=float(values[3]),
                      pitch=float(values[4]),
                      yaw=float(values[5]),
                      vn=float(values[6]),
                      ve=float(values[7]),
                      vf=float(values[8]),
                      vl=float(values[9]),
                      vu=float(values[10]),
                      ax=float(values[11]),
                      ay=float(values[12]),
                      az=float(values[13]),
                      af=float(values[14]),
                      al=float(values[15]),
                      au=float(values[16]),
                      wx=float(values[17]),
                      wy=float(values[18]),
                      wz=float(values[19]),
                      wf=float(values[20]),
                      wl=float(values[21]),
                      wu=float(values[22]),
                      pos_accuracy=float(values[23]),
                      vel_accuracy=float(values[24]),
                      navstat=int(float(values[25])),
                      numsats=int(float(values[26])),
                      posmode=int(float(values[27])),
                      velmode=int(float(values[28])),
                      orimode=int(float(values[29])))


def parse_oxts(oxts_dir) -> list[OxtsPacket]:
    """Load all packets of one drive's oxts folder, one per video frame: packet i is frame i."""
    data_dir = os.path.join(oxts_dir, "data")
    packets = []
    for frame in range(len(os.listdir(data_dir))):
        packets.append(parse_oxts_packet(os.path.join(data_dir, f"{frame:010d}.txt")))
    return packets
