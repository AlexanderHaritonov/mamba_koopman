"""World BEV map of one drive: all tracks from above, in the drive's world frame (x East, y North).

Parked cars should be dots. A labeled track is parked if all its positions stay within PARKED_SPREAD of their mean.
The parked panels show each parked car's positions around their mean, at true scale, with a DOT_RMS circle.
With a frame, the map also shows each car at that frame, and the predictions made at that frame.

The SVGs have no title and no legend: world_bev_site.py puts them into HTML pages with both.

Script: one drive, and a table of its parked tracks in the terminal.
Output: plots/world_bev_<drive>[_frame<F>[_<predictor>]].svg, plots/world_bev_<drive>_parked.svg
Reads samples/vehicle_poses_world.csv, with --predictor also samples/samples_<split>.npz and
predictions/<predictor>_<split>.npz (split from data_model/drive_split.py).
"""
import argparse
import io
import math
import os
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from data_model.car_frame import car_frame_to_world
from data_model.data_classes import Predictions, Samples, VehicleTrack
from data_model.drive_split import TEST_DRIVES, VALIDATION_DRIVES
from data_model.poses_io import read_vehicle_tracks
from data_model.predictions_io import load_predictions
from data_model.samples_io import read_samples

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")

PARKED_SPREAD = 1.0  # m, largest distance from the mean position of a parked track, as STATIC_SPREAD in step 3
DOT_RMS = 0.2  # m, a parked car is a dot if its RMS distance from the mean is below this
PANEL_HALF_SIZE = 0.5  # m, parked panels show 1 m x 1 m
PANELS_PER_ROW = 8
HEADING_LENGTH = 2.0  # m, line from a car's position towards its nose
ELLIPSE_POINT_COUNT = 37

FIGURE_WIDTH = 8.0  # inches
PANEL_ROW_HEIGHT = 1.5  # inches
THUMBNAIL_WIDTH = 4.0  # inches
THUMBNAIL_HEIGHT = 3.0  # inches
THUMBNAIL_DPI = 100

# Reference palette of the plotting guide
SURFACE_COLOR = "#fcfcfb"
EGO_COLOR = "#0b0b0b"
MOVING_COLOR = "#2a78d6"
PARKED_COLOR = "#eb6834"
PREDICTION_COLOR = "#1baf7a"
TRUE_FUTURE_COLOR = "#52514e"
TEXT_COLOR = "#52514e"
MUTED_COLOR = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"

# Marks, in points. The HTML legend uses the same values.
TRACK_LINE_WIDTH = 0.6
TRACK_DOT_SIZE = 1.2  # diameter
CAR_DOT_SIZE = 5.0  # diameter
CAR_DOT_EDGE_WIDTH = 0.8
HEADING_LINE_WIDTH = 1.5
MU_LINE_WIDTH = 1.2
SIGMA_LINE_WIDTH = 0.5
SIGMA_OPACITY = 0.7
TRUE_FUTURE_LINE_WIDTH = 0.9
TRUE_FUTURE_DASH = (3.0, 2.0)  # dash, gap
PANEL_DOT_SIZE = 1.5  # diameter
DOT_CIRCLE_LINE_WIDTH = 0.6


# Geometry: numpy only, no plotting

def offsets_from_mean(track: VehicleTrack) -> np.ndarray:
    """x, y of every frame minus the track's mean x, y. (N, 2), metres."""
    xy = track.positions[:, :2]
    return xy - xy.mean(axis=0)


def track_spread(track: VehicleTrack) -> tuple[float, float]:
    """RMS and largest distance of the track's positions from their mean, metres."""
    offsets = offsets_from_mean(track)
    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    rms = float(np.sqrt(np.mean(distances ** 2)))
    largest = float(distances.max())
    return rms, largest


def is_parked(track: VehicleTrack) -> bool:
    """A labeled track whose positions all stay within PARKED_SPREAD of their mean."""
    if track.object_type == "Ego":
        return False
    rms, largest = track_spread(track)
    return largest < PARKED_SPREAD


def parked_tracks(tracks: list[VehicleTrack]) -> list[VehicleTrack]:
    result = []
    for track in tracks:
        if is_parked(track):
            result.append(track)
    return result


def count_dots(parked: list[VehicleTrack]) -> int:
    """Parked tracks with RMS distance from their mean under DOT_RMS."""
    dot_count = 0
    for track in parked:
        rms, largest = track_spread(track)
        if rms < DOT_RMS:
            dot_count += 1
    return dot_count


@dataclass
class DriveSummary:
    """Numbers of one drive, for titles."""
    labeled_track_count: int
    first_frame: int
    last_frame: int
    parked_count: int
    dot_count: int  # parked tracks under DOT_RMS


def drive_summary(tracks: list[VehicleTrack]) -> DriveSummary:
    labeled_track_count = 0
    first_frame = int(tracks[0].frames[0])
    last_frame = int(tracks[0].frames[-1])
    for track in tracks:
        first_frame = min(first_frame, int(track.frames[0]))
        last_frame = max(last_frame, int(track.frames[-1]))
        if track.object_type != "Ego":
            labeled_track_count += 1
    parked = parked_tracks(tracks)
    return DriveSummary(labeled_track_count=labeled_track_count,
                        first_frame=first_frame,
                        last_frame=last_frame,
                        parked_count=len(parked),
                        dot_count=count_dots(parked))


@dataclass
class CarAtFrame:
    """One vehicle at one frame, in the world."""
    track: VehicleTrack
    xy: np.ndarray  # (2,)
    yaw: float  # radians, counter-clockwise from East


def cars_at_frame(tracks: list[VehicleTrack], frame: int) -> list[CarAtFrame]:
    """Every track that has this frame, with its position and yaw there."""
    cars = []
    for track in tracks:
        if frame < track.frames[0] or frame > track.frames[-1]:
            continue
        index = frame - track.frames[0]  # frames of a track are consecutive
        cars.append(CarAtFrame(track=track, xy=track.positions[index, :2], yaw=float(track.yaws[index])))
    return cars


def heading_segment(car: CarAtFrame) -> np.ndarray:
    """(2, 2): the car's position, and the point HEADING_LENGTH ahead of it."""
    nose = car.xy + HEADING_LENGTH * np.array([math.cos(car.yaw), math.sin(car.yaw)])
    return np.array([car.xy, nose])


@dataclass
class PredictionInWorld:
    """The prediction of one sample, in the world."""
    track_id: int
    start: np.ndarray  # (2,) x, y of the car at t
    mu: np.ndarray  # (output_steps, 2) predicted x, y
    true_future: np.ndarray  # (output_steps, 2) x, y
    ellipses: list[np.ndarray] | None  # one (ELLIPSE_POINT_COUNT, 2) 1σ ellipse per output step. None: no sigma


def sigma_ellipse_in_car_frame(center: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Points of the ellipse with half axes sigma x, sigma y around center, in the car frame. (ELLIPSE_POINT_COUNT, 2)"""
    angles = np.linspace(0, 2 * math.pi, ELLIPSE_POINT_COUNT)
    x = center[0] + sigma[0] * np.cos(angles)
    y = center[1] + sigma[1] * np.sin(angles)
    return np.column_stack([x, y])


def predictions_in_world(samples: Samples, predictions: Predictions,
                         drive: str, frame: int) -> list[PredictionInWorld]:
    """The predictions of all samples of the drive with t at this frame.

    sigma is along the car's x and y axes, so the ellipses turn with the car.
    """
    sample_indices = np.nonzero((samples.drive == drive) & (samples.frame == frame))[0]
    result = []
    for sample_index in sample_indices:
        anchor_pose = samples.anchor_pose[sample_index]
        mu_car = predictions.mu[sample_index]
        future_car = samples.future[sample_index, :, :2]
        step_count = mu_car.shape[0]
        zero_yaws = np.zeros(step_count)
        mu_world, _ = car_frame_to_world(mu_car, zero_yaws, anchor_pose)
        future_world, _ = car_frame_to_world(future_car, zero_yaws, anchor_pose)

        ellipses = None
        if predictions.sigma is not None:
            ellipses = []
            for step in range(step_count):
                ellipse_car = sigma_ellipse_in_car_frame(mu_car[step], predictions.sigma[sample_index, step])
                ellipse_world, _ = car_frame_to_world(ellipse_car, np.zeros(ELLIPSE_POINT_COUNT), anchor_pose)
                ellipses.append(ellipse_world)

        result.append(PredictionInWorld(track_id=int(samples.track_id[sample_index]),
                                        start=anchor_pose[:2],
                                        mu=mu_world,
                                        true_future=future_world,
                                        ellipses=ellipses))
    return result


# Drawing: matplotlib, each function takes an axes

def track_color(track: VehicleTrack) -> str:
    if track.object_type == "Ego":
        return EGO_COLOR
    if is_parked(track):
        return PARKED_COLOR
    return MOVING_COLOR


def draw_tracks(axes: Axes, tracks: list[VehicleTrack], with_labels: bool):
    """All tracks from start to end: a dot per frame on a thin line. Labels: the track id at the last position."""
    for track in tracks:
        color = track_color(track)
        xy = track.positions[:, :2]
        axes.plot(xy[:, 0], xy[:, 1], color=color, linewidth=TRACK_LINE_WIDTH,
                  marker="o", markersize=TRACK_DOT_SIZE, markeredgewidth=0)
        if not with_labels:
            continue
        if track.object_type == "Ego":
            label = "ego"
        else:
            label = str(track.track_id)
        axes.annotate(label, xy=(xy[-1, 0], xy[-1, 1]), xytext=(3, 3), textcoords="offset points",
                      fontsize=6, color=TEXT_COLOR)


def draw_cars_at_frame(axes: Axes, tracks: list[VehicleTrack], frame: int):
    """Each car at the frame: a dot, and a line towards its nose."""
    for car in cars_at_frame(tracks, frame):
        color = track_color(car.track)
        segment = heading_segment(car)
        axes.plot(segment[:, 0], segment[:, 1], color=color, linewidth=HEADING_LINE_WIDTH, solid_capstyle="round")
        axes.plot(car.xy[0], car.xy[1], marker="o", markersize=CAR_DOT_SIZE, color=color,
                  markeredgecolor=SURFACE_COLOR, markeredgewidth=CAR_DOT_EDGE_WIDTH)


def draw_predictions(axes: Axes, predictions: list[PredictionInWorld]):
    """True future dashed, predicted mu as a line, 1σ ellipses if there is sigma. All start at the car at t."""
    for prediction in predictions:
        true_line = np.vstack([prediction.start, prediction.true_future])
        mu_line = np.vstack([prediction.start, prediction.mu])
        axes.plot(true_line[:, 0], true_line[:, 1], color=TRUE_FUTURE_COLOR, linewidth=TRUE_FUTURE_LINE_WIDTH,
                  linestyle=(0, TRUE_FUTURE_DASH))
        axes.plot(mu_line[:, 0], mu_line[:, 1], color=PREDICTION_COLOR, linewidth=MU_LINE_WIDTH)
        if prediction.ellipses is not None:
            for ellipse in prediction.ellipses:
                axes.plot(ellipse[:, 0], ellipse[:, 1], color=PREDICTION_COLOR, linewidth=SIGMA_LINE_WIDTH,
                          alpha=SIGMA_OPACITY)


def draw_parked_panel(axes: Axes, track: VehicleTrack):
    """Positions of one parked track around their mean, 1 m x 1 m, true scale, with a DOT_RMS circle."""
    offsets = offsets_from_mean(track)
    rms, largest = track_spread(track)
    angles = np.linspace(0, 2 * math.pi, 73)
    axes.plot(DOT_RMS * np.cos(angles), DOT_RMS * np.sin(angles), color=MUTED_COLOR,
              linewidth=DOT_CIRCLE_LINE_WIDTH, linestyle=(0, TRUE_FUTURE_DASH))
    axes.plot(offsets[:, 0], offsets[:, 1], linestyle="none", marker="o", markersize=PANEL_DOT_SIZE,
              color=PARKED_COLOR, markeredgewidth=0)
    axes.set_xlim(-PANEL_HALF_SIZE, PANEL_HALF_SIZE)
    axes.set_ylim(-PANEL_HALF_SIZE, PANEL_HALF_SIZE)
    axes.set_aspect("equal")
    axes.set_xticks([-PANEL_HALF_SIZE, 0, PANEL_HALF_SIZE])
    axes.set_yticks([-PANEL_HALF_SIZE, 0, PANEL_HALF_SIZE])
    axes.tick_params(labelsize=5, colors=MUTED_COLOR, length=2)
    for spine in axes.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.5)
    axes.set_title(f"{track.track_id} {track.object_type}\nRMS {rms:.2f} m\nmax {largest:.2f} m",
                   fontsize=6, color=TEXT_COLOR)


def style_map_axes(axes: Axes):
    axes.set_facecolor(SURFACE_COLOR)
    axes.set_aspect("equal", adjustable="datalim")
    axes.set_xlabel("x East (m)", fontsize=8, color=TEXT_COLOR)
    axes.set_ylabel("y North (m)", fontsize=8, color=TEXT_COLOR)
    axes.tick_params(labelsize=7, colors=MUTED_COLOR)
    axes.grid(color=GRID_COLOR, linewidth=0.5)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.5)


# Figures

def svg_text(figure: Figure) -> str:
    """The figure as SVG text, without a date in it."""
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", metadata={"Date": None})
    return buffer.getvalue()


def map_height(tracks: list[VehicleTrack]) -> float:
    """Height of the map in inches, from the drive's shape, between 4 and 12."""
    all_xy = np.concatenate([track.positions[:, :2] for track in tracks])
    extent_x = all_xy[:, 0].max() - all_xy[:, 0].min()
    extent_y = all_xy[:, 1].max() - all_xy[:, 1].min()
    return min(max(FIGURE_WIDTH * extent_y / max(extent_x, 1.0), 4.0), 12.0)


def world_bev_svg(tracks: list[VehicleTrack], frame: int | None,
                  predictions_world: list[PredictionInWorld]) -> str:
    """The map of one drive as SVG text. With a frame, also the cars at that frame and the predictions."""
    figure = Figure(figsize=(FIGURE_WIDTH, map_height(tracks)), layout="constrained", facecolor=SURFACE_COLOR)
    axes = figure.add_subplot()
    style_map_axes(axes)
    draw_tracks(axes, tracks, with_labels=True)
    if frame is not None:
        draw_cars_at_frame(axes, tracks, frame)
        draw_predictions(axes, predictions_world)
    return svg_text(figure)


def parked_panels_svg(parked: list[VehicleTrack]) -> str:
    """One panel per parked track, PANELS_PER_ROW in a row, as SVG text."""
    row_count = math.ceil(len(parked) / PANELS_PER_ROW)
    figure = Figure(figsize=(FIGURE_WIDTH, row_count * PANEL_ROW_HEIGHT), layout="constrained",
                    facecolor=SURFACE_COLOR)
    panel_axes = figure.subplots(row_count, PANELS_PER_ROW, squeeze=False)
    for panel_index in range(row_count * PANELS_PER_ROW):
        axes = panel_axes[panel_index // PANELS_PER_ROW, panel_index % PANELS_PER_ROW]
        if panel_index < len(parked):
            axes.set_facecolor(SURFACE_COLOR)
            draw_parked_panel(axes, parked[panel_index])
        else:
            axes.axis("off")
    return svg_text(figure)


def write_thumbnail_png(png_path: str, tracks: list[VehicleTrack]) -> tuple[int, int]:
    """The whole drive, no axes, no text. Returns width and height in pixels."""
    figure = Figure(figsize=(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), facecolor=SURFACE_COLOR)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_aspect("equal", adjustable="datalim")
    axes.axis("off")
    draw_tracks(axes, tracks, with_labels=False)
    figure.savefig(png_path, dpi=THUMBNAIL_DPI)
    return round(THUMBNAIL_WIDTH * THUMBNAIL_DPI), round(THUMBNAIL_HEIGHT * THUMBNAIL_DPI)


# Script

def tracks_of_drive(tracks: list[VehicleTrack], drive: str) -> list[VehicleTrack]:
    result = []
    for track in tracks:
        if track.drive == drive:
            result.append(track)
    if len(result) == 0:
        raise ValueError(f"no tracks for drive {drive}")
    return result


def split_of_drive(drive: str) -> str:
    if drive in VALIDATION_DRIVES:
        return "validation"
    if drive in TEST_DRIVES:
        return "test"
    return "training"


def print_parked_table(parked: list[VehicleTrack]):
    print(f"parked tracks (all positions within {PARKED_SPREAD:g} m of their mean):")
    print(f"{'track':>5}  {'type':<5} {'frames':>6} {'RMS m':>6} {'max m':>6}")
    for track in parked:
        rms, largest = track_spread(track)
        print(f"{track.track_id:>5}  {track.object_type:<5} {len(track.frames):>6} {rms:>6.2f} {largest:>6.2f}")
    print(f"{count_dots(parked)} of {len(parked)} under {DOT_RMS:g} m RMS")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--drive", default=TEST_DRIVES[0], help=f"drive to plot (default: {TEST_DRIVES[0]})")
    parser.add_argument("--frame", type=int, default=None, help="also show the cars at this frame")
    parser.add_argument("--predictor", default=None,
                        help="also show the predictions made at --frame, from predictions/<predictor>_<split>.npz")
    args = parser.parse_args()
    if args.predictor is not None and args.frame is None:
        parser.error("--predictor needs --frame")

    tracks = tracks_of_drive(read_vehicle_tracks(), args.drive)
    summary = drive_summary(tracks)
    print(f"{args.drive}: {summary.labeled_track_count} labeled tracks and the ego car, "
          f"frames {summary.first_frame}-{summary.last_frame}")
    parked = parked_tracks(tracks)
    print_parked_table(parked)

    predictions_world = []
    if args.predictor is not None:
        split = split_of_drive(args.drive)
        samples = read_samples(split)
        predictions = load_predictions(args.predictor, split, samples.past.shape[0], samples.future.shape[1])
        predictions_world = predictions_in_world(samples, predictions, args.drive, args.frame)
        print(f"frame {args.frame}: {len(predictions_world)} samples with t at this frame ({split} split)")

    name = f"world_bev_{args.drive}"
    if args.frame is not None:
        name += f"_frame{args.frame}"
    if args.predictor is not None:
        name += f"_{args.predictor}"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    map_path = os.path.join(PLOTS_DIR, name + ".svg")
    with open(map_path, "w") as f:
        f.write(world_bev_svg(tracks, args.frame, predictions_world))
    print(f"wrote {map_path}")
    if len(parked) > 0:
        panels_path = os.path.join(PLOTS_DIR, f"world_bev_{args.drive}_parked.svg")
        with open(panels_path, "w") as f:
            f.write(parked_panels_svg(parked))
        print(f"wrote {panels_path}")


if __name__ == "__main__":
    main()
