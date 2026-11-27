"""Static website of the world BEV maps: an index page with one thumbnail per drive, and one page per drive.

A drive page shows the map at one frame with the predictions made at that frame, zoomable (svg-pan-zoom),
and the parked cars below. The frame: the most turning samples, then the most samples, then the earliest.
The legend and all text are HTML. The SVGs and thumbnails come from plot_world_bev.py.
Pages come from the Jinja templates in templates/; style.css and map_zoom.js are copied as they are.

Output: site/world_bev/: index.html, style.css, map_zoom.js, drives/drive_<number>.html, thumbnails/drive_<number>.png
Upload: wrangler pages deploy site (this site is then under /world_bev/)
Reads samples/vehicle_poses_world.csv, samples/samples_<split>.npz and predictions/<predictor>_<split>.npz.
"""
import argparse
import os
import re
import shutil
from dataclasses import dataclass

import jinja2
import numpy as np

import visualization.world_bev.plot_world_bev as world_bev
from data_model.data_classes import Samples
from data_model.poses_io import read_vehicle_tracks
from data_model.predictions_io import load_predictions
from data_model.samples_io import read_samples
from evaluation.metrics import sample_groups

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SITE_DIR = os.path.join(PROJECT_DIR, "site", "world_bev")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
STATIC_FILES = ["style.css", "map_zoom.js"]  # copied from TEMPLATES_DIR into the site as they are
SVG_PAN_ZOOM_URL = "https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"

SWATCH_WIDTH = 28  # px
SWATCH_HEIGHT = 14  # px
LEGEND_SCALE = 2.0  # px per point of the plot marks, so the colors are easy to see


# Legend: data for the macros in legend.html. Sizes in px.

@dataclass
class SwatchLine:
    color: str
    width: float
    x_start: float
    x_end: float
    y: float
    dash: str | None  # SVG stroke-dasharray
    opacity: float


@dataclass
class SwatchDot:
    color: str
    x: float
    y: float
    radius: float
    edge_color: str | None
    edge_width: float


@dataclass
class SwatchEllipse:
    color: str
    x: float
    y: float
    radius_x: float
    radius_y: float
    width: float
    opacity: float


@dataclass
class SwatchRectangle:
    color: str
    x: float  # left
    y: float  # top
    width: float
    height: float
    line_width: float
    fill_opacity: float


@dataclass
class LegendEntry:
    label: str
    lines: list[SwatchLine]
    dots: list[SwatchDot]
    ellipses: list[SwatchEllipse]
    rectangles: list[SwatchRectangle]
    tooltip: str = ""  # longer text, shown on hover


def swatch_line(color: str, line_width: float, x_start: float, x_end: float,
                dash: tuple[float, float] | None = None, opacity: float = 1.0) -> SwatchLine:
    """A horizontal line through the middle of the swatch. line_width and dash in points of the plot."""
    dash_text = None
    if dash is not None:
        dash_text = f"{dash[0] * LEGEND_SCALE:g} {dash[1] * LEGEND_SCALE:g}"
    return SwatchLine(color=color, width=round(line_width * LEGEND_SCALE, 2), x_start=x_start, x_end=x_end,
                      y=SWATCH_HEIGHT / 2, dash=dash_text, opacity=opacity)


def swatch_dot(color: str, diameter: float, x: float,
               edge_color: str | None = None, edge_width: float = 0.0) -> SwatchDot:
    """A filled circle on the middle line of the swatch. diameter and edge_width in points of the plot."""
    return SwatchDot(color=color, x=x, y=SWATCH_HEIGHT / 2, radius=round(diameter * LEGEND_SCALE / 2, 2),
                     edge_color=edge_color, edge_width=round(edge_width * LEGEND_SCALE, 2))


def swatch_ellipse(color: str, line_width: float, opacity: float) -> SwatchEllipse:
    """An ellipse outline in the middle of the swatch. line_width in points of the plot."""
    return SwatchEllipse(color=color, x=SWATCH_WIDTH / 2, y=SWATCH_HEIGHT / 2,
                         radius_x=SWATCH_WIDTH / 2 - 4, radius_y=SWATCH_HEIGHT / 2 - 3,
                         width=round(line_width * LEGEND_SCALE, 2), opacity=opacity)


def swatch_rectangle(color: str, line_width: float, fill_opacity: float,
                     x_start: float, x_end: float, height: float) -> SwatchRectangle:
    """A rectangle on the middle line of the swatch, lightly filled. line_width in points of the plot."""
    return SwatchRectangle(color=color, x=x_start, y=(SWATCH_HEIGHT - height) / 2, width=x_end - x_start,
                           height=height, line_width=round(line_width * LEGEND_SCALE, 2), fill_opacity=fill_opacity)


# Legend entries, one per mark of the map

def legend_track(color: str, label: str) -> LegendEntry:
    """A track: a line with a dot per frame."""
    dots = []
    for x in [7, 14, 21]:
        dots.append(swatch_dot(color, world_bev.TRACK_DOT_SIZE, x))
    return LegendEntry(label=label,
                       lines=[swatch_line(color, world_bev.TRACK_LINE_WIDTH, 2, SWATCH_WIDTH - 2)],
                       dots=dots,
                       ellipses=[],
                       rectangles=[])


def legend_ego() -> LegendEntry:
    return legend_track(world_bev.EGO_COLOR, "ego car")


def legend_moving() -> LegendEntry:
    """Three short pieces of track, in the first three track colors."""
    lines = []
    dots = []
    for index in range(3):
        color = world_bev.TRACK_COLORS[index]
        x_start = 2 + index * 8
        lines.append(swatch_line(color, world_bev.TRACK_LINE_WIDTH, x_start, x_start + 8))
        dots.append(swatch_dot(color, world_bev.TRACK_DOT_SIZE, x_start + 4))
    return LegendEntry(label="moving vehicles",
                       lines=lines,
                       dots=dots,
                       ellipses=[],
                       rectangles=[],
                       tooltip=f"moving vehicles, one color per track, {len(world_bev.TRACK_COLORS)} colors in turn")


def legend_parked() -> LegendEntry:
    return legend_track(world_bev.PARKED_COLOR, "parked vehicle")


def legend_car_at_frame() -> LegendEntry:
    """A box pointing right, with the line from its middle to the nose."""
    return LegendEntry(label="vehicle at the frame",
                       lines=[swatch_line(world_bev.MUTED_COLOR, world_bev.CAR_LINE_WIDTH, 14, 23)],
                       dots=[],
                       ellipses=[],
                       rectangles=[swatch_rectangle(world_bev.MUTED_COLOR, world_bev.CAR_LINE_WIDTH,
                                                    world_bev.CAR_FILL_OPACITY, 5, 23, 8)],
                       tooltip="vehicle at the frame: its box from above, in the vehicle's color, "
                               "line from its position to the nose")


def legend_mu(predictor: str) -> LegendEntry:
    return LegendEntry(label=f"{predictor} μ",
                       lines=[swatch_line(world_bev.MUTED_COLOR, world_bev.MU_LINE_WIDTH, 2, SWATCH_WIDTH - 2)],
                       dots=[],
                       ellipses=[],
                       rectangles=[],
                       tooltip=f"{predictor} prediction μ, in the vehicle's color")


def legend_sigma(predictor: str) -> LegendEntry:
    return LegendEntry(label=f"{predictor} 1σ per step",
                       lines=[],
                       dots=[],
                       ellipses=[swatch_ellipse(world_bev.MUTED_COLOR, world_bev.SIGMA_LINE_WIDTH,
                                                world_bev.SIGMA_OPACITY)],
                       rectangles=[],
                       tooltip=f"{predictor} 1σ ellipse, one per step, in the vehicle's color")


def legend_true_future() -> LegendEntry:
    return LegendEntry(label="true future",
                       lines=[swatch_line(world_bev.TRUE_FUTURE_COLOR, world_bev.TRUE_FUTURE_LINE_WIDTH,
                                          2, SWATCH_WIDTH - 2, world_bev.TRUE_FUTURE_DASH)],
                       dots=[],
                       ellipses=[],
                       rectangles=[])


def legend_entries(with_frame: bool, predictor: str, with_sigma: bool) -> list[LegendEntry]:
    entries = [legend_ego(), legend_moving(), legend_parked()]
    if with_frame:
        entries.append(legend_car_at_frame())
        entries.append(legend_mu(predictor))
        if with_sigma:
            entries.append(legend_sigma(predictor))
        entries.append(legend_true_future())
    return entries


# Pages

@dataclass
class DriveCard:
    """One thumbnail on the index page."""
    file_name: str  # of the drive page and the thumbnail, without extension
    label: str
    labeled_track_count: int
    frame: int | None  # None: no samples
    thumbnail_width: int  # px
    thumbnail_height: int  # px


def drive_label(drive: str) -> str:
    """2011_09_26_drive_0011 -> drive 0011"""
    return "drive " + drive.split("_drive_")[1]


def drive_file_name(drive: str) -> str:
    """2011_09_26_drive_0011 -> drive_0011"""
    return "drive_" + drive.split("_drive_")[1]


def inline_svg(svg: str) -> str:
    """The <svg> element of an SVG file, for putting into HTML: no XML header, no width and height (CSS sets them)."""
    start = svg.index("<svg")
    root_end = svg.index(">", start)
    root_tag = re.sub(r'\s(width|height)="[^"]*"', "", svg[start:root_end])
    return root_tag + svg[root_end:]


def template_environment() -> jinja2.Environment:
    return jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
                              autoescape=True,
                              undefined=jinja2.StrictUndefined,
                              trim_blocks=True,
                              lstrip_blocks=True)


# Script

def frame_with_most_turning_samples(samples: Samples, drive: str) -> int | None:
    """Frame of t with the most turning samples of the drive; if tied, the most samples, then the earliest.

    None: the drive has no samples.
    """
    turning = sample_groups(samples)["turning"]
    sample_indices = np.nonzero(samples.drive == drive)[0]
    frames = samples.frame[sample_indices]
    turning_of_drive = turning[sample_indices]
    best_frame = None
    best_turning_count = -1
    best_sample_count = -1
    for frame in np.unique(frames):  # ascending, so ties keep the earliest
        at_frame = frames == frame
        turning_count = int((turning_of_drive & at_frame).sum())
        sample_count = int(at_frame.sum())
        if (turning_count, sample_count) > (best_turning_count, best_sample_count):
            best_frame = int(frame)
            best_turning_count = turning_count
            best_sample_count = sample_count
    return best_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictor", default="kalman", help="predictions to show (default: kalman)")
    parser.add_argument("--out-dir", default=SITE_DIR, help="folder of the site (default: site/world_bev)")
    args = parser.parse_args()

    all_tracks = read_vehicle_tracks()
    drives = []
    for track in all_tracks:
        if track.drive not in drives:
            drives.append(track.drive)
    drives.sort()

    samples_by_split = {}
    predictions_by_split = {}
    for split in ["training", "validation", "test"]:
        samples = read_samples(split)
        samples_by_split[split] = samples
        predictions_by_split[split] = load_predictions(args.predictor, split,
                                                       samples.past.shape[0], samples.future.shape[1])

    environment = template_environment()
    drive_template = environment.get_template("drive.html")
    index_template = environment.get_template("index.html")
    os.makedirs(os.path.join(args.out_dir, "drives"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "thumbnails"), exist_ok=True)
    for filename in STATIC_FILES:
        shutil.copyfile(os.path.join(TEMPLATES_DIR, filename), os.path.join(args.out_dir, filename))

    cards = []
    file_names = []
    for drive in drives:
        file_name = drive_file_name(drive)
        if file_name in file_names:
            raise ValueError(f"two drives with the file name {file_name}")
        file_names.append(file_name)
        tracks = world_bev.tracks_of_drive(all_tracks, drive)
        summary = world_bev.drive_summary(tracks)
        split = world_bev.split_of_drive(drive)
        samples = samples_by_split[split]
        predictions = predictions_by_split[split]

        frame = frame_with_most_turning_samples(samples, drive)
        predictions_world = []
        vehicle_count = 0
        ego_xy = None  # the page starts zoomed in on it
        if frame is not None:
            predictions_world = world_bev.predictions_in_world(samples, predictions, drive, frame)
            cars = world_bev.cars_at_frame(tracks, frame)
            vehicle_count = len(cars)
            for car in cars:
                if car.track.object_type == "Ego":
                    ego_xy = (float(car.xy[0]), float(car.xy[1]))

        map_svg = world_bev.world_bev_svg(tracks, frame, predictions_world, with_axes=False)
        parked = world_bev.parked_tracks(tracks)
        panels_svg = None
        if len(parked) > 0:
            panels_svg = inline_svg(world_bev.parked_panels_svg(parked))

        page = drive_template.render(root="../",
                                     label=drive_label(drive),
                                     summary=summary,
                                     frame=frame,
                                     vehicle_count=vehicle_count,
                                     predictor=args.predictor,
                                     prediction_count=len(predictions_world),
                                     legend_entries=legend_entries(frame is not None, args.predictor,
                                                                   predictions.sigma is not None),
                                     swatch_width=SWATCH_WIDTH,
                                     swatch_height=SWATCH_HEIGHT,
                                     map_svg=inline_svg(map_svg.svg),
                                     map_area=map_svg,
                                     ego_xy=ego_xy,
                                     label_offset=world_bev.TRACK_LABEL_OFFSET,
                                     panels_svg=panels_svg,
                                     dot_rms=world_bev.DOT_RMS,
                                     svg_pan_zoom_url=SVG_PAN_ZOOM_URL)
        with open(os.path.join(args.out_dir, "drives", f"{file_name}.html"), "w") as f:
            f.write(page)

        thumbnail_path = os.path.join(args.out_dir, "thumbnails", f"{file_name}.png")
        thumbnail_width, thumbnail_height = world_bev.write_thumbnail_png(thumbnail_path, tracks)
        cards.append(DriveCard(file_name=file_name,
                               label=drive_label(drive),
                               labeled_track_count=summary.labeled_track_count,
                               frame=frame,
                               thumbnail_width=thumbnail_width,
                               thumbnail_height=thumbnail_height))
        print(f"{drive}: {split}, frame {frame}, {len(predictions_world)} predictions")

    index_path = os.path.join(args.out_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(index_template.render(root="", cards=cards, predictor=args.predictor))
    print(f"wrote {index_path} and {len(drives)} drive pages")


if __name__ == "__main__":
    main()
