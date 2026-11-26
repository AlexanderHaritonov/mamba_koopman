# World BEV website

A static website on Cloudflare Pages: one thumbnail per drive. A click on a thumbnail opens a static page with a zoomable SVG of one frame with Kalman predictions.
The legend and all text are HTML, not part of the SVG.

## Files

All in `visualization/world_bev/`. Run from the project folder:
`python -m visualization.world_bev.world_bev_site` (the site), `python -m visualization.world_bev.plot_world_bev` (one drive to `plots/`).

| File | Holds |
|---|---|
| `plot_world_bev.py` | geometry (numpy only), drawing (matplotlib), the SVG and thumbnail writers |
| `world_bev_site.py` | data for the pages (legend entries, cards), renders the templates, builds the site |
| `templates/` | Jinja templates `base.html`, `index.html`, `drive.html`, `legend.html` (macros); `style.css`, `map_zoom.js` copied as they are |

## SVG side: `plot_world_bev.py`

- Line widths, dot sizes and the dash pattern become named constants next to the colors. The HTML legend uses the same values.
- `world_bev_svg(tracks, frame, predictions_world, with_axes)` → `MapSvg`: SVG text and the area it shows (m). Tracks, cars at the frame, predictions. No title, no legend.
  Site: `with_axes=False`, the map fills the SVG, no ticks, no grid, transparent. `plots/`: with axes.
- `parked_panels_svg(parked_tracks)` → SVG text: the strip of parked panels. A separate SVG, so it does not zoom with the map.
- Both return text, not a file: the site puts it into the page, `main` writes it to `plots/`.
- `write_thumbnail_png(png_path, tracks)`: the whole drive, no axes, no text, about 400 px wide.
- `main`: one drive to `plots/`, SVG only.
- No full-size PNG any more.
- `draw_legend()` is removed. Title lines and the parked-car number move to HTML.

## HTML side: `world_bev_site.py`

- No HTML in Python: the text and markup are in the templates, Python passes data.
- Legend, one function per entry, each returns a `LegendEntry` (label, swatch lines, dots, ellipses; px) in the plot's colors and line widths:
  `legend_ego()`, `legend_moving()`, `legend_parked()`, `legend_car_at_frame()`, `legend_mu(predictor)`, `legend_sigma(predictor)`, `legend_true_future()`.
  - Pieces: `swatch_line()`, `swatch_dot()`, `swatch_ellipse()`. The macros in `legend.html` draw them.
- `drive.html`: name and track count, frame line, legend, the map SVG inlined and zoomable, the parked panels below.
- `index.html`: a grid of thumbnails (`DriveCard`), each a link to its drive page.
- `main`: loops over all drives, picks the frame, renders the pages, copies `style.css` and `map_zoom.js`.

## Site

One folder per website under `site/`, this one in `site/world_bev/`:

```
site/world_bev/index.html
site/world_bev/style.css
site/world_bev/map_zoom.js
site/world_bev/drives/<drive>.html
site/world_bev/thumbnails/<drive>.png
```

- About 37 drive pages, about 0.5 MB each. Cloudflare Pages allows 25 MiB per file.
- Zoom: mouse wheel zooms, dragging pans, on the inlined SVG.
  Lines and dots keep their size on screen: CSS `vector-effect: non-scaling-stroke` for lines,
  and on every zoom, JS scales the dot shapes (in `<defs>`, used by every dot) by 1 / zoom.
  A CSS transform on those shapes did not reach the dots in Chrome; the `transform` attribute does.
  Track labels: matplotlib writes each as glyphs in a group with `translate(x y) scale(s -s)`. On every zoom JS shrinks the group by 1 / zoom
  around the track end; the end is (x, y) minus `TRACK_LABEL_OFFSET` (`data-label-offset` on `#map`).
- Start view (load, refresh, window resize): the ego car at the frame in the middle, 60 m on the shorter side (`START_VIEW_METRES` in JS).
  The ego position comes as `data-ego-x`, `data-ego-y` on `#map`. Without it (no samples) or for a smaller drive: the whole drive.
  Reset shows the whole drive.
- Axes and grid: not in the map SVG, so they do not zoom away. Two empty `<svg>` strips (left, bottom) and a grid `<svg>` behind the transparent map.
  On every pan and zoom (`onUpdatedCTM`), JS works out the visible metres from the map limits (`data-` attributes on `#map`) and redraws ticks and grid lines,
  one step of 1, 2 or 5 × 10ⁿ m, at least 80 px apart.
- Drive page height: the top (heading, legend, map with axes, hint) fills the window, the map takes the height left, at least 300 px.
  The parked panels are below, one scroll down; a link under the map ("Parked cars: 6 of 7 under 0.2 m RMS ↓") leads there.
- Names: "drive 0011". Index cards: track count and frame, no split.
  Drive page: name, frame and counts in one line.
  Legend always in one line, scrolls sideways on narrow screens. Short labels; the longer text is a tooltip (`LegendEntry.tooltip`).

## Decisions (2026-11-26)

1. Frame per drive: the frame with the most turning samples; if tied, the most samples, then the earliest. Drive 0011 gives frame 98.
   Drive 0048 has no samples: its page shows the map without predictions.
2. Zoom code: svg-pan-zoom 3.6.1 from a CDN.
3. `site/`: in `.gitignore`, uploaded with `wrangler pages deploy site`.
   2026-11-26: the site moved into `site/world_bev/`, so other websites can follow. Deployed from `site/`, it is under `/world_bev/`.
4. Parked panels on the drive page: kept under the map, as the 0.2 m check.
