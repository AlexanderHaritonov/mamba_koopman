# KITTI BEV trajectory prediction: data and baselines

Predicting car tracks in a bird's-eye view (BEV), from the KITTI raw drives.

## Setup

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

Scripts run from this folder: `venv/bin/python -m <folder>.<script>`.

## World BEV website

Rebuild the static website, all drives (about 30 s):

```bash
venv/bin/python -m visualization.world_bev.world_bev_site
```

Output in `site/world_bev/`: open `site/world_bev/index.html`.
It reads `samples/` and `predictions/`, made by the earlier steps.
