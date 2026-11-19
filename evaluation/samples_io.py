"""Samples file, one per split, written by data_prep/cut_tracks_into_samples.py."""
import os

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")


def samples_path(split: str) -> str:
    """samples/samples_<split>.npz"""
    return os.path.join(SAMPLES_DIR, f"samples_{split}.npz")


def read_samples(split: str) -> dict[str, np.ndarray]:
    """All arrays of one samples file."""
    with np.load(samples_path(split)) as data:
        samples = dict(data)
    return samples
