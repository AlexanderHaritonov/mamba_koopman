"""Samples file, one per split, written by data_prep/cut_tracks_into_samples.py."""
import os

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPARED_DIR = os.path.join(PROJECT_DIR, "prepared")


def samples_path(split: str) -> str:
    """prepared/samples_<split>.npz"""
    return os.path.join(PREPARED_DIR, f"samples_{split}.npz")


def read_samples(split: str) -> dict[str, np.ndarray]:
    """All arrays of one samples file."""
    with np.load(samples_path(split)) as data:
        samples = dict(data)
    return samples
