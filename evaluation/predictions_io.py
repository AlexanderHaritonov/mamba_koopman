"""Predictions file (file contract 3 of 3).

mu: (N, output steps, 2), predicted x, y in the target-car frame.
sigma: optional, same shape as mu, spread of x and y. A file without sigma is drawn as a line.
Rows are in the order of the samples file of the same split.
"""
import os

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(PROJECT_DIR, "predictions")


def predictions_path(predictor: str, split: str) -> str:
    """predictions/<predictor>_<split>.npz"""
    return os.path.join(PREDICTIONS_DIR, f"{predictor}_{split}.npz")


def check_shapes(mu: np.ndarray, sigma: np.ndarray | None, name: str):
    """mu is (N, steps, 2); sigma is None or the same shape as mu, all values > 0."""
    if mu.ndim != 3 or mu.shape[2] != 2:
        raise ValueError(f"{name}: mu has shape {mu.shape}, expected (N, output steps, 2)")
    if sigma is None:
        return
    if sigma.shape != mu.shape:
        raise ValueError(f"{name}: sigma has shape {sigma.shape}, mu has {mu.shape}")
    if not np.all(sigma > 0):
        raise ValueError(f"{name}: sigma has values <= 0")


def save_predictions(filepath: str, mu: np.ndarray, sigma: np.ndarray | None = None):
    """Write one predictions file. Without sigma, the file has only mu."""
    check_shapes(mu, sigma, filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if sigma is None:
        np.savez_compressed(filepath, mu=mu.astype(np.float32))
    else:
        np.savez_compressed(filepath, mu=mu.astype(np.float32), sigma=sigma.astype(np.float32))


def load_predictions(filepath: str, sample_count: int,
                     output_steps: int) -> tuple[np.ndarray, np.ndarray | None]:
    """mu and sigma of one predictions file. sigma is None if the file has none."""
    with np.load(filepath) as data:
        mu = data["mu"]
        sigma = None
        if "sigma" in data:
            sigma = data["sigma"]
    check_shapes(mu, sigma, filepath)
    if mu.shape[0] != sample_count or mu.shape[1] != output_steps:
        raise ValueError(f"{filepath}: mu has shape {mu.shape}, "
                         f"the samples file has {sample_count} samples and {output_steps} output steps")
    return mu, sigma
