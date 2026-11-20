"""Predictions file (file contract 3 of 3): predictions/<predictor>_<split>.npz with mu and sigma of Predictions.

A file without sigma has only mu.
"""
import os

import numpy as np

from data_model.data_classes import Predictions

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(PROJECT_DIR, "predictions")


def predictions_path(predictor: str, split: str) -> str:
    """predictions/<predictor>_<split>.npz"""
    return os.path.join(PREDICTIONS_DIR, f"{predictor}_{split}.npz")


def check_shapes(predictions: Predictions, name: str):
    """mu is (N, steps, 2); sigma is None or the same shape as mu, all values > 0."""
    mu = predictions.mu
    sigma = predictions.sigma
    if mu.ndim != 3 or mu.shape[2] != 2:
        raise ValueError(f"{name}: mu has shape {mu.shape}, expected (N, output steps, 2)")
    if sigma is None:
        return
    if sigma.shape != mu.shape:
        raise ValueError(f"{name}: sigma has shape {sigma.shape}, mu has {mu.shape}")
    if not np.all(sigma > 0):
        raise ValueError(f"{name}: sigma has values <= 0")


def save_predictions(filepath: str, predictions: Predictions):
    """Write one predictions file. Without sigma, the file has only mu."""
    check_shapes(predictions, filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if predictions.sigma is None:
        np.savez_compressed(filepath, mu=predictions.mu.astype(np.float32))
    else:
        np.savez_compressed(filepath, mu=predictions.mu.astype(np.float32),
                            sigma=predictions.sigma.astype(np.float32))


def load_predictions(filepath: str, sample_count: int, output_steps: int) -> Predictions:
    """One predictions file. sigma is None if the file has none."""
    with np.load(filepath) as data:
        mu = data["mu"]
        sigma = None
        if "sigma" in data:
            sigma = data["sigma"]
    predictions = Predictions(mu=mu, sigma=sigma)
    check_shapes(predictions, filepath)
    if mu.shape[0] != sample_count or mu.shape[1] != output_steps:
        raise ValueError(f"{filepath}: mu has shape {mu.shape}, "
                         f"the samples file has {sample_count} samples and {output_steps} output steps")
    return predictions
