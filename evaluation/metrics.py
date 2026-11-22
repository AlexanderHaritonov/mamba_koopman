"""Metrics of predictions files, for groups of samples.

ADE: mean distance between predicted and true position, over all output steps.
FDE: mean distance at the last output step.
NLL and 1σ coverage: only for predictors with sigma, "-" for the others.

Script: one table per split, with a row per group and predictor.
Output: printed in the terminal only, no file. Reads samples/samples_<split>.npz and predictions/<predictor>_<split>.npz.
"""
import argparse
import math
import os

import numpy as np

from data_model.data_classes import Samples
from data_model.predictions_io import load_predictions, predictions_path
from data_model.samples_io import read_samples

# Decides whether car is moving or parking
MOVING_DISPLACEMENT_THRESHOLD = 1.5 # m, from the first input to the last output position.

# Decides whether a moving car is turning or driving straight
TURN_YAW_CHANGE_THRESHOLD = math.radians(15)  # from the first input to the last output position

def displacement_errors(predicted_mu: np.ndarray, actual_future: np.ndarray) -> np.ndarray:
    """Distance between predicted and actual x, y, in meters.

    predicted_mu, actual_future: (N, output steps, 2), x, y.
    Returns (N, output steps).
    """
    difference = predicted_mu - actual_future
    return np.hypot(difference[:, :, 0], difference[:, :, 1])

# ADE, in meters, averaged over samples and over all (output) steps
def average_displacement_error(predicted_mu: np.ndarray, actual_future: np.ndarray) -> float:
    return float(displacement_errors(predicted_mu, actual_future).mean())

# FDE, in meters, averaged over samples, but only on last (output) step
def final_displacement_error(predicted_mu: np.ndarray, actual_future: np.ndarray) -> float:
    return float(displacement_errors(predicted_mu, actual_future)[:, -1].mean())

# Gaussian NLL of one position, x and y summed, averaged over samples and (output) steps. Lower is better.
def negative_log_likelihood(predicted_mu: np.ndarray, predicted_sigma: np.ndarray,
                            actual_future: np.ndarray) -> float:
    """predicted_mu, predicted_sigma, actual_future: (N, output steps, 2), x, y."""
    error_in_sigmas = (actual_future - predicted_mu) / predicted_sigma
    nll_per_axis = 0.5 * np.log(2 * np.pi) + np.log(predicted_sigma) + 0.5 * error_in_sigmas ** 2
    return float(nll_per_axis.sum(axis=2).mean())

# Share of x and y errors smaller than sigma. 0.683 if sigma is right.
def one_sigma_coverage(predicted_mu: np.ndarray, predicted_sigma: np.ndarray,
                       actual_future: np.ndarray) -> float:
    """predicted_mu, predicted_sigma, actual_future: (N, output steps, 2), x, y."""
    inside = np.abs(actual_future - predicted_mu) < predicted_sigma
    return float(inside.mean())


def sample_groups(samples: Samples) -> dict[str, np.ndarray]:
    """One True/False mask per group: all, parked, moving, turning, labeled, ego.

    Turning samples are also moving. Parked = not moving.
    """
    moving = samples.displacement > MOVING_DISPLACEMENT_THRESHOLD
    turning = moving & (np.abs(samples.yaw_change) > TURN_YAW_CHANGE_THRESHOLD)
    ego = samples.object_type == "Ego"
    return {"all": np.ones(len(moving), dtype=bool),
            "parked": ~moving,
            "moving": moving,
            "turning": turning,
            "labeled": ~ego,
            "ego": ego}


def print_metrics_table(split: str, predictors: list[str]):
    """ADE and FDE of each predictor, for each group of one split."""
    samples = read_samples(split)
    future_xy = samples.future[:, :, :2]
    sample_count = future_xy.shape[0]
    output_steps = future_xy.shape[1]
    groups = sample_groups(samples)

    predictions_by_predictor = {}
    for predictor in predictors:
        filepath = predictions_path(predictor, split)
        if not os.path.exists(filepath):
            print(f"no file {os.path.basename(filepath)}, skipped")
            continue
        predictions_by_predictor[predictor] = load_predictions(predictor, split, sample_count, output_steps)

    print(f"\n{split}: {sample_count} samples, {output_steps} output steps")
    print(f"{'group':<8} {'samples':>7}  {'predictor':<18} {'ADE m':>6} {'FDE m':>6} {'NLL':>6} {'cover':>6}")
    for group in groups:
        mask = groups[group]
        group_count = int(mask.sum())
        if group_count == 0:
            print(f"{group:<8} {0:>7}")
            continue
        for predictor in predictions_by_predictor:
            mu = predictions_by_predictor[predictor].mu
            sigma = predictions_by_predictor[predictor].sigma
            ade = average_displacement_error(mu[mask], future_xy[mask])
            fde = final_displacement_error(mu[mask], future_xy[mask])
            if sigma is None:
                nll_text = "-"
                coverage_text = "-"
            else:
                nll = negative_log_likelihood(mu[mask], sigma[mask], future_xy[mask])
                coverage = one_sigma_coverage(mu[mask], sigma[mask], future_xy[mask])
                nll_text = f"{nll:.2f}"
                coverage_text = f"{coverage:.2f}"
            print(f"{group:<8} {group_count:>7}  {predictor:<18} {ade:>6.2f} {fde:>6.2f} "
                  f"{nll_text:>6} {coverage_text:>6}")


def main():
    default_predictors = ["constant_position", "constant_velocity", "kalman"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--splits", nargs="+", default=["validation", "test"],
                        help="splits to score (default: validation test)")
    parser.add_argument("--predictors", nargs="+", default=default_predictors,
                        help=f"predictors to score (default: {' '.join(default_predictors)})")
    args = parser.parse_args()

    for split in args.splits:
        print_metrics_table(split, args.predictors)


if __name__ == "__main__":
    main()
