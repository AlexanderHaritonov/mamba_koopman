"""Constant position and constant velocity baselines. They write mu only, no sigma.

Constant position: the car stays where it is at t.
Constant velocity: a straight line is fitted by least squares through the last positions up to t, and continued for the output steps.

Output: predictions/constant_position_<split>.npz and predictions/constant_velocity_<split>.npz, for training, validation and test.
"""
import argparse

import numpy as np

from data_model.data_classes import Predictions
from data_model.predictions_io import predictions_path, save_predictions
from data_model.samples_io import read_samples
from evaluation.metrics import average_displacement_error, final_displacement_error, sample_groups


def predict_constant_position(past: np.ndarray, output_steps: int) -> np.ndarray:
    """The position at t, repeated.

    past: (N, input steps, 3), x, y, yaw. Yaw is not used.
    Returns mu: (N, output_steps, 2), x, y.
    """
    position_at_t = past[:, -1, :2]
    mu = np.zeros((past.shape[0], output_steps, 2))
    for step in range(output_steps):
        mu[:, step] = position_at_t
    return mu


def least_squares_velocity(positions: np.ndarray) -> np.ndarray:
    """Slope of the least-squares line through the positions, in metres per step.

    positions: (N, fit steps, 2), x, y, in time order. At least 2 steps.
    Returns velocity: (N, 2), x, y.
    """
    fit_steps = positions.shape[1]
    steps = np.arange(fit_steps, dtype=np.float64)
    steps_from_mean = steps - steps.mean()
    positions_from_mean = positions - positions.mean(axis=1, keepdims=True)

    velocity = np.zeros((positions.shape[0], 2))
    for step_index in range(fit_steps):
        velocity += steps_from_mean[step_index] * positions_from_mean[:, step_index]
    velocity /= np.sum(steps_from_mean ** 2)
    return velocity


def predict_constant_velocity(past: np.ndarray, output_steps: int) -> np.ndarray:
    """The line through all past positions, continued.

    past: (N, fit steps, 3), x, y, yaw. At least 2 steps, the last one is t. Yaw is not used.
    Returns mu: (N, output_steps, 2), x, y.
    """
    positions = past[:, :, :2]  # (N, fit steps, 2)
    fit_steps = past.shape[1]
    velocity = least_squares_velocity(positions)
    # The line goes through the mean position at the middle step; t is (fit_steps - 1) / 2 steps later
    line_at_t = positions.mean(axis=1) + velocity * (fit_steps - 1) / 2

    mu = np.zeros((past.shape[0], output_steps, 2))
    for step in range(output_steps):
        mu[:, step] = line_at_t + velocity * (step + 1)
    return mu


def print_velocity_steps_check(split: str):
    """ADE and FDE of constant velocity for every number of positions in the line fit."""
    samples = read_samples(split)
    past = samples.past
    future_xy = samples.future[:, :, :2]
    output_steps = future_xy.shape[1]
    groups = sample_groups(samples)
    moving = groups["moving"]
    print(f"\nconstant velocity on {split}, with N past positions (t-N+1 .. t) in the line fit, in metres:")
    print(f"{'N past positions':>16}  {'all ADE':>7} {'all FDE':>7}  {'moving ADE':>10} {'moving FDE':>10}")
    for velocity_steps in range(2, past.shape[1] + 1):
        mu = predict_constant_velocity(past[:, -velocity_steps:], output_steps)
        print(f"{velocity_steps:>16}  "
              f"{average_displacement_error(mu, future_xy):>7.3f} {final_displacement_error(mu, future_xy):>7.3f}  "
              f"{average_displacement_error(mu[moving], future_xy[moving]):>10.3f} "
              f"{final_displacement_error(mu[moving], future_xy[moving]):>10.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--velocity-steps", type=int, default=4,
                        help="positions in the line fit of constant velocity, up to and with t (default: 4)")
    args = parser.parse_args()

    for split in ["training", "validation", "test"]:
        samples = read_samples(split)
        past = samples.past
        output_steps = samples.future.shape[1]
        if args.velocity_steps < 2 or args.velocity_steps > past.shape[1]:
            raise ValueError(f"--velocity-steps must be 2..{past.shape[1]}")

        save_predictions("constant_position", split,
                         Predictions(mu=predict_constant_position(past, output_steps), sigma=None))
        save_predictions("constant_velocity", split,
                         Predictions(mu=predict_constant_velocity(past[:, -args.velocity_steps:], output_steps), sigma=None))
        print(f"{split}: wrote {predictions_path('constant_position', split)} "
              f"and {predictions_path('constant_velocity', split)}")

    print_velocity_steps_check("training")
    print_velocity_steps_check("validation")


if __name__ == "__main__":
    main()
