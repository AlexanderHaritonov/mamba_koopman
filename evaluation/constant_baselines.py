"""Constant position and constant velocity baselines. They write mu only, no sigma.

Constant position: the car stays where it is at t.
Constant velocity: a straight line is fitted by least squares through the last positions up to t, and continued for the output steps.

Output: predictions/constant_position_<split>.npz and predictions/constant_velocity_<split>.npz, for training, validation and test.
"""
import argparse
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)  # for data_model/
from data_model.data_classes import Predictions
from data_model.predictions_io import predictions_path, save_predictions
from metrics import average_displacement_error, final_displacement_error, sample_groups
from samples_io import read_samples


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


def predict_constant_velocity(past: np.ndarray, output_steps: int, velocity_steps: int) -> np.ndarray:
    """The line through the last velocity_steps positions, continued.

    past: (N, input steps, 3), x, y, yaw. Yaw is not used.
    velocity_steps: 2..input steps, positions up to and with t.
    Returns mu: (N, output_steps, 2), x, y.
    """
    positions = past[:, -velocity_steps:, :2]  # (N, velocity_steps, 2)
    steps = np.arange(velocity_steps, dtype=np.float64)  # step number, t is the last one
    steps_from_mean = steps - steps.mean()
    positions_from_mean = positions - positions.mean(axis=1, keepdims=True)

    # Least-squares slope (metres per step) and the line's position at t, for x and y
    velocity = np.zeros((past.shape[0], 2))
    for step_index in range(velocity_steps):
        velocity += steps_from_mean[step_index] * positions_from_mean[:, step_index]
    velocity /= np.sum(steps_from_mean ** 2)
    line_at_t = positions.mean(axis=1) + velocity * steps_from_mean[-1]

    mu = np.zeros((past.shape[0], output_steps, 2))
    for step in range(output_steps):
        mu[:, step] = line_at_t + velocity * (step + 1)
    return mu


def print_velocity_steps_check(split: str):
    """ADE and FDE of constant velocity for every number of positions in the line fit."""
    samples = read_samples(split)
    past = samples["past"]
    future_xy = samples["future"][:, :, :2]
    output_steps = future_xy.shape[1]
    groups = sample_groups(samples)
    moving = groups["moving"]
    print(f"\nconstant velocity on {split}, with N past positions (t-N+1 .. t) in the line fit, in metres:")
    print(f"{'N past positions':>16}  {'all ADE':>7} {'all FDE':>7}  {'moving ADE':>10} {'moving FDE':>10}")
    for velocity_steps in range(2, past.shape[1] + 1):
        mu = predict_constant_velocity(past, output_steps, velocity_steps)
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
        past = samples["past"]
        output_steps = samples["future"].shape[1]
        if args.velocity_steps < 2 or args.velocity_steps > past.shape[1]:
            raise ValueError(f"--velocity-steps must be 2..{past.shape[1]}")

        constant_position_path = predictions_path("constant_position", split)
        save_predictions(constant_position_path,
                         Predictions(mu=predict_constant_position(past, output_steps), sigma=None))
        constant_velocity_path = predictions_path("constant_velocity", split)
        save_predictions(constant_velocity_path,
                         Predictions(mu=predict_constant_velocity(past, output_steps, args.velocity_steps), sigma=None))
        print(f"{split}: wrote {constant_position_path} and {constant_velocity_path}")

    print_velocity_steps_check("training")
    print_velocity_steps_check("validation")


if __name__ == "__main__":
    main()
