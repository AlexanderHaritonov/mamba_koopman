"""Grid sweep of the two noise values of the Kalman filter.

measurement variance r: noise of one labeled position, m².
process variance q: velocity variance added per step, m² per step³.
speed noise factor a: per sample, the process variance is q + a · speed², speed in m per step. a is in 1/step.

Script: one row per (r, q, a) with ADE, FDE, NLL and 1σ coverage on one split. Then the triple with the lowest NLL, also on a second split.
Output: printed in the terminal only, no file. Reads samples/samples_<split>.npz.
"""
import argparse
import math

import numpy as np

from data_model.samples_io import read_samples
from evaluation.kalman import predict_kalman
from evaluation.metrics import (average_displacement_error, final_displacement_error,
                                negative_log_likelihood, one_sigma_coverage)

# around the best (r, q) of the r, q sweep on the cleaned data (r 1e-5–0.1, q 0–0.01), so the run takes about 20 min
DEFAULT_MEASUREMENT_VARIANCES = [3e-5, 1e-4, 3e-4, 1e-3]
DEFAULT_PROCESS_VARIANCES = [0.0, 3e-5, 1e-4, 3e-4, 1e-3]
DEFAULT_SPEED_NOISE_FACTORS = [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03]


def kalman_scores(past: np.ndarray, future_xy: np.ndarray, measurement_variance: float,
                  process_variance: float, speed_noise_factor: float) -> dict[str, float]:
    """ADE, FDE, NLL, coverage and median sigma of the last step, for one (r, q, a).

    past: (N, input steps, 3). future_xy: (N, output steps, 2).
    """
    sample_count = past.shape[0]
    output_steps = future_xy.shape[1]

    mu = np.zeros((sample_count, output_steps, 2))
    sigma = np.zeros((sample_count, output_steps, 2))
    for sample_index in range(sample_count):
        prediction = predict_kalman(past[sample_index], output_steps, measurement_variance,
                                    process_variance, speed_noise_factor)
        mu[sample_index] = prediction.mu
        sigma[sample_index] = prediction.sigma

    return {"ade": average_displacement_error(mu, future_xy),
            "fde": final_displacement_error(mu, future_xy),
            "nll": negative_log_likelihood(mu, sigma, future_xy),
            "coverage": one_sigma_coverage(mu, sigma, future_xy),
            "last_sigma": float(np.median(sigma[:, -1, 0]))}


def print_header():
    print(f"{'r m²':>7} {'q m²/step³':>10} {'a 1/step':>8}  "
          f"{'ADE m':>6} {'FDE m':>6} {'NLL':>6} {'cover':>6} {'σ last m':>8}")


def print_scores_row(measurement_variance: float, process_variance: float, speed_noise_factor: float,
                     scores: dict[str, float]):
    print(f"{measurement_variance:>7.3g} {process_variance:>10.3g} {speed_noise_factor:>8.3g}  "
          f"{scores['ade']:>6.3f} {scores['fde']:>6.3f} {scores['nll']:>6.3f} "
          f"{scores['coverage']:>6.3f} {scores['last_sigma']:>8.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", default="training", help="split for the sweep (default: training)")
    parser.add_argument("--check-split", default="validation",
                        help="split for the best pair (default: validation)")
    parser.add_argument("--measurement-variances", type=float, nargs="+",
                        default=DEFAULT_MEASUREMENT_VARIANCES, help="r values, m²")
    parser.add_argument("--process-variances", type=float, nargs="+",
                        default=DEFAULT_PROCESS_VARIANCES, help="q values, m² per step³")
    parser.add_argument("--speed-noise-factors", type=float, nargs="+",
                        default=DEFAULT_SPEED_NOISE_FACTORS, help="a values, 1/step")
    args = parser.parse_args()

    samples = read_samples(args.split)
    past = samples.past
    future_xy = samples.future[:, :, :2]
    print(f"\n{args.split}: {past.shape[0]} samples, {past.shape[1]} past steps, {future_xy.shape[1]} output steps")
    print_header()

    best_nll = math.inf
    best_measurement_variance = None
    best_process_variance = None
    best_speed_noise_factor = None
    for measurement_variance in args.measurement_variances:
        for process_variance in args.process_variances:
            for speed_noise_factor in args.speed_noise_factors:
                scores = kalman_scores(past, future_xy, measurement_variance, process_variance, speed_noise_factor)
                print_scores_row(measurement_variance, process_variance, speed_noise_factor, scores)
                if scores["nll"] < best_nll:
                    best_nll = scores["nll"]
                    best_measurement_variance = measurement_variance
                    best_process_variance = process_variance
                    best_speed_noise_factor = speed_noise_factor

    print(f"\nlowest NLL on {args.split}: r = {best_measurement_variance:g} m², q = {best_process_variance:g} m²/step³, "
          f"a = {best_speed_noise_factor:g} 1/step")

    check_samples = read_samples(args.check_split)
    check_past = check_samples.past
    check_future_xy = check_samples.future[:, :, :2]
    print(f"\n{args.check_split}: {check_past.shape[0]} samples")
    print_header()
    check_scores = kalman_scores(check_past, check_future_xy, best_measurement_variance,
                                 best_process_variance, best_speed_noise_factor)
    print_scores_row(best_measurement_variance, best_process_variance, best_speed_noise_factor, check_scores)


if __name__ == "__main__":
    main()
