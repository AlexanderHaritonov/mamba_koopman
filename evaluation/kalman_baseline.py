"""Kalman filter baseline with a constant-velocity model. It writes mu and sigma.

The filter starts from the first two past positions, filters the other past positions up to t, then predicts the output steps.

Output: predictions/kalman_<split>.npz, for training, validation and test.
"""
import argparse

import numpy as np

from data_model.data_classes import Predictions
from data_model.predictions_io import predictions_path, save_predictions
from data_model.samples_io import read_samples
from evaluation.kalman import predict_kalman

# obtained from best NLL (negative log-likelihood) on training set, from evaluation/kalman_sweep.py
MEASUREMENT_VARIANCE = 0.0001  # m²
PROCESS_VARIANCE = 0.0003  # m² per step³


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args()

    for split in ["training", "validation", "test"]:
        samples = read_samples(split)
        sample_count = samples.past.shape[0]
        output_steps = samples.future.shape[1]

        mu = np.zeros((sample_count, output_steps, 2))
        sigma = np.zeros((sample_count, output_steps, 2))
        for sample_index in range(sample_count):
            prediction = predict_kalman(samples.past[sample_index], output_steps,
                                        MEASUREMENT_VARIANCE, PROCESS_VARIANCE)
            mu[sample_index] = prediction.mu
            sigma[sample_index] = prediction.sigma

        save_predictions("kalman", split, Predictions(mu=mu, sigma=sigma))
        print(f"{split}: wrote {predictions_path('kalman', split)}")


if __name__ == "__main__":
    main()
