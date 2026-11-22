from dataclasses import dataclass
import numpy as np

class Kalman:
    def __init__(self, measurement_variance: float):
        self.measurement_noise = np.array(
            [[measurement_variance, 0       ],
            [0,        measurement_variance]])

    @dataclass
    class Estimate:
        state: np.ndarray        # (4,)   [px, py, vx, vy]
        covariance: np.ndarray   # (4, 4)

        @property
        def position(self):
            return self.state[0:2]

        @property
        def velocity(self):
            return self.state[2:4]

    # order: posx, posy, velx, vely
    # one frame per step
    motion_model = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    # picks posx, posy out of the state
    measurement_model = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    def predict(self, estimate: Estimate) -> Estimate:
        new_state = self.motion_model @ estimate.state
        new_covariance = self.motion_model @ estimate.covariance @ self.motion_model.T

        return Kalman.Estimate(new_state, new_covariance)

    def gain(self, estimate: Estimate) -> np.ndarray:
        # H P Hᵀ + R
        # covariance of (measurement - predicted position), (2, 2)
        innovation_covariance = (
            self.measurement_model @ estimate.covariance @ self.measurement_model.T
            + self.measurement_noise
        )

        # K = P Hᵀ S⁻¹
        # (4, 2)
        K = estimate.covariance @ self.measurement_model.T @ np.linalg.inv(innovation_covariance)
        return K
    
    def update(self, prediction: Estimate, measurement: np.ndarray) -> Estimate:
        K = self.gain(prediction) # (4, 2)

        residual = measurement - prediction.position

        new_state = prediction.state + K @ residual
        new_covariance = (np.identity(4) - K @ self.measurement_model) @ prediction.covariance

        return Kalman.Estimate(new_state, new_covariance)
'''
    def predict(self, estimate):
        posx = estimate.position[0] + estimate.velocity[0]
        posy = estimate.position[1] + estimate.velocity[1]
        velx = estimate.velocity[0]
        vely = estimate.velocity[1]

        new_state = np.array([posx, posy, velx, vely])

        # order: posx, posy, velx, vely
        old = estimate.covariance

        posx_posx = old[0, 0] + 2 * old[0, 2] + old[2, 2]
        posy_posy = old[1, 1] + 2 * old[1, 3] + old[3, 3]
        posx_posy = old[0, 1] + old[0, 3] + old[2, 1] + old[2, 3]
        posx_velx = old[0, 2] + old[2, 2]
        posx_vely = old[0, 3] + old[2, 3]
        posy_velx = old[1, 2] + old[3, 2]
        posy_vely = old[1, 3] + old[3, 3]
        velx_velx = old[2, 2]
        vely_vely = old[3, 3]
        velx_vely = old[2, 3]

        new_covariance = np.array([
            [posx_posx, posx_posy, posx_velx, posx_vely],
            [posx_posy, posy_posy, posy_velx, posy_vely],
            [posx_velx, posy_velx, velx_velx, velx_vely],
            [posx_vely, posy_vely, velx_vely, vely_vely],
        ])

        return Kalman.Estimate(new_state, new_covariance)
'''