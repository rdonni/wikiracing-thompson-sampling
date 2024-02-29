import numpy as np


class Drift:
    def __init__(self, drift_method):
        self.drift_method = drift_method
        self.num_iter = 0

        if self.drift_method == 'linear':
            self.growth_rate = np.random.uniform(-0.3, 0.3)

        if self.drift_method == 'brownian':
            self.noise_var = 1
            self.cumulative_noise = 0

    def apply_drift(self, reward):
        if self.drift_method == 'linear':
            drifted_reward = reward + self.num_iter * self.growth_rate
            self.num_iter += 1
            return drifted_reward

        if self.drift_method == 'brownian':
            new_noise = np.random.normal(0, self.noise_var)
            self.cumulative_noise += new_noise
            drifted_reward = reward + self.cumulative_noise
            self.num_iter += 1
            return drifted_reward





