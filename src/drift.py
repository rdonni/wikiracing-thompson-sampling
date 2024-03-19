import numpy as np


class Drift:
    def __init__(self, drift_method: str, num_iters: int) -> None:
        # TODO: changer le drift pour qu'il y ait pas de valeurs négatives
        self.drift_method = drift_method
        self.num_iters = num_iters
        self.num_arm_played = 0

        if self.drift_method == 'linear':
            self.growth_rate = np.random.uniform(-0.05, 0.05)

        if self.drift_method == 'brownian':
            self.noise_var = 2

        self.predicted_drifts = self.predict_drift()

    def apply_drift(self, reward: float, current_iteration: int) -> float:
        # TODO: changer le drift pour qu'il soit adapté à des lois log normales
        drifted_reward = reward + self.predicted_drifts[current_iteration]
        #if drifted_reward < 10:
        #    drifted_reward = 10
        return drifted_reward

    def predict_drift(self) -> np.ndarray[float]:
        # TODO: rajouter un mode de drift escalier (ici il n'y a que du drift 'smooth')
        if self.drift_method == 'linear':
            predicted_drifts = np.array([i * self.growth_rate for i in range(self.num_iters)])
        if self.drift_method == 'brownian':
            predicted_drifts = np.cumsum(np.random.normal(0, self.noise_var, self.num_iters))
        return predicted_drifts
