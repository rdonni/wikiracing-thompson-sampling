import concurrent.futures
from typing import List, Union

import numpy as np

from src.drift import Drift
from src.latency_estimate import get_url_from_article_title, estimate_url_latency


class WikipediaArm:
    def __init__(self,
                 path: List[str]):
        self.path = path

    def get_observation(self):
        if self.theoretical_params is None:
            path_urls = [get_url_from_article_title(title) for title in self.path]
            max_workers = 11
            with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
                url_latencies = list(executor.map(estimate_url_latency, path_urls))
            path_latencies = np.sum(url_latencies)
            return path_latencies
        else:
            # TODO: intégrer la loi log normale
            return np.random.normal(self.theoretical_params[0], self.theoretical_params[1])

    def set_synthetic_distribution(self, synthetic_distributions: tuple[float, float], ) -> None:
        self.theoretical_params = synthetic_distributions

    def set_drift(self, drift: Drift) -> None:
        self.drift = drift

    def apply_drift(self, observation: float, current_iteration: int):
        return self.drift.apply_drift(observation, current_iteration)


class UnknownMeanStdGaussianTSArm(WikipediaArm):
    def __init__(self,
                 initial_params: List[float],
                 path: List[str],
                 discount_factor: Union[float, None]) -> None:
        WikipediaArm.__init__(self, path)
        self.initial_params = initial_params
        self.µ, self.v, self.α, self.β = self.initial_params
        self.var = self.β / (self.α + 1)
        self.discount_factor = discount_factor

    def sample_prior(self) -> float:
        precision = np.random.gamma(self.α, 1 / self.β)
        if precision == 0 or self.v == 0:
            precision = 0.001

        estimated_variance = (1 / precision)
        sample = np.random.normal(self.μ, np.sqrt(estimated_variance))
        return sample

    def compute_posterior(self, observation: float) -> None:
        # TODO: intégrer la loi log normale
        if self.discount_factor is None:
            self.µ = (self.v * self.µ + observation) / (self.v + 1)
        else:
            self.µ = (self.v * self.µ * self.discount_factor + observation) / (self.v * self.discount_factor + 1)
        self.v += 1
        self.α += 1 / 2
        self.β += (self.v / (self.v + 1)) * (((observation - self.µ) ** 2) / 2)
        self.var = self.β / (self.α + 1)

    def reset(self) -> None:
        self.µ, self.v, self.α, self.β = self.initial_params
        self.var = self.β / (self.α + 1)

    def get_params(self) -> tuple[float, float]:
        return self.µ, self.var ** 0.5


class UnknownMeanGaussianTSArm(WikipediaArm):

    def __init__(self,
                 initial_params: List[float],
                 path: List[str],
                 discount_factor: Union[float, None]) -> None:
        WikipediaArm.__init__(self, path)
        self.initial_params = initial_params
        self.discount_factor = discount_factor
        self.μ, self.v = self.initial_params
        self.τ = 1 / self.v
        self.τ_0 = 0.0001  # the sum of all individual precisions
        self.N = 0

    def sample_prior(self):
        return np.random.normal(self.μ, np.sqrt(1 / self.τ_0))

    def compute_posterior(self, x):
        if self.discount_factor is None:
            self.μ = ((self.τ_0 * self.μ) + (self.τ * x)) / (self.τ_0 + self.τ)
            self.τ_0 += self.τ
        else:
            self.μ = ((self.τ_0 * self.μ * self.discount_factor) + (self.τ * x)) / (self.τ_0 * self.discount_factor + self.τ)
            self.τ_0 += self.τ

    def reset(self) -> None:
        self.µ, self.v = self.initial_params
        self.τ = 1 / self.v
        self.τ_0 = 0.0001

    def get_params(self) -> tuple[float, float]:
        return self.µ, self.v ** 0.5


class EpsGreedyArm(WikipediaArm):
    def __init__(self,
                 path: List[str]) -> None:
        WikipediaArm.__init__(self, path)
        self.mean = 0
        self.N = 0

    def update(self, observation: float) -> None:
        self.mean = (self.N * self.mean + observation) / (self.N + 1)
        self.N += 1

    def reset(self) -> None:
        self.mean = 0
        self.N = 0

    def get_params(self) -> tuple[float]:
        return self.mean,


class UCBArm(WikipediaArm):
    def __init__(self,
                 confidence_level: float,
                 path: List[str]) -> None:
        WikipediaArm.__init__(self, path)
        self.confidence_level = confidence_level
        self.mean = 0
        self.N = 0

    def sample(self) -> float:
        num_iters = self.N
        if self.N == 0:
            num_iters = 1
        ucb_values = -self.mean + self.confidence_level * np.sqrt(np.log(num_iters) / num_iters)
        return ucb_values

    def update(self, observation: float) -> None:
        self.mean = (self.N * self.mean + observation) / (self.N + 1)
        self.N += 1

    def reset(self) -> None:
        self.mean = 0
        self.N = 0

    def get_params(self) -> tuple[float]:
        return self.mean,


class MultiArmedBandit:
    def __init__(self,
                 arms: List[Union[UnknownMeanGaussianTSArm, UnknownMeanStdGaussianTSArm, EpsGreedyArm, UCBArm]],
                 name: str,
                 type: str,
                 use_synthetic_distributions: bool,
                 use_drift: bool,
                 epsilon: Union[float, None] = None) -> None:
        self.arms = arms
        self.name = name
        self.type = type
        self.latencies = []
        self.use_synthetic_distributions = use_synthetic_distributions
        self.use_drift = use_drift
        if epsilon is not None:
            self.epsilon = epsilon
        if self.use_synthetic_distributions:
            self.regrets = []

    def run_one_iteration(self, current_iteration: int) -> None:
        if self.type in ['unknown-mean-std-thompson-sampling',
                         'unknown-mean-thompson-sampling',
                         'unknown-mean-std-discounted-thompson-sampling',
                         'unknown-mean-discounted-thompson-sampling']:
            self.run_one_iteration_thompson_sampling(current_iteration)
        elif self.type == 'epsilon-greedy':
            self.run_one_iteration_epsilon_greedy(current_iteration)
        elif self.type == 'ucb':
            self.run_one_iteration_ucb(current_iteration)
        else:
            raise ValueError('Not a valid type.')

    def run_one_iteration_thompson_sampling(self, current_iteration: int) -> None:
        sampled_values = [arm.sample_prior() for arm in self.arms]
        best_arm_index = np.argmin(sampled_values)
        if self.use_synthetic_distributions:
            regret = self.compute_regret(best_arm_index, current_iteration)
            self.regrets.append(regret)
        observation = self.arms[best_arm_index].get_observation()
        if self.use_drift:
            observation = self.arms[best_arm_index].apply_drift(observation, current_iteration)
        self.arms[best_arm_index].compute_posterior(observation)
        self.latencies.append(observation)

    def run_one_iteration_epsilon_greedy(self, current_iteration: int) -> None:
        if np.random.random() < self.epsilon:
            best_arm_index = np.random.choice(len(self.arms))
        else:
            best_arm_index = np.argmin([arm.mean for arm in self.arms])
        observation = self.arms[best_arm_index].get_observation()
        if self.use_drift:
            observation = self.arms[best_arm_index].apply_drift(observation, current_iteration)
        self.arms[best_arm_index].update(observation)
        self.latencies.append(observation)
        if self.use_synthetic_distributions:
            regret = self.compute_regret(best_arm_index, current_iteration)
            self.regrets.append(regret)

    def run_one_iteration_ucb(self, current_iteration: int) -> None:
        sampled_values = [arm.sample() for arm in self.arms]
        best_arm_index = np.argmax(sampled_values)
        observation = self.arms[best_arm_index].get_observation()
        if self.use_drift:
            observation = self.arms[best_arm_index].apply_drift(observation, current_iteration)
        self.arms[best_arm_index].update(observation)
        self.latencies.append(observation)
        if self.use_synthetic_distributions:
            regret = self.compute_regret(best_arm_index, current_iteration)
            self.regrets.append(regret)

    def reset(self) -> None:
        for arm in self.arms:
            arm.reset()
        self.latencies = []
        self.regrets = []

    def get_params(self) -> List[Union[tuple[float, float], tuple[float]]]:
        return [arm.get_params() for arm in self.arms]

    def get_best_path(self) -> List[str]:
        path_average_time = [params[0] for params in self.get_params()]
        best_path_index = np.argmin(path_average_time)
        return self.arms[best_path_index].path

    def set_synthetic_distributions(self, synthetic_distributions: List[tuple[float, float]]):
        for i, arm in enumerate(self.arms):
            arm.set_synthetic_distribution(synthetic_distributions[i])

    def set_drifts(self, drifts: List[Drift]):
        for i, arm in enumerate(self.arms):
            arm.set_drift(drifts[i])

    def compute_regret(self, chosen_arm_index: np.ndarray[int], current_iteration: int) -> float:
        theoretical_params = [arm.theoretical_params for arm in self.arms]
        if None in theoretical_params:
            raise ValueError('theoretical_params is None, cannot compute regret')
        else:
            means = [param[0] for param in theoretical_params]
            if self.use_drift:
                means = [arm.apply_drift(means[i], current_iteration) for i, arm in enumerate(self.arms)]
            optimal_arm_index = np.argmin(means)
            chosen_arm_mean = means[chosen_arm_index]
            regret = chosen_arm_mean - means[optimal_arm_index]
            return regret

    def compute_rewards_average_with_drift(self):
        theoretical_params = [arm.theoretical_params for arm in self.arms]
        means = [param[0] for param in theoretical_params]
        predicted_drifts = np.array([arm.drift.predicted_drifts for arm in self.arms])
        return np.expand_dims(means, axis=1) + predicted_drifts
