import concurrent.futures
from typing import List, Union

import numpy as np

from src.drift import Drift
from src.latency_estimate import get_url_from_article_title, estimate_url_latency


class WikipediaArm:
    def __init__(self, path, theoretical_params, drift: Drift):
        self.path = path
        self.theoretical_params = theoretical_params
        self.drift = drift

    def get_observation(self):
        if self.theoretical_params is None:
            path_urls = [get_url_from_article_title(title) for title in self.path]
            max_workers = 11
            with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
                url_latencies = list(executor.map(estimate_url_latency, path_urls))
            path_latencies = np.sum(url_latencies)
            return path_latencies
        else:
            return np.random.normal(self.theoretical_params[0], self.theoretical_params[1])

    def apply_drift(self, observation):
        self.drift.apply_drift(observation)


class GaussianTSArm(WikipediaArm):
    def __init__(self,
                 initial_params: List[float],
                 path: List[str],
                 theoretical_params: List[float] = None,
                 drift: Drift = None) -> None:
        WikipediaArm.__init__(self, path, theoretical_params, drift)
        self.initial_params = initial_params
        self.µ, self.v, self.α, self.β = self.initial_params
        self.var = self.β / (self.α + 1)

    def sample_prior(self) -> float:
        precision = np.random.gamma(self.α, 1 / self.β)
        if precision == 0 or self.v == 0:
            precision = 0.001

        estimated_variance = (1 / precision)
        sample = np.random.normal(self.μ, np.sqrt(estimated_variance))
        return sample

    def compute_posterior(self, observation: float) -> None:
        self.µ = (self.v * self.µ + observation) / (self.v + 1)
        self.v += 1
        self.α += 1 / 2
        self.β += (self.v / (self.v + 1)) * (((observation - self.µ) ** 2) / 2)
        self.var = self.β / (self.α + 1)

    def reset(self) -> None:
        self.µ, self.v, self.α, self.β = self.initial_params
        self.var = self.β / (self.α + 1)

    def get_params(self) -> tuple[float, float]:
        return self.µ, self.var ** 0.5


class EpsGreedyArm(WikipediaArm):
    def __init__(self,
                 path: List[str],
                 theoretical_params: List[float] = None,
                 drift: Drift = None) -> None:
        WikipediaArm.__init__(self, path, theoretical_params, drift)
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
                 path: List[str],
                 theoretical_params: List[float] = None,
                 drift: Drift = None) -> None:
        WikipediaArm.__init__(self, path, theoretical_params, drift)
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
                 arms: List[Union[GaussianTSArm, EpsGreedyArm, UCBArm]],
                 name: str,
                 use_synthetic_distributions: bool = False,
                 epsilon: Union[float, None] = None) -> None:
        self.arms = arms
        self.name = name
        self.latencies = []
        self.use_synthetic_distributions = use_synthetic_distributions
        if epsilon is not None:
            self.epsilon = epsilon
        if self.use_synthetic_distributions:
            self.regrets = []

    def run_one_iteration(self) -> None:
        if 'thompson-sampling' in self.name:
            self.run_one_iteration_thompson_sampling()
        elif 'epsilon-greedy' in self.name:
            self.run_one_iteration_epsilon_greedy()
        elif 'ucb' in self.name:
            self.run_one_iteration_ucb()
        else:
            raise ValueError('Not a valid name')

    def run_one_iteration_thompson_sampling(self) -> None:
        sampled_values = [arm.sample_prior() for arm in self.arms]
        best_arm_index = np.argmin(sampled_values)
        if self.use_synthetic_distributions:
            regret = self.compute_regret(best_arm_index)
            self.regrets.append(regret)
        observation = self.arms[best_arm_index].get_observation()
        self.arms[best_arm_index].compute_posterior(observation)
        self.latencies.append(observation)

    def run_one_iteration_epsilon_greedy(self) -> None:
        if np.random.random() < self.epsilon:
            best_arm_index = np.random.choice(len(self.arms))
        else:
            best_arm_index = np.argmin([arm.mean for arm in self.arms])
        observation = self.arms[best_arm_index].get_observation()
        self.arms[best_arm_index].update(observation)
        self.latencies.append(observation)
        if self.use_synthetic_distributions:
            regret = self.compute_regret(best_arm_index)
            self.regrets.append(regret)

    def run_one_iteration_ucb(self) -> None:
        sampled_values = [arm.sample() for arm in self.arms]
        best_arm_index = np.argmax(sampled_values)
        observation = self.arms[best_arm_index].get_observation()
        self.arms[best_arm_index].update(observation)
        self.latencies.append(observation)
        if self.use_synthetic_distributions:
            regret = self.compute_regret(best_arm_index)
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

    def compute_regret(self, chosen_arm_index: np.ndarray[int]) -> float:
        theoretical_params = [arm.theoretical_params for arm in self.arms]
        if None in theoretical_params:
            raise ValueError('theoretical_params is None, cannot compute regret')
        else:
            means = [param[0] for param in theoretical_params]

            optimal_arm_index = np.argmin(means)
            chosen_arm_mean = self.arms[chosen_arm_index].get_params()[0]
            regret = chosen_arm_mean - means[optimal_arm_index]
            return regret if regret > 0 else 0
