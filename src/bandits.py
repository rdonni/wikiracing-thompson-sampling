import concurrent.futures

import numpy as np
from scipy.stats import invgamma

from src.latency_estimate import get_url_from_article_title, estimate_url_latency


class WikipediaArm:
    def __init__(self, path, num_observations):
        self.path = path
        self.num_observations = num_observations

    def get_observation(self):
        path_urls = [get_url_from_article_title(title) for title in self.path]
        path_urls *= self.num_observations

        max_workers = 11
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            url_latencies = list(executor.map(estimate_url_latency, path_urls))
        url_latencies = np.array_split(url_latencies, self.num_observations)
        path_latencies = np.sum(url_latencies, axis=-1)
        return list(path_latencies)


class GaussianTSArm(WikipediaArm):
    def __init__(self, initial_params, path, num_observations=1):
        WikipediaArm.__init__(self, path, num_observations)
        self.µ, self.v, self.α, self.β = initial_params
        self.var = self.β / (self.α + 1)
        self.observations_list = []

    def sample_prior(self):
        mean = self.µ
        normalization_factor = self.v
        if self.v == 0:
            normalization_factor = self.num_observations
        std = np.sqrt(invgamma(self.α, self.β).rvs(1)[0] / normalization_factor)
        return np.random.normal(mean, std, 1)

    def compute_posterior(self, observations):
        self.µ = (self.v * self.µ + np.sum(observations)) / (self.v + self.num_observations)
        self.v += self.num_observations
        self.α += self.num_observations / 2
        self.β += np.std(observations) * 0.5 * self.num_observations + (
                (self.num_observations * self.v) / (self.num_observations + self.v)) * (
                          ((np.mean(observations) - self.µ) ** 2) / 2)

        self.var = self.β / (self.α + 1)

    def get_params(self):
        return self.µ, self.var ** 0.5


class EpsGreedyArm(WikipediaArm):
    def __init__(self, path, num_observations=1):
        WikipediaArm.__init__(self, path, num_observations)
        self.mean = 0
        self.N = 0

    def update(self, observations):
        self.mean = (self.N * self.mean + np.sum(observations)) / (self.N + self.num_observations)
        self.N += self.num_observations

    def get_params(self):
        return [self.mean]


class UCBArm(WikipediaArm):
    def __init__(self, confidence_level, path, num_observations=1):
        WikipediaArm.__init__(self, path, num_observations)
        self.confidence_level = confidence_level
        self.mean = 0
        self.N = 0

    def sample(self):
        num_iters = self.N
        if self.N == 0:
            num_iters = self.num_observations
        ucb_values = -self.mean + self.confidence_level * np.sqrt(np.log(num_iters) / num_iters)
        return ucb_values

    def update(self, observations):
        self.mean = (self.N * self.mean + np.sum(observations)) / (self.N + self.num_observations)
        self.N += self.num_observations

    def get_params(self):
        return [self.mean]


class MultiArmedBandit:
    def __init__(self, arms, name, epsilon=None):
        self.arms = arms
        self.name = name
        self.latencies = []
        if epsilon is not None:
            self.epsilon = epsilon

    def run_one_iteration(self):
        if 'thompson-sampling' in self.name:
            self.run_one_iteration_thompson_sampling()
        elif 'epsilon-greedy' in self.name:
            self.run_one_iteration_epsilon_greedy()
        elif 'ucb' in self.name:
            self.run_one_iteration_ucb()
        else:
            raise ValueError('Not a valid name')

    def run_one_iteration_thompson_sampling(self):
        sampled_values = [arm.sample_prior() for arm in self.arms]
        best_arm_index = np.argmin(sampled_values)
        observations = self.arms[best_arm_index].get_observation()
        self.arms[best_arm_index].compute_posterior(observations)
        self.latencies += observations

    def run_one_iteration_epsilon_greedy(self):
        if np.random.random() < self.epsilon:
            best_arm_index = np.random.choice(len(self.arms))
        else:
            best_arm_index = np.argmin([arm.mean for arm in self.arms])
        observations = self.arms[best_arm_index].get_observation()
        self.arms[best_arm_index].update(observations)
        self.latencies += observations

    def run_one_iteration_ucb(self):
        sampled_values = [arm.sample() for arm in self.arms]
        best_arm_index = np.argmax(sampled_values)
        observations = self.arms[best_arm_index].get_observation()
        self.arms[best_arm_index].update(observations)
        self.latencies += observations

    def get_params(self):
        return [arm.get_params() for arm in self.arms]

    def get_best_path(self):
        path_average_time = [params[0] for params in self.get_params()]
        best_path_index = np.argmin(path_average_time)
        return self.arms[best_path_index].path
