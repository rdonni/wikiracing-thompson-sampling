import json
import os

import numpy as np
import plotly as px
import plotly.graph_objs as go
from tqdm import tqdm

from src.drift import Drift


class Simulation:
    def __init__(self,
                 mabs,
                 nb_iterations,
                 nb_simulations,
                 use_synthetic_distributions,
                 synthetic_data_config,
                 use_drift,
                 drift_method,
                 results_path,
                 plots_path,
                 show_plots,
                 display_ci) -> None:
        self.mabs = mabs
        self.nb_simulations = nb_simulations
        self.use_synthetic_distributions = use_synthetic_distributions
        self.synthetic_data_config = synthetic_data_config
        self.use_drift = use_drift
        self.drift_method = drift_method
        self.nb_iterations = nb_iterations
        self.results_path = results_path
        self.plots_path = plots_path
        self.show_plots = show_plots
        self.display_ci = display_ci

        self.n_iter = 0

    def simulation(self):

        results = {}
        average_latencies = {}
        cumulative_latencies = {}
        average_regrets = {}

        for sim_num in range(self.nb_simulations):
            print(f'Running simulation number {sim_num}...')
            # We reset the distributions between each simulation
            self.reset()
            self.set_synthetic_distributions()
            self.set_drifts()

            if self.use_drift and self.use_synthetic_distributions:
                self.plot_average_reward_per_arm(sim_num)

            for i in tqdm(range(self.nb_iterations)):
                for mab in self.mabs:
                    mab.run_one_iteration(i)
                self.n_iter += 1
            for mab in self.mabs:
                print(
                    f"At iteration {self.n_iter}, for {mab.name}, cumulative latency is equal to {np.sum(mab.latencies)} the best path is {mab.get_best_path()}.")
            results[sim_num] = [list(mab.latencies) for mab in self.mabs]

            if self.use_synthetic_distributions:
                sim_regrets = [list(mab.regrets) for mab in self.mabs]
                average_regrets[sim_num] = np.cumsum(sim_regrets, axis=1) / (np.arange(self.nb_iterations) + 1)

            # Plots for the current simulation
            cumulative_latencies[sim_num] = np.cumsum(results[sim_num], axis=1)
            average_latencies[sim_num] = np.cumsum(results[sim_num], axis=1) / (np.arange(self.nb_iterations) + 1)
            self.generate_plots(cumulative_latencies, 'Cumulative loading time', sim_num)
            self.generate_plots(average_latencies, 'Average loading time', sim_num)
            if self.use_synthetic_distributions:
                self.generate_plots(average_regrets, 'Regrets', sim_num)

        # Aggregation plots to sum up all simulations
        self.generate_plots(cumulative_latencies, 'Cumulative loading time')
        self.generate_plots(average_latencies, 'Average loading time')
        if self.use_synthetic_distributions:
            self.generate_plots(average_regrets, 'Regrets')

        # Save results as a json file
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)
        with open(f"{self.results_path}/results.json", "w") as results_path:
            json.dump(results, results_path)

    def reset(self):
        for mab in self.mabs:
            mab.reset()
        self.n_iter = 0

    def set_synthetic_distributions(self):
        num_paths = len(self.mabs[0].arms)
        if self.use_synthetic_distributions:
            # We model the average loading time of each path by a normal variable
            synthetic_means = np.random.uniform(self.synthetic_data_config['min_mean'],
                                                self.synthetic_data_config['max_mean'],
                                                num_paths)
            synthetic_stds = np.random.uniform(self.synthetic_data_config['min_std'],
                                               self.synthetic_data_config['max_std'],
                                               num_paths)
            synthetic_params = list(zip(synthetic_means, synthetic_stds))
        else:
            # We use real distributions
            synthetic_params = [None] * num_paths

        for i, mab in enumerate(self.mabs):
            mab.set_synthetic_distributions(synthetic_params)

    def set_drifts(self):
        num_paths = len(self.mabs[0].arms)
        if self.use_drift:
            drifts = [Drift(self.drift_method, self.nb_iterations) for _ in range(num_paths)]
        else:
            drifts = [None] * num_paths
        for i, mab in enumerate(self.mabs):
            mab.set_drifts(drifts)

    def generate_plots(self, results, title: str, sim_num: int = None) -> None:

        # Cumulative loading time plot
        fig = go.Figure()
        for mab_index in range(len(self.mabs)):
            if sim_num is None:
                # If no sim_num is given, we extract the mab results of all simulations as a matrix
                mab_results = [results[sim_num][mab_index] for sim_num in results.keys()]

                ci = 1.96 * np.std(mab_results, axis=0) / np.sqrt(self.nb_simulations)
                mab_results = np.mean(mab_results, axis=0)
                upper_bound = mab_results + ci
                lower_bound = mab_results - ci

            else:
                # If a sim_num is given, we extract the mab results only for the asked simulation as a list
                mab_results = results[sim_num][mab_index]

            fig.add_trace(go.Scatter(x=list(range(len(mab_results))),
                                     y=mab_results,
                                     mode='lines',
                                     name=self.mabs[mab_index].name,
                                     line=dict(color=px.colors.qualitative.Plotly[mab_index])))

            if (sim_num is None) and self.display_ci:
                fig.add_trace(go.Scatter(x=list(range(len(mab_results))),
                                         y=upper_bound,
                                         mode='lines',
                                         marker=dict(color=hex_to_rgba(px.colors.qualitative.Plotly[mab_index], 0.2)),
                                         line=dict(width=0),
                                         showlegend=False))
                fig.add_trace(go.Scatter(x=list(range(len(mab_results))),
                                         y=lower_bound,
                                         marker=dict(color=hex_to_rgba(px.colors.qualitative.Plotly[mab_index], 0.2)),
                                         line=dict(width=0),
                                         mode='lines',
                                         fillcolor=hex_to_rgba(px.colors.qualitative.Plotly[mab_index], 0.2),
                                         fill='tonexty',
                                         showlegend=False))
            fig.update_layout(title_text=f"{title} at iteration : {self.n_iter}")
            if self.show_plots and (sim_num is None):
                fig.show()

        if sim_num is None:
            fig_path = os.path.join(self.plots_path, "aggregated")
        else:
            fig_path = os.path.join(self.plots_path, f"simulation_{sim_num}")
        if not os.path.exists(fig_path):
            if not os.path.exists(self.plots_path):
                os.mkdir(self.plots_path)
            os.mkdir(fig_path)
        if 'Cumulative' in title:
            fig.write_image(f"{fig_path}/cumulative_{self.n_iter}.png", scale=4)
        elif 'Average' in title:
            fig.write_image(f"{fig_path}/average_{self.n_iter}.png", scale=4)
        elif 'Regrets' in title:
            fig.write_image(f"{fig_path}/regrets_{self.n_iter}.png", scale=4)
        else:
            raise ValueError('Not a valid figure title name.')

    def plot_average_reward_per_arm(self, sim_num) -> None:
        random_mab = self.mabs[0]
        rewards_with_drift = random_mab.compute_rewards_average_with_drift()

        traces = []
        for i, rewards in enumerate(rewards_with_drift):
            trace = go.Scatter(
                x=list(range(len(rewards))),
                y=rewards,
                mode='lines',
                name=f'Rewards of arm {i}'
            )
            traces.append(trace)

        layout = go.Layout(
            title='Arms rewards with drift',
        )

        fig = go.Figure(data=traces, layout=layout)

        fig_path = os.path.join(self.plots_path, f"simulation_{sim_num}")
        if not os.path.exists(fig_path):
            if not os.path.exists(self.plots_path):
                os.mkdir(self.plots_path)
            os.mkdir(fig_path)
        fig.write_image(f"{fig_path}/rewards_with_drift.png", scale=4)
        if self.show_plots:
            fig.show()


def hex_to_rgba(hex_color, alpha=1.0):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'rgba({r},{g},{b},{alpha})'
