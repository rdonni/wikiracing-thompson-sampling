import json
import os

import numpy as np
import plotly as px
import plotly.graph_objs as go
from tqdm import tqdm


class Simulation:
    def __init__(self, mabs, nb_iterations, nb_simulations, eval_iterations, results_path, plots_path, show_plots, display_ci):
        self.mabs = mabs
        self.nb_simulations = nb_simulations
        self.nb_iterations = nb_iterations
        self.eval_iterations = eval_iterations
        self.results_path = results_path
        self.plots_path = plots_path
        self.show_plots = show_plots
        self.display_ci = display_ci

        self.n_iter = 0

    def simulation(self):

        results = {}
        for sim_num in range(self.nb_simulations):
            print(f'Running simulation number {sim_num}...')
            # We reset the distributions between each simulation
            self.reset()
            for _ in tqdm(range(self.nb_iterations)):
                for mab in self.mabs:
                    mab.run_one_iteration()
                self.n_iter += 1
            for mab in self.mabs:
                print(f"At iteration {self.n_iter}, for {mab.name}, cumulative latency is equal to {np.sum(mab.latencies)} the best path is {mab.get_best_path()}.")
            results[sim_num] = [list(mab.latencies) for mab in self.mabs]
            self.generate_plots(results, sim_num)
        self.generate_plots(results)

        # Save results as a json file
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)
        with open(f"{self.results_path}/results.json", "w") as results_path:
            json.dump(results, results_path)

    def reset(self):
        for mab in self.mabs:
            mab.reset()
        self.n_iter = 0

    def generate_plots(self, results, sim_num: int = None) -> None:

        fig = go.Figure()
        for mab_index in range(len(self.mabs)):
            if sim_num is None:
                # If no sim_num is given, we extract the mab results of all simulations as a matrix
                mab_results = [results[sim_num][mab_index] for sim_num in results.keys()]
                mab_results = np.cumsum(mab_results, axis=1)

                ci = 1.96 * np.std(mab_results, axis=0) / np.sqrt(self.nb_simulations)
                mab_results = np.mean(mab_results, axis=0)
                upper_bound = mab_results + ci
                lower_bound = mab_results - ci

            else:
                # If a sim_num is given, we extract the mab results only for the asked simulation as a list
                mab_results = results[sim_num][mab_index]
                mab_results = np.cumsum(mab_results)

            fig.add_trace(go.Scatter(x=list(range(len(mab_results))),
                                     y=mab_results,
                                     mode='lines',
                                     name=self.mabs[mab_index].name))

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
            fig.update_layout(title_text=f"Cumulative loading time at iteration : {self.n_iter}")
            if self.show_plots:
                fig.show()

            fig_2 = go.Figure()
            for mab_index in range(len(self.mabs)):
                if sim_num is None:
                    # If no sim_num is given, we extract the mab results of all simulations as a matrix
                    mab_results = [results[sim_num][mab_index] for sim_num in results.keys()]
                    mab_results = np.cumsum(mab_results, axis=1) / (np.arange(self.nb_iterations) + 1)

                    ci = 1.96 * np.std(mab_results, axis=0) / np.sqrt(self.nb_simulations)
                    mab_results = np.mean(mab_results, axis=0)
                    upper_bound = mab_results + ci
                    lower_bound = mab_results - ci

                else:
                    # If a sim_num is given, we extract the mab results only for the asked simulation as a list
                    mab_results = results[sim_num][mab_index]
                    mab_results = np.cumsum(mab_results) / (np.arange(self.nb_iterations) + 1)

                fig_2.add_trace(go.Scatter(x=list(range(len(mab_results))),
                                         y=mab_results,
                                         mode='lines',
                                         name=self.mabs[mab_index].name))

                if (sim_num is None) and self.display_ci:
                    fig_2.add_trace(go.Scatter(x=list(range(len(mab_results))),
                                             y=upper_bound,
                                             mode='lines',
                                             marker=dict(
                                                 color=hex_to_rgba(px.colors.qualitative.Plotly[mab_index], 0.2)),
                                             line=dict(width=0),
                                             showlegend=False))
                    fig_2.add_trace(go.Scatter(x=list(range(len(mab_results))),
                                             y=lower_bound,
                                             marker=dict(
                                                 color=hex_to_rgba(px.colors.qualitative.Plotly[mab_index], 0.2)),
                                             line=dict(width=0),
                                             mode='lines',
                                             fillcolor=hex_to_rgba(px.colors.qualitative.Plotly[mab_index], 0.2),
                                             fill='tonexty',
                                             showlegend=False))
                fig_2.update_layout(title_text=f"Cumulative loading time at iteration : {self.n_iter}")
                if self.show_plots:
                    fig_2.show()

            if sim_num is None:
                fig_path = os.path.join(self.plots_path, "aggregated")
            else:
                fig_path = os.path.join(self.plots_path, f"simulation_{sim_num}")
            if not os.path.exists(fig_path):
                if not os.path.exists(self.plots_path):
                    os.mkdir(self.plots_path)
                os.mkdir(fig_path)
            fig.write_image(f"{fig_path}/cumulative_{self.n_iter}.png", scale=4)
            fig_2.write_image(f"{fig_path}/average_{self.n_iter}.png", scale=4)


def hex_to_rgba(hex_color, alpha=1.0):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'rgba({r},{g},{b},{alpha})'
