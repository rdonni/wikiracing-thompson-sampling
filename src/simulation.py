import os

import numpy as np
import plotly.graph_objs as go
from tqdm import tqdm
import json


class Simulation:
    def __init__(self, mabs, nb_iterations, eval_iterations, results_path, plots_path, show_plots):
        self.mabs = mabs
        self.nb_iterations = nb_iterations
        self.eval_iterations = eval_iterations
        self.results_path = results_path
        self.plots_path = plots_path
        self.show_plots = show_plots

        self.n_iter = 0

    def simulation(self):
        for i in tqdm(range(self.nb_iterations)):
            for mab in self.mabs:
                mab.run_one_iteration()
            self.n_iter += 1
            if i % self.eval_iterations == self.eval_iterations - 1:
                for mab in self.mabs:
                    print(f"At iteration {self.n_iter}, the best path is {mab.get_best_path()} for {mab.name}")
                self.generate_plots(show=self.show_plots)

        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)
        results = {mab.name: list(mab.latencies) for mab in self.mabs}
        with open(f"{self.results_path}/results.json", "w") as results_path:
            json.dump(results, results_path)

    def generate_plots(self, show=True):
        fig = go.Figure()

        for mab in self.mabs:
            fig.add_trace(go.Scatter(x=list(range(len(mab.latencies))),
                                     y=np.cumsum(mab.latencies), #/ (np.arange(len(mab.latencies)) + 1),
                                     mode='lines',
                                     name=mab.name))

        fig.update_layout(title_text=f"Cumulative latency at iteration : {self.n_iter}")
        if show:
            fig.show()
        if not os.path.exists(self.plots_path):
            os.mkdir(self.plots_path)
        fig.write_image(f"{self.plots_path}/iteration_{self.n_iter}.png")
