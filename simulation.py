import json

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.bandits import GaussianTSArm, MultiArmedBandit, EpsGreedyArm, UCBArm
from src.dfs import dfs
from src.drift import Drift
from src.simulation import Simulation


@hydra.main(version_base=None, config_path="./config", config_name="simulation")
def main(cfg: DictConfig) -> None:
    print('Configuration:\n', OmegaConf.to_yaml(cfg))

    with open(cfg.graph_path) as graph:
        graph = json.loads(graph.read())

    # Extract all paths from start_node to end_node using depth-first-search algorithms
    paths = dfs(graph, cfg.start_node, cfg.end_node, max_depth=cfg.path_max_depth)[:cfg.max_num_paths]
    print(f"Found {len(paths)} paths between {cfg.start_node} and {cfg.end_node}, using max_depth={cfg.path_max_depth}")

    # TODO: A chaque simulation on doit changer les valeurs synthétiques
    if cfg.use_synthetic_distributions:
        # We model the average loading time of each path by a normal variable
        synthetic_means = np.random.uniform(500, 1500, len(paths))
        synthetic_stds = np.random.uniform(10, 100, len(paths))
        synthetic_params = list(zip(synthetic_means, synthetic_stds))
    else:
        # We use real distributions
        synthetic_params = [None] * len(paths)

    # Set drift for all arms
    if cfg.use_drift:
        drifts = [Drift(cfg.drift_method, cfg.num_iters) for _ in range(len(paths))]
    else:
        drifts = [None] * len(paths)

    mabs = []
    for alg in cfg.algorithms:
        if 'thompson-sampling' in alg.name:
            ts_arms = [GaussianTSArm(initial_params=alg.initial_parameters,
                                     path=paths[i],
                                     theoretical_params=synthetic_params[i],
                                     drift=drifts[i]) for i in range(len(paths))]
            mabs.append(MultiArmedBandit(ts_arms,
                                         name=alg.name,
                                         use_synthetic_distributions=cfg.use_synthetic_distributions,
                                         use_drift=cfg.use_drift))

        elif 'ucb' in alg.name:
            ucb_arms = [UCBArm(confidence_level=alg.confidence_level,
                               path=paths[i],
                               theoretical_params=synthetic_params[i],
                               drift=drifts[i]) for i in range(len(paths))]
            mabs.append(MultiArmedBandit(ucb_arms,
                                         name=alg.name,
                                         use_synthetic_distributions=cfg.use_synthetic_distributions,
                                         use_drift=cfg.use_drift))

        elif 'epsilon-greedy' in alg.name:
            eps_arms = [EpsGreedyArm(path=paths[i],
                                     theoretical_params=synthetic_params[i],
                                     drift=drifts[i]) for i in range(len(paths))]
            mabs.append(MultiArmedBandit(eps_arms,
                                         name=alg.name,
                                         use_synthetic_distributions=cfg.use_synthetic_distributions,
                                         use_drift=cfg.use_drift,
                                         epsilon=alg.epsilon))
        else:
            raise ValueError('Not a valid algorithm name')

    print(f'Using following algorithms : {[mab.name for mab in mabs]}.')


    # TODO: rajouter le paramètre use_theoretical_distribs
    simulation = Simulation(mabs=mabs,
                            nb_simulations=cfg.num_simulations,
                            nb_iterations=cfg.num_iters,
                            use_drift=cfg.use_drift,
                            results_path=cfg.results_path,
                            plots_path=cfg.plots_path,
                            show_plots=cfg.show_plots,
                            display_ci=cfg.display_ci)
    simulation.simulation()


if __name__ == '__main__':
    main()
