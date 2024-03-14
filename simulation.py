import json

import hydra
from omegaconf import DictConfig, OmegaConf

from src.bandits import GaussianTSArm, MultiArmedBandit, EpsGreedyArm, UCBArm
from src.dfs import dfs
from src.simulation import Simulation


@hydra.main(version_base=None, config_path="./config", config_name="simulation")
def main(cfg: DictConfig) -> None:
    print('Configuration:\n', OmegaConf.to_yaml(cfg))

    with open(cfg.graph_path) as graph:
        graph = json.loads(graph.read())

    # Extract all paths from start_node to end_node using depth-first-search algorithms
    paths = dfs(graph, cfg.start_node, cfg.end_node, max_depth=cfg.path_max_depth)[:cfg.max_num_paths]
    print(f"Found {len(paths)} paths between {cfg.start_node} and {cfg.end_node}, using max_depth={cfg.path_max_depth}")

    mabs = []
    for alg in cfg.algorithms:
        if 'thompson-sampling' in alg.name:
            ts_arms = [GaussianTSArm(initial_params=alg.initial_parameters,
                                     path=paths[i]) for i in range(len(paths))]
            mabs.append(MultiArmedBandit(ts_arms,
                                         name=alg.name,
                                         use_synthetic_distributions=cfg.use_synthetic_distributions,
                                         use_drift=cfg.use_drift))

        elif 'ucb' in alg.name:
            ucb_arms = [UCBArm(confidence_level=alg.confidence_level,
                               path=paths[i]) for i in range(len(paths))]
            mabs.append(MultiArmedBandit(ucb_arms,
                                         name=alg.name,
                                         use_synthetic_distributions=cfg.use_synthetic_distributions,
                                         use_drift=cfg.use_drift))

        elif 'epsilon-greedy' in alg.name:
            eps_arms = [EpsGreedyArm(path=paths[i]) for i in range(len(paths))]
            mabs.append(MultiArmedBandit(eps_arms,
                                         name=alg.name,
                                         use_synthetic_distributions=cfg.use_synthetic_distributions,
                                         use_drift=cfg.use_drift,
                                         epsilon=alg.epsilon))
        else:
            raise ValueError('Not a valid algorithm name')

    print(f'Using following algorithms : {[mab.name for mab in mabs]}.')

    simulation = Simulation(mabs=mabs,
                            nb_simulations=cfg.num_simulations,
                            nb_iterations=cfg.num_iters,
                            use_synthetic_distributions=cfg.use_synthetic_distributions,
                            synthetic_data_config=cfg.synthetic_data_config,
                            use_drift=cfg.use_drift,
                            drift_method=cfg.drift_method,
                            results_path=cfg.results_path,
                            plots_path=cfg.plots_path,
                            show_plots=cfg.show_plots,
                            display_ci=cfg.display_ci)
    simulation.simulation()


if __name__ == '__main__':
    main()
