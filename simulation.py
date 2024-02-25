import json

from src.bandits import GaussianTSArm, MultiArmedBandit, EpsGreedyArm, UCBArm
from src.dfs import dfs
from src.simulation import Simulation

GRAPH_PATH = './graph.json'
PATH_MAX_DEPTH = 3
START_NODE = 'Thompson sampling'
END_NODE = 'Albert Einstein'

NUM_ITERS = 400
EVAL_ITERS = 500
PLOTS_PATH = './plots'
RESULTS_PATH = './results'

if __name__ == '__main__':
    with open(GRAPH_PATH) as graph:
        graph = json.loads(graph.read())

    paths = dfs(graph, START_NODE, END_NODE, max_depth=PATH_MAX_DEPTH)
    print(f"Found {len(paths)} paths between {START_NODE} and {END_NODE}")

    ts_arms = [GaussianTSArm(initial_params=(500, 0, 1, 1000), path=paths[i]) for i in range(len(paths))]
    ts_mab = MultiArmedBandit(ts_arms, name='thompson')

    ucb_arms = [UCBArm(confidence_level=1, path=paths[i]) for i in range(len(paths))]
    ucb_mab = MultiArmedBandit(ucb_arms, name='ucb')

    eps_arms_0_05 = [EpsGreedyArm(path=paths[i]) for i in range(len(paths))]
    eps_greedy_mab_0_05 = MultiArmedBandit(eps_arms_0_05, name='epsilon_greedy_0_05', epsilon=0.05)

    simulation = Simulation(mabs=[ts_mab, ucb_mab, eps_greedy_mab_0_05],
                            nb_iterations=NUM_ITERS,
                            eval_iterations=EVAL_ITERS,
                            results_path=RESULTS_PATH,
                            plots_path=PLOTS_PATH,
                            show_plots=False)
    simulation.simulation()

