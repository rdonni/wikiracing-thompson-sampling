import hydra
from omegaconf import DictConfig, OmegaConf

from src.graph import build_graph, save_graph


@hydra.main(version_base=None, config_path="./config", config_name="graph")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    graph = build_graph(cfg.start_node, cfg.max_num_nodes, cfg.max_workers)
    save_graph(graph, cfg.graph_path)


if __name__ == '__main__':
    main()
