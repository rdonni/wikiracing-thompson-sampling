from src.graph import build_graph, save_graph

START_NODE = 'Thompson sampling'
MAX_NUM_NODES = 1000
MAX_WORKERS = 11
GRAPH_PATH = './graph.json'


if __name__ == '__main__':
    graph = build_graph(START_NODE, MAX_NUM_NODES, MAX_WORKERS)
    save_graph(graph, GRAPH_PATH)
