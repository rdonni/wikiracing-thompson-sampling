from collections import deque
import pywikibot
from tqdm import tqdm
import concurrent.futures
import json


def build_graph(start_node: str, max_graph_nodes: int, max_workers: int) -> dict:
    graph = {}

    current_pages = [start_node]
    waiting_nodes = deque()
    nb_iter = max_graph_nodes // max_workers

    for _ in tqdm(range(nb_iter)):
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            page_links = list(executor.map(extract_linked_pages, current_pages))

        graph.update({current_pages[i]: page_links[i] for i in range(len(page_links))})
        page_links = [l for links in page_links for l in links if l not in graph]
        waiting_nodes += page_links

        current_pages = [waiting_nodes.popleft() for _ in range(max_workers)]

    graph = remove_unknown_neighbors_nodes(graph)
    return graph


def extract_linked_pages(article_title, language='en'):
    site = pywikibot.Site(language, 'wikipedia')
    page = pywikibot.Page(site, article_title)
    linked_pages = page.linkedPages()

    linked_article_titles = [linked_page.title() for linked_page in linked_pages\
                             if (linked_page.namespace() == 0) and ('identifier' not in linked_page.title()) and
                             (linked_page.exists())]
    return linked_article_titles


def remove_unknown_neighbors_nodes(graph):
    known_neighbors_nodes = list(graph.keys())
    pruned_graph = {key: [v for v in value if v in known_neighbors_nodes] for key, value in graph.items()}
    return pruned_graph


def save_graph(graph, file_path) -> None:
    with open(file_path, "w") as outfile:
        json.dump(graph, outfile)


def load_graph(file_path):
    with open(file_path) as graph:
        graph = json.loads(graph.read())
    return graph
