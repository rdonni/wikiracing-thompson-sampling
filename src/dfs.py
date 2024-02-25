def dfs(graph, start, end, max_depth, path=None):
    if path is None:
        path = []
    path = path + [start]

    if start == end:
        return [path]

    if max_depth <= 0:
        return []

    if start not in graph:
        return []

    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = dfs(graph, node, end, max_depth - 1, path)
            for new_path in new_paths:
                paths.append(new_path)
    return paths



