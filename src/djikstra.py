import heapq


def dijkstra(graph, start, end, latencies):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    queue = [(0, start)]
    predecessors = {node: None for node in graph}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == end:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = predecessors[current_node]
            return distances, list(reversed(path))

        if current_distance > distances[current_node]:
            continue

        for neighbor in graph[current_node]:
            weight = latencies[neighbor]
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    return float('inf'), None