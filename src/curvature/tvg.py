import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np
import portion as P
import ot

from typing import Callable
from itertools import combinations

import soap_parser.tvg as tvg
from soap_parser.matrix import IntervalMatrix

INF = float("inf")
# INF = 10 ** 30

class TVG(tvg.TVG):

    def get_ball(self, u: int, t: float,  K: float) -> list[int]:
        graph = self.get_graph_at(t)

        nodes: list[int] = []
        for node in graph:
            dist = nx.shortest_path_length(graph, u, node)
            if dist <= K:
                nodes.append(node)
        return nodes

    def get_cone(self, u: int, t: float, r: float, s: float) -> list[int]:
        
        # print(f"{t = }, {r = }, {s = }")
        l = int((s - t) / r)

        if t <= s < t + r:
            return [u]
        elif (t + r) <= s < (t + 2 * r):
            return self.get_ball(u, t + r, 1)
        elif (t + l * r) <= s < (t + (l + 1) * r):
            nodes: set[int] = set()
            for v in self.get_cone(u, t, r, t + (l - 1) * r):
                nodes = nodes | set(self.get_ball(v, t + (l - 1) * r, 1))
            return list(nodes)
        else:
            raise ValueError()

    # TODO : finish; incomplete
    def get_temporal_cost(self, u: int, v: int, t: float, r: float) -> float:
        times = self.get_critical_times()
        # print(f"{times = }")
        distances = [s for s in times if v in self.get_cone(u, t, r, s)]
        return min(distances)

    def calculate_distances(self, 
        r: float,
        truncate: bool = False
    ) -> list[list[list[float]]]:
        nodes = self.graph.nodes()
        n = len(nodes)

        critical_times = self.get_critical_times()
        start, end = critical_times[0], critical_times[-1]
        sample_times = list(np.arange(start, end, r))

        adjacency_matrices = []
        for time in sample_times:
            adjacency_matrices.append(
                np.array(self.get_adjacency_matrix_at(time))
            )

        distance_matrices = []
        last_distance_matrix = np.full((n, n), INF, dtype = np.float64)

        # first graph
        np.fill_diagonal(last_distance_matrix, 0)
        last_distance_matrix[adjacency_matrices[-1] == 1] = r
        distance_matrices.append(last_distance_matrix)

        # rest of graphs
        for adjacency_matrix in reversed(adjacency_matrices[1:]):
            distance_matrix = np.full((n, n), INF)
            np.fill_diagonal(distance_matrix, 0)

            for row in nodes:
                distance_matrix[row] = np.minimum(
                    last_distance_matrix[row] + r,
                    np.where(adjacency_matrix[row] == 1,
                        r,
                        last_distance_matrix[row] + r
                    )
                )

                for column in np.where(adjacency_matrix[row] == 1)[0]:
                    distance_matrix[row] = np.minimum(
                        distance_matrix[row],
                        last_distance_matrix[column] + r
                    )

            distance_matrices.append(distance_matrix)
            last_distance_matrix = distance_matrix
        
        return [matrix.tolist() for matrix in reversed(distance_matrices)]

    # TODO : think summary graph is based on sample_times?
    def get_summary_graph_thickness(self) -> list[float]:
        critical_times = self.get_critical_times()
        # start, end = critical_times[0], critical_times[-1]
        lifetime = critical_times[-1] - critical_times[0]

        alive = lambda u, v : sum([i.upper - i.lower 
            for i in self.graph[u][v]["contacts"]]) / lifetime

        return [alive(u, v) for u, v in self.graph.edges]

    def get_summary_graph_colors(self,
        distance_matrices: list[list[list[float]]],
        kernels: list[list[list[float]]],
        calculate_curvature: Callable[[list[list[float]], list[list[float]], int, int, float, float], float],
        K: float = 1,
        r: float = 1
    ) -> list[float]:

        assert len(distance_matrices) == len(kernels)

        # temporary truncation code
        T = 0
        for matrix in distance_matrices:
            if any(value >= INF for _, value in np.ndenumerate(matrix)):
                break
            else:
                T += 1

        # print(f"Truncation {T = }")

        distance_matrices = distance_matrices[0:T]
        kernels = kernels[0:T]

        curvature: dict[tuple[int, int], list[float]] = {}
        for matrix, kernel in zip(distance_matrices, kernels):
            for u, v in self.graph.edges:
                c = calculate_curvature(matrix, kernel, u, v, K, r)
                curvature.setdefault((u, v), []).append(c)
                # curvature[(u, v)].append(c)

        return [sum(curvature[(u, v)]) / T for u, v in self.graph.edges]

    def bandpass_filter(self,
        sample_times: list[float],
        distance: Callable[[float, int, int], float],
        threshold_low: float = -INF,
        threshold_high: float = INF
    ) -> "TVG":
        nodes = self.graph.nodes
        n = len(nodes)
        matrix = IntervalMatrix(n, n)

        for (start, end) in zip(sample_times, sample_times[1:]):
            interval = P.closed(start, end)

            for u, v in list(combinations(nodes, 2)):
                print(f"distance[{start}][{u}][{v}] = {distance(start, u, v)}")
                if threshold_low <= distance(start, u, v) <= threshold_high:
                    matrix[u, v] = matrix[u, v] | interval
        print(matrix)

        return TVG(matrix)

    def get_summary_graph(self, K: float = 1, r: float = 1) -> nx.Graph:
        critical_times = self.get_critical_times()
        start, end = critical_times[0], critical_times[-1]
        
        for source, target in self.graph.edges:
            self.graph[source][target]["curvature"] = 0
            
            contacts = self.graph[source][target]["contacts"]
            count: float = 0
            for interval in contacts:
                count += interval.upper - interval.lower
            self.graph[source][target]["alive"] = count / (end - start)

        return self.graph

TemporalNetwork = TVG

def build_cycle_tvg(n: int, start: float = -P.inf, end: float = P.inf) -> TVG:
    matrix = IntervalMatrix(n, n, labels = [str(k) for k in range(n)])

    for k in range(n - 1):
        matrix[k, k + 1] = P.closed(start, end)
    matrix[0, n - 1] = P.closed(start, end)

    # print(matrix)

    return TVG(matrix)

if __name__ == "__main__":
    network = build_cycle_tvg(n := 5, start = 0, end = 5)
    # TODO : add tests

def build_complete_tvg(
    n: int,
    start: float = -P.inf,
    end: float = P.inf
) -> TVG:
    matrix = IntervalMatrix(n, n, labels = [str(k) for k in range(n)])

    for u, v in combinations(list(range(n)), 2):
        matrix[u, v] = P.closed(start, end)

    return TVG(matrix)

if __name__ == "__main__":
    network = build_complete_tvg(n := 5, start = 0, end = 5)
    # TODO : add tests


def index(sample_times: list[float], time: float) -> int:

    for idx, (start, end) in enumerate(zip(sample_times, sample_times[1:])):
        if start <= time < end:
            return idx

    raise ValueError(f"{time = } is outside the range of `sample_times`")

if __name__ == "__main__":
    sample_times: list[float] = [0, 1, 4, 5, 8]
    assert index(sample_times, time = 0) == 0
    assert index(sample_times, time = 3) == 1
    assert index(sample_times, time = 6) == 3

if __name__ == "__main__":
    network = build_cycle_tvg(n := 20, start = 0, end = 5)

    # nodes = network.get_ball(0, 1, 0.5)
    assert [0, 1, 19] == network.get_ball(0, 0.5, 1)
    # print(f"{nodes = }")

    assert [0] == network.get_cone(0, 0, 1, 0)
    assert [0, 1, 19] == network.get_cone(0, 0, 1, 1)
    assert [0, 1, 2, 18, 19] == sorted(network.get_cone(0, 0, 1, 2))
    # print(f"{network.get_cone(0, 0, 1, 0)}")
    # print(f"{network.get_cone(0, 0, 1, 1)}")
    # print(f"{network.get_cone(0, 0, 1, 2)}")

    # TODO : test `get_temporal_cost`
    # print(f"{network.get_temporal_cost(0, 2, 0, 1)}")
    network.get_summary_graph()