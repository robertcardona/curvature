from curvature.tvg import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

# INF = 10 ** 30
INF = float("inf")

def radius_1_uniform_kernel(tvg: TVG,
    sample_times: list[float]
) -> list[list[list[float]]]:

    n = len(tvg.graph.nodes())

    kernel_list = []
    for t in sample_times:
        adjacency_matrix = np.array(tvg.get_adjacency_matrix_at(t))
        # adjacency_matrix = np.array(
        #     tvg.get_adjacency_matrix_at(t),
        #     dtype = np.float64
        # )
        adjacency_matrix += np.eye(n, dtype = np.int64)

        normalized_matrix = adjacency_matrix / adjacency_matrix.sum(
            axis = 1,
            keepdims = True
        )
        kernel_list.append(normalized_matrix.tolist())

    return kernel_list

def calculate_curvature(
    distance_matrix: list[list[float]],
    kernel: list[list[float]],
    source: int,
    target: int,
    K: float,
    r: float
) -> float:

    # source_measure = kernel[source]
    # target_measure = kernel[target]
    
    distance = distance_matrix[source][target]

    if distance == 0:
        return 0

    W = ot.emd2(
        source_measure := kernel[source],
        target_measure := kernel[target],
        distance_matrix,
        processes = 1,
        numItermax = 100_00,
        log = False,
        return_matrix = False,
        center_dual = True,
        numThreads = 1
    )

    curvature = INF
    if W <= INF / 10 and distance <= INF / 10:
        curvature = (distance - W) / distance

    return curvature

# TODO : check : length of first two related to r, maybe remove r
def calculate_curvature_matrices(
    distance_matrices,
    kernels: list[list[list[float]]],
    K: float,
    r: float
) -> list[list[list[float]]]:
    assert len(distance_matrices) == len(kernels)
    n = len(distance_matrices[0])

    curvature_matrices = []
    for distance_matrix, kernel in zip(distance_matrices, kernels):
        curvature_matrix = np.zeros((n, n))
        for i, j in IntervalMatrix.get_indices(n, n):
            # TODO : possibly replace
            curvature_matrix[i][j] = calculate_curvature(
                distance_matrix,
                kernel,
                i,
                j,
                K,
                r
            )
        curvature_matrices.append(curvature_matrix.tolist())
    return curvature_matrices

# TODO : move draw functions to `visualization` module

# TODO : `draw_scatter_plot` related to summary graph

# TODO : pass in `sample_times` instead of `K` and `r`
def draw_summary_graph(
    tvg: TVG,
    title = None,
    K = 1,
    r = 1
) -> None:

    scale = 2

    critical_times = tvg.get_critical_times()
    start_time, end_time = critical_times[0], critical_times[-1]
    sample_times = np.arange(start_time, end_time, r).tolist()

    distance_matrices = tvg.calculate_distances(r)
    # for dm in distance_matrices:
    #     print(f"{dm = }")
    kernels = radius_1_uniform_kernel(tvg, sample_times)
    # for kernel in kernels:
    #     print(f"{kernel = }")
    
    colors = tvg.get_summary_graph_colors(
        distance_matrices,
        kernels,
        lambda matrix, kernel, u, v, K, r: calculate_curvature(matrix, kernel, u, v, K, r),
        K = K,
        r = r
    )
    weights = [w * scale for w in tvg.get_summary_graph_thickness()]

    vmin, vmax = min(colors), max(colors)
    if vmin < vmax:
        norm = Normalize(vmin = min(colors), vmax = max(colors))
    else:
        norm = Normalize(vmin = 0, vmax = 1)
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    # cmap = matplotlib.colormaps.get_cmap("viridis")
    sm = ScalarMappable(norm = norm, cmap = cmap)
    edge_colors = [sm.to_rgba(np.array(color)) for color in colors]

    # build pos
    pos: dict[int, tuple[int, int]] = dict()
    sg = tvg.graph
    n = len(sg.nodes())
    for node in sg.nodes():
        pos[node] = (np.cos(2 * np.pi * node / n), np.sin(2 * np.pi * node / n))
        # print(f"{node = }")
    # print(f"{pos = }")

    # post = nx.shell_layout(sg)
    nx.draw(sg, pos = pos, with_labels=True, font_weight='bold', edge_color=edge_colors, width=weights)
    if title is not None:
        plt.title(title)
    plt.colorbar(sm, label = "Average Curvature", ax = plt.gca())
    plt.savefig(f"{title}.png")
    plt.show()

    return None

if __name__ == "__main__":
    # network.calculate_distances(1)

    n = 20
    start, end = 0, 100
    sample_times = np.arange(start, end, 1).tolist()

    network = build_cycle_tvg(n = n, start = start, end = end)
    # network = build_complete_tvg(n := 15, start = start, end = end)
    distance_matrices = network.calculate_distances(r := 1)
    for m in distance_matrices:
        # print(m)
        pass

    # draw_summary_graph(network, title = "Scenario")

    distance = lambda t, u, v: distance_matrices[index(sample_times, t)][u][v]
    # network_filtered = network.bandpass_filter(sample_times, distance, threshold_high = 1)

    # tvg.draw_reeb_graph(network_filtered.get_reeb_graph(sample_times = sample_times))

    network = build_cycle_tvg(n := 15, start = start, end = end)
    draw_summary_graph(network, title = f"Cycle Graph : {n}")

    network = build_complete_tvg(n = n, start = start, end = end)
    # draw_summary_graph(network, title = f"Complete Graph : {n}")