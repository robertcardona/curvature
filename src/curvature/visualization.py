
from curvature.tvg import *
from curvature.curvature_utils import *

from soap_parser.visualization import convert_figure, save_gif, circular_pos, show_gif

import io
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from PIL import Image, ImageFile

# plt.rcParams["figure.figsize"] = (5, 5)

def draw_summary_graph(
    tvg: TVG,
    title = None,
    K = 1,
    r = 1
) -> ImageFile.ImageFile:
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
    
    # TODO : maybe add curvature matrix and replace lambda function with that
    #   NO : takes too long for this specific function, as we only need edges
    # curvature_matrices = calculate_curvature_matrices(
    #     distance_matrices,
    #     kernels,
    #     K = K,
    #     r = r
    # )
    # for cm in curvature_matrices:
    #     print(f"{cm = }")
    #     break

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
    # pos: dict[int, tuple[int, int]] = dict()
    # sg = tvg.graph
    # n = len(sg.nodes())
    # for node in sg.nodes():
    #     pos[node] = (np.cos(2 * np.pi * node / n), np.sin(2 * np.pi * node / n))
        # print(f"{node = }")
    # print(f"{pos = }")
    pos = circular_pos(sg := tvg.graph)

    # post = nx.shell_layout(sg)
    nx.draw(sg, pos = pos, with_labels=True, font_weight='bold', edge_color=edge_colors, width=weights)
    plt.colorbar(sm, label = "Average Curvature", ax = plt.gca())
    # plt.savefig(f"{title}.png")
    # plt.show()

    if title is not None:
        plt.title(f"Summary Graph {title}")
    
    figure = plt.gcf()
    # plt.clf()

    image = convert_figure(figure)
    # image.show()
    # image.save(f"{title}.png", "png")
    plt.clf()
    plt.cla()
    plt.close()

    return image
# image = convert_figure(plt.gcf())

def save_tvg(
    tvg: TVG,
    filename: str,
    sample_times: list[float] | None = None
) -> None:

    if sample_times is None:
        sample_times = tvg.get_critical_times()

    pos = circular_pos(tvg.get_graph_at(sample_times[0]))

    images: list[ImageFile.ImageFile] = []

    for t in sample_times:
        figure = plt.figure(figsize = (5, 5))
        g = tvg.get_graph_at(t)
        # pos = circular_pos(g)

        nx.draw(g, pos = pos, with_labels=True, font_weight='bold')

        plt.title(f"{t = }")
        figure = plt.gcf()
        # plt.show()
        images.append(convert_figure(figure))
        plt.clf()
        plt.cla()
        plt.close()

    save_gif(filename, images)

    return None

if __name__ == "__main__":
    # network.calculate_distances(1)

    n = 20
    start, end = 0, 100
    sample_times = np.arange(start, end, 1).tolist()

    network = build_cycle_tvg(n = n, start = start, end = end)
    save_tvg(network, f"cycle.gif", sample_times = sample_times)
    # display(show_gif(f"cycle.gif"))
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
    sg_cg = draw_summary_graph(network, title = f"Cycle Graph : {n}")

    network = build_complete_tvg(n = n, start = start, end = end)
    sg_kg = draw_summary_graph(network, title = f"Complete Graph : {n}")

    filename = f"aaa_summary_graphs_{n}.gif"
    save_gif(filename, [sg_cg, sg_kg] * 10)