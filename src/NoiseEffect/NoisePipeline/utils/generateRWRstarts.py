import numpy as np
import random


def generateRWRstarts(random_seed_list, G):
    number_of_starts = len(random_seed_list)
    degree_sequence = sorted(
        G.nodes(),
        key=lambda n: G.degree(n),
        reverse=True,
    )
    # Single node start points
    start_point_indices = np.linspace(
        0, len(degree_sequence) - 1, number_of_starts, dtype=int
    )
    single_start_points = [[degree_sequence[i]] for i in start_point_indices]
    # Single node plus first neighbors
    first_neighbors_full = []
    count = 0
    for node in single_start_points:
        count += 1
        neighbors = list(G.neighbors(node[0]))
        first_neighbors_full.append(list(set(node + neighbors)))
    # print(f"Generated first neighbor start points for {count} nodes.")

    second_neighbors_full = []
    count = 0
    for list_of_nodes in first_neighbors_full:
        count += 1
        second_neighbors = []
        for first_neighbor in list_of_nodes[1:]:
            second_neighbors.extend(G.neighbors(first_neighbor))
        second_neighbors_full.append(list(set(list_of_nodes + second_neighbors)))
    # print(f"Generated second neighbor start points for {count} nodes.")

    first_neighbors_start_points = [
        random.sample(_, max(1, len(_) // 2)) for _ in first_neighbors_full
    ]
    second_neighbors_start_points = [
        random.sample(_, max(1, len(_) // 2)) for _ in second_neighbors_full
    ]

    all_start_points = (
        single_start_points
        + first_neighbors_start_points
        + second_neighbors_start_points
    )

    return all_start_points
