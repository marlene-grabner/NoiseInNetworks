import igraph as ig
import networkx as nx
import itertools
from NoiseEffect.BenchmarkAlgorithms.utils import convertPartitionToLabels, getMetrics
from NoiseEffect.BenchmarkAlgorithms.detection_algorithms import leidenAlgorithmPartioning, infomapAlgorithmPartioning


def benchmarkAlgorithm(nx_graph, list_of_seeds, algorithm, parameters = {}):
    """
    Runs a community detection algorithm multiple times to benchmark its stability.

    Args:
        nx_graph (networkx.Graph): The input graph.
        list_of_seeds (list[int]): A list of random seeds to run the algorithm with.
        algorithm (str): The name of the algorithm to use ('leiden', 'infomap').
        parameters (dict, optional): Extra parameters for the algorithm,
                                     e.g., {'n_iterations': 5}. Defaults to {}.

    Returns:
        list[dict]: A list of metric dictionaries, comparing all pairs
                    of runs. Each dict contains 'ari', 'ami', etc.
    """

    ig_graph = ig.Graph.from_networkx(nx_graph)
    num_nodes = ig_graph.vcount()

    if algorithm == 'leiden':
        n_iterations = parameters.get('n_iterations', 2) # Get specified n_iterations or use default chosed by leidenalg package (which is 2)
        partitions = leidenAlgorithmPartioning(ig_graph, list_of_seeds, n_iterations=n_iterations)
    elif algorithm == 'louvain':
        raise NotImplementedError("Louvain algorithm not yet implemented.")
    elif algorithm == 'infomap':
        n_iterations = parameters.get('n_iterations', 1) # Get specified n_iterations or use default value for infomap package which is 1
        partitions = infomapAlgorithmPartioning(ig_graph, list_of_seeds, n_iterations=n_iterations)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    
    labels = []
    for seed, partition in partitions.items():
        labels.append(convertPartitionToLabels(partition, num_nodes))

    all_ordered_pairs = itertools.combinations(labels, 2)

    results = []
    for pair in all_ordered_pairs:
        # *pair unpacks the tuple (e.g., (10, 20)) into two arguments
        result = getMetrics(*pair)
        results.append(result)
    return results




