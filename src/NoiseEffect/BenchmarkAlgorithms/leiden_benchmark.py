import leidenalg as la
import igraph as ig
import networkx as nx
import itertools
from NoiseEffect.BenchmarkAlgorithms.utils import convertPartitionToLabels, getMetrics


def benchmarkLeidenAlgorithm(nx_graph, list_of_seeds):
    ig_graph = ig.Graph.from_networkx(nx_graph)
    num_nodes = ig_graph.vcount()
    partitions = leidenAlgorithmPartioning(ig_graph, list_of_seeds)
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


def leidenAlgorithmPartioning(ig_graph, list_of_seeds):
        """
        Finds communities using the Leiden algorithm on an igraph object.

        Args:
            ig_graph (igraph.Graph): The graph to analyze.

        Returns:
            list[set[int]]: A list of sets, where each set contains the integer
                            vertex IDs of a community.
        """
        partitions = {}
        for seed in list_of_seeds:
            partition_ig = la.find_partition(
                ig_graph, la.ModularityVertexPartition, seed=seed
            )
            partitions[seed] = [set(community) for community in partition_ig]

        # Return the full partition, converted to a list of sets of integer IDs.
        return partitions

