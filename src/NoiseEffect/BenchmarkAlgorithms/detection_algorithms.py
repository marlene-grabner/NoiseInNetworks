import leidenalg as la
from infomap import Infomap
import igraph as ig
import random
import numpy as np


def leidenAlgorithmPartioning(ig_graph, list_of_seeds, n_iterations):
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
            ig_graph, la.ModularityVertexPartition, seed=seed, n_iterations=n_iterations
        )
        partitions[seed] = [set(community) for community in partition_ig]

    # Return the full partition, converted to a list of sets of integer IDs.
    return partitions


def infomapAlgorithmPartioning(ig_graph, list_of_seeds, n_iterations):
    """
    Finds communities using the Infomap algorithm on an igraph object.

    Args:
        ig_graph (igraph.Graph): The graph to analyze.

    Returns:
        list[set[int]]: A list of sets, where each set contains the integer
                        vertex IDs of a community.
    """
    all_partitions = {}

    # Convert edges to list of tuples
    edges = [(e.source, e.target) for e in ig_graph.es]

    for seed in list_of_seeds:
        # Initialize Infomap
        infomap_wrapper = Infomap(
            f"--two-level --silent --seed {seed} --num-trials {n_iterations}"
        )

        # Add links
        for src, tgt in edges:
            infomap_wrapper.add_link(src, tgt)

        # Run Infomap
        infomap_wrapper.run()

        # Extract modules
        communities = {}
        for node_id, module_id in infomap_wrapper.get_modules().items():
            communities.setdefault(module_id, []).append(node_id)

        all_partitions[seed] = [set(nodes) for nodes in communities.values()]

    return all_partitions


def louvainPartioning(ig_graph, list_of_seeds):
    """
    Finds communities using the Louvain algorithm (Multilevel).
    Uses vertex attributes to track original IDs through the permutation.
    """
    all_partitions = {}

    # 1. Attach original IDs to the graph so they survive the shuffle
    # We use a temporary attribute '_orig_id'
    if "_orig_id" not in ig_graph.vertex_attributes():
        ig_graph.vs["_orig_id"] = range(ig_graph.vcount())

    for seed in list_of_seeds:
        random.seed(seed)

        # 2. Create random permutation
        perm = list(range(ig_graph.vcount()))
        random.shuffle(perm)

        # 3. Permute the graph
        # This reorders vertices AND their attributes (including _orig_id)
        g_permuted = ig_graph.permute_vertices(perm)

        # 4. Run Louvain on the shuffled graph
        vertex_partition = g_permuted.community_multilevel(weights=None)

        # 5. Extract communities using the attached '_orig_id'
        communities = []
        for subgraph in vertex_partition:
            # subgraph contains indices for g_permuted (e.g., 0, 1, 2...)
            # We look up what '_orig_id' these nodes hold
            original_nodes = {
                g_permuted.vs[node_idx]["_orig_id"] for node_idx in subgraph
            }
            communities.append(original_nodes)

        all_partitions[seed] = communities

    # Cleanup: Remove the temporary attribute from the original graph
    if "_orig_id" in ig_graph.vertex_attributes():
        del ig_graph.vs["_orig_id"]

    return all_partitions


def labelPropagationPartitioning(ig_graph, list_of_seeds):
    """
    Label Propagation: Fast but unstable.
    Great for proving your stability metric works (should have low ARI on noise).
    """
    all_partitions = {}

    # 1. Attach original IDs
    if "_orig_id" not in ig_graph.vertex_attributes():
        ig_graph.vs["_orig_id"] = range(ig_graph.vcount())

    for seed in list_of_seeds:
        random.seed(seed)

        # 2. Shuffle to trigger stochastic behavior
        perm = list(range(ig_graph.vcount()))
        random.shuffle(perm)
        g_permuted = ig_graph.permute_vertices(perm)

        # 3. Run Label Propagation
        vertex_partition = g_permuted.community_label_propagation()

        # 4. Map back
        communities = []
        for subgraph in vertex_partition:
            original_nodes = {
                g_permuted.vs[node_idx]["_orig_id"] for node_idx in subgraph
            }
            communities.append(original_nodes)

        all_partitions[seed] = communities

    # Cleanup
    if "_orig_id" in ig_graph.vertex_attributes():
        del ig_graph.vs["_orig_id"]

    return all_partitions
