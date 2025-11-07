import leidenalg as la
from infomap import Infomap

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
            infomap_wrapper = Infomap(f"--two-level --silent --seed {seed} --num-trials {n_iterations}")

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