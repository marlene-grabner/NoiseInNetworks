import networkx as nx
import random


class PerturbedEdges:
    def __init__(self, original_network, noise_information):
        self.original_network = original_network
        self.noise_information = noise_information
        self.random_added_edges_dict = {}
        self.random_removed_edges_dict = {}

    ######### 2. Add noise to the original network #########
    # Edge Addition (self.random_added_edges_dict)
    # Edge Removal (self.random_removed_edges_dict)
    #
    # Dictionary for each type of modification, with specified noise level as key
    # will be created in the class attributes
    #
    # User can specify number of neworks per noise level that are created
    ######################################################

    def generateNoisyNetworkSets(self):
        modification_percentages = self.noise_information.get(
            "noise_levels", [0.1, 0.5]
        )
        num_noise_repeats = self.noise_information.get("num_repeats", 3)

        for percentage in modification_percentages:
            num_modify = self._calcualteNumberOfEdgesToModify(percentage)

            for repetition in range(num_noise_repeats):
                # Mark noise level and repetition in the key
                marker = f"{percentage}_{repetition}"

                self.random_added_edges_dict[marker] = self._randomEdgesToAdd(
                    num_modify
                )
                self.random_removed_edges_dict[marker] = self._randomEdgesToRemove(
                    num_modify
                )

    ### 1 Calcualtes how many edges shall be modified ###
    def _calcualteNumberOfEdgesToModify(self, percentage):
        num_edges = len(self.original_network.edges)
        num_modify = int(num_edges * percentage)  # Number of edges to modify

        # Check if results make sense
        if num_modify <= 0:
            raise ValueError("Number of edges to modify can not be negative or zero.")

        if num_modify >= num_edges:
            raise ValueError("Can not remove all edges in the graph.")

        return num_modify

    ### 2.1 Adds edges randomly ###
    def _randomEdgesToAdd(self, num_modify):
        # 1. Get all possible edges that don't already exist.
        possible_edges_to_add = list(nx.non_edges(self.original_network))

        # Raise error if more edges than possible are to be added
        if num_modify > len(possible_edges_to_add):
            raise ValueError("More edges to be added than complete network.")

        # 2. Directly sample unique edges to add.
        edges_to_add = random.sample(possible_edges_to_add, num_modify)
        return edges_to_add

    ### 2.2 Removed edges randomly ###
    def _randomEdgesToRemove(self, num_modify):
        edges = list(self.original_network.edges)
        # Directly sample unique edges to remove
        edges_to_remove = random.sample(edges, num_modify)
        return edges_to_remove
