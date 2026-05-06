import networkx as nx
import random
import numpy as np

##################################################
# Code generates networks with artificially,
# randomly introduced and removed edges
##################################################


# 1. Main function
def generateNoiseNetworksFromBaseline(
    path_to_edgelist: str,
    folder_to_save_perturbed: str,
    noise_levels: list[float],
    noise_types: list[str] = [
        "added_edges",
        "removed_edges",
        "targeted_hub_addition",
        "targeted_hub_removal",
        "targeted_periphery_addition",
        "targeted_periphery_removal",
    ],
    num_repeats_per_noise_level: int = 10,
    network_name: str = "network",
):
    """
    Orchestrates the process of generating networks with randomly
    added and removed edges from a baseline network.

    :param path_to_edgelist: Path to the edgelist file of the baseline network
    :type path_to_edgelist: str
    :param folder_to_save_perturbed: Folder where perturbed networks will be saved
    :type folder_to_save_perturbed: str
    :param noise_levels: List of noise levels to apply
    :type noise_levels: list[float]
    :param num_repeats_per_noise_level: Number of repeats per noise level. Optional, defaults to 10
    :type num_repeats_per_noise_level: int
    :param network_name: Information on the main network, will be included in the file name of the pertrubed networks. Optional, defaults to "network"
    :type network_name: str
    """
    # Load the baseline network
    g, idx_to_node, node_to_idx = _loadBaseline(path_to_edgelist)
    for noise_level in noise_levels:
        num_edges_to_modify = _calcualteNumberOfEdgesToModify(
            g, noise_level, noise_types
        )
        for repeat in range(num_repeats_per_noise_level):
            for noise_type in noise_types:
                # Generate perturbed graph based on the action word
                if "add" in noise_type:
                    G_perturbed = _addEdgesToNetwork(g, num_edges_to_modify, noise_type)
                elif "remov" in noise_type:
                    G_perturbed = _removeEdgesFromNetwork(
                        g, num_edges_to_modify, noise_type
                    )
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")

                # Save the graph
                _saveEdgelists(
                    G_perturbed,
                    idx_to_node,
                    folder_to_save_perturbed,
                    noise_level,
                    repeat,
                    network_name,
                    modification_type=noise_type,
                )


############################################################
# 2. Loading the baseline network


def _loadBaseline(path_to_edgelist: str):
    """
    Loads the baseline network from an edgelist file and creates
    mappings between original node labels and integer indices.
    Returns a networkx graph with integer-labeled nodes.

    :param path_to_edgelist: Path to the edgelist file of the baseline network
    :type path_to_edgelist: str
    """
    # Required params: path (file path to edgelist)
    g_raw = nx.read_edgelist(path_to_edgelist)
    # Sorting the nodes to make the mapping always the same
    original_node_labels = sorted(list(g_raw.nodes()))
    # Create the forward and reverse mappings
    idx_to_node = {i: label for i, label in enumerate(original_node_labels)}
    node_to_idx = {label: i for i, label in enumerate(original_node_labels)}
    # Creating an integer-labeled graph
    g = nx.relabel_nodes(g_raw, node_to_idx)
    return g, idx_to_node, node_to_idx


############################################################
# 3. Introduce noise by adding and removing edges


def _addEdgesToNetwork(g: nx.Graph, num_edges_to_modify: int, noise_type: str):
    """
    Overall function to introduce noise by adding edges.

    :param g: Original network
    :type g: nx.Graph
    :param num_edges_to_modify: Number of edges to add and remove
    :type num_edges_to_modify: int
    """
    # Map the string to the correct function and target
    dispatch = {
        "added_edges": lambda: _randomEdgesToAdd(g, num_edges_to_modify),
        "targeted_hub_addition": lambda: _targetedAddition(
            g, num_edges_to_modify, target="hubs"
        ),
        "targeted_periphery_addition": lambda: _targetedAddition(
            g, num_edges_to_modify, target="periphery"
        ),
    }

    if noise_type not in dispatch:
        raise ValueError(f"Addition strategy for '{noise_type}' is not defined.")

    # Execute the mapped function
    edges_to_add = dispatch[noise_type]()

    # Add the new edges to the graph
    G_added = g.copy()
    G_added.add_edges_from(edges_to_add)
    return G_added


def _removeEdgesFromNetwork(g: nx.Graph, num_edges_to_modify: int, noise_type: str):
    """
    Overall function to introduce noise by removing edges.

    :param g: Original network
    :type g: nx.Graph
    :param num_edges_to_modify: Number of edges to add and remove
    :type num_edges_to_modify: int
    """
    dispatch = {
        "removed_edges": lambda: _randomEdgesToRemove(g, num_edges_to_modify),
        "targeted_hub_removal": lambda: _targetedRemoval(
            g, num_edges_to_modify, target="hubs"
        ),
        "targeted_periphery_removal": lambda: _targetedRemoval(
            g, num_edges_to_modify, target="periphery"
        ),
    }

    if noise_type not in dispatch:
        raise ValueError(f"Removal strategy for '{noise_type}' is not defined.")

    edges_to_remove = dispatch[noise_type]()

    G_removed = g.copy()
    G_removed.remove_edges_from(edges_to_remove)
    return G_removed


def _calcualteNumberOfEdgesToModify(g, noise_level, noise_types):
    """
    Calculates how many edges shall be modified based on the noise level.

    :param g: Original network
    :type g: nx.Graph
    :param noise_level: Noise level to apply
    :type noise_level: float
    :param noise_types: Types of noise to apply (e.g., ["added_edges", "removed_edges"])
    :type noise_types: list[str]
    """

    num_edges = len(g.edges)
    num_modify = int(num_edges * noise_level)  # Number of edges to modify

    # Check if results make sense
    if num_modify <= 0:
        raise ValueError("Number of edges to modify can not be negative or zero.")
    if "removed_edges" in noise_types and num_modify >= num_edges:
        raise ValueError("Can not remove all edges in the graph.")

    return num_modify


############################################################
# 5. Random noise introduction


def _randomEdgesToAdd(g, num_modify):
    """
    Calcualtes all non-exisitng edges in the graph and randomly samples edges to add.

    :param g: Original network
    :type g: nx.Graph
    :param num_modify: Number of edges to add
    :type num_modify: int
    """
    # 1. Get all possible edges that don't already exist.
    possible_edges_to_add = list(nx.non_edges(g))

    # Raise error if more edges than possible are to be added
    if num_modify > len(possible_edges_to_add):
        raise ValueError("More edges to be added than complete network.")

    # 2. Directly sample unique edges to add.
    edges_to_add = random.sample(possible_edges_to_add, num_modify)
    return edges_to_add


def _randomEdgesToRemove(g, num_modify):
    """
    Gets all edges in the original graph and randomly samples edges to remove from the graph.

    :param g: Original network
    :type g: nx.Graph
    :param num_modify: Number of edges to remove
    :type num_modify: int
    """
    edges = list(g.edges)
    # Directly sample unique edges to remove
    edges_to_remove = random.sample(edges, num_modify)
    return edges_to_remove


############################################################
# 5. Targeted noise introduction


def _targetedAddition(g: nx.Graph, num_modify: int, target: str):
    """
    Randomly samples edges to remove, heavily weighting edges connected to high-degree hubs.
    """
    edges = list(g.edges())
    degrees = dict(g.degree())

    # Calculate weights based on the product of degrees
    if target == "hubs":
        weights = np.array([degrees[u] * degrees[v] for u, v in edges], dtype=float)
    elif target == "periphery":
        # max(..., 1) prevents DivisionByZero if isolated nodes exist.
        weights = np.array(
            [1.0 / (max(degrees[u], 1) * max(degrees[v], 1)) for u, v in edges],
            dtype=float,
        )

    # Normalize to create a probability distribution
    probabilities = weights / weights.sum()

    # Sample unique indices based on probabilities
    chosen_indices = np.random.choice(
        len(edges), size=num_modify, replace=False, p=probabilities
    )
    edges_to_remove = [edges[i] for i in chosen_indices]

    return edges_to_remove


def _targetedRemoval(g: nx.Graph, num_modify: int, target: str):
    """
    Randomly samples non-edges to add, heavily weighting edges between high-degree nodes.
    """
    possible_edges_to_add = list(nx.non_edges(g))

    if num_modify > len(possible_edges_to_add):
        raise ValueError("More edges to be added than available non-edges.")

    degrees = dict(g.degree())

    # Calculate weights
    if target == "hubs":
        weights = np.array(
            [degrees[u] * degrees[v] for u, v in possible_edges_to_add], dtype=float
        )
    elif target == "periphery":
        # max(..., 1) prevents DivisionByZero if isolated nodes exist.
        weights = np.array(
            [
                1.0 / (max(degrees[u], 1) * max(degrees[v], 1))
                for u, v in possible_edges_to_add
            ],
            dtype=float,
        )

    probabilities = weights / weights.sum()

    chosen_indices = np.random.choice(
        len(possible_edges_to_add), size=num_modify, replace=False, p=probabilities
    )
    edges_to_add = [possible_edges_to_add[i] for i in chosen_indices]

    return edges_to_add


############################################################
# 4. Save the perturbed edgelists


def _saveEdgelists(
    g: nx.Graph,
    idx_to_node: dict[int, str],
    folder_to_save_perturbed: str,
    noise_level: float,
    repeat: int,
    network_name: str,
    modification_type: str,
):
    """
    Saves the perturbed edgelist to a specified folder with a specific naming convention.

    :param g: Perturbed network
    :type g: nx.Graph
    :param idx_to_node: Mapping from node indices to original node labels
    :type idx_to_node: dict[int, str]
    :param folder_to_save_perturbed: Folder path to save the perturbed edgelists
    :type folder_to_save_perturbed: str
    :param noise_level: Noise level applied to the network
    :type noise_level: float
    :param repeat: Repeat index for the noise application
    :type repeat: int
    :param network_name: Name of the original network
    :type network_name: str
    :param modification_type: Description of the modification type (e.g., "added_edges" or "removed_edges")
    :type modification_type: str
    """
    # Translate noise level to string suitable for file name
    if str(noise_level).find("."):
        noise_level = str(noise_level).replace(".", "p")

    # Translate node id back to the original labels
    g = nx.relabel_nodes(g, idx_to_node)

    # Create filename & save file
    file_name = f"{folder_to_save_perturbed}/{network_name}_{modification_type}_noise{noise_level}_repeat{repeat}.txt"
    nx.write_edgelist(G=g, path=file_name, delimiter="\t", data=False)
