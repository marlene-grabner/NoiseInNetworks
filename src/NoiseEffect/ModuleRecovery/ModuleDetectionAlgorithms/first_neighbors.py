import networkx as nx
from ..module_result import ModuleResult


def firstNeighbors(G: nx.Graph, seed_nodes: list[str]) -> ModuleResult:
    """
    Get all first neighbors of seed nodes.

    Returns
    -------
    ModuleResult
        Standardized result object containing set of nodes.
    """

    # Collect all neighbors
    neighbors = set()  # Do not include seeds themselves
    for seed in seed_nodes:
        neighbors.update(G.neighbors(seed))

    # RETURN ModuleResult object
    return ModuleResult(
        nodes_set=neighbors,
        algorithm_type="set",
        metadata={
            "algorithm": "FirstNeighbors",
            "n_valid_seeds": len(seed_nodes),
            "module_size": len(neighbors),
        },
    )
