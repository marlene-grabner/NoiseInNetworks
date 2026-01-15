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
    # Validate seeds
    valid_seeds = [s for s in seed_nodes if s in G.nodes()]

    if not valid_seeds:
        raise ValueError("No valid seed nodes found in graph!")

    # Collect all neighbors
    neighbors = set()  # Do not include seeds themselves
    for seed in valid_seeds:
        neighbors.update(G.neighbors(seed))

    # RETURN ModuleResult object
    return ModuleResult(
        nodes_set=neighbors,
        algorithm_type="set",
        metadata={"n_valid_seeds": len(valid_seeds), "module_size": len(neighbors)},
    )
