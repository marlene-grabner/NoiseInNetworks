import logging
import networkx as nx
from .ModuleDetectionAlgorithms.random_walk_with_restart_row_normalization import (
    randomWalkWithRestartRowNormalization,
)
from .ModuleDetectionAlgorithms.random_walk_with_restart_symmetric_normalization import (
    randomWalkWithRestartSymmetricNormalization,
)
from .ModuleDetectionAlgorithms.domino import domino
from .ModuleDetectionAlgorithms.first_neighbors import firstNeighbors
from .module_result import ModuleResult

# Create the logging channel for this file
logger = logging.getLogger(__name__)


def startAlgorithm(algorithm: str, G: nx.Graph, seed_nodes: list[str]) -> ModuleResult:
    if algorithm == "1stNeighbors":
        results = firstNeighbors(G=G, seed_nodes=seed_nodes)

    elif algorithm == "DIAMOnD":
        logger.warning("DIAMOnD is not implemented yet.")
        results = ModuleResult(nodes_set=set(), algorithm_type="ranked")

    elif algorithm == "DOMINO":
        results = domino(G=G, seeds=seed_nodes)

    elif algorithm == "ROBUST":
        print("Not implemented yet")
        results = ModuleResult(nodes_set=set(), algorithm_type="ranked")

    elif algorithm == "ROBUST(bias_aware)":
        print("Not implemented yet")
        results = ModuleResult(nodes_set=set(), algorithm_type="ranked")

    elif algorithm == "RandomWalkWithRestartRowNormalization":
        results = randomWalkWithRestartRowNormalization(G=G, seed_nodes=seed_nodes)

    elif algorithm == "RandomWalkWithRestartSymmetricNormalization":
        results = randomWalkWithRestartSymmetricNormalization(
            G=G, seed_nodes=seed_nodes
        )

    else:
        raise ValueError(f"Unknown algorithm specified: {algorithm}")

    return results
