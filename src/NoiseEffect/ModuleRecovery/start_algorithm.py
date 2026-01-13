import logging
import networkx as nx
from .ModuleDetectionAlgorithms.random_walk_with_restart import randomWalkWithRestart

# Create the logging channel for this file
logger = logging.getLogger(__name__)


def startAlgorithm(algorithm: str, G: nx.Graph, seed_nodes: list[str]):
    print(
        "TODO: run_algorithm; what kind of results should be returned - are they all the same?"
    )
    if algorithm == "1stNeighbors":
        print("Not implemented yet")
        results = []
    elif algorithm == "DIAMOnD":
        logger.warning("DIAMOnD is not implemented yet.")
        print("Not implemented yet")
        results = []
    elif algorithm == "DOMINO":
        print("Not implemented yet")
        results = []
    elif algorithm == "ROBUST":
        print("Not implemented yet")
        results = []
    elif algorithm == "ROBUST(bias_aware)":
        print("Not implemented yet")
        results = []
    elif algorithm == "RandomWalkWithRestart":
        results = randomWalkWithRestart(G=G, seed_nodes=seed_nodes)
    else:
        raise ValueError(f"Unknown algorithm specified: {algorithm}")

    return results
