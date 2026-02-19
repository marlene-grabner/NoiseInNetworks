import networkx as nx
import logging
import numpy as np
import json
import gzip
from pathlib import Path
from .utils import _saveRawToDisk
from .start_algorithm import startAlgorithm
from .module_result import ModuleResult


# Create the logging channel for this file
logger = logging.getLogger(__name__)


# Run algorithm and compare to baseline
def _runAlgorithmAndSaveResultsToFile(
    perturbed_G: nx.Graph,
    algorithm_name: str,
    noise_type: str,
    noise_level: str,
    repeat: str,
    filename: str,
    seed_nodes: list[str],
    seed_id: str,
    domino_env_path: str = None,
):
    if len(seed_nodes) == 0:
        results = _handleEmptySeeds(
            algorithm_name=algorithm_name,
            noise_type=noise_type,
            noise_level=noise_level,
            repeat=repeat,
            seed_id=seed_id,
            seed_nodes=seed_nodes,
        )
        return results

    logger.info(f"Running {algorithm_name} on {filename} with seed {seed_id}")

    # 1. Recover Modules
    perturbedResult_obj = startAlgorithm(
        algorithm=algorithm_name,
        G=perturbed_G,
        seed_nodes=seed_nodes,
        domino_env_path=domino_env_path,
    )

    # Log convergence info if available
    if algorithm_name in ["RandomWalkWithRestart"]:
        converged = perturbedResult_obj.metadata.get("converged")
        if not converged:
            logger.warning(
                f"Warning: {algorithm_name} did not converge for {filename} with seed {seed_id}."
            )
            print(
                f"Warning: {algorithm_name} did not converge for {filename} with seed {seed_id}."
            )

    # 2. Save the results to disk
    restults_dict = _dictForSaving(
        results_obj=perturbedResult_obj,
        algorithm_name=algorithm_name,
        noise_type=noise_type,
        noise_level=noise_level,
        repeat=repeat,
        seed_id=seed_id,
        seed_nodes=seed_nodes,
    )

    return restults_dict


def _dictForSaving(
    results_obj: ModuleResult,
    algorithm_name: str,
    noise_type: str,
    noise_level: str,
    repeat: str,
    seed_id: str,
    seed_nodes: list[str],
):
    # Arrange the results format for saving to disk

    # Get the returned modules
    if results_obj.algorithm_type == "set":
        module_results = results_obj.nodes_set
    elif results_obj.algorithm_type == "ranked":
        module_results = results_obj.nodes_ranked
    elif results_obj.algorithm_type == "diamond":
        module_results = results_obj.nodes_diamond

    # Metadata about this run
    run_metadata = results_obj.metadata

    results = {
        "metadata_network": {
            "noise_type": noise_type,
            "noise_level": noise_level,
            "repeat": repeat,
        },
        "metadata_seed": {"seed_id": seed_id, "seeds_in_network": seed_nodes},
        "metadata_run": run_metadata,
        "module_results": module_results,
    }

    return results


def _handleEmptySeeds(
    algorithm_name: str,
    noise_type: str,
    noise_level: str,
    repeat: str,
    seed_id: str,
    seed_nodes: list[str],
):
    if algorithm_name in [
        "RandomWalkWithRestartRowNormalization",
        "RandomWalkWithRestartSymmetricNormalization",
    ]:
        # Return empty ranked list
        perturbedResult_obj = ModuleResult(
            nodes_ranked={},
            algorithm_type="ranked",
            metadata={
                "converged": "No Seeds",
                "n_valid_seeds": 0,
                "module_size": 0,
            },
        )
    else:
        # Return empty set
        perturbedResult_obj = ModuleResult(
            nodes_set=set(),
            algorithm_type="set",
            metadata={"n_valid_seeds": 0, "module_size": 0},
        )
    results = _dictForSaving(
        results_obj=perturbedResult_obj,
        algorithm_name=algorithm_name,
        noise_type=noise_type,
        noise_level=noise_level,
        repeat=repeat,
        seed_id=seed_id,
        seed_nodes=seed_nodes,
    )
    return results
