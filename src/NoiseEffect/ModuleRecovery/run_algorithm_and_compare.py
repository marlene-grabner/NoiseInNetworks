import networkx as nx
import logging
from .utils import _saveRawToDisk
from .compare_results import compareResults
from .start_algorithm import startAlgorithm


# Create the logging channel for this file
logger = logging.getLogger(__name__)


# Run algorithm and compare to baseline
def _runAlgorithmAndCompareToBaseline(
    perturbed_G: nx.Graph,
    algorithm_name: str,
    noise_type: str,
    noise_level: str,
    repeat: str,
    filename: str,
    seed_nodes: list[str],
    seed_id: str,
    baseline_cache: dict,
    save_raw_modules: bool = False,
):
    ###### NEEDS CHECK FOR WHICH SEEDS ARE STILL IN THE GRAPH ######
    logger.info(f"Running {algorithm_name} on {filename} with seed {seed_id}")

    # 1. Recover Modules
    recovered_module = startAlgorithm(
        algorithm=algorithm_name, G=perturbed_G, seed_nodes=seed_nodes
    )
    print(recovered_module)

    # 2. Compare to Baseline immediately
    logger.info(
        f"Running comparisons to baseline for {algorithm_name} on {filename} with seed {seed_id}"
    )
    baseline_mod = baseline_cache[(algorithm_name, seed_id)]
    print(baseline_mod)

    # Calculate metrics (Replace with your actual comparison functions)
    # jaccard = calculate_jaccard(baseline_mod, recovered_mod)

    metrics = compareResults(baseline_mod, recovered_module)

    row = [
        algorithm_name,
        noise_type,
        noise_level,
        repeat,
        filename,
        seed_id,
        len(seed_nodes),
        metrics["jaccard"],
        metrics["overlap_size"],
        metrics["precision"],
        metrics["recall"],
    ]

    # Optional: Save raw rank list to disk if needed (separate files)
    if save_raw_modules:
        _saveRawToDisk(recovered_module, filename, algorithm_name, seed_id)

    return row
