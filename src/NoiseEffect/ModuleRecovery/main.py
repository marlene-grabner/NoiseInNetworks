import networkx as nx
import os
import csv
import logging
from tqdm import tqdm
import json
import gzip
from pathlib import Path
from .run_algorithm_and_compare import _runAlgorithmAndSaveResultsToFile
from .utils import _setupOutputCSV, _networkMapFromDirectory
from .seeds_preprocessing import filterForSeedsInNetwork
from .start_algorithm import startAlgorithm

# Create the logging channel for this file
logger = logging.getLogger(__name__)


#############################################
# Master Function
#############################################


def benchmarkModuleDetectionAlgorithms(
    baseline_network_path: str,
    perturbed_networks_directory: str,
    algorithms_config: dict[str, bool],
    seed_groups: dict[str, list[str]],
    output_file_location: str,
    experiment_identifier: str,
    domino_env_path: str = None,
):
    # 1.
    # Initialize output CSV
    # This is where all results will be stored
    # _setupOutputCSV(output_csv_path)

    # 2.
    # Check seed sets for presence in baseline network
    baseline_G = nx.read_edgelist(baseline_network_path)
    cleaned_seed_groups = filterForSeedsInNetwork(
        G=baseline_G, seed_groups=seed_groups, network_name="baseline"
    )

    # 2.
    # Loading the baseline network to which the perturbed networks will be compared
    # Get the results for the baseline once
    print(f"--- Loading Baseline: {baseline_network_path} ---")
    baseline_cache = _computeBaselineModules(
        baseline_network_path=baseline_network_path,
        baseline_G=baseline_G,
        algorithms_config=algorithms_config,
        seed_groups=cleaned_seed_groups,
        output_file_location=output_file_location,
        experiment_identifier=experiment_identifier,
        domino_env_path=domino_env_path,
    )
    print("--- Baseline processing complete. Loading perturbed networks. ---")

    # 3.
    # Generate a list of perturbed networks to process
    # Format: List of tuples (filename, noise_type, noise_level, repeat_id, network_name)
    tasks = _networkMapFromDirectory(perturbed_networks_directory)

    print(
        f"--- {len(tasks)} Perturbed Networks Found. Starting module detection algorithms ---"
    )
    # 4.
    # Calculate the modules on all perturbed networks and compare each to the baseline
    # Results of comparisons are appended to the output CSV
    # Optionally raw outputs of the algorithms can be saved to disk
    _computeModulesOnPerturbedNetworks(
        perturbed_networks_directory=perturbed_networks_directory,
        tasks=tasks,
        algorithms_config=algorithms_config,
        seed_groups=cleaned_seed_groups,
        output_file_location=output_file_location,
        experiment_identifier=experiment_identifier,
        domino_env_path=domino_env_path,
    )


#############################################
# Compute the baseline on the unperturbed network
#############################################


def _computeBaselineModules(
    baseline_network_path: str,
    baseline_G: nx.Graph,
    algorithms_config: dict[str, bool],
    seed_groups: dict[str, list[str]],
    output_file_location: str,
    experiment_identifier: str,
    domino_env_path: str = None,
):
    logger.info("Starting algorithms on baseline network...")
    baseline_cache = {}

    for algo in tqdm([algo for algo, active in algorithms_config.items() if active]):
        algorithm_cache = []
        for seed_id, seed_nodes in seed_groups.items():
            logger.info(
                f"Running {algo} on baseline network ({baseline_network_path}) with seed {seed_id}"
            )

            # Run algorithm on network
            results = startAlgorithm(
                algorithm=algo,
                G=baseline_G,
                seed_nodes=seed_nodes,
                domino_env_path=domino_env_path,
            )

            # Get the returned modules
            if results.algorithm_type == "set":
                module_results = results.nodes_set
            else:
                # Algorihm type is 'ranked'
                module_results = results.nodes_ranked

            # Metadata about this run
            if algo in [
                "RandomWalkWithRestartRowNormalization",
                "RandomWalkWithRestartSymmetricNormalization",
            ]:
                run_metadata = results.metadata
            else:
                run_metadata = {}

            metrics_dict = {
                "metadata_network": {
                    "noise_type": "baseline",
                    "noise_level": 0,
                    "repeat": "rep0",
                },
                "metadata_seed": {"seed_id": seed_id, "seeds_in_network": seed_nodes},
                "metadata_run": run_metadata,
                "module_results": module_results,
            }

            algorithm_cache.append(metrics_dict)

        _saveBatchToDisk(
            results_batch=algorithm_cache,
            outputfile_location=output_file_location,
            algorithm_name=algo,
            experiment_identifier=experiment_identifier,
        )

    logger.info("Algorithms on baseline network complete.")
    return baseline_cache


#############################################
# Compute the outcome of the algorithms on the perturbed networks
# Then compare the results to the baseline and write to CSV
#############################################
def _computeModulesOnPerturbedNetworks(
    perturbed_networks_directory: str,
    tasks: list[tuple[str, str, str, str, str]],
    algorithms_config: dict[str, bool],
    seed_groups: dict[str, list[str]],
    output_file_location: str,
    experiment_identifier: str,
    domino_env_path: str = None,
):
    # Iterate through files (Outer Loop)
    for filename, noise_type, noise_level, repeat, network_name in tqdm(tasks):
        logger.info(f"Loading perturbed network: {filename}")

        # 1. Load Perturbed Network
        perturbed_G = _loadPerturbedNetworkFromFile(
            perturbed_networks_directory, filename
        )
        # If the file is missing, skip
        if perturbed_G is None:
            print(f"Skipping missing perturbed network file: {filename}")
            logger.warning(
                f"Perturbed network file not found: {filename}. File skipped."
            )
            continue

        # 2. Filter seeds to only those present in the network
        # Only necessary when edges were removed (nodes may have been isolated/removed)
        if noise_type == "removed":
            seed_groups_in_network = filterForSeedsInNetwork(
                perturbed_G, seed_groups, filename
            )
        else:
            seed_groups_in_network = seed_groups

        # 3. Run all algorithms on this specific graph
        for algo in [algo for algo, active in algorithms_config.items() if active]:
            # Also start the algorithm on each individual seed group
            batch_results = []
            for seed_id, seed_nodes in seed_groups_in_network.items():
                # Get the modules on this perturbed network and then compare to baseline
                row_of_results = _runAlgorithmAndSaveResultsToFile(
                    perturbed_G=perturbed_G,
                    algorithm_name=algo,
                    noise_type=noise_type,
                    noise_level=noise_level,
                    repeat=repeat,
                    filename=filename,
                    seed_nodes=seed_nodes,
                    seed_id=seed_id,
                    domino_env_path=domino_env_path,
                )
                batch_results.append(row_of_results)

            _saveBatchToDisk(
                batch_results, output_file_location, algo, experiment_identifier
            )

        # Explicitly delete graph to ensure memory is freed
        del perturbed_G
        batch_results.clear()
    print("--- Benchmarking Complete ---")


# Load a perturbed network
def _loadPerturbedNetworkFromFile(perturbed_networks_directory: str, filename: str):
    full_path = os.path.join(perturbed_networks_directory, filename)
    try:
        perturbed_G = nx.read_edgelist(full_path, delimiter="\t")
        return perturbed_G
    except FileNotFoundError:
        print(f"Skipping missing file: {full_path}")
        logger.warning(f"Perturbed network file not found: {full_path}. File skipped.")
        return None


def _saveBatchToDisk(
    results_batch: list[dict],
    outputfile_location: str,
    algorithm_name: str,
    experiment_identifier: str,
):
    # Folder for output files
    # Create if it does not yet exist
    folder_path = Path(outputfile_location)
    folder_path.mkdir(parents=True, exist_ok=True)
    filename = f"results_{experiment_identifier}_{algorithm_name}.jsonl.gz"
    filepath = folder_path / filename

    # json.dumps(default=list) makes sure that objects which can't be serialized otherwise
    # are handeled. In this case this is for the 'set' object, which are turned into lists.
    with gzip.open(filepath, "ab") as f:
        for one_result in results_batch:
            line = json.dumps(one_result, default=list).encode("utf-8") + b"\n"
            f.write(line)
