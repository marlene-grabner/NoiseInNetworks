import re
import logging
from pathlib import Path
from tqdm import tqdm
import networkx as nx
from .seed_structural_metrics import _getPropertiesOfSeedSetsOnNetwork

# Create the logging channel for this file
logger = logging.getLogger(__name__)


def _getPropertiesOfSeedSetsOnPerturbedNetworks(
    perturbed_networks_directory: str,
    seed_groups: dict[str, list[str]],
    all_nodes_in_baseline: set,
):
    # 1. Create the list of perturbed networks to iterate through
    tasks = _networkMapFromDirectory(perturbed_networks_directory)

    perturbed_networks_seed_stats = []
    # 2. Iterate through each perturbed network and collect seed set properties
    for task in tqdm(tasks):
        # Generate a dict for the information about this network
        filename, p_type, p_level, repeat_id, network_name = task
        network_info = {
            "network_name": filename,
            "perturbation_type": p_type,
            "noise_level": p_level,
            "repeat_id": repeat_id,
        }
        # Full path to the perturbed network
        network_path = str(Path(perturbed_networks_directory) / filename)

        # Calculate seed set properties on this perturbed network
        perturbed_stats = _getPropertiesOfSeedSetsOnNetwork(
            network_path,
            seed_groups,
            network_info=network_info,
            all_nodes_in_baseline=all_nodes_in_baseline,
        )

        # Append to overall results
        perturbed_networks_seed_stats.extend(perturbed_stats)

    return perturbed_networks_seed_stats


def _networkMapFromDirectory(directory_path: str):
    """
    Scans a directory and automatically builds the structured map
    of perturbed networks based on filenames.

    Expected format: *_{type}_edges_noise{level}_repeat{N}.txt
    e.g. "autocore_ppi_added_edges_noise0p05_repeat0.txt"
    """
    tasks = []

    # Regex to capture: (added/removed), (0p05), and the full filename
    # Looks for: "added_edges" or "removed_edges" followed by "noise"
    pattern = re.compile(r"(.*)_(added|removed)_edges_noise(\d+p\d+)_repeat(\d+)")

    # Get all .txt files in the directory
    path_obj = Path(directory_path)
    files = sorted([f.name for f in path_obj.glob("*.txt")])

    for filename in files:
        match = pattern.search(filename)
        if match:
            network_name = match.group(1)
            p_type = match.group(2)  # 'added' or 'removed'
            p_level_str = match.group(3)  # '0p05'
            p_level = float(p_level_str.replace("p", "."))
            repeat = match.group(4)
            repeat_id = f"rep{repeat}"
            tasks.append((filename, p_type, p_level, repeat_id, network_name))
    return tasks
