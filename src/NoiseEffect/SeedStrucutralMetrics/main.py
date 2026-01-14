import re
import logging
import pandas as pd
import networkx as nx
from pathlib import Path
from .seed_structural_metrics import _getPropertiesOfSeedSetsOnNetwork
from .perturbed_networks_workflow import _getPropertiesOfSeedSetsOnPerturbedNetworks

# Create the logging channel for this file
logger = logging.getLogger(__name__)


def computeSeedStrucutralMetrics(
    baseline_network_path: str,
    perturbed_networks_directory: str,
    seed_groups: dict[str, list[str]],
    seed_statistics_csv_path: str,
):
    # 1. Gather all nodes from baseline for isolated node addition
    baseline_G = nx.read_edgelist(baseline_network_path)
    all_nodes_in_baseline = set(baseline_G.nodes())

    # 2. Get the properties of seed sets on the baseline network
    baseline_info = {
        "network_name": Path(baseline_network_path).name,
        "perturbation_type": "None",
        "noise_level": 0,
        "repeat_id": "rep0",
    }
    baseline_stats = _getPropertiesOfSeedSetsOnNetwork(
        baseline_network_path,
        seed_groups,
        network_info=baseline_info,
        all_nodes_in_baseline=all_nodes_in_baseline,
    )

    # 3. Analyze perturbed networks
    perturbed_stats = _getPropertiesOfSeedSetsOnPerturbedNetworks(
        perturbed_networks_directory=perturbed_networks_directory,
        seed_groups=seed_groups,
        all_nodes_in_baseline=all_nodes_in_baseline,
    )

    # Save baseline and perturbed statistics together
    all_stats = baseline_stats + perturbed_stats
    df = pd.DataFrame(all_stats)
    df.to_csv(seed_statistics_csv_path, index=False)
    logger.info(f"All seed statistics saved to: {seed_statistics_csv_path}")
