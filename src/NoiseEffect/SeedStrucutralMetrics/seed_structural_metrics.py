import logging
import numpy as np
import networkx as nx
from pathlib import Path

# Create the logging channel for this file
logger = logging.getLogger(__name__)


def _getPropertiesOfSeedSetsOnNetwork(
    network_path: str,
    seed_groups: dict[str, list[str]],
    network_info: dict,
    all_nodes_in_baseline: set,
):
    logger.info(f"Analyzing seed sturcutral metrics on network: {network_path}")

    # Load network
    G = nx.read_edgelist(network_path)
    # Add any isolated nodes that might be missing from the edgelist
    G.add_nodes_from(all_nodes_in_baseline)

    # Identify all connected components and create node-to-component mapping
    components = list(nx.connected_components(G))
    component_sizes = {i: len(comp) for i, comp in enumerate(components)}

    # Create mapping from node to component index
    node_to_component = {}
    for i, comp in enumerate(components):
        for node in comp:
            node_to_component[node] = i

    # Identify largest component
    lcc_idx = max(component_sizes, key=component_sizes.get)
    lcc = components[lcc_idx]
    lcc_size = len(lcc)

    logger.info(f"Network has {len(components)} components. LCC size: {lcc_size}")

    # Analyze each seed group
    results = []

    for seed_id, seed_nodes in seed_groups.items():
        # Track component sizes for seeds in this group
        seed_component_sizes = []
        seeds_in_lcc = 0
        seeds_not_in_network = 0
        seeds_in_network = 0
        zero_degree_seeds = 0

        for node in seed_nodes:
            if node not in G:
                seeds_not_in_network += 1
            else:
                seeds_in_network += 1
                comp_idx = node_to_component[node]
                comp_size = component_sizes[comp_idx]
                if comp_size == 1:
                    zero_degree_seeds += 1
                seed_component_sizes.append(comp_size)

                if node in lcc:
                    seeds_in_lcc += 1

        # Calculate statistics
        total_seeds = len(seed_nodes)

        if seed_component_sizes:
            mean_component_size = np.mean(seed_component_sizes)
            std_component_size = np.std(seed_component_sizes)
            median_component_size = np.median(seed_component_sizes)
            min_component_size = np.min(seed_component_sizes)
            max_component_size = np.max(seed_component_sizes)
        else:
            mean_component_size = 0
            std_component_size = 0
            median_component_size = 0
            min_component_size = 0
            max_component_size = 0

        # Fraction metrics
        frac_in_lcc = seeds_in_lcc / total_seeds if total_seeds > 0 else 0
        frac_in_network = seeds_in_network / total_seeds if total_seeds > 0 else 0
        frac_not_in_network = (
            seeds_not_in_network / total_seeds if total_seeds > 0 else 0
        )

        result = {
            "seed_group": seed_id,
            "network_name": network_info.get("network_name", None),
            "perturbation_type": network_info.get("perturbation_type", None),
            "noise_level": network_info.get("noise_level", None),
            "repeat_id": network_info.get("repeat_id", None),
            "total_seeds": total_seeds,
            "seeds_in_network": seeds_in_network,
            "seeds_not_in_network": seeds_not_in_network,
            "zero_degree_seeds": zero_degree_seeds,
            "seeds_in_lcc": seeds_in_lcc,
            "frac_in_network": frac_in_network,
            "frac_not_in_network": frac_not_in_network,
            "frac_in_lcc": frac_in_lcc,
            "mean_component_size": mean_component_size,
            "std_component_size": std_component_size,
            "median_component_size": median_component_size,
            "min_component_size": min_component_size,
            "max_component_size": max_component_size,
            "lcc_size": lcc_size,
            "total_components": len(components),
        }

        results.append(result)

    return results
