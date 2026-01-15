import logging
import networkx as nx

# Create the logging channel for this file
logger = logging.getLogger(__name__)


def filterForSeedsInNetwork(
    G: nx.Graph(), seed_groups: dict[str, list[str]], network_name: str
):
    cleaned_seed_groups = {}
    all_network_nodes = set(G.nodes())

    # Iterate through all seed groups, remove nodes not on baseline network
    # duplicates and remove seed groups that become empty
    for seed_id, seed_nodes in seed_groups.items():
        valid_seed_nodes = list(set(seed_nodes).intersection(all_network_nodes))

        # Skip seed groups that become empty
        if len(valid_seed_nodes) == 0:
            logger.warning(
                f"Warning: Seed group {seed_id} has no valid seed nodes on {network_name} and will be skipped."
            )
            print(
                f"Warning: Seed group {seed_id} has no valid seed nodes on {network_name} and will be skipped."
            )
            continue

        # Alert if some seeds are not present in the baseline network
        if len(valid_seed_nodes) < len(seed_nodes):
            difference_sets = set(seed_nodes).difference(all_network_nodes)
            if len(difference_sets) == 0:
                print(f'Warning: "{seed_id}" seed set contains duplicates.')
                logger.warning(f'"{seed_id}" seed set contains duplicates.')
            else:
                print(
                    f'Warning: Seeds "{difference_sets}" not found in {network_name}.'
                )
                logger.warning(
                    f'Seeds "{difference_sets}" not found in {network_name}.'
                )

        # Add cleaned seed group to the output dictionary
        cleaned_seed_groups[seed_id] = valid_seed_nodes
    return cleaned_seed_groups
