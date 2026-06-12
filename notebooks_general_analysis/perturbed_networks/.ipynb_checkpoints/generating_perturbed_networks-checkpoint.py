import NoiseEffect as na
from pathlib import Path
import sys

baseline_key = sys.argv[1]

baseline_files = {"test": f"./data/baseline_networks/test_network/karate_club.tsv"}

noise_levels_removed = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 0.95]
noise_levels_added = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

output_folders_random = {"test": "./data/perturbed_networks/test_network/perturbed_random_target"}
output_folders_hub_targeted = {
    "test": "./data/perturbed_networks/test_network/perturbed_hub_target"
}
output_folders_periphery_targeted = {
    "test": "./data/perturbed_networks/test_network/perturbed_periphery_target"
}

""""
input_networks = {
    "chloe_ppi": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/chloe_ppi/chloe_ppi_lcc_2026_02_23.tsv",
    "config_model_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/configuration_models/configuration_model_0.tsv",
    "config_model_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/configuration_models/configuration_model_1.tsv",
    "config_model_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/configuration_models/configuration_model_2.tsv",
    "erdos_renyi_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/erdos_renyi_models/erdos_renyi_model_0.tsv",
    "erdos_renyi_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/erdos_renyi_models/erdos_renyi_model_1.tsv",
    "erdos_renyi_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/erdos_renyi_models/erdos_renyi_model_2.tsv",
    "sbm_degree_not_preserved_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/classic_sbm_models/sbm_network_degree_not_preserved_307_blocks_0.tsv",
    "sbm_degree_not_preserved_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/classic_sbm_models/sbm_network_degree_not_preserved_307_blocks_1.tsv",
    "sbm_degree_not_preserved_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/classic_sbm_models/sbm_network_degree_not_preserved_307_blocks_2.tsv",
    "sbm_degree_preserved_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/degree_pres_sbm_models/sbm_network_307_blocks_0.tsv",
    "sbm_degree_preserved_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/degree_pres_sbm_models/sbm_network_307_blocks_1.tsv",
    "sbm_degree_preserved_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/degree_pres_sbm_models/sbm_network_307_blocks_2.tsv",
    "hgg_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/hgg/hgg_model_0.tsv",
    "hgg_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/hgg/hgg_model_1.tsv",
    "hgg_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/hgg/hgg_model_2.tsv",
    "caida_as": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/real_world/caida_as.tsv",
    "western_us_power_grid": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/originals/real_world/western_us_power_grid.tsv",
}

output_folders_random = {
    # "chloe_ppi": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/chloe_ppi",
    # "config_model_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/configuration_models/configuration_model_0",
    "config_model_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/configuration_models/configuration_model_1",
    "config_model_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/configuration_models/configuration_model_2",
    "erdos_renyi_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/erdos_renyi_models/erdos_renyi_model_0",
    "erdos_renyi_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/erdos_renyi_models/erdos_renyi_model_1",
    "erdos_renyi_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/erdos_renyi_models/erdos_renyi_model_2",
    "sbm_degree_not_preserved_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/classic_sbm_models/sbm_degree_not_preserved_0",
    "sbm_degree_not_preserved_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/classic_sbm_models/sbm_degree_not_preserved_1",
    "sbm_degree_not_preserved_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/classic_sbm_models/sbm_degree_not_preserved_2",
    "sbm_degree_preserved_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/degree_pres_sbm_models/sbm_degree_preserved_0",
    "sbm_degree_preserved_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/degree_pres_sbm_models/sbm_degree_preserved_1",
    "sbm_degree_preserved_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/degree_pres_sbm_models/sbm_degree_preserved_2",
    "hgg_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/hgg/hgg_model_0",
    "hgg_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/hgg/hgg_model_1",
    "hgg_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/hgg/hgg_model_2",
    "caida_as": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/real_world/caida_as",
    "western_us_power_grid": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_random_target/real_world/western_us_power_grid",
}

output_folders_hub_targeted = {
    # "chloe_ppi": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/chloe_ppi",
    # "config_model_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/configuration_models/configuration_model_0",
    "config_model_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/configuration_models/configuration_model_1",
    "config_model_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/configuration_models/configuration_model_2",
    "erdos_renyi_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/erdos_renyi_models/erdos_renyi_model_0",
    "erdos_renyi_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/erdos_renyi_models/erdos_renyi_model_1",
    "erdos_renyi_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/erdos_renyi_models/erdos_renyi_model_2",
    "sbm_degree_not_preserved_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/classic_sbm_models/sbm_degree_not_preserved_0",
    "sbm_degree_not_preserved_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/classic_sbm_models/sbm_degree_not_preserved_1",
    "sbm_degree_not_preserved_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/classic_sbm_models/sbm_degree_not_preserved_2",
    "sbm_degree_preserved_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/degree_pres_sbm_models/sbm_degree_preserved_0",
    "sbm_degree_preserved_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/degree_pres_sbm_models/sbm_degree_preserved_1",
    "sbm_degree_preserved_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/degree_pres_sbm_models/sbm_degree_preserved_2",
    "hgg_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/hgg/hgg_model_0",
    "hgg_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/hgg/hgg_model_1",
    "hgg_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/hgg/hgg_model_2",
    "caida_as": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/real_world/caida_as",
    "western_us_power_grid": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_hub_target/real_world/western_us_power_grid",
}


output_folders_periphery_targeted = {
    # "chloe_ppi": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/chloe_ppi",
    # "config_model_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/configuration_models/configuration_model_0",
    "config_model_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/configuration_models/configuration_model_1",
    "config_model_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/configuration_models/configuration_model_2",
    "erdos_renyi_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/erdos_renyi_models/erdos_renyi_model_0",
    "erdos_renyi_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/erdos_renyi_models/erdos_renyi_model_1",
    "erdos_renyi_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/erdos_renyi_models/erdos_renyi_model_2",
    "sbm_degree_not_preserved_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/classic_sbm_models/sbm_degree_not_preserved_0",
    "sbm_degree_not_preserved_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/classic_sbm_models/sbm_degree_not_preserved_1",
    "sbm_degree_not_preserved_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/classic_sbm_models/sbm_degree_not_preserved_2",
    "sbm_degree_preserved_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/degree_pres_sbm_models/sbm_degree_preserved_0",
    "sbm_degree_preserved_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/degree_pres_sbm_models/sbm_degree_preserved_1",
    "sbm_degree_preserved_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/degree_pres_sbm_models/sbm_degree_preserved_2",
    "hgg_0": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/hgg/hgg_model_0",
    "hgg_1": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/hgg/hgg_model_1",
    "hgg_2": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/hgg/hgg_model_2",
    "caida_as": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/real_world/caida_as",
    "western_us_power_grid": "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/paper_notebooks/comparison_network_creation/networks_2026_05_06/perturbed_periphery_target/real_world/western_us_power_grid",
}
"""

# Perturbing

PERTURBATIONS = {
    "random": {
        "folder": output_folders_random[baseline_key],
        "noise_types": ["added_edges", "removed_edges"]
    },
    "hub": {
        "folder": output_folders_hub_targeted[baseline_key],
        "noise_types": ["targeted_hub_addition", "targeted_hub_removal"]
    },
    "periphery": {
        "folder": output_folders_periphery_targeted[baseline_key],
        "noise_types": ["targeted_periphery_addition", "targeted_periphery_removal"]
    }
}

    
for perturbation_type in PERTURBATIONS.keys():
    # Create the folder if it does not exist
    if not Path(PERTURBATIONS[perturbation_type]["folder"]).exists():
        Path(PERTURBATIONS[perturbation_type]["folder"]).mkdir(parents=True, exist_ok=True)
    # Generate the perturbed networks
    # Addition
    na.generateNoiseNetworksFromBaseline(
        path_to_edgelist=baseline_files[baseline_key],
        folder_to_save_perturbed=PERTURBATIONS[perturbation_type]["folder"],
        noise_levels=noise_levels_added,
        noise_types=[PERTURBATIONS[perturbation_type]["noise_types"][0]],
        num_repeats_per_noise_level=100,
        network_name=baseline_key,
    )

    # Removal
    na.generateNoiseNetworksFromBaseline(
        path_to_edgelist=baseline_files[baseline_key],
        folder_to_save_perturbed=PERTURBATIONS[perturbation_type]["folder"],
        noise_levels=noise_levels_removed,
        noise_types=[PERTURBATIONS[perturbation_type]["noise_types"][1]],
        num_repeats_per_noise_level=100,
        network_name=baseline_key,
    )
