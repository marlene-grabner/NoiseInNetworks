# %%
import networkx as nx
import os
import csv
from NoiseEffect.ModuleRecovery.utils import _setupOutputCSV


# 1. Specify all networks modules shall be defined upon (both baseline and perturbed)
def benchmarkModuleDetectionAlgorithms(
    baseline_network_path: str,
    perturbed_networks_directory: str,
    perturbed_files_map: dict[str, dict[float, list[str]]],
    algorithms_config: dict[str, bool],
    seed_groups: dict[str, list[str]],
    output_csv_path: str,
    save_raw_modules: bool = False,
):
    # Initialize output CSV
    _setupOutputCSV(output_csv_path)
    # Load baseline network to compare to
    print(f"--- Loading Baseline: {baseline_network_path} ---")
    baseline_G = nx.read_edgelist(path=baseline_network_path, delimiter="\t")
    baseline_cache = _computeBaselineModules(baseline_G, algorithms_config, seed_groups)
    print(baseline_cache)
    print("--- Baseline processing complete. Starting Perturbations ---")
    _computeModulesonPerturbedNetworks(
        perturbed_networks_directory,
        perturbed_files_map,
        algorithms_config,
        seed_groups,
        baseline_cache,
        save_raw_modules,
    )

    # A. Get the results for baseline once
    # B. for algorithm in algorithms_to_test:
    #     for noise_type in perturbed_networks_instructions:
    #         for noise_level in perturbed_networks_instructions[noise_type]:
    #             for repeat_file in perturbed_networks_instructions[noise_type][noise_level]:
    #                 # load perturbed network
    #                 for starting_points in algorithm_starting_points[algorithm]:
    #                       # recover modules
    #                       # compare to baseline


def _computeBaselineModules(
    baseline_G: nx.Graph,
    algorithms_config: dict[str, bool],
    seed_groups: dict[str, list[str]],
):
    baseline_cache = {}
    for algo in [algo for algo, active in algorithms_config.items() if active]:
        for seed_id, seed_nodes in seed_groups.items():
            # modules = run_algorithm(baseline_G, algo_name, seed_nodes)

            # MOCK RESULTS
            modules = ["geneA", "geneB"]

            baseline_cache[(algo, seed_id)] = modules
    return baseline_cache


def _computeModulesonPerturbedNetworks(
    perturbed_networks_directory: str,
    perturbed_files_map: dict[str, dict[float, list[str]]],
    algorithms_config: dict[str, bool],
    seed_groups: dict[str, list[str]],
    baseline_cache: dict,
    save_raw_modules: bool = False,
):
    # Flatten the nested dictionary for clean iteration
    #### OR CHANGE THE CALLING FUNCTION TO RETURN IT LIKE THIS DIRECTLY
    tasks = []
    for noise_type, levels in perturbed_files_map.items():
        for level, files in levels.items():
            for filename in files:
                tasks.append((noise_type, level, filename))

    # Iterate through files (Outer Loop)
    for noise_type, noise_level, filename in tasks:
        full_path = os.path.join(perturbed_networks_directory, filename)

        # A. Load Graph ONCE per file
        try:
            perturbed_G = nx.read_edgelist(full_path, delimiter="\t")
        except FileNotFoundError:
            print(f"Skipping missing file: {full_path}")
            continue

        # B. Run all algorithms on this specific graph
        batch_results = []

        for algo in [algo for algo, active in algorithms_config.items() if active]:
            for seed_id, seed_nodes in seed_groups.items():
                # 1. Recover Modules
                # recovered_mod = run_algorithm(perturbed_G, algo_name, seed_nodes)
                recovered_mod = ["geneA", "geneC"]  # Mock

                # 2. Compare to Baseline immediately
                baseline_mod = baseline_cache[(algo, seed_id)]

                # Calculate metrics (Replace with your actual comparison functions)
                # jaccard = calculate_jaccard(baseline_mod, recovered_mod)
                jaccard = 0.5  # Mock
                overlap = len(set(baseline_mod).intersection(set(recovered_mod)))

                # 3. Collect Result Row
                row = [
                    algo,
                    noise_type,
                    noise_level,
                    filename,
                    seed_id,
                    len(seed_nodes),
                    jaccard,
                    overlap,
                    0.0,
                ]
                batch_results.append(row)

                # Optional: Save raw rank list to disk if needed (separate files)
                if save_raw_modules:
                    save_raw_to_disk(recovered_mod, filename, algo, seed_id)

        # C. Write batch to CSV (Reduces I/O overhead compared to writing 1 by 1)
        with open(output_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(batch_results)

        # Explicitly delete graph to ensure memory is freed
        del perturbed_G
    print("--- Benchmarking Complete ---")


def save_raw_to_disk(module_list, filename, algo, seed_id):
    # Construct a filename that identifies the run
    # e.g., results/raw/autocore_noise0.05_diamond_seed1.txt
    pass


# 2. Code to get the module recovery results once


def recoverModules(G: nx.Graph, algorithm_name: str, starting_points: list):
    # if and elifs to select the correct algorithm, run it and return the ranked list of nodes
    # uses the functions in ModuleDetectionAlgorithms
    pass


def compareModules(baseline_modules: list, recovered_modules: list):
    # orchestrator to compare two lists of modules
    # runs a few different methods of comparison (probably complete ranking and also mainly top k ranking)
    # Uses the functions in ModuleComparisonAlgorithms
    pass


# 3. Specify starting points for the module detection algorithms

# 4. Get the results for baseline once

# 5. Iterate over all perturbed networks, recover the modules, compare to baseline

# %%

##################################################
# TESTING
################################################

baseline_network_path = (
    "/Users/marlene/Documents/data/Data/Networks/PPIs/AutoCore_ppi/ppi2017_elist.txt"
)

perturbed_networks_directory = (
    "/Users/marlene/Documents/data/Data/Networks/Perturbed_networks/autocore_ppis/"
)


perturbed_networks_instructions = {
    "added": {
        0.05: [
            "autocore_ppi_added_edges_noise0p05_repeat0.txt",
            "autocore_ppi_added_edges_noise0p05_repeat1.txt",
            "autocore_ppi_added_edges_noise0p05_repeat2.txt",
            "autocore_ppi_added_edges_noise0p05_repeat3.txt",
            "autocore_ppi_added_edges_noise0p05_repeat4.txt",
            "autocore_ppi_added_edges_noise0p05_repeat5.txt",
            "autocore_ppi_added_edges_noise0p05_repeat6.txt",
            "autocore_ppi_added_edges_noise0p05_repeat7.txt",
            "autocore_ppi_added_edges_noise0p05_repeat8.txt",
            "autocore_ppi_added_edges_noise0p05_repeat9.txt",
        ],
        0.1: [
            "autocore_ppi_added_edges_noise0p1_repeat0.txt",
            "autocore_ppi_added_edges_noise0p1_repeat1.txt",
            "autocore_ppi_added_edges_noise0p1_repeat2.txt",
            "autocore_ppi_added_edges_noise0p1_repeat3.txt",
            "autocore_ppi_added_edges_noise0p1_repeat4.txt",
            "autocore_ppi_added_edges_noise0p1_repeat5.txt",
            "autocore_ppi_added_edges_noise0p1_repeat6.txt",
            "autocore_ppi_added_edges_noise0p1_repeat7.txt",
            "autocore_ppi_added_edges_noise0p1_repeat8.txt",
            "autocore_ppi_added_edges_noise0p1_repeat9.txt",
        ],
        0.15: [
            "autocore_ppi_added_edges_noise0p15_repeat0.txt",
            "autocore_ppi_added_edges_noise0p15_repeat1.txt",
            "autocore_ppi_added_edges_noise0p15_repeat2.txt",
            "autocore_ppi_added_edges_noise0p15_repeat3.txt",
            "autocore_ppi_added_edges_noise0p15_repeat4.txt",
            "autocore_ppi_added_edges_noise0p15_repeat5.txt",
            "autocore_ppi_added_edges_noise0p15_repeat6.txt",
            "autocore_ppi_added_edges_noise0p15_repeat7.txt",
            "autocore_ppi_added_edges_noise0p15_repeat8.txt",
            "autocore_ppi_added_edges_noise0p15_repeat9.txt",
        ],
    },
    "removed": {
        0.05: [
            "autocore_ppi_removed_edges_noise0p05_repeat0.txt",
            "autocore_ppi_removed_edges_noise0p05_repeat1.txt",
            "autocore_ppi_removed_edges_noise0p05_repeat2.txt",
            "autocore_ppi_removed_edges_noise0p05_repeat3.txt",
            "autocore_ppi_removed_edges_noise0p05_repeat4.txt",
            "autocore_ppi_removed_edges_noise0p05_repeat5.txt",
            "autocore_ppi_removed_edges_noise0p05_repeat6.txt",
            "autocore_ppi_removed_edges_noise0p05_repeat7.txt",
            "autocore_ppi_removed_edges_noise0p05_repeat8.txt",
            "autocore_ppi_removed_edges_noise0p05_repeat9.txt",
        ],
        0.1: [
            "autocore_ppi_removed_edges_noise0p1_repeat0.txt",
            "autocore_ppi_removed_edges_noise0p1_repeat1.txt",
            "autocore_ppi_removed_edges_noise0p1_repeat2.txt",
            "autocore_ppi_removed_edges_noise0p1_repeat3.txt",
            "autocore_ppi_removed_edges_noise0p1_repeat4.txt",
            "autocore_ppi_removed_edges_noise0p1_repeat5.txt",
            "autocore_ppi_removed_edges_noise0p1_repeat6.txt",
            "autocore_ppi_removed_edges_noise0p1_repeat7.txt",
            "autocore_ppi_removed_edges_noise0p1_repeat8.txt",
            "autocore_ppi_removed_edges_noise0p1_repeat9.txt",
        ],
        0.15: [
            "autocore_ppi_removed_edges_noise0p15_repeat0.txt",
            "autocore_ppi_removed_edges_noise0p15_repeat1.txt",
            "autocore_ppi_removed_edges_noise0p15_repeat2.txt",
            "autocore_ppi_removed_edges_noise0p15_repeat3.txt",
            "autocore_ppi_removed_edges_noise0p15_repeat4.txt",
            "autocore_ppi_removed_edges_noise0p15_repeat5.txt",
            "autocore_ppi_removed_edges_noise0p15_repeat6.txt",
            "autocore_ppi_removed_edges_noise0p15_repeat7.txt",
            "autocore_ppi_removed_edges_noise0p15_repeat8.txt",
            "autocore_ppi_removed_edges_noise0p15_repeat9.txt",
        ],
    },
}

"""
One could consider making the map automatically like this:
def generate_perturbation_map(directory_path: str) -> dict:
    
    Scans a directory and automatically builds the structured map 
    of perturbed networks based on filenames.
    
    Expected format: *_{type}_edges_noise{level}_repeat{N}.txt
    e.g. "autocore_ppi_added_edges_noise0p05_repeat0.txt"
   
    # Structure: Map[NoiseType][NoiseLevel] = List of Files
    file_map = {}
    
    # Regex to capture: (added/removed), (0p05), and the full filename
    # Looks for: "added_edges" or "removed_edges" followed by "noise"
    pattern = re.compile(r"(added|removed)_edges_noise(\d+p\d+)_repeat(\d+)")

    path_obj = Path(directory_path)
    
    # Get all .txt files
    files = sorted([f.name for f in path_obj.glob("*.txt")])
    
    for filename in files:
        match = pattern.search(filename)
        if match:
            p_type = match.group(1)  # "added" or "removed"
            p_level_str = match.group(2) # "0p05"
            
            # Convert "0p05" -> 0.05 (float)
            p_level = float(p_level_str.replace("p", "."))
            
            # Initialize dict structure if missing
            if p_type not in file_map:
                file_map[p_type] = {}
            if p_level not in file_map[p_type]:
                file_map[p_type][p_level] = []
            
            file_map[p_type][p_level].append(filename)
            
    return file_map
"""

# Would be nice to have both ways of degree normalizing in RWR as options
algorithms_to_test = {
    "1stNeighbors": False,
    "DIAMOnD": True,
    "DOMINO": False,
    "ROBUST": False,
    "ROBUST(bias_aware)": False,
    "RandomWalkwithRestart": True,
}

algorithm_starting_points = {
    "seed_A_size_1": ["66008"],
    "seed_A_size_5": ["5347", "66008", "5347", "66008", "5347"],
}

output_csv_path = "/Users/marlene/Documents/data/Projects/23_noise_in_networks_systematic/Code/NoiseInNetworks/notebooks/testing_algorithms/test.csv"


# %%

benchmarkModuleDetectionAlgorithms(
    baseline_network_path=baseline_network_path,
    perturbed_networks_directory=perturbed_networks_directory,
    perturbed_files_map=perturbed_networks_instructions,
    algorithms_config=algorithms_to_test,
    seed_groups=algorithm_starting_points,
    output_csv_path=output_csv_path,
    c
)

# %%
