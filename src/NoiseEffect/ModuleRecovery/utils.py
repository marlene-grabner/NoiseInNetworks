import os
import csv
import re
from pathlib import Path
import networkx as nx


def _setupOutputCSV(output_csv_path: str):
    columns = [
        "Algorithm",
        "NoiseType",
        "NoiseLevel",
        "RepeatFile",
        "FileName",
        "SeedGroupID",
        "SeedsOnNetwork_Length",
        "SeedsOnNetwork_List",
        "Overlap_SizeJaccard",
        "TopK_Precision",
        "TopK_Recall",
        "TopK_F1",
        "AUPRC",
    ]

    # Initalize file with headers if it does not exist
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
    else:
        # raise FileExistsError(f"Output CSV already exists: {output_csv_path}")
        print(
            "forget not to uncomment the making an error if the csv file already exists"
        )


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


def _saveRawToDisk(module_list, filename, algo, seed_id):
    # Construct a filename that identifies the run
    # e.g., results/raw/autocore_noise0.05_diamond_seed1.txt
    # TODO
    pass
