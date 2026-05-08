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
    """
    tasks = []

    # Group 1: Network Name
    # Group 2: Perturbation Type
    # Group 3: Noise Level (e.g., 0p05 or just 1)
    # Group 4: Repeat number
    regex_str = r"(.*)_(targeted_hub_addition|targeted_hub_removal|targeted_periphery_addition|targeted_periphery_removal|added_edges|removed_edges)_noise(\d+(?:p\d+)?)_repeat(\d+)"
    pattern = re.compile(regex_str)

    path_obj = Path(directory_path)

    # Grab both .txt and .tsv just to be completely safe
    files = list(path_obj.glob("*.txt")) + list(path_obj.glob("*.tsv"))
    files = sorted([f.name for f in files])

    for filename in files:
        # Ignore ghost files
        if filename.startswith("._"):
            continue

        match = pattern.search(filename)
        if match:
            network_name = match.group(1)
            p_type = match.group(2)  # e.g., 'targeted_hub_addition'

            p_level_str = match.group(3)  # e.g., '0p05' or '1'
            p_level = float(p_level_str.replace("p", "."))

            repeat = match.group(4)
            repeat_id = f"rep{repeat}"

            tasks.append((filename, p_type, p_level, repeat_id, network_name))
        else:
            print(
                f"Warning: File '{filename}' was ignored because it didn't match the regex."
            )

    return tasks


def _saveRawToDisk(module_list, filename, algo, seed_id):
    # Construct a filename that identifies the run
    # e.g., results/raw/autocore_noise0.05_diamond_seed1.txt
    # TODO
    pass
