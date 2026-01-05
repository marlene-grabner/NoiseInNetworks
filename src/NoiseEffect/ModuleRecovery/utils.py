import os
import csv


def _setupOutputCSV(output_csv_path: str):
    columns = [
        "Algorithm",
        "NoiseType",
        "NoiseLevel",
        "RepeatFile",
        "FileName",
        "SeedGroupID",
        "SeedSize",
        "Jaccard",
        "Overlap_Size",
        "TopK_Precision",
    ]

    # Initalize file with headers if it does not exist
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
