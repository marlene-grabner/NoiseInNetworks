import pandas as pd
import json


class ModuleAnalyzer:
    def __init__(self):
        self.df = None
        self.baselines = {}
        self.baseline_seeds = {}

    def _loadData(self, file_paths: list[str]):
        records = []

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for path in file_paths:
            with open(path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        algorithm = entry.get("metadata_run", {}).get(
                            "algorithm", "unknown_algorithm"
                        )
                        raw_results = entry.get("module_results", None)

                        if algorithm == "DIAMOnD":
                            result_data = raw_results
                            result_type = "DIAMOnD"

                        # Are the results ranked indicating RWR?
                        elif isinstance(raw_results, dict):
                            result_data = raw_results
                            result_type = "rwr"

                        # Are the results just a set of nodes? e.g. 1st Neigbors or DOMINO
                        elif isinstance(raw_results, list):
                            if len(raw_results) == 0:
                                result_data = set()
                                result_type = "empty"
                            elif isinstance(raw_results[0], list):
                                # If we have a list of lists, we convert to a list of sets
                                result_data = [
                                    set(sub_module) for sub_module in raw_results
                                ]
                                result_type = "multi_module"
                            else:
                                # If we only have a single list, convert to single set
                                result_data = set(raw_results)
                                result_type = "single_module"
                        else:
                            print(
                                f"Unexpected type for 'module_results' in line: {line}"
                            )
                            continue

                        # 2. Flatten Metadata
                        meta_net = entry.get("metadata_network", {})
                        meta_seed = entry.get("metadata_seed", {})

                        record = {
                            # Identifiers
                            "seed_id": meta_seed.get("seed_id"),
                            "repeat": meta_net.get("repeat", "rep0"),
                            # Perturbation Info
                            "noise_type": meta_net.get("noise_type"),
                            "noise_level": meta_net.get("noise_level"),
                            # The Data
                            "result_type": result_type,
                            "results": result_data,
                            # Seed nodes
                            "seeds_in_network": set(
                                meta_seed.get("seeds_in_network", [])
                            ),
                        }
                        records.append(record)
                    except json.JSONDecodeError:
                        continue

        self.df = pd.DataFrame(records)
        self._buildBaseline()

        # Drop any duplicate records
        subset_cols = ["seed_id", "noise_type", "noise_level", "repeat"]
        self.df.drop_duplicates(subset=subset_cols, keep="first", inplace=True)

        print(f"Loaded {len(self.df)} records.")

    def _buildBaseline(self):
        mask = (self.df["noise_level"] == 0) | (self.df["noise_type"] == "baseline")
        baseline_df = self.df[mask]
        self.baselines = dict(zip(baseline_df["seed_id"], baseline_df["results"]))
        self.baseline_seeds = dict(
            zip(baseline_df["seed_id"], baseline_df["seeds_in_network"])
        )
        print(f"Indexed baselines for {len(self.baselines)} seeds.")
