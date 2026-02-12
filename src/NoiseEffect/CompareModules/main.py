from NoiseEffect.CompareModules.module_analyzer import ModuleAnalyzer
from NoiseEffect.CompareModules.jaccard import applyJaccard
from NoiseEffect.CompareModules.remove_seeds import _removeSeeds
import pandas as pd


def compareModules(file_paths: list[str], metric: str, top_k: int):

    analyzer = ModuleAnalyzer()
    analyzer._loadData(file_paths)

    # Removing potential leftover seeds from baseline and perturbed results
    _removeSeeds(analyzer)

    if metric == "jaccard":
        analyzer.df["metric"] = applyJaccard(analyzer.df, analyzer.baselines, top_k)

    # Summarize per noise level and seed
    # Convert None to NaN
    analyzer.df["metric"] = pd.to_numeric(analyzer.df["metric"], errors="coerce")
    summary_df = (
        analyzer.df.groupby(["seed_id", "noise_type", "noise_level"])["metric"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    return summary_df
