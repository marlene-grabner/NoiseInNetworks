import logging
import numpy as np
from .module_result import ModuleResult

# Create the logging channel for this file
logger = logging.getLogger(__name__)

"""
Key Comparison Strategies by Algorithm Type

Ranked outputs (RWR, DIAMOnD, ROBUST):

AUPRC: Best metric for evaluating ranking quality
Top-K Jaccard/Precision/Recall: Take top-K nodes where K = size of baseline
Rank Biased Overlap (RBO): If you want to compare two rankings


Set outputs (DOMINO, 1st Neighbors):

Jaccard Index: Standard overlap metric
Precision/Recall/F1: Classification-style metrics
Size difference: Track if methods produce larger/smaller modules


Mixed comparisons (your typical case):

Baseline is usually ground truth (set)
Take Top-K from ranked methods to compare as sets
Keep AUPRC as additional metric for ranked methods
    """


def compareResults(baseline_module: ModuleResult, recovered_module: ModuleResult):
    """
    baseline_module: ALWAYS a set/list of nodes (The Truth).
    recovered_module: Can be a Dict (Ranked) OR a Set/List (Unranked).
    """
    baseline_set = baseline_module.as_set()
    k = len(baseline_set)  # The "Target Size" for Top-K

    metrics = {
        "jaccard": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "overlap_size": 0,
        "auprc": np.nan,
        "baseline_size": k,
        "recovered_size": recovered_module.size(),
    }

    # --- CASE 1: RANKED RESULTS (RWR, DIAMOnD) ---
    if recovered_module.algorithm_type == "ranked":
        # Calculate AUPRC using full ranking
        metrics["auprc"] = _calculate_auprc(baseline_set, recovered_module.nodes_ranked)

        # Calculate Top-K metrics (take top-k where k = baseline size)
        recovered_top_k = recovered_module.get_top_k(k)

    # --- CASE 2: SET RESULTS (DOMINO, 1stNeighbors) ---
    else:
        recovered_top_k = recovered_module.as_set()

    # Calculate overlap metrics
    intersection = len(baseline_set.intersection(recovered_top_k))
    union = len(baseline_set.union(recovered_top_k))

    metrics["jaccard"] = intersection / union if union > 0 else 0.0
    metrics["overlap_size"] = intersection
    metrics["precision"] = (
        intersection / len(recovered_top_k) if len(recovered_top_k) > 0 else 0.0
    )
    metrics["recall"] = intersection / k if k > 0 else 0.0

    # Calculate F1
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = (
            2
            * (metrics["precision"] * metrics["recall"])
            / (metrics["precision"] + metrics["recall"])
        )

    return metrics


def _calculate_auprc(true_positives: set, ranked_results: dict) -> float:
    """Calculate Area Under Precision-Recall Curve."""
    if not ranked_results or not true_positives:
        return 0.0

    ranked_nodes = list(ranked_results.keys())
    n_positives = len(true_positives)

    precisions = []
    recalls = []
    tp_count = 0

    for i, node in enumerate(ranked_nodes, 1):
        if node in true_positives:
            tp_count += 1

        precision = tp_count / i
        recall = tp_count / n_positives

        precisions.append(precision)
        recalls.append(recall)

    # Calculate area using trapezoidal rule
    auprc = np.trapz(precisions, recalls)
    return float(auprc)


"""
def compareResults(algorithm: str, baseline_module: dict, recovered_module: dict):
    print("TODO: compareResults; what kind of comparison metrics are needed?")
    if algorithm == "1stNeighbors":
        print("Not implemented yet")
        results = []
    elif algorithm == "DIAMOnD":
        logger.warning("DIAMOnD is not implemented yet.")
        print("Not implemented yet")
        results = []
    elif algorithm == "DOMINO":
        print("Not implemented yet")
        results = []
    elif algorithm == "ROBUST":
        print("Not implemented yet")
        results = []
    elif algorithm == "ROBUST(bias_aware)":
        print("Not implemented yet")
        results = []
    elif algorithm == "RandomWalkWithRestart":
        print("x")
    else:
        raise ValueError(f"Unknown algorithm specified: {algorithm}")

    return results

"""
