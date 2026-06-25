import argparse
import itertools
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from NoiseEffect.CommunityDetection.utils import convertPartitionToLabels, getMetrics
from NoiseEffect.CommunityDetection.detection_algorithms import (
    leidenAlgorithmPartioning,
    infomapAlgorithmPartioning,
    louvainPartioning,
    labelPropagationPartitioning,
)


NOISE_TYPES = ["noise_type_a", "noise_type_b", "noise_type_c"]  # replace with yours
N_NOISE_LEVELS = 20
SEEDS = list(range(5))  # 5 seeds → 10 within-pairs; tune to your time budget

# --------------------------------------------------------------------------- #
# Community detection dispatcher                                               #
# --------------------------------------------------------------------------- #

def run_algorithm(ig_graph: ig.Graph, algo: str, seeds: list[int]) -> dict[int, list[set]]:
    if algo == "leiden":
        return leidenAlgorithmPartioning(ig_graph, seeds, n_iterations=2)
    elif algo == "louvain":
        return louvainPartioning(ig_graph, seeds)
    elif algo == "infomap":
        return infomapAlgorithmPartioning(ig_graph, seeds, n_iterations=1)
    elif algo == "label_propagation":
        return labelPropagationPartitioning(ig_graph, seeds)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


# --------------------------------------------------------------------------- #
# Metric helpers                                                                #
# --------------------------------------------------------------------------- #

def safe_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """ARI with guards for degenerate partitions."""
    u_a, u_b = np.unique(labels_a), np.unique(labels_b)
    if len(u_a) == 1 or len(u_b) == 1:
        return np.nan
    if len(u_a) == len(labels_a) or len(u_b) == len(labels_b):
        return np.nan
    try:
        return float(adjusted_rand_score(labels_a, labels_b))
    except Exception:
        return np.nan


def pairwise_ari_stats(label_matrix: np.ndarray) -> tuple[float, float]:
    """
    Given shape (k, n_nodes), compute mean/std ARI over all k(k-1)/2 pairs.
    Returns (mean, std) — NaNs ignored.
    """
    scores = [
        safe_ari(label_matrix[i], label_matrix[j])
        for i, j in itertools.combinations(range(len(label_matrix)), 2)
    ]
    arr = np.array(scores, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.mean(valid)), float(np.std(valid))


def cross_ari_stats(
    label_matrix: np.ndarray,      # (k_perturbed, n_nodes) — current run
    baseline_labels: np.ndarray,   # (k_baseline, n_nodes) — precomputed
) -> tuple[float, float]:
    """
    Mean/std ARI across all k_perturbed × k_baseline pairs.
    """
    scores = [
        safe_ari(label_matrix[i], baseline_labels[j])
        for i in range(len(label_matrix))
        for j in range(len(baseline_labels))
    ]
    arr = np.array(scores, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.mean(valid)), float(np.std(valid))


# --------------------------------------------------------------------------- #
# Per-network worker (called in parallel)                                      #
# --------------------------------------------------------------------------- #

def process_one_network(
    repeat_id: int,
    edges: list[tuple[int, int]],
    n_nodes: int,
    algo: str,
    seeds: list[int],
    baseline_labels: np.ndarray,   # (k_baseline, n_nodes)
    graph_id: int,
    noise_type: str,
    noise_level: float,
) -> dict:
    # Rebuild igraph (cheap, avoids sharing graph objects across processes)
    g = ig.Graph(n=n_nodes, edges=edges, directed=False)

    partitions = run_algorithm(g, algo, seeds)

    label_matrix = np.stack([
        convertPartitionToLabels(partitions[s], n_nodes) for s in seeds
    ])  # (k, n_nodes)

    within_mean, within_std = pairwise_ari_stats(label_matrix)
    vs_base_mean, vs_base_std = cross_ari_stats(label_matrix, baseline_labels)

    mean_n_communities = float(np.mean([len(partitions[s]) for s in seeds]))

    return {
        "graph_id":           graph_id,
        "noise_type":         noise_type,
        "noise_level":        noise_level,
        "repeat_id":          repeat_id,
        "algorithm":          algo,
        "within_ari_mean":    within_mean,
        "within_ari_std":     within_std,
        "vs_baseline_ari_mean": vs_base_mean,
        "vs_baseline_ari_std":  vs_base_std,
        "mean_n_communities": mean_n_communities,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def load_parquet_as_graphs(parquet_path: Path) -> list[tuple[int, list, int]]:
    """
    Returns list of (repeat_id, edge_list, n_nodes).
    Adjust column names to match your parquet schema.
    """
    df = pd.read_parquet(parquet_path)
    networks = []
    for repeat_id, group in df.groupby("repeat"):
        edges = list(zip(group["source"], group["target"]))
        n_nodes = int(group[["source", "target"]].max().max()) + 1
        networks.append((int(repeat_id), edges, n_nodes))
    return networks


