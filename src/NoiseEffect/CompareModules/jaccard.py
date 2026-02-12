import numpy as np
from scipy.optimize import linear_sum_assignment


def applyJaccard(df, baselines, k=100):
    """
    Function called by main.py.
    Searches for the baseline corresponding to each row's seed_id and applies Jaccard comparison.
    """

    def row_processor(row):
        seed_id = row.get("seed_id")
        if seed_id not in baselines:
            return None

        baseline_data = baselines[seed_id]
        perturbed_data = row["results"]

        return _computeScore(baseline_data, perturbed_data, k)

    return df.apply(row_processor, axis=1)


def _computeScore(baseline, perturbed, k):
    """
    Checks data type and selects the appropriate logic for each case.
    """
    # 1. Handle Ranked Results (RWRs)
    # Get converted to sets, and top-k are picked
    if isinstance(baseline, list) and isinstance(perturbed, list) | isinstance(
        baseline, set
    ) & isinstance(perturbed, list) | isinstance(baseline, list) & isinstance(
        perturbed, set
    ):
        return _jaccardForMultiModule(baseline, perturbed)

    elif isinstance(baseline, dict) and isinstance(perturbed, dict):
        base_set = _getTopkSet(baseline, k)
        pert_set = _getTopkSet(perturbed, k)

    # 2. Handle single sets (1st Neighbors)
    elif isinstance(baseline, set) and isinstance(perturbed, set):
        base_set = baseline
        pert_set = perturbed

    intersection = len(base_set.intersection(pert_set))
    union = len(base_set.union(pert_set))

    return intersection / union if union > 0 else 0.0


########## List of Sets case ##########


def _jaccardForMultiModule(baseline_modules, perturbed_modules):
    """
    Computes the optimal one to one matching score between lists of sets.
    """
    # 1. Handle the empty case
    n_base = len(baseline_modules)
    n_pert = len(perturbed_modules)

    if n_base == 0 and n_pert == 0:
        return "both empty"  # Both empty = perfect match
    if n_base == 0 or n_pert == 0:
        return "one empty"  # One empty = failure

    # Build Cost Matrix
    cost_matrix = np.zeros((n_base, n_pert))

    for i, b_mod in enumerate(baseline_modules):
        for j, p_mod in enumerate(perturbed_modules):
            # negative Jaccard because linear_sum_assignment minimizes cost
            cost_matrix[i, j] = -jaccard_set(b_mod, p_mod)

    # 3. Solve Assignment Problem (The Hungarian Algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. Sum the Jaccard scores of the optimal pairs
    # the sum is negated back to positive
    total_jaccard = -cost_matrix[row_ind, col_ind].sum()

    # 5. Normalizing
    # Dividing by max(n_base, n_pert) penalizes:
    # - Missing modules (unmatched baseline rows)
    # - Extra noise modules (unmatched perturbed columns)
    return total_jaccard / max(n_base, n_pert)


########## Helper for List of Sets case ##########
def jaccard_set(s1, s2):
    """Standard Jaccard for two sets"""
    if not s1 and not s2:
        return 0.0
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return inter / union if union > 0 else 0.0


########## Helper for RWR case ##########


def _getTopkSet(ranked_data, k):
    """
    Extracts top K keys from a dictionary based on values.
    """
    if not isinstance(ranked_data, dict):
        return set()
    # Sort by score descending and take top k keys
    top_keys = sorted(ranked_data, key=ranked_data.get, reverse=True)[:k]
    return set(top_keys)
