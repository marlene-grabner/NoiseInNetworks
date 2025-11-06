from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
import ast
from NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.random_walk import (
    randomWalkWithRestart,
)
import numpy as np

####################################
# Local Neighborhood Analysis #####
# Master function #
####################################


def localNeighborhoodAnalysis(modified_network_nx, original_neighborhood):
    similarity_results = {}
    for start_str in original_neighborhood.keys():
        start = ast.literal_eval(start_str)
        new_neighborhood = randomWalkWithRestart(
            G=modified_network_nx,
            seed_nodes=start,
        )
        similarity_results[start_str] = _calculateSimilarityMetrics(
            original_neighborhood=original_neighborhood[start_str],
            new_neighborhood=new_neighborhood,
            seed_nodes=start,
            num_nodes=modified_network_nx.number_of_nodes(),
        )
    return similarity_results


####################################
###### Similarity Metrics ##########
####################################


def _calculateSimilarityMetrics(
    original_neighborhood, new_neighborhood, seed_nodes, num_nodes
):
    # Remove seed nodes from both distributions
    original_neighborhood_minus_seed = original_neighborhood.copy()
    new_neighborhood_minus_seed = new_neighborhood.copy()
    for seed in seed_nodes:
        original_neighborhood_minus_seed.pop(seed, None)
        new_neighborhood_minus_seed.pop(seed, None)

    p = _rwrResultsToVectors(p=original_neighborhood_minus_seed, n=num_nodes)
    q = _rwrResultsToVectors(p=new_neighborhood_minus_seed, n=num_nodes)

    p_sum = p.sum()
    q_sum = q.sum()

    # Renormalize after removing seed nodes
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum

    if p_sum == 0 or q_sum == 0:
        print(
            "Warning: One of the distributions is zero after removing seed nodes. Similarity metrics may be undefined."
        )
        for seed in seed_nodes:
            print(
                f"For seed node {seed} original visiting probability was {original_neighborhood.get(seed)} and new visiting probability was {new_neighborhood.get(seed)}"
            )
        return {
            "jsd": "isolated",
            "spearman": "isolated",
            "top_50_jaccard": "isolated",
            "l2_norm": "isolated",
        }

    else:
        jsd = jensenshannon(
            p, q
        )  # For JSD we need to ensure a proper probability distribution, meaning both p and q must sum to 1
        spearman_corr, spearman_p = spearmanr(p, q)
        top_50_jaccard = _calculateTop50JaccardSimilarity(
            original_neighborhood_minus_seed, new_neighborhood_minus_seed
        )
        l2_norm = np.linalg.norm(p - q)
        return {
            "jsd": jsd,
            "spearman": (spearman_corr, spearman_p),
            "top_50_jaccard": top_50_jaccard,
            "l2_norm": l2_norm,
        }


####################################
#### Top 50 Jaccard Similarity #####
####################################


def _calculateTop50JaccardSimilarity(original_neighborhood, new_neighborhood):
    top50_orig = set(
        sorted(original_neighborhood, key=original_neighborhood.get, reverse=True)[:50]
    )
    top50_new = set(
        sorted(new_neighborhood, key=new_neighborhood.get, reverse=True)[:50]
    )
    return len(top50_orig.intersection(top50_new)) / len(top50_orig.union(top50_new))


####################################
##### RWR results to vectors #######
####################################


def _rwrResultsToVectors(p, n):
    vec = np.zeros(n)
    for node_id in range(n):
        vec[node_id] = p.get(node_id, 0.0)
    return vec
