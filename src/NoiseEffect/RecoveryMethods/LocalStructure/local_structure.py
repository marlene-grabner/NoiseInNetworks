from NoiseEffect.RecoveryMethods.LocalStructure.community_detection_algorithms import (
    CommunityDetectionAlgorithms,
)
from NoiseEffect.RecoveryMethods.LocalStructure.community_comparison_metrics import (
    CommunityComparisonMetrics,
)
from NoiseEffect.RecoveryMethods.LocalStructure.heuristic_comparison import (
    CompareHeuristicClusterings,
)


def localStructureAnalysis(G_nx, G_ig, original_communities, random_seed_list):
    recovery_results = {}
    # Leiden Algorithm Comparison
    leiden_clusters = CommunityDetectionAlgorithms.leidenAlgorithmPartioning(
        G_ig, list_of_seeds=random_seed_list
    )
    original_leiden_clusters = original_communities["leiden_algorithm"]

    comparison_obj = CompareHeuristicClusterings(
        original_communities=original_leiden_clusters,
        perturbed_communities=leiden_clusters,
    )
    comparison_obj.makeComparison()
    comparison_results = comparison_obj.comparison_results
    recovery_results["leiden_algorithm"] = comparison_results

    # Infomap Algorithm Comparison
    infomap_clusters = CommunityDetectionAlgorithms.infomapAlgorithmPartioning(
        G_ig, list_of_seeds=random_seed_list
    )
    original_infomap_clusters = original_communities["infomap_algorithm"]

    comparison_obj = CompareHeuristicClusterings(
        original_communities=original_infomap_clusters,
        perturbed_communities=infomap_clusters,
    )
    comparison_obj.makeComparison()
    comparison_results = comparison_obj.comparison_results
    recovery_results["infomap_algorithm"] = comparison_results
    return recovery_results
