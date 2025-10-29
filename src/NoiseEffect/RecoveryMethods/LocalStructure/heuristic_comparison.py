from NoiseEffect.RecoveryMethods.LocalStructure.community_comparison_metrics import (
    CommunityComparisonMetrics,
)
from collections import Counter
import numpy as np


class CompareHeuristicClusterings:
    def __init__(self, original_communities, perturbed_communities):
        self.original_communities = original_communities
        self.perturbed_communities = perturbed_communities
        self.comparison_results = {}

    def makeComparison(self):
        metrics_summary = {}
        # Go through the obtained clusterings for each random seed
        for seed in self.perturbed_communities:
            original_partition = self.original_communities[seed]
            perturbed_partition = self.perturbed_communities[seed]

            # Generate labels for both partitions
            original_labels = CommunityComparisonMetrics.convertPartitionToLabels(
                original_partition
            )
            perturbed_labels = CommunityComparisonMetrics.convertPartitionToLabels(
                perturbed_partition
            )

            # Compute comparison metrics
            metrics_dict = CommunityComparisonMetrics.getMetrics(
                true_labels=original_labels,
                noisy_labels=perturbed_labels,
            )

            metrics_summary[seed] = metrics_dict

        self.summarizeResults(metrics_summary)

    def summarizeResults(self, metrics_dict):
        ari_values = []
        ami_values = []
        status_list = []
        num_clusters_list = []

        for results in metrics_dict.values():
            # Collect the numeric values only if present
            if isinstance(results.get("ari"), (int, float)):
                ari_values.append(results["ari"])
            if isinstance(results.get("ami"), (int, float)):
                ami_values.append(results["ami"])

            status_list.append(results.get("status"))
            num_clusters_list.append(results.get("num_clusters"))

        summary = {
            "ari": {
                "values": ari_values,
                "mean": float(np.mean(ari_values)) if ari_values else None,
                "std": float(np.std(ari_values, ddof=1))
                if len(ari_values) > 1
                else None,
            },
            "ami": {
                "values": ami_values,
                "mean": float(np.mean(ami_values)) if ami_values else None,
                "std": float(np.std(ami_values, ddof=1))
                if len(ami_values) > 1
                else None,
            },
            "status_counts": dict(Counter(status_list)),
            "num_clusters_counts": dict(Counter(num_clusters_list)),
        }

        self.comparison_results = summary
