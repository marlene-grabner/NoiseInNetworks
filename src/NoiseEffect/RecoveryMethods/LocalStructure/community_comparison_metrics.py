from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
import numpy as np


class CommunityComparisonMetrics:
    """A collection of static methods for community comparison metrics"""

    @staticmethod
    def convertPartitionToLabels(partition):
        """
        Converts a partition (list of sets of node IDs) to a labels array.

        This version infers the total number of nodes by finding the maximum
        node ID in the partition. It assumes nodes are indexed from 0.
        """
        # Handle the edge case of an empty or invalid partition
        if not partition or not any(partition):
            return np.array([], dtype=int)

        # 1. Find the highest node ID to determine the required array size.
        num_nodes = max(max(community) for community in partition if community) + 1

        # 2. Create the labels array. Using np.full with -1 is often safer
        #    to make it obvious if a node was missed.
        labels = np.full(num_nodes, -1, dtype=int)

        # 3. Populate the array with cluster IDs.
        for cluster_id, community in enumerate(partition):
            for node in community:
                labels[node] = cluster_id

        return labels

    @staticmethod
    def getMetrics(true_labels, noisy_labels):
        """
        Computes Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI)
        between two cluster labelings.
        """

        n_true_labels = len(np.unique(true_labels))
        n_noisy_labels = len(np.unique(noisy_labels))

        # Handle there being only one cluster in either parition
        if n_true_labels == 1 and n_noisy_labels == 1:
            return {
                "status": "trivial_one_cluster",
                "num_clusters": n_noisy_labels,
                "ari": np.nan,
                "ami": np.nan,
            }

        # Handle all nodes being singletons in either partition
        if n_true_labels == len(true_labels) or n_noisy_labels == len(noisy_labels):
            return {
                "status": "trivial_all_singletons",
                "num_clusters": n_noisy_labels,
                "ari": np.nan,
                "ami": np.nan,
            }

        try:
            # Calculate metrics
            ari = adjusted_rand_score(labels_true=true_labels, labels_pred=noisy_labels)
            ami = adjusted_mutual_info_score(
                labels_true=true_labels, labels_pred=noisy_labels
            )
            return {
                "status": "success",
                "num_clusters": n_noisy_labels,
                "ari": ari,
                "ami": ami,
            }
        except Exception as e:
            return {
                "status": "error",
                "num_clusters": n_noisy_labels,
                "ari": np.nan,
                "ami": np.nan,
            }
