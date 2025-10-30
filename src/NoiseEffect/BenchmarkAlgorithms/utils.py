import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)


def convertPartitionToLabels(partition, num_nodes):
        """
        Converts a partition (list of sets of node IDs) to a labels array.

        This version infers the total number of nodes by finding the maximum
        node ID in the partition. It assumes nodes are indexed from 0.
        """
        # Handle the edge case of an empty or invalid partition
        if not partition or not any(partition):
            return np.array([], dtype=int)

        # 1. Create the labels array. Using np.full with -1 is often safer
        #    to make it obvious if a node was missed.
        labels = np.full(num_nodes, -1, dtype=int)

        # 2. Populate the array with cluster IDs.
        for cluster_id, community in enumerate(partition):
            for node in community:
                labels[node] = cluster_id

        return labels


def getMetrics(clustering_1, clustering_2):
    """
    Computes Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI)
    between two cluster labelings.
    """

    n_clustering_1 = len(np.unique(clustering_1))
    n_clustering_2 = len(np.unique(clustering_2))

    # Handle there being only one cluster in either parition
    if n_clustering_1 == 1 and n_clustering_2 == 1:
        return {
            "status": "trivial_one_cluster",
            "num_clusters_1": n_clustering_1,
            "num_clusters_2": n_clustering_2,
            "ari": np.nan,
            "ami": np.nan,
        }

    # Handle all nodes being singletons in either partition
    if n_clustering_1 == len(clustering_1) or n_clustering_2 == len(clustering_2):
        return {
            "status": "trivial_all_singletons",
            "num_clusters_1": n_clustering_1,
            "num_clusters_2": n_clustering_2,
            "ari": np.nan,
            "ami": np.nan,
        }

    try:
        # Calculate metrics
        ari = adjusted_rand_score(labels_true=clustering_1, labels_pred=clustering_2)
        ami = adjusted_mutual_info_score(
            labels_true=clustering_1, labels_pred=clustering_2
        )
        return {
            "status": "success",
            "num_clusters_1": n_clustering_1,
            "num_clusters_2": n_clustering_2,
            "ari": ari,
            "ami": ami,
        }
    except Exception as e:
        return {
            "status": "error",
            "num_clusters_1": n_clustering_1,
            "num_clusters_2": n_clustering_2,
            "ari": np.nan,
            "ami": np.nan,
        }
    

