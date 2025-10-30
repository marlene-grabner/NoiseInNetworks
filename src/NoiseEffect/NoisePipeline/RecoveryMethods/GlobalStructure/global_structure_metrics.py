import random
import os
import numpy as np
import networkx as nx
from collections import Counter
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, csgraph


def generateGlobalStructureMetrics(
    G, samples=100, eigenvector_spectra_dir=None, network_identifier=""
):
    metrics = {}
    metrics["number_of_components"] = _numberOfComponents(G)
    metrics["average_shortest_path_approx_lcc"] = _averageShortestPathApproximateOnLCC(
        G, samples=samples
    )
    metrics["average_clustering_coeff_approx_lcc"] = (
        _averageClusteringCoeffApproximateOnLCC(G, samples=samples)
    )
    # Eigenvectors get saved immediately to avoid large memory usage
    os.makedirs("global_structure_spectra", exist_ok=True)
    _calculateAndSaveEigenvectorSpectrum(
        G,
        k=min(300, G.number_of_nodes() - 2),
        normalized_laplacian=True,
        eigenvector_spectra_dir=eigenvector_spectra_dir,
        network_identifier=network_identifier,
    )
    return metrics


####### Individual global structure metrics ##########


def _numberOfComponents(G):
    sizes = [len(c) for c in nx.connected_components(G)]
    return dict(Counter(sizes))


def _averageShortestPathApproximateOnLCC(G, samples=100):
    # Only use LCC to approximate path lengths
    lcc = max(nx.connected_components(G), key=len)
    LCC_G = G.subgraph(lcc)
    # Calculate average shortest path using a sample of nodes
    nodes = random.sample(list(LCC_G.nodes()), min(samples, len(LCC_G)))
    total = 0
    count = 0
    for n in nodes:
        lengths = nx.single_source_shortest_path_length(LCC_G, n)
        total += sum(lengths.values())
        count += len(lengths) - 1
    return total / count


def _averageClusteringCoeffApproximateOnLCC(G, samples=1000):
    # Only use LCC to approximate path lengths
    lcc = max(nx.connected_components(G), key=len)
    LCC_G = G.subgraph(lcc)
    # Calculate average clustering on the subsample
    nodes = random.sample(list(LCC_G.nodes()), min(samples, len(LCC_G)))
    triads = triangles = 0
    for v in nodes:
        neighbors = list(G.neighbors(v))
        k = len(neighbors)
        if k < 2:
            continue
        triads += k * (k - 1) / 2
        sub = G.subgraph(neighbors)
        triangles += len(sub.edges())
    return (3 * triangles) / triads if triads > 0 else 0


def _calculateAndSaveEigenvectorSpectrum(
    G,
    k=300,
    normalized_laplacian=True,
    eigenvector_spectra_dir=None,
    network_identifier="",
):
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    if (A != A.T).nnz != 0:
        print("[WARN] Adjacency is not symmetric. Symmetrizing...")
        # make adjacency symmetric to avoid issues in eigensolvers
        A = (A + A.T) / 2

    # ensure k is smaller than the matrix dimension
    n_nodes = A.shape[0]
    if k >= n_nodes:
        k = max(1, n_nodes - 1)

    vals_A, vecs_A = eigsh(A, k=k, which="LA")
    np.save(
        f"{eigenvector_spectra_dir}/{network_identifier}_adjacency_vals_k{k}.npy",
        vals_A,
    )
    np.savez_compressed(
        f"{eigenvector_spectra_dir}/{network_identifier}_adjacency_vecs_k{k}.npz",
        vecs_A,
    )

    L = csgraph.laplacian(A, normed=normalized_laplacian)
    vals_L, vecs_L = eigsh(L, k=k, which="SM")
    np.save(
        f"{eigenvector_spectra_dir}/{network_identifier}_laplacian_vals_k{k}.npy",
        vals_L,
    )
    np.savez_compressed(
        f"{eigenvector_spectra_dir}/{network_identifier}_laplacian_vecs_k{k}.npz",
        vecs_L,
    )
