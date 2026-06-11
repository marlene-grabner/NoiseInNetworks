import networkx as nx
import numpy as np
import scipy.stats as stats
import scipy.sparse.linalg as sla
import igraph as ig
import random


def get_network_profile(G):
    """
    Calculates a comprehensive topology profile for a NetworkX graph G.
    Uses SciPy and igraph for heavy computations to ensure extreme efficiency.
    """
    # Ensure graph is undirected and simple for baseline metrics
    G = nx.Graph(G)

    # 1. MACROSCOPIC PROPERTIES (Fast natively in NetworkX)
    print("Calculating macroscopic properties...")
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    density = nx.density(G)

    # Giant Connected Component (GCC) fraction
    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc_fraction = len(gcc_nodes) / nodes

    # 2. DEGREE STATISTICS (Fast natively with NumPy/SciPy)
    print("Calculating degree statistics...")
    degrees = np.array([d for n, d in G.degree()])
    deg_skew = stats.skew(degrees)
    deg_cv = np.std(degrees) / np.mean(degrees) if np.mean(degrees) > 0 else 0

    # Assortativity (NetworkX native is usually fast enough, O(E))
    assortativity = nx.degree_assortativity_coefficient(G)

    # 3. SPECTRAL PROPERTIES (Fast with SciPy Sparse Linear Algebra)
    # Use the unnormalized Laplacian matrix
    print("Calculating spectral properties...")
    L = nx.laplacian_matrix(G).astype(float)

    # We look for the 3 smallest algebraic eigenvalues (SA).
    # lambda_1 is always ~0. lambda_2 is algebraic connectivity.
    try:
        # tol=1e-3 speeds up convergence drastically since we don't need perfect float precision
        eigenvalues, _ = sla.eigsh(L, k=3, which="SA", tol=1e-3)
        eigenvalues = np.sort(eigenvalues)
        alg_connectivity = eigenvalues[1]
        spectral_gap = eigenvalues[2] - eigenvalues[1]
    except Exception as e:
        # Fallback if eigsh fails to converge (rare, but happens on perfectly disconnected graphs)
        alg_connectivity, spectral_gap = 0.0, 0.0

    # 4. MESOSCOPIC & MICROSCOPIC PROPERTIES (Fast with igraph C-core)
    # Convert NetworkX to igraph for the heavy lifting
    print("Calculating mesoscopic and microscopic properties with igraph...")
    ig_g = ig.Graph.from_networkx(G)

    # Transitivity (Global clustering) and Average Local Clustering
    transitivity = ig_g.transitivity_undirected()
    clustering = ig_g.transitivity_avglocal_undirected()

    # Modularity using Louvain (Multilevel)
    # This takes milliseconds in igraph compared to minutes in NetworkX
    partition = ig_g.community_multilevel()
    modularity = partition.modularity

    print("Calculating path lengths and efficiency...")
    # 5. PATH LENGTH / EFFICIENCY (Approximation via Sampling)
    # Exact all-pairs shortest paths on 20k nodes is O(V*E) and takes too long.
    # We sample 500 nodes from the GCC to get a highly accurate approximation.
    sample_size = min(500, len(gcc_nodes))
    sampled_nodes = random.sample(list(gcc_nodes), sample_size)

    path_lengths = []
    for node in sampled_nodes:
        # nx.single_source_shortest_path_length is highly optimized C-like BFS in NetworkX
        lengths = nx.single_source_shortest_path_length(G, node)
        path_lengths.extend(lengths.values())

    avg_path_length = np.mean(path_lengths) if path_lengths else 0.0

    # Compile results
    profile = {
        "Nodes": nodes,
        "Edges": edges,
        "Density": density,
        "GCC_Fraction": gcc_fraction,
        "Degree_Skew": deg_skew,
        "Degree_CV": deg_cv,
        "Assortativity": assortativity,
        "Transitivity": transitivity,
        "Clustering_Coefficient": clustering,
        "Modularity_Louvain": modularity,
        "Algebraic_Connectivity": alg_connectivity,
        "Spectral_Gap": spectral_gap,
        "Avg_Path_Length_Sampled": avg_path_length,
    }

    return profile
