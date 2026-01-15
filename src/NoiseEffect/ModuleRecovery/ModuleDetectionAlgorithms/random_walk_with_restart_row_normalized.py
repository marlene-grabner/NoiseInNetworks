import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from ..module_result import ModuleResult


def randomWalkWithRestartRowNormalized(
    G: nx.Graph,
    seed_nodes: list[str],
    restart: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> ModuleResult:
    """
    Perform a random walk with restart (RWR) on graph G starting from a set of seed nodes.

    Parameters
    ----------
    G : networkx.Graph
        The input undirected graph.
    seed_nodes : list or array-like
        Indices of seed nodes to initialize the walk.
    restart : float, optional
        Restart probability (default: 0.85). High values emphasize local proximity to seed nodes.
    tol : float, optional
        Convergence tolerance (default: 1e-6). The iteration stops when the L1 norm change is below this value.
    max_iter : int, optional
        Maximum number of iterations allowed (default: 100).

    Returns
    -------
    p : np.ndarray
        Steady-state visiting probabilities for each node in the graph.
    """
    # Get consistent node ordering
    nodelist = list(G.nodes())
    n = len(nodelist)

    """
    print(
        "TODO: RWR currently checks by itself if seed nodes are in the graph. This might move to doing it once before calling each algorithm."
    )

    # Creating an internal map of string node names to int indices
    node_to_idx = {node: idx for idx, node in enumerate(nodelist)}
    # Convert string seeds to integer indices using the internal map
    valid_seed_indices = []
    for seed in seed_nodes:
        if seed in node_to_idx:
            valid_seed_indices.append(node_to_idx[seed])
        else:
            print(f"Warning: Seed {seed} not found in this graph.")
    if not valid_seed_indices:
        raise ValueError(
            "No valid seed nodes found in the graph. Check RWR implementation!!"
        )
    """

    # Building the matrix using the same order as nodelist
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, format="csr", dtype=float)

    """
    # Add self-loops to handle nodes with zero degree
    print(
        "#TODO: Currently I am adding self loops to nodes which don't have connections, in case they are a start. But I think they would not even be in the network anymore with the current implementation. So this might need fixing."
    )

    # Handle isolated nodes (degree 0)
    d_orig = A @ np.ones(n)
    # Identify the nodes with zero degree
    sink_nodes = np.where(d_orig == 0)[0]
    # Add a self loop to these nodes
    if len(sink_nodes) > 0:
        A_lil = A.tolil()
        A_lil[sink_nodes, sink_nodes] = 1.0
        A = A_lil.tocsr()
    """

    # Compute the transition probability matrix P
    # d[i] = sum of row i = degree of node i
    d = A @ np.ones(n)
    D_inv = csr_matrix(np.diag(1 / d))
    P = D_inv @ A

    # Initialize starting probability vector
    p0 = np.zeros(n)
    seed_indices = [nodelist.index(seed) for seed in seed_nodes]
    p0[seed_indices] = 1.0 / len(seed_indices)

    # Iterative RWR
    p = p0.copy()
    converged = False

    for i in range(max_iter):
        p_next = (1 - restart) * P.T @ p + restart * p0
        if np.linalg.norm(p_next - p, ord=1) < tol:
            converged = True
            break
        p = p_next

    """
    if not converged:
        print("Warning: RWR did not converge within the maximum number of iterations.")
    """

    # Create ranked dicitonary
    nodes_to_probs = dict(zip(nodelist, p))

    # Remove the seed nodes from the results
    for seed in seed_nodes:
        nodes_to_probs.pop(seed, None)

    # Sort by descending probability
    nodes_to_probs_sorted = dict(
        sorted(nodes_to_probs.items(), key=lambda item: item[1], reverse=True)
    )

    return ModuleResult(
        nodes_ranked=nodes_to_probs_sorted,
        algorithm_type="ranked",
        metadata={
            "converged": converged,
            "iterations": i + 1,
            "restart_prob": restart,
            "n_valid_seeds": len(seed_indices),
        },
    )
