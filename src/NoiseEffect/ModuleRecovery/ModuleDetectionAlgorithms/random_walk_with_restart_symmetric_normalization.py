import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from ..module_result import ModuleResult


def randomWalkWithRestartSymmetricNormalization(
    G: nx.Graph,
    seed_nodes: list[str],
    restart: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> ModuleResult:
    """
    Perform a random walk with restart (RWR) on graph G starting from a set of seed nodes.
    Normalization is done symmetrically. S = D^(-1/2) * A * D^(-1/2)

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

    # Building the matrix using the same order as nodelist
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, format="csr", dtype=float)

    # Compute symmetric normalization: S = D^(-1/2) * A * D^(-1/2)
    # Calculate degrees d
    d = A @ np.ones(n)

    # D^(-1/2)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = csr_matrix(np.diag(d_inv_sqrt))

    # S = D^(-1/2) * A * D^(-1/2)
    S = D_inv_sqrt @ A @ D_inv_sqrt

    # Initialize starting probability vector
    p0 = np.zeros(n)
    seed_indices = [nodelist.index(seed) for seed in seed_nodes]
    p0[seed_indices] = 1.0 / len(seed_indices)

    # Iterative RWR
    # For symmetric normalization, S can be used directly (not S.T)
    # because S is symmetric: S.T = S
    p = p0.copy()
    converged = False

    for i in range(max_iter):
        p_next = (1 - restart) * S @ p + restart * p0
        if np.linalg.norm(p_next - p, ord=1) < tol:
            converged = True
            break
        p = p_next

    # Create ranked dicitonary
    # Remember, this is now no loger a visting probability but a steady-state score
    nodes_to_scores = dict(zip(nodelist, p))

    # Remove the seed nodes from the results
    for seed in seed_nodes:
        nodes_to_scores.pop(seed, None)

    # Sort by descending scores
    nodes_to_scores_sorted = dict(
        sorted(nodes_to_scores.items(), key=lambda item: item[1], reverse=True)
    )

    return ModuleResult(
        nodes_ranked=nodes_to_scores_sorted,
        algorithm_type="ranked",
        metadata={
            "converged": converged,
            "iterations": i + 1,
            "restart_prob": restart,
            "n_valid_seeds": len(seed_nodes),
        },
    )
