from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx


def randomWalkWithRestart(G, seed_nodes, restart=0.85, tol=1e-6, max_iter=1000):
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
    nodelist = list(G.nodes())
    n = len(nodelist)

    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    d_orig = A @ np.ones(n)
    # Identify the nodes with zero degree
    sink_nodes = np.where(d_orig == 0)[0]
    # Add a self loop to these nodes
    if len(sink_nodes) > 0:
        A_lil = A.tolil()
        A_lil[sink_nodes, sink_nodes] = 1.0
        A = A_lil.tocsr()

    # Recompute degrees after adding self-loops
    d = A @ np.ones(n)

    D_inv = csr_matrix(np.diag(1 / d))
    P = D_inv @ A

    # Initialize starting probability vector
    p0 = np.zeros(n)
    p0[seed_nodes] = 1.0 / len(seed_nodes)

    p = p0.copy()
    for _ in range(max_iter):
        p_next = (1 - restart) * P.T @ p + restart * p0
        if np.linalg.norm(p_next - p, ord=1) < tol:
            break
        p = p_next

        if _ == max_iter - 1:
            print(
                "Warning: RWR did not converge within the maximum number of iterations."
            )

    nodes_to_probs = dict(zip(nodelist, p))
    nodes_to_probs_sorted = dict(
        sorted(nodes_to_probs.items(), key=lambda item: item[1], reverse=True)
    )
    return nodes_to_probs_sorted
