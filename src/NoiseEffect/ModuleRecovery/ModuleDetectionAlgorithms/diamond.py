import scipy.stats
import scipy.special
import numpy as np
import networkx as nx
from collections import defaultdict
from ..module_result import ModuleResult
import logging

logger = logging.getLogger(__name__)

###########################################################################
# DIAMOnD Algorithm Implementation
# Reference: Ghiassian SD, Menche J, Barabási AL. A
# A DIseAse MOdule Detection (DIAMOnD) Algorithm Derived from a Systematic Analysis of Connectivity Patterns of Disease Proteins in the Human Interactome. PLoS Comput Biol. 2015 Apr 8;11(4):e1004120.
#
# This implementation is adapted from: https://github.com/dinaghiassian/DIAMOnD/tree/master
###########################################################################


# ================================================================================
# Core Math Functions (unchanged logic, fixed warnings)
# ================================================================================


def compute_all_gamma_ln(N):
    """
    precomputes all logarithmic gammas
    """
    gamma_ln = {}
    for i in range(1, N + 1):
        gamma_ln[i] = scipy.special.gammaln(i)
    return gamma_ln


def logchoose(n, k, gamma_ln):
    if n - k + 1 <= 0:
        return -np.inf  # Fixed: returns negative infinity for log prob
    lgn1 = gamma_ln[n + 1]
    lgk1 = gamma_ln[k + 1]
    lgnk1 = gamma_ln[n - k + 1]
    return lgn1 - (lgnk1 + lgk1)  # Fixed: bracket syntax error in original


def gauss_hypergeom(x, r, b, n, gamma_ln):
    return np.exp(
        logchoose(r, x, gamma_ln)
        + logchoose(b, n - x, gamma_ln)
        - logchoose(r + b, n, gamma_ln)
    )


def pvalue(kb, k, N, s, gamma_ln):
    r"""
    -------------------------------------------------------------------
    Computes the p-value for a node that has kb out of k links to
    seeds, given that there's a total of s sees in a network of N nodes.

    p-val = \sum_{n=kb}^{k} HypergemetricPDF(n,k,N,s)
    -------------------------------------------------------------------
    """
    p = 0.0
    for n in range(kb, k + 1):
        if n > s:
            break
        prob = gauss_hypergeom(n, s, N - s, k, gamma_ln)
        p += prob
    return min(p, 1.0)  # Ensure p <= 1


def get_neighbors_and_degrees(G):
    neighbors = {}
    all_degrees = {}
    for node in G.nodes():
        neighbors[node] = set(G.neighbors(node))
        all_degrees[node] = G.degree(node)
    return neighbors, all_degrees


def reduce_not_in_cluster_nodes(
    all_degrees, neighbors, G, not_in_cluster, cluster_nodes, alpha
):
    reduced_not_in_cluster = {}
    kb2k = defaultdict(dict)

    for node in not_in_cluster:
        k = all_degrees[node]
        kb = 0
        # Going through all neighbors and counting the number of module neighbors
        for neighbor in neighbors[node]:
            if neighbor in cluster_nodes:
                kb += 1

        # adding wights to the the edges connected to seeds
        k += (alpha - 1) * kb
        kb += (alpha - 1) * kb
        kb2k[kb][k] = node

    # Going to choose the node with largest kb, given k
    k2kb = defaultdict(dict)
    for kb, k2node in kb2k.items():
        min_k = min(k2node.keys())
        node = k2node[min_k]
        k2kb[min_k][kb] = node

    for k, kb2node in k2kb.items():
        max_kb = max(kb2node.keys())
        node = kb2node[max_kb]
        reduced_not_in_cluster[node] = (max_kb, k)

    return reduced_not_in_cluster


# ======================================================================================
#  Main Algorithm
# ======================================================================================
def diamond(G, S, X=200, alpha=1):
    """
    Directly accepts a NetworkX graph and a set/list of seed genes.
    Returns a list of added nodes, and their corresponding p-values.


    Parameters:
    ----------
    - G:     graph
    - S:     seeds
    - X:     the number of iterations, i.e only the first X gened will be
             pulled in
    - alpha: seeds weight

    Returns:
    --------

    - added_nodes: ordered list of nodes in the order by which they
      are agglomerated. Each entry has 4 info:
      * name : dito
      * k    : degree of the node
      * kb   : number of +1 neighbors
      * p    : p-value at agglomeration
    """

    #  Modified: Prune seeds not in network
    all_genes = set(G.nodes())
    S_orig = len(S)
    S = set(S) & all_genes
    if len(S) < S_orig:
        print(
            f"DIAMOnD found {S_orig - len(S)} seed genes that are not in the network and will be ignored."
        )

    if len(S) == 0:
        print("DIAMOnD Warning: No seed genes found in network.")
        return []

    N = G.number_of_nodes()
    added_nodes_data = []  # Modified: Store the node name, and its p-value

    # ------------------------------------------------------------------
    # Setting up dictionaries with all neighbor lists
    # and all degrees
    # ------------------------------------------------------------------
    neighbors, all_degrees = get_neighbors_and_degrees(G)

    # ------------------------------------------------------------------
    # Setting up initial set of nodes in cluster
    # ------------------------------------------------------------------
    cluster_nodes = set(S)
    not_in_cluster = set()
    s0 = len(cluster_nodes)

    # Weight adjustments
    s0 += (alpha - 1) * s0
    N += (alpha - 1) * s0

    # ------------------------------------------------------------------
    # precompute the logarithmic gamma functions
    # ------------------------------------------------------------------
    gamma_ln = compute_all_gamma_ln(N + 1)

    # ------------------------------------------------------------------
    # Setting initial set of nodes not in cluster
    # ------------------------------------------------------------------
    for node in cluster_nodes:
        not_in_cluster |= neighbors[node]
    not_in_cluster -= cluster_nodes

    # ------------------------------------------------------------------
    #
    # M A I N     L O O P
    #
    # ------------------------------------------------------------------

    all_p = {}

    logger.info(f"Starting DIAMOnD with {len(S)} seeds, aiming to add {X} nodes...")

    while len(added_nodes_data) < X:
        # ------------------------------------------------------------------
        #
        # Going through all nodes that are not in the cluster yet and
        # record k, kb and p
        #
        # ------------------------------------------------------------------
        if not not_in_cluster:
            break

        info = {}
        pmin = 10
        next_node = None  # Modified: Change to pythonic way of saying nothing

        reduced_not_in_cluster = reduce_not_in_cluster_nodes(
            all_degrees, neighbors, G, not_in_cluster, cluster_nodes, alpha
        )

        for node, kbk in reduced_not_in_cluster.items():
            # Getting the p-value of this kb,k
            # combination and save it in all_p, so computing it only once!
            kb, k = kbk

            # Memoization of p-values
            if (
                (k, kb, s0) in all_p
            ):  # Modified: Checking if already calcualted, instead of try except logic
                p = all_p[(k, kb, s0)]
            else:
                p = pvalue(kb, k, N, s0, gamma_ln)
                all_p[(k, kb, s0)] = p

            # recording the node with smallest p-value
            if p < pmin:
                pmin = p
                next_node = node

            info[node] = (k, kb, p)

        if (
            next_node is None
        ):  # Modified: If no valid node is found, the loop is stopped
            break

        # ---------------------------------------------------------------------
        # Adding node with smallest p-value to the list of aaglomerated nodes
        # ---------------------------------------------------------------------
        added_nodes_data.append((next_node, pmin))

        # Updating the list of cluster nodes and s0
        cluster_nodes.add(next_node)
        s0 = len(cluster_nodes)
        not_in_cluster |= neighbors[next_node] - cluster_nodes
        not_in_cluster.remove(next_node)

    return ModuleResult(
        nodes_diamond=added_nodes_data,
        algorithm_type="diamond",
        metadata={"algorithm": "DIAMOnD", "alpha": alpha, "n_requested": X},
    )
