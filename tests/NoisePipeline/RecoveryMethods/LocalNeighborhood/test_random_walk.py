import pytest
import networkx as nx
import numpy as np

from NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.random_walk import (
    randomWalkWithRestart,
)

# --- Fixture: A reusable graph for testing ---


@pytest.fixture
def sample_graph():
    """
    Creates a simple 4-node graph:
    - Nodes 0, 1, 2 form a path
    - Node 3 is completely isolated
    """
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    G.add_node(3)  # Isolated node
    return G


# --- Test Cases ---


def test_rwr_normalization(sample_graph):
    """
    Test 1: Check if probabilities sum to 1.
    """
    seed_nodes = [0]
    result = randomWalkWithRestart(sample_graph, seed_nodes)

    assert result is not None, "Function returned None"
    assert len(result) == 4, "Result dictionary doesn't have 4 nodes"

    total_prob = sum(result.values())
    assert np.isclose(total_prob, 1.0), "Probabilities do not sum to 1.0"


def test_rwr_isolated_seed_node(sample_graph):
    """
    Test 2: This is the specific test for your bug!
    Seed the walk on the isolated node.
    """
    seed_nodes = [3]  # Start on the isolated node
    result = randomWalkWithRestart(sample_graph, seed_nodes)

    # The RWR on an isolated node should converge to p = p0
    # So, p = [0, 0, 0, 1.0]
    assert np.isclose(result[3], 1.0), "Isolated seed node score is not 1.0"
    assert np.isclose(result[0], 0.0), "Non-seed node score is not 0.0"
    assert np.isclose(result[1], 0.0), "Non-seed node score is not 0.0"
    assert np.isclose(result[2], 0.0), "Non-seed node score is not 0.0"


def test_rwr_multiple_seeds_symmetry(sample_graph):
    """
    Test 3: Seed on nodes 0 and 2.
    Due to symmetry, their scores should be equal.
    """
    seed_nodes = [0, 2]
    result = randomWalkWithRestart(sample_graph, seed_nodes)

    # Check normalization
    assert np.isclose(sum(result.values()), 1.0)

    # Check that the isolated node gets no score
    assert np.isclose(result[3], 0.0)

    # Check that seeds (0, 2) have highest scores
    assert result[0] > result[1]
    assert result[2] > result[1]

    # Check symmetry
    assert np.isclose(result[0], result[2]), "Symmetric nodes do not have equal scores"


def test_rwr_math_on_two_nodes():
    """
    Test 4: Manually check the math on a simple 2-node graph.
    p_k+1 = (1-c) * P.T @ p_k + c * p0
    """
    G = nx.Graph()
    G.add_edge(0, 1)
    seed_nodes = [0]
    restart = 0.5  # Use c = 0.5 for simple math

    # p0 = [1, 0]
    # D = [[1, 0], [0, 1]] -> D_inv = [[1, 0], [0, 1]]
    # A = [[0, 1], [1, 0]] -> P = D_inv @ A = [[0, 1], [1, 0]]
    # P.T = P

    # Steady state p = [a, b]:
    # [a, b] = (0.5) * [[0, 1], [1, 0]] @ [a, b] + (0.5) * [1, 0]
    # [a, b] = [0.5b, 0.5a] + [0.5, 0]
    #
    # Eq 1: a = 0.5b + 0.5
    # Eq 2: b = 0.5a
    #
    # Sub (2) into (1):
    # a = 0.5(0.5a) + 0.5
    # a = 0.25a + 0.5
    # 0.75a = 0.5  => a = 0.5 / 0.75 = 2/3
    # b = 0.5 * (2/3) = 1/3

    expected_a = 2 / 3
    expected_b = 1 / 3

    result = randomWalkWithRestart(G, seed_nodes, restart=restart)

    assert np.isclose(result[0], expected_a)
    assert np.isclose(result[1], expected_b)
