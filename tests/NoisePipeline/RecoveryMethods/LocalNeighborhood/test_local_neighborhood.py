import numpy as np
import numpy.testing as npt
from unittest.mock import patch, MagicMock
from NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.local_neighborhood import (
    _rwrResultsToVectors,
    _calculateTop50JaccardSimilarity,
    _calculateSimilarityMetrics,
    localNeighborhoodAnalysis,
)


def test_rwrResultsToVectors_basic():
    """
    Tests that the dictionary is correctly converted to a vector.
    """
    p_dict = {
        0: 0.5,
        2: 0.25,
        # Note: Node 1 is missing
    }
    n = 4

    expected_vector = np.array([0.5, 0.0, 0.25, 0.0])

    result_vector = _rwrResultsToVectors(p_dict, n)

    npt.assert_array_equal(result_vector, expected_vector)


def test_rwrResultsToVectors_empty():
    """
    Tests that an empty dictionary results in a zero vector.
    """
    p_dict = {}
    n = 3

    expected_vector = np.array([0.0, 0.0, 0.0])

    result_vector = _rwrResultsToVectors(p_dict, n)

    npt.assert_array_equal(result_vector, expected_vector)


def test_top50_jaccard_perfect_overlap():
    """
    Tests the Jaccard similarity when top 50 are identical.
    """
    # Create dicts with 60 nodes; nodes 10-59 are the top 50
    p1 = {i: i for i in range(60)}
    p2 = {i: i for i in range(60)}

    result = _calculateTop50JaccardSimilarity(p1, p2)

    assert result == 1.0


def test_top50_jaccard_no_overlap():
    """
    Tests the Jaccard similarity when top 50 are disjoint.
    """
    # p1: top 50 are nodes 50-99
    p1 = {i: i for i in range(100)}
    # p2: top 50 are nodes 0-49
    p2 = {i: (100 - i) for i in range(100)}

    result = _calculateTop50JaccardSimilarity(p1, p2)

    assert result == 0.0


def test_top50_jaccard_partial_overlap_and_small_graph():
    """
    Tests partial overlap on a graph with fewer than 50 nodes.
    """
    # The 'top 50' slice will just be all nodes
    p1 = {"a": 3, "b": 2, "c": 1}  # Top set: {'a', 'b', 'c'}
    p2 = {"b": 3, "c": 2, "d": 1}  # Top set: {'b', 'c', 'd'}

    # Intersection: {'b', 'c'} (size 2)
    # Union: {'a', 'b', 'c', 'd'} (size 4)
    expected_jaccard = 2 / 4

    result = _calculateTop50JaccardSimilarity(p1, p2)

    assert result == expected_jaccard


@patch(
    "NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.local_neighborhood._rwrResultsToVectors"
)
@patch(
    "NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.local_neighborhood.jensenshannon"
)
@patch(
    "NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.local_neighborhood.spearmanr"
)
@patch(
    "NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.local_neighborhood._calculateTop50JaccardSimilarity"
)
@patch(
    "NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.local_neighborhood.np.linalg.norm"
)
def test_calculateSimilarityMetrics_orchestration(
    mock_rwr_to_vectors,
    mock_jensenshannon,
    mock_spearmanr,
    mock_top50_jaccard,
    mock_l2_norm,
    mock_np_delete,  # <-- Add the new mock argument
):
    """
    Tests that _calculateSimilarityMetrics correctly calls its
    dependencies and assembles their results.
    """
    # 1. Setup Inputs
    orig_hood = {"a": 0.1, "b": 0.9}
    new_hood = {"a": 0.2, "b": 0.8}
    seed_nodes = [0]  # <-- Define the new argument
    num_nodes = 2

    # 2. Setup Mock Return Values

    # Mock the *original* vectors from _rwrResultsToVectors
    mock_p_vec_orig = np.array([0.1, 0.9])
    mock_q_vec_orig = np.array([0.2, 0.8])
    mock_rwr_to_vectors.side_effect = [mock_p_vec_orig, mock_q_vec_orig]

    # Mock the *deleted* vectors returned by np.delete
    mock_p_vec_deleted = np.array([0.9])  # Only node 'b' remains
    mock_q_vec_deleted = np.array([0.8])  # Only node 'b' remains
    mock_np_delete.side_effect = [mock_p_vec_deleted, mock_q_vec_deleted]

    # Mock the final similarity results
    mock_jensenshannon.return_value = 0.5
    mock_spearmanr.return_value = (0.8, 0.05)
    mock_top50_jaccard.return_value = 0.75
    mock_l2_norm.return_value = 0.99

    # 3. Run the Function
    # Call with all 4 arguments
    result = _calculateSimilarityMetrics(orig_hood, new_hood, seed_nodes, num_nodes)

    # 4. Assertions

    # Assert _rwrResultsToVectors was called correctly
    assert mock_rwr_to_vectors.call_count == 2
    mock_rwr_to_vectors.assert_any_call(p=orig_hood, n=num_nodes)
    mock_rwr_to_vectors.assert_any_call(p=new_hood, n=num_nodes)

    # Assert np.delete was called on the *original* vectors
    assert mock_np_delete.call_count == 2
    # We use ANY or np.array_equal because comparing numpy arrays
    # with '==' is ambiguous. ANY is simplest here.
    mock_np_delete.assert_any_call(ANY, seed_nodes)

    # Assert JSD and L2 were called with the *renormalized* vectors.
    # After masking [0.1, 0.9] -> [0.0, 0.9] -> renorm -> [0.0, 1.0]
    # After masking [0.2, 0.8] -> [0.0, 0.8] -> renorm -> [0.0, 1.0]
    renorm_p = np.array([0.0, 1.0])
    renorm_q = np.array([0.0, 1.0])

    mock_jensenshannon.assert_called_once()
    assert np.array_equal(mock_jensenshannon.call_args[0][0], renorm_p)
    assert np.array_equal(mock_jensenshannon.call_args[0][1], renorm_q)

    mock_l2_norm.assert_called_once()
    assert np.array_equal(mock_l2_norm.call_args[0][0], renorm_p - renorm_q)

    # Assert spearmanr was called with the *deleted* vectors
    mock_spearmanr.assert_called_once()
    assert np.array_equal(mock_spearmanr.call_args[0][0], mock_p_vec_deleted)
    assert np.array_equal(mock_spearmanr.call_args[0][1], mock_q_vec_deleted)

    # Assert internal helper was called with the dicts AND seeds
    mock_top50_jaccard.assert_called_once_with(
        orig_hood,
        new_hood,
        seed_nodes,  # <-- Check for 3rd arg
    )

    # Assert the final dictionary is assembled correctly
    expected_result = {
        "jsd": 0.5,
        "spearman": (0.8, 0.05),
        "top_50_jaccard": 0.75,
        "l2_norm": 0.99,
    }
    assert result == expected_result


@patch(
    "NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.local_neighborhood.randomWalkWithRestart"
)
@patch(
    "NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.local_neighborhood._calculateSimilarityMetrics"
)
@patch(
    "NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.local_neighborhood.ast.literal_eval"
)
def test_localNeighborhoodAnalysis_orchestration(
    mock_literal_eval, mock_calculate_similarity, mock_rwr
):
    """
    Tests that localNeighborhoodAnalysis loops correctly, calls RWR
    and similarity metrics for each seed.
    """
    # 1. Setup Inputs
    mock_network = MagicMock()
    mock_network.number_of_nodes.return_value = 100
    original_neighborhood = {
        "[0]": {"a": 0.1},
        "[1, 2]": {"b": 0.2},
    }

    # 2. Setup Mock Return Values

    # --- FIX: Define seed lists as variables ---
    seeds_1 = [0]
    seeds_2 = [1, 2]
    # --- End Fix ---

    # Mock ast.literal_eval using the variables
    mock_literal_eval.side_effect = [seeds_1, seeds_2]

    mock_rwr_result_1 = {"new_a": 0.5}
    mock_rwr_result_2 = {"new_b": 0.6}
    mock_rwr.side_effect = [mock_rwr_result_1, mock_rwr_result_2]

    mock_sim_result_1 = {"jsd": 0.1}
    mock_sim_result_2 = {"jsd": 0.2}
    mock_calculate_similarity.side_effect = [mock_sim_result_1, mock_sim_result_2]

    # 3. Run the Function
    result = localNeighborhoodAnalysis(mock_network, original_neighborhood)

    # 4. Assertions
    assert mock_literal_eval.call_count == 2
    mock_literal_eval.assert_any_call("[0]")
    mock_literal_eval.assert_any_call("[1, 2]")

    assert mock_rwr.call_count == 2
    mock_rwr.assert_any_call(G=mock_network, seed_nodes=seeds_1)
    mock_rwr.assert_any_call(G=mock_network, seed_nodes=seeds_2)

    assert mock_calculate_similarity.call_count == 2

    # --- FIX: Use the variables in the assertion ---
    mock_calculate_similarity.assert_any_call(
        original_neighborhood["[0]"],
        mock_rwr_result_1,
        seeds_1,  # Use the variable
        100,
    )
    mock_calculate_similarity.assert_any_call(
        original_neighborhood["[1, 2]"],
        mock_rwr_result_2,
        seeds_2,  # Use the variable
        100,
    )
    # --- End Fix ---

    expected_result = {"[0]": mock_sim_result_1, "[1, 2]": mock_sim_result_2}
    assert result == expected_result
