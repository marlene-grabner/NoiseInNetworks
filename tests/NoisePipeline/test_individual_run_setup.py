import pytest
from unittest.mock import patch, MagicMock

# The class we want to test
from NoiseEffect.NoisePipeline.individual_run_setup import IndividualAnalysis


# Sample data for testing
@pytest.fixture
def sample_network_request():
    """Provides a sample network request dictionary for tests."""
    return {"type": "barabasi_albert", "nodes": 100, "m": 2, "instance": 1}


@pytest.fixture
def sample_noise_information():
    """Provides a sample noise information dictionary for tests."""
    return {"noise_levels": [0.1], "num_repeats": 1}


@pytest.fixture
def list_of_seeds():
    """Provides a sample list of random seeds for tests."""
    return [42, 7, 99]


def test_create_identifier(
    sample_network_request, sample_noise_information, list_of_seeds
):
    """
    Tests if the unique identifier is created correctly based on the network request.
    """
    analysis = IndividualAnalysis(
        sample_network_request, sample_noise_information, list_of_seeds
    )
    # The implementation sorts parameters by key, ensuring a stable identifier.
    expected_identifier = "barabasi_albert_m=2_nodes=100_1"
    assert analysis.identifier == expected_identifier


@patch("NoiseEffect.NoisePipeline.individual_run_setup.NoisyNetworkRecovery")
@patch("NoiseEffect.NoisePipeline.individual_run_setup.PerturbedEdges")
@patch("NoiseEffect.NoisePipeline.individual_run_setup.OriginalNetwork")
def test_run_orchestration(
    mock_original_network_class,
    mock_perturbed_edges_class,
    mock_noisy_recovery_class,
    sample_network_request,
    sample_noise_information,
    list_of_seeds,
):
    """
    Tests that the run() method correctly orchestrates the analysis by calling
    the right methods on its dependencies in the correct order. This is a unit
    test that mocks all external dependencies.
    """
    # --- 1. Setup Mocks ---
    # Configure the mock classes to return a mock instance when they are called.
    mock_original_network_instance = MagicMock()
    mock_original_network_class.return_value = mock_original_network_instance

    mock_perturbed_edges_instance = MagicMock()
    mock_perturbed_edges_class.return_value = mock_perturbed_edges_instance

    mock_noisy_recovery_instance = MagicMock()
    # Define a sample result that our mock recovery object will "produce".
    mock_noisy_recovery_instance.recovery_results = {"accuracy": 0.9}
    mock_noisy_recovery_class.return_value = mock_noisy_recovery_instance

    # --- 2. Run the Code Under Test ---
    # Instantiate the class we are testing
    analysis = IndividualAnalysis(
        sample_network_request, sample_noise_information, list_of_seeds
    )

    # Call the main method that orchestrates the workflow
    analysis.run()

    # --- 3. Assertions ---
    # Verify that each step of the orchestration happened as expected.

    # Check that OriginalNetwork was instantiated correctly and its method was called.
    mock_original_network_class.assert_called_once_with(
        sample_network_request, list_of_seeds
    )
    mock_original_network_instance.generateEvaluationBaseline.assert_called_once()

    # Check that PerturbedEdges was instantiated with the correct network and noise info.
    mock_perturbed_edges_class.assert_called_once_with(
        mock_original_network_instance.original_network_nx, sample_noise_information
    )
    mock_perturbed_edges_instance.generateNoisyNetworkSets.assert_called_once()

    # Check that NoisyNetworkRecovery was instantiated with the correct objects.
    mock_noisy_recovery_class.assert_called_once_with(
        original_network_obj=mock_original_network_instance,
        noisy_network_sets_obj=mock_perturbed_edges_instance,
    )
    mock_noisy_recovery_instance.recoverOriginalModules.assert_called_once()

    # Finally, check that the results from the recovery step were stored correctly.
    assert analysis.results == {"accuracy": 0.9}
