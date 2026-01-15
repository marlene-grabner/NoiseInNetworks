import pytest
import networkx as nx
import igraph as ig
from unittest.mock import patch, MagicMock

from NoiseEffect.NoisePipeline.baseline import OriginalNetwork


@pytest.fixture
def ba_network_request():
    """Provides a sample request for a Barabasi-Albert graph."""
    return {"type": "barabasi_albert", "nodes": 50, "m": 2}


@pytest.fixture
def random_seed_list():
    """Provides a sample list of random seeds."""
    return [42, 7, 99]


@pytest.fixture
def personal_network_request(tmp_path):
    """Creates a temporary edgelist file and provides a request for it."""
    edgelist_content = "A B\nA C\nB C\nD E\nD F\nE F"
    edgelist_file = tmp_path / "test_edgelist.txt"
    edgelist_file.write_text(edgelist_content)
    return {"type": "personal_network", "path": str(edgelist_file)}


def test_init(ba_network_request, random_seed_list):
    """Tests that the OriginalNetwork class is initialized correctly."""
    network = OriginalNetwork(ba_network_request, random_seed_list)
    assert network.network_request == ba_network_request
    assert network.random_seed_list == random_seed_list
    assert network.original_network_nx is None
    assert network.original_communities == {}
    assert network.original_num_communities == {}


def test_create_barabasi_albert(ba_network_request, random_seed_list):
    """Tests the creation of a Barabasi-Albert graph."""
    network = OriginalNetwork(ba_network_request, random_seed_list)
    network.createOriginalNetwork()
    assert isinstance(network.original_network_nx, nx.Graph)
    assert network.original_network_nx.number_of_nodes() == 50
    assert isinstance(network.original_network_ig, ig.Graph)


def test_create_personal_network(personal_network_request, random_seed_list):
    """Tests creating a graph from a personal edgelist file."""
    network = OriginalNetwork(personal_network_request, random_seed_list)
    network.createOriginalNetwork()
    assert isinstance(network.original_network_nx, nx.Graph)
    assert network.original_network_nx.number_of_nodes() == 6
    # Check that nodes were correctly mapped to integers
    assert set(network.original_network_nx.nodes()) == set(range(6))
    # Check that the mapping is stored correctly
    assert "A" in network.node_to_idx
    assert network.idx_to_node[network.node_to_idx["A"]] == "A"


def test_create_unknown_network_raises_error(random_seed_list):
    """Tests that an unknown network type raises a ValueError."""
    request = {"type": "non_existent_type"}
    network = OriginalNetwork(request, random_seed_list)
    with pytest.raises(ValueError, match="Unknown network type: non_existent_type"):
        network.createOriginalNetwork()


@patch("NoiseEffect.NoisePipeline.baseline.CommunityDetectionAlgorithms")
def test_get_baseline_community_structure(mock_convert_to_igraph, mock_community_algos):
    """
    Tests the getBaselineCommunityStructure method's logic by mocking
    all its external dependencies.
    """
    # --- 1. Setup Mocks (The "Arrange") ---

    # Create a specific mock object for the igraph
    # This lets us check that *this exact object* is passed around.
    mock_ig_graph = MagicMock(spec=ig.Graph)
    mock_convert_to_igraph.return_value = mock_ig_graph

    # Define the mock return values for the community detection
    mock_leiden_result = [{0, 1}, {2, 3}]
    mock_infomap_result = [{0}, {1, 2, 3}]
    mock_community_algos.leidenAlgorithmPartioning.return_value = mock_leiden_result
    mock_community_algos.infomapAlgorithmPartioning.return_value = mock_infomap_result

    # --- 2. Setup Test Subject (The "Act" part 1) ---

    # Instantiate the class, bypassing the complex __init__
    # We pass None to satisfy the __init__ signature without doing real work.
    network = OriginalNetwork(network_request=None, random_seed_list=None)

    # Manually create the *input* graph
    # This is the only "Given" state our method needs
    g_nx = nx.Graph()
    g_nx.add_edges_from([(0, 1), (2, 3)])
    network.original_network_nx = g_nx

    # Manually create the *output* dictionaries (which __init__ normally does)
    network.original_communities = {}
    network.original_num_communities = {}  # As in your example

    # --- 3. Run Code (The "Act" part 2) ---
    network.getBaselineCommunityStructure()

    # --- 4. Assertions (The "Assert") ---

    # Assert that our conversion mock was called *once* with the nx graph
    mock_convert_to_igraph.assert_called_once_with(g_nx)

    # Assert that the *output* of the conversion (our mock_ig_graph)
    # was correctly assigned to the instance attribute
    assert network.original_network_ig is mock_ig_graph

    # Assert that the community algorithms were called *once*
    # with the mock_ig_graph
    mock_community_algos.leidenAlgorithmPartioning.assert_called_once_with(
        mock_ig_graph
    )
    mock_community_algos.infomapAlgorithmPartioning.assert_called_once_with(
        mock_ig_graph
    )

    # Assert that the *return values* from the community mocks
    # were stored in the correct attributes
    assert network.original_communities["leiden_algorithm"] == mock_leiden_result
    assert network.original_num_communities["infomap_algorithm"] == mock_infomap_result


def test_get_baseline_no_edges_raises_error(ba_network_request, random_seed_list):
    """
    Tests that getBaselineCommunityStructure raises a ValueError for a graph with no edges.
    """
    network = OriginalNetwork(ba_network_request, random_seed_list)
    # Create a graph with nodes but no edges
    network.original_network_nx = nx.Graph()
    network.original_network_nx.add_nodes_from(range(10))

    with pytest.raises(
        ValueError, match="Network without edges can not have communities."
    ):
        network.getBaselineCommunityStructure()
