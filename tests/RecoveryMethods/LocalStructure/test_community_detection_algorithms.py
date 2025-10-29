import pytest
import igraph as ig
from NoiseEffect.RecoveryMethods.LocalStructure.community_detection_algorithms import (
    CommunityDetectionAlgorithms,
)  # Import your class


@pytest.fixture
def barbell_graph():
    """
    Creates a simple igraph "barbell" graph fixture.
    This graph has two obvious communities: {0, 1, 2} and {3, 4, 5}.

    NOTE: This is now much simpler. It no longer needs to deal with
    string names or the '_nx_name' attribute.
    """
    g = ig.Graph()
    g.add_vertices(6)  # Add 6 vertices, with IDs 0 through 5.

    # Edges are defined directly with integer vertex IDs.
    edges = [
        (0, 1),
        (0, 2),
        (1, 2),  # First cluster
        (3, 4),
        (3, 5),
        (4, 5),  # Second cluster
        (2, 3),  # Bridge connecting them
    ]
    g.add_edges(edges)
    return g


def test_leidenAlgorithmPartioning(barbell_graph):
    """
    Tests that the Leiden algorithm partitioner finds one of the expected communities.
    """
    # 1. SETUP is done by the barbell_graph fixture.

    # 2. EXECUTE the function we are testing.
    # Note: We need to access the static method via the class.
    found_community = CommunityDetectionAlgorithms.leidenAlgorithmPartioning(
        barbell_graph
    )

    # 3. ASSERT the result is what we expect.
    expected_community_1 = {0, 1, 2}
    expected_community_2 = {3, 4, 5}

    # Because your function returns only the FIRST community ([0]), and we can't
    # guarantee which one that will be, we check if the result is one of the possibilities.
    assert found_community in [expected_community_1, expected_community_2]

    # Let's also check that it's NOT the other one (a sanity check)
    if found_community == expected_community_1:
        assert found_community != expected_community_2
    else:
        assert found_community == expected_community_2


def test_infomapAlgorithmPartioning(barbell_graph):
    """
    Tests that the Infomap algorithm partitioner finds one of the expected communities.
    """
    # 1. SETUP is done by the barbell_graph fixture.

    # 2. EXECUTE the function we are testing.
    # Note: We need to access the static method via the class.
    found_community = CommunityDetectionAlgorithms.infomapAlgorithmPartioning(
        barbell_graph
    )
    print(found_community)

    # 3. ASSERT the result is what we expect.
    expected_community_1 = {0, 1, 2}
    expected_community_2 = {3, 4, 5}

    # Because your function returns only the FIRST community ([0]), and we can't
    # guarantee which one that will be, we check if the result is one of the possibilities.
    assert found_community in [expected_community_1, expected_community_2]

    # Let's also check that it's NOT the other one (a sanity check)
    if found_community == expected_community_1:
        assert found_community != expected_community_2
    else:
        assert found_community == expected_community_2


###### ONCE I AM RETURNING ALL COMMUNITIES, I CAN USE THIS TEST ######
'''
def test_find_communities_leiden(barbell_graph):
    """
    Tests that the refactored Leiden function finds the correct full partition
    of integer vertex IDs.
    """
    # 1. ARRANGE: The fixture provides the graph.
    # The expected result is now the full partition of integer IDs.
    expected_partition = [{0, 1, 2}, {3, 4, 5}]

    # 2. ACT: Call the new, refactored function.
    found_partition = community_algorithms.find_communities_leiden(barbell_graph)

    # 3. ASSERT: Check if the found partition matches the expected one.
    #
    # IMPORTANT: The order of communities in the result list is not guaranteed.
    # We can't just do `assert found_partition == expected_partition`.
    # To compare them regardless of order, we sort the elements within each set,
    # and then sort the outer list to create a stable, canonical representation.
    
    sorted_found = sorted([sorted(list(c)) for c in found_partition])
    sorted_expected = sorted([sorted(list(c)) for c in expected_partition])

    assert sorted_found == sorted_expected


def test_find_communities_infomap(barbell_graph):
    """
    Tests that the refactored Infomap function finds the correct full partition
    of integer vertex IDs.
    """
    # ARRANGE
    expected_partition = [{0, 1, 2}, {3, 4, 5}]

    # ACT
    found_partition = community_algorithms.find_communities_infomap(barbell_graph)

    # ASSERT (using the same order-independent comparison method)
    sorted_found = sorted([sorted(list(c)) for c in found_partition])
    sorted_expected = sorted([sorted(list(c)) for c in expected_partition])

    assert sorted_found == sorted_expected
'''
