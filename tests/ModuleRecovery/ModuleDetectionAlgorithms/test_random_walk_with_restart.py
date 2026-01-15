import pytest
import networkx as nx
import numpy as np

# Adjust this import to match your actual file structure
from NoiseEffect.ModuleRecovery.ModuleDetectionAlgorithms.random_walk_with_restart import (
    randomWalkWithRestart,
)


class TestRWRIndexAlignment:
    def test_rwr_disconnected_components(self):
        """
        CRITICAL TEST: Verifies that seeds are mapped to the correct matrix indices.

        Setup: Two disconnected components: (A-B) and (C-D).
        Action: Seed on 'A'.
        Expectation:
            - 'A' and 'B' have scores > 0.
            - 'C' and 'D' have score == 0.

        If the internal mapping is wrong (e.g., sorted vs unsorted mismatch),
        probability might 'teleport' to C or D.
        """
        # 1. Create a disconnected graph
        G = nx.Graph()
        G.add_edge("A", "B")  # Island 1
        G.add_edge("C", "D")  # Island 2 (The Decoy)

        # 2. Run RWR seeding ONLY on "A"
        # We assume string IDs here
        results = randomWalkWithRestart(G, seed_nodes=["A"], restart=0.7)

        print(f"\nRWR Results: {results}")

        # 3. Assertions
        # Island 1 should be "hot"
        assert results["A"] > 0, "Seed node A has 0 probability!"
        assert results["B"] > 0, "Neighbor B received no probability!"

        # Island 2 should be "cold" (Floating point tolerance check)
        assert results.get("C", 0.0) < 1e-9, (
            "Leakage detected! Node C (disconnected) got probability."
        )
        assert results.get("D", 0.0) < 1e-9, (
            "Leakage detected! Node D (disconnected) got probability."
        )

    def test_rwr_ordering_robustness(self):
        """
        Verifies that node input order doesn't break the mapping.
        We force 'Z' to be added first, but it is alphabetically last.
        """
        G = nx.Graph()
        # Add nodes in "weird" order
        G.add_edge("Z", "Y")
        G.add_edge("A", "B")

        # Seed on "Z" (which might be index 0 in insertion, but last in alpha sort)
        results = randomWalkWithRestart(G, seed_nodes=["Z"], restart=0.5)

        # Probability should stay on Z-Y edge
        assert results["Z"] > 0
        assert results["Y"] > 0
        assert results.get("A") < 1e-9

    def test_missing_seed_handling(self):
        """
        Ensures the function doesn't crash if a seed is missing
        (common in perturbed networks).
        """
        G = nx.Graph()
        G.add_edge("A", "B")

        # 'Ghost' does not exist in G
        try:
            results = randomWalkWithRestart(G, seed_nodes=["A", "Ghost"], restart=0.5)
        except Exception as e:
            pytest.fail(f"RWR crashed on missing seed: {e}")

        # Should still work for A
        assert results["A"] > 0

    def test_single_node_self_loop(self):
        """
        Edge case: A graph with 1 isolated node.
        RWR should handle the sink node logic (adding self-loop internally).
        """
        G = nx.Graph()
        G.add_node("Lonely")

        results = randomWalkWithRestart(G, seed_nodes=["Lonely"], restart=0.5)

        # With restart, it should just stay on itself (prob = 1.0)
        assert np.isclose(results["Lonely"], 1.0), (
            "Single node probability should be 1.0"
        )
