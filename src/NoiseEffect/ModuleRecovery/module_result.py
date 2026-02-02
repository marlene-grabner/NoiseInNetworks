from dataclasses import dataclass
from typing import Dict, Set, List, Optional
import numpy as np
import logging

# Create the logging channel for this file
logger = logging.getLogger(__name__)


# Currently all algorithms should not return the seeds
@dataclass
class ModuleResult:
    """
    Standardized output format for all module detection algorithms.
    """

    nodes_ranked: Optional[Dict[str, float]] = None  # For RWR, DIAMOnD, ROBUST
    nodes_set: Optional[Set[str]] = None  # For DOMINO, 1st Neighbors
    algorithm_type: str = "ranked"  # 'ranked' or 'set'
    metadata: Optional[Dict] = None

    def get_top_k(self, k: int) -> Set[str]:
        """Extract top-k nodes regardless of algorithm type."""
        if self.algorithm_type == "ranked":
            return set(list(self.nodes_ranked.keys())[:k])
        else:
            # For set-based, just return the set (can't subset without ranking)
            return self.nodes_set

    def as_set(self) -> Set[str]:
        """Convert to set regardless of type."""
        if self.algorithm_type == "ranked":
            return set(self.nodes_ranked.keys())
        else:
            return self.nodes_set

    def size(self) -> int:
        """Get number of nodes in the module."""
        if self.algorithm_type == "ranked":
            return len(self.nodes_ranked)
        else:
            return len(self.nodes_set)
