import networkx as nx
import random
import os
import numpy as np
from NoiseEffect.baseline import OriginalNetwork
from NoiseEffect.perturbation import PerturbedEdges
from NoiseEffect.recovery import NoisyNetworkRecovery


class IndividualAnalysis:
    def __init__(self, network_request, noise_information, random_seed_list):
        # Initialize with the following parameters
        self.network_request = network_request
        self.noise_information = noise_information
        self.random_seed_list = random_seed_list

        # Following attributes house subclasses and results used during analysis
        self.original_network = None
        self.noisy_network_sets = None
        self.noisy_networks_module_recovery = None

        self.network_list = None
        self.results = {}
        self.identifier = self._createIdentifier()
        self.eigenvector_spectra_dir = None
        self.original_network = None
        self.random_added_edges_dict = {}
        self.random_removed_edges_dict = {}
        self.noise_applicators = {
            "add_edges": nx.Graph.add_edges_from,
            "remove_edges": nx.Graph.remove_edges_from,
            # "rewire_edges": some_rewire_function,
        }

        print(f"-> Initialized analysis for: {self.network_request.get('type')}")

    def run(self):
        # 0. Folder for eigenvector spectra
        self._createEigenvectorSpectraDir()
        # 1. Create ground truth network
        self.original_network = OriginalNetwork(
            self.network_request, self.random_seed_list, self.eigenvector_spectra_dir
        )
        self.original_network.generateEvaluationBaseline()

        # 2. Create derived networks with noise
        self.noisy_network_sets = PerturbedEdges(
            self.original_network.original_network_nx, self.noise_information
        )
        self.noisy_network_sets.generateNoisyNetworkSets()

        # 3. Trying to recover the original communities on the noisy networks
        self.noisy_networks_module_recovery = NoisyNetworkRecovery(
            original_network_obj=self.original_network,
            noisy_network_sets_obj=self.noisy_network_sets,
        )
        self.noisy_networks_module_recovery.recoverOriginalModules()

        # 4. Collect results
        self.results = self.noisy_networks_module_recovery.recovery_results

    ######### Creates unique identifier ##########
    # Called upon creation of class object
    # Will identify each network in the final table
    ##############################################

    def _createIdentifier(self):
        """Creates a unique string key from the network parameters."""

        name = self.network_request.get("type", "unknown")
        # Add all other parameters, sorted by key for consistency
        params = [
            f"{k}={v}"
            for k, v in sorted(self.network_request.items())
            if k not in ["type", "instance"]
        ]
        print(params)
        print()
        print(self.network_request.items())
        base_name = f"{name}_" + "_".join(params)

        # Append the instance number if it exists
        instance_num = self.network_request.get("instance")
        if instance_num is not None:
            return f"{base_name}_{instance_num}"

        return base_name

    def _createEigenvectorSpectraDir(self):
        os.makedirs(f"global_structure_spectra_{self.identifier}", exist_ok=True)
        self.eigenvector_spectra_dir = f"global_structure_spectra_{self.identifier}"
