import networkx as nx
from NoiseEffect.RecoveryMethods.LocalNeighborhood.random_walk import (
    randomWalkWithRestart,
)
import leidenalg as la
import numpy as np
import igraph as ig
from infomap import Infomap
from NoiseEffect.RecoveryMethods.LocalStructure.community_detection_algorithms import (
    CommunityDetectionAlgorithms,
)
from NoiseEffect.RecoveryMethods.GlobalStructure.global_structure_metrics import (
    generateGlobalStructureMetrics,
)
import random
from NoiseEffect.utils.generateRWRstarts import generateRWRstarts


class OriginalNetwork:
    def __init__(self, network_request, random_seed_list, eigenvector_spectra_dir):
        self.network_request = network_request
        self.random_seed_list = random_seed_list
        self.eigenvector_spectra_dir = eigenvector_spectra_dir
        self.original_network_nx = None
        self.original_network_ig = None
        self.idx_to_node = None
        self.node_to_idx = None
        self.original_communities = {}
        self.original_num_communities = {}
        self.original_neighborhood = {}
        self.original_global_structure = {}

    def generateEvaluationBaseline(self):
        # 1. Create ground truth network
        self.createOriginalNetwork()
        # 2. Get the baseline community structure
        self.getBaselineCommunityStructure()
        # 3. Get the baseline neighborhood structure
        self.getBaselineNeighborhoodStructure()
        # 4. Get some global network properties
        self.getBaselineGlobalNetworkProperties()

    ###########################################
    # 1. Creation of ground truth network #####
    ###########################################

    def createOriginalNetwork(self):
        """
        Generates a network based on the parameters specified by the user in the request dictionary.
        """

        network_type = self.network_request.get("type")

        if network_type == "erdos_renyi":
            # Required params: n (nodes), p (probability)
            g = nx.erdos_renyi_graph(
                n=self.network_request["nodes"], p=self.network_request["p"]
            )
            self.idx_to_node = {i: i for i in g.nodes()}
            self.node_to_idx = {i: i for i in g.nodes()}

        elif network_type == "watts_strogatz":
            # Required params: n (nodes), k (neighbors), p (rewire probability)
            g = nx.watts_strogatz_graph(
                n=self.network_request["nodes"],
                k=self.network_request["k"],
                p=self.network_request["p"],
            )
            self.idx_to_node = {i: i for i in g.nodes()}
            self.node_to_idx = {i: i for i in g.nodes()}

        elif network_type == "barabasi_albert":
            # Required params: n (nodes), m (edges to attach)
            g = nx.barabasi_albert_graph(
                n=self.network_request["nodes"], m=self.network_request["m"]
            )
            self.idx_to_node = {i: i for i in g.nodes()}
            self.node_to_idx = {i: i for i in g.nodes()}

        elif network_type == "lfr_benchmark_grah":
            # Required params: n (nodes), tau1, tau2, mu, average_degree, min_community
            g = nx.LFR_benchmark_graph(
                n=self.network_request["nodes"],
                tau1=self.network_request["tau1"],
                tau2=self.network_request["tau2"],
                mu=self.network_request["mu"],
                average_degree=self.network_request["average_degree"],
                min_community=self.network_request["min_community"],
            )
            self.idx_to_node = {i: i for i in g.nodes()}
            self.node_to_idx = {i: i for i in g.nodes()}

        # NOT REALLY TOO SURE IF THIS WORKS WITH THE COMMUNITY DETECTION METHODS; MIGHT HAVE ISSUES WITH THE NODE LABELS
        elif network_type == "personal_network":
            # Required params: path (file path to edgelist)
            g_raw = nx.read_edgelist(self.network_request["path"])
            # Sorting the nodes to make the mapping always the same
            original_node_labels = sorted(list(g_raw.nodes()))
            # Create the forward and reverse mappings
            self.idx_to_node = {
                i: label for i, label in enumerate(original_node_labels)
            }
            self.node_to_idx = {
                label: i for i, label in enumerate(original_node_labels)
            }
            # Creating an integer-labeled graph
            g = nx.relabel_nodes(g_raw, self.node_to_idx)

        # If anything else is requested, throw an error
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # Store the networkx version
        self.original_network_nx = g

        # Store igraph version as well
        self.original_network_ig = ig.Graph.from_networkx(g)

    ##################################################
    #### 2. Get the baseline community structure #####
    ##################################################

    def getBaselineCommunityStructure(self):
        if self.original_network_nx.number_of_edges() == 0:
            raise ValueError("Network without edges can not have communities.")

        self.original_communities["leiden_algorithm"] = (
            CommunityDetectionAlgorithms.leidenAlgorithmPartioning(
                self.original_network_ig, list_of_seeds=self.random_seed_list
            )
        )
        self.original_communities["infomap_algorithm"] = (
            CommunityDetectionAlgorithms.infomapAlgorithmPartioning(
                self.original_network_ig, list_of_seeds=self.random_seed_list
            )
        )

    #####################################################
    #### 3. Get the baseline neighborhood structure #####
    #####################################################

    def getBaselineNeighborhoodStructure(self):
        start_points = generateRWRstarts(
            self.random_seed_list, self.original_network_nx
        )
        for start in start_points:
            p = randomWalkWithRestart(
                G=self.original_network_nx,
                seed_nodes=start,
            )
            self.original_neighborhood[str(start)] = p

    #####################################################
    #### 4. Get some global network properties ##########
    #####################################################

    def getBaselineGlobalNetworkProperties(self):
        self.original_global_structure = generateGlobalStructureMetrics(
            self.original_network_nx,
            samples=100,
            eigenvector_spectra_dir=self.eigenvector_spectra_dir,
            network_identifier="baseline",
        )
