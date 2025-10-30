from sklearn.metrics import roc_auc_score, average_precision_score
from NoiseEffect.NoisePipeline.RecoveryMethods.GlobalStructure.global_structure_metrics import (
    generateGlobalStructureMetrics,
)
from NoiseEffect.NoisePipeline.RecoveryMethods.LocalStructure.local_structure import (
    localStructureAnalysis,
)
from NoiseEffect.NoisePipeline.RecoveryMethods.LocalNeighborhood.local_neighborhood import (
    localNeighborhoodAnalysis,
)
import igraph as ig


class NoisyNetworkRecovery:
    def __init__(self, original_network_obj, noisy_network_sets_obj):
        # Input objects
        self.original_network_obj = original_network_obj
        self.noisy_network_sets_obj = noisy_network_sets_obj
        # Results
        self.recovery_results = {}

    def recoverOriginalModules(self):
        # 2. Added edges analysis
        self._changesBasedOnModification(modification_type="added_edges")
        # 3. Removed edges analysis
        self._changesBasedOnModification(modification_type="removed_edges")

    ###############################################
    # Function to pass the different modification #
    # types to, iterate through the noise levels ##
    # and evaluate the influence on outcomes ######
    ###############################################

    def _changesBasedOnModification(self, modification_type):
        self.recovery_results[modification_type] = {}

        if modification_type == "added_edges":
            edge_dict = self.noisy_network_sets_obj.random_added_edges_dict
        elif modification_type == "removed_edges":
            edge_dict = self.noisy_network_sets_obj.random_removed_edges_dict
        else:
            raise ValueError(
                "modification_type must be 'added_edges' or 'removed_edges'"
            )

        for perturbation in edge_dict:
            # Create entry in the results dictionary
            self.recovery_results[modification_type][f"noise_{perturbation}"] = {}

            # Generating the modified network
            edges_to_modify = edge_dict[perturbation]

            if modification_type == "added_edges":
                modified_network_nx, modified_network_ig = (
                    self._NetworkWithAddedEdgesObject(edges_to_modify)
                )
            elif modification_type == "removed_edges":
                modified_network_nx, modified_network_ig = (
                    self._NetworkWithRemovedEdgesObject(edges_to_modify)
                )

            # Recover Local Structure
            self._recoverLocalStructure(
                G_nx=modified_network_nx,
                G_ig=modified_network_ig,
                modification_type=modification_type,
                perturbation=perturbation,
            )

            # Recover Local Neighborhood
            self._recoverLocalNeighborhood(
                G_nx=modified_network_nx,
                modification_type=modification_type,
                perturbation=perturbation,
            )

            # Recover Global Structure
            self._recoverGlobalStructure(
                G_nx=modified_network_nx,
                modification_type=modification_type,
                perturbation=perturbation,
            )

    ###############################################
    ####### Network properties evaluated ##########
    ###############################################

    def _recoverLocalStructure(self, G_nx, G_ig, modification_type, perturbation):
        structure_results = localStructureAnalysis(
            G_nx=G_nx,
            G_ig=G_ig,
            original_communities=self.original_network_obj.original_communities,
            random_seed_list=self.original_network_obj.random_seed_list,
        )
        self.recovery_results[modification_type][f"noise_{perturbation}"][
            "local_structure"
        ] = structure_results

    def _recoverLocalNeighborhood(self, G_nx, modification_type, perturbation):
        original_neighborhood = self.original_network_obj.original_neighborhood
        neighborhood_results = localNeighborhoodAnalysis(
            modified_network_nx=G_nx,
            original_neighborhood=original_neighborhood,
        )
        self.recovery_results[modification_type][f"noise_{perturbation}"][
            "local_neighborhood"
        ] = neighborhood_results

    def _recoverGlobalStructure(
        self,
        G_nx,
        modification_type,
        perturbation,
    ):
        global_structure_results = generateGlobalStructureMetrics(
            G_nx,
            samples=100,
            eigenvector_spectra_dir=self.original_network_obj.eigenvector_spectra_dir,
            network_identifier=f"{modification_type}_noise_{perturbation}",
        )
        self.recovery_results[modification_type][f"noise_{perturbation}"][
            "global_structure"
        ] = global_structure_results

    ##### Add/Remove edges to/from original network #####
    def _NetworkWithAddedEdgesObject(self, edges_to_modify):
        modified_network_nx = self.original_network_obj.original_network_nx.copy()
        modified_network_nx.add_edges_from(edges_to_modify)
        return modified_network_nx, ig.Graph.from_networkx(modified_network_nx)

    def _NetworkWithRemovedEdgesObject(self, edges_to_modify):
        modified_network_nx = self.original_network_obj.original_network_nx.copy()
        modified_network_nx.remove_edges_from(edges_to_modify)
        return modified_network_nx, ig.Graph.from_networkx(modified_network_nx)
