import os
import sys
import subprocess
import tempfile
import shutil
import networkx as nx
import logging
from ..module_result import ModuleResult


def domino(
    G: nx.Graph,
    seeds: list,
    keep_files=False,
    DOMINO_PYTHON: str = "/opt/miniconda3/envs/domino-env/bin/python",  # on my machine
):
    """
    Runs DOMINO on a NetworkX graph and a list of seed nodes.

    Args:
        G: The NetworkX graph (node IDs can be strings or ints).
        seeds: List of seed node IDs.
        keep_files: If True, won't delete temp folder (for debugging).

    Returns:
        set: A set of gene IDs (strings) found in the module.
    """

    # --- CONFIGURATION ---
    # 1. Path to the Python executable inside your domino-env

    # 2. Path to the scripts (slicer and your patched serial runner)
    #    Ideally, put 'domino_serial.py' in the same folder as this wrapper.
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DOMINO_SCRIPT = os.path.join(CURRENT_DIR, "domino_serial.py")
    # Slicer is usually in the bin folder of the domino-env python
    SLICER_EXE = os.path.join(os.path.dirname(DOMINO_PYTHON), "slicer")

    # 1. Setup Temporary Workspace
    temp_dir = tempfile.mkdtemp(prefix="domino_run_")

    try:
        # --- PREPARATION ---
        # Convert Graph to SIF with 'g' prefix (Safety for DOMINO)
        sif_path = os.path.join(temp_dir, "network.sif")
        seed_path = os.path.join(temp_dir, "active_genes.txt")
        slices_path = os.path.join(temp_dir, "network.slices")
        output_dir = os.path.join(temp_dir, "output")

        # Write Network
        with open(sif_path, "w") as f:
            for u, v in G.edges():
                # We use 'pp' as the interaction type
                f.write(f"g{u}\tpp\tg{v}\n")

        # Write Seeds (must match 'g' prefix)
        with open(seed_path, "w") as f:
            for s in seeds:
                f.write(f"g{s}\n")

        # --- STEP 1: SLICER ---
        cmd_slicer = [
            SLICER_EXE,
            "--network_file",
            sif_path,
            "--output_file",
            slices_path,
        ]

        # We capture output to avoid spamming the console
        subprocess.run(cmd_slicer, check=True, capture_output=True)

        # --- STEP 2: DOMINO ---
        cmd_domino = [
            DOMINO_PYTHON,  # Use the specific environment python
            DOMINO_SCRIPT,  # Run the patched script
            "--active_genes_files",
            seed_path,
            "--network_file",
            sif_path,
            "--slices_file",
            slices_path,
            "--output_folder",
            output_dir,
        ]

        res = subprocess.run(cmd_domino, capture_output=True, text=True)

        # --- PARSING RESULTS ---
        modules = set()

        # Check if successful (Exit code 0)
        if res.returncode == 0:
            result_file = os.path.join(output_dir, "active_genes", "modules.out")

            if os.path.exists(result_file):
                modules = []
                with open(result_file, "r") as f:
                    # DOMINO output is just gene names separated by newlines/spaces
                    # We must strip the 'g' prefix we added

                    for line in f:
                        clean_line = line.strip().strip("[]")
                        if clean_line:
                            current_list = [
                                item.strip().replace("g", "")
                                for item in clean_line.split(",")
                            ]
                            modules.append(current_list)
            else:
                logging.warning(
                    "DOMINO ran but produced no 'modules.out'. (Likely no significant module found)"
                )
        else:
            # Handle the "Empty List" crash gracefully (It just means no result)
            if "union_all to an empty list" in res.stderr:
                logging.info(
                    "DOMINO found 0 significant modules (Standard statistical filter)."
                )
            else:
                logging.error(
                    f"DOMINO Failed!\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
                )

        module_sizes = [len(m) for m in modules] if modules else 0

        return ModuleResult(
            nodes_set=modules,
            algorithm_type="set",
            metadata={"n_valid_seeds": len(seeds), "module_sizes": module_sizes},
        )

    finally:
        # --- CLEANUP ---
        if not keep_files and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
