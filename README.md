# Network Noise Analysis Framework

This module systematically analyzes how noise (random edge additions/removals) affects
community detection performance across different network types using parallel processing.

## WHAT THIS CODE DOES:

1. Generates multiple instances of different network types (e.g., Barabási-Albert, Erdős-Rényi)
2. Applies various levels of edge noise to each network multiple times
3. Runs community detection algorithms on both original and noisy networks
4. Compares community structures to measure noise robustness
5. Collects results across all networks and noise levels for analysis

## USER-CONFIGURABLE PARAMETERS:

Network Type Parameters (in network_requests):
type: str - Network model to generate ("barabasi_albert", "erdos_renyi", "watts_strogatz", etc.)
nodes: int

- Number of nodes in the network
  Model-specific parameters: - m (Barabási-Albert): Number of edges to attach from new node - p (Erdős-Rényi): Probability of edge creation between any two nodes - k, p (Watts-Strogatz): k=number of nearest neighbors, p=rewiring probability

Noise Parameters (in noise_information):
noise_levels: list of float - Fractions of edges to add/remove as noise (e.g., [0.05, 0.1, 0.2]) - 0.1 = add/remove 10% of the original number of edges
num_repeats: int - How many times to apply SAME noise level to SAME network topology - Controls statistical sampling of noise randomness

Analysis Parameters:
num_instances: int (NetworkNoiseAnalysis parameter) - How many DIFFERENT network realizations to generate per network type

- Controls statistical sampling of topology randomness
  num_cores: int (runAnalysis parameter) - Number of CPU cores for parallel processing - Auto-detects SLURM allocation or uses (CPU_count - 1) if not specified

## EXAMPLE ANALYSIS SCOPE:

With the current settings:

- 2 network types × 5 instances = 10 different network topologies
- 2 noise levels × 2 repeats = 4 noise applications per network
- Total: 10 networks × 4 noise tests = 40 individual analyses
- Run in parallel across specififed number of CPU cores

## OUTPUT:

Results saved to 'noise_addition_results_table.json' containing community detection
performance metrics for each network-noise combination, organized by unique identifiers.

## PARAMETER INTERACTION EXAMPLE:

num_instances=5, num_repeats=3, noise_levels=[0.1, 0.2]:

- Generate 5 different Barabási-Albert networks (same n,m but different topology)
- For each network: Apply 10% noise 3 times + Apply 20% noise 3 times
- Total: 5 networks × 2 noise levels × 3 repeats = 30 measurements
- This separates topology variation (instances) from noise variation (repeats)

## Difference between num_instances and num_repeats:

num_instances (in NetworkNoiseAnalysis.**init**): - Controls how many INDEPENDENT NETWORK REALIZATIONS are generated for each network type - Each instance creates a completely new random network with the same parameters - Used for statistical robustness across different network topologies - Example: If num_instances=5 for a Barabási-Albert network, you get 5 different
BA networks, each with different random structure but same (n=100, m=3)

num_repeats (in noise_information): - Controls how many times NOISE IS APPLIED to each individual network - Each repeat applies the same noise level to the SAME network topology - Used for statistical robustness of noise effects on a single network - Example: If num_repeats=3 for noise_level=0.1, the same network gets
10% noise applied 3 times with different random edge additions/removals

## Combined Effect:

With num_instances=5 and num_repeats=3:

- Generate 5 different network topologies
- Apply each noise level 3 times to each topology
- Total: 5 networks × 3 noise applications = 15 measurements per noise level

This separates topology variation (num_instances) from noise variation (num_repeats).
