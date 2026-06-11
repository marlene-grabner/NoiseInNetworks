import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def plot_degree_distribution(
    network_input,
    num_bins=20,
    log_binning=True,
    fit_trend=False,
    save_fig=None,
    ax=None,
    title=None,
    **kwargs,
):
    """
    Plots the degree distribution P(k) vs k on a log-log scale.
    Supports both logarithmic binning (for heavy-tailed networks) and
    linear integer binning (for single-scale networks like the Power Grid).
    """
    # 1. Parse Input
    if isinstance(network_input, nx.Graph):
        degrees = np.array([d for n, d in network_input.degree()])
    else:
        degrees = np.array(network_input)

    degrees = degrees[degrees > 0]
    if len(degrees) == 0:
        raise ValueError("The input contains no degrees greater than 0.")

    min_deg = np.min(degrees)
    max_deg = np.max(degrees)

    # 2. Generate Bins Based on Network Type
    if log_binning:
        # Exponentially growing bins for heavy-tailed networks
        bin_edges = np.logspace(np.log10(min_deg), np.log10(max_deg), num=num_bins + 1)
        k_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        # Linear discrete integer bins for single-scale networks (Power Grid)
        bin_edges = np.arange(min_deg, max_deg + 2)
        k_centers = bin_edges[:-1]

    # 3. Compute and Normalize Histogram
    counts, _ = np.histogram(degrees, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    total_nodes = len(degrees)

    non_zero_indices = counts > 0
    counts = counts[non_zero_indices]
    bin_widths = bin_widths[non_zero_indices]
    pk = counts / (total_nodes * bin_widths)
    k_centers = k_centers[non_zero_indices]

    # 4. Plotting Setup
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    plot_styles = {
        "color": "black",
        "marker": "o",
        "alpha": 0.75,
        "edgecolors": "none",
        "s": 40,
    }
    plot_styles.update(kwargs)

    # Extract color for the trend line to match the scatter points
    scatter_color = plot_styles.get("color", "black")

    # Plot empirical points
    ax.scatter(k_centers, pk, **plot_styles)

    # 5. Fit and Plot Trend Line (Optional)
    if fit_trend and len(k_centers) > 1:
        # Fit a straight line in log-log space: log10(P(k)) = slope * log10(k) + intercept
        log_k = np.log10(k_centers)
        log_pk = np.log10(pk)
        slope, intercept = np.polyfit(log_k, log_pk, 1)

        # Generate line coordinates spanning the data range
        k_fit = np.logspace(
            np.log10(np.min(k_centers)), np.log10(np.max(k_centers)), 100
        )
        pk_fit = (10**intercept) * (k_fit**slope)

        # Plot the trend line
        ax.plot(
            k_fit,
            pk_fit,
            linestyle="--",
            color=scatter_color,
            alpha=0.8,
        )

    # Enforce logarithmic axes
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Degree $k$", fontsize=11)
    ax.set_ylabel(r"Probability Density $P(k)$", fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if title:
        ax.set_title(title, fontsize=15)

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")

    return ax
