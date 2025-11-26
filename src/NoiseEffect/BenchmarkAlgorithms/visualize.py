import matplotlib.pyplot as plt
import numpy as np
import math
import warnings


def plotStabilityResults(
    results_data, algorithm_name="Undefined", extra_info=None, save_path=False
):
    """
    Plots the stability results as a 2x2 matplotlib chart.
    """

    # 1. Process the data
    # Get graph names (e.g., 'Erdos-Renyi', 'PPI', ...)
    graph_names = list(results_data.keys())

    # Define the metrics we want to pull from the data
    metric_keys = {
        "ARI": "ari",
        "AMI": "ami",
        "Clusters 1": "num_clusters_1",
        "Clusters 2": "num_clusters_2",
    }

    plot_data = {}
    for plot_title, data_key in metric_keys.items():
        means = []
        stds = []
        for graph in graph_names:
            # Create a list of all values for this metric & graph
            values = [
                run[data_key]
                for run in results_data[graph]
                if run["status"] == "success"
            ]

            # Convert to numpy array to handle NaNs easily
            values = np.array(values, dtype=float)

            # Check if we have valid data before calculating
            # (Silences "Mean of empty slice" warnings)
            if len(values) == 0 or np.all(np.isnan(values)):
                means.append(np.nan)
                stds.append(np.nan)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    means.append(np.nanmean(values))
                    stds.append(np.nanstd(values))

        plot_data[plot_title] = {"means": means, "stds": stds}

    # 2. Create the 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Flatten the 2x2 axes array into a 1D array for easy looping
    axes = axes.flatten()

    # Get the x-axis positions
    x_positions = np.arange(len(graph_names))

    # 3. Draw each of the 4 subplots
    for i, (title, data) in enumerate(plot_data.items()):
        ax = axes[i]

        current_means = data["means"]
        current_stds = data["stds"]

        # Create the bar plot with error bars
        ax.bar(
            x_positions,
            current_means,
            yerr=current_stds,  # This adds the standard deviation
            capsize=5,  # Adds the little caps on the error bars
            align="center",
            alpha=0.7,
            color="midnightblue",
        )

        # Iterate through our data. If a mean is NaN, place text on the axis.
        for x, val in zip(x_positions, current_means):
            if np.isnan(val):
                ax.text(
                    x,
                    0,
                    "N/A",
                    ha="center",
                    va="bottom",
                    color="darkred",
                    fontweight="bold",
                    fontsize=16,
                )

        ax.set_title(title)
        ax.set_ylabel("Mean Value")

        # Set the x-axis labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(graph_names, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # 4. Finalize and show the plot
    fig.suptitle(f"Algorithm Stability for {algorithm_name}", fontsize=16)
    fig.text(0.5, 0.93, extra_info, ha="center", fontsize=12) if extra_info else None
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plotSpreadOfStabilityResults(
    results_dict, measurement="ari", title=None, save_path=False
):
    # get keys and layout
    keys = list(results_dict.keys())
    n = len(keys)
    if n == 0:
        raise RuntimeError("results_dict is empty.")

    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # flatten

    for i, name in enumerate(keys):
        res = results_dict[name]
        ari = [r[measurement] for r in res]
        ax = axes[i]

        # --- IF ONLY NANs ---
        # Convert to float and remove NaNs
        ari = np.asarray(ari, dtype=float)
        ari = ari[~np.isnan(ari)]

        if ari is None or len(ari) == 0:
            ax.text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                color="darkred",
                fontweight="bold",
                fontsize=16,
            )
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax.hist(ari, bins=20, color="#71C5C9", edgecolor="w", alpha=0.8)
        ax.axvline(
            ari.mean(),
            color="#6B1403",
            linestyle="--",
            linewidth=3,
            label=f"mean={ari.mean():.3f}",
        )
        ax.set_title(f"{name} (n={len(ari)})")
        ax.set_xlabel("ARI")
        ax.set_ylabel("Count")
        ax.legend()

    # hide any unused axes
    for j in range(n, rows * cols):
        axes[j].axis("off")

    if title is not None:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
