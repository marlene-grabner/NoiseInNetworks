import matplotlib.pyplot as plt
import numpy as np

def plotStabilityResults(results_data, algorithm_name="Undefined"):
    """
    Plots the stability results as a 2x2 matplotlib chart.
    """
    
    # 1. Process the data
    # Get graph names (e.g., 'Erdos-Renyi', 'PPI', ...)
    graph_names = list(results_data.keys())
    
    # Define the metrics we want to pull from the data
    metric_keys = {
        'ARI': 'ari',
        'AMI': 'ami',
        'Clusters 1': 'num_clusters_1',
        'Clusters 2': 'num_clusters_2'
    }
    
    plot_data = {}
    for plot_title, data_key in metric_keys.items():
        means = []
        stds = []
        for graph in graph_names:
            # Create a list of all values for this metric & graph
            values = [run[data_key] for run in results_data[graph] 
                      if run['status'] == 'success']
            
            # Calculate mean and standard deviation
            means.append(np.mean(values))
            stds.append(np.std(values))
            
        plot_data[plot_title] = {'means': means, 'stds': stds}

    # 2. Create the 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flatten the 2x2 axes array into a 1D array for easy looping
    axes = axes.flatten()
    
    # Get the x-axis positions
    x_positions = np.arange(len(graph_names))
    
    # 3. Draw each of the 4 subplots
    for i, (title, data) in enumerate(plot_data.items()):
        ax = axes[i]
        
        # Create the bar plot with error bars
        ax.bar(
            x_positions,
            data['means'],
            yerr=data['stds'],  # This adds the standard deviation
            capsize=5,          # Adds the little caps on the error bars
            align='center',
            alpha=0.7,
            color='midnightblue'
        )
        
        ax.set_title(title)
        ax.set_ylabel('Mean Value')
        
        # Set the x-axis labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(graph_names, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Finalize and show the plot
    fig.suptitle(f'Algorithm Stability for {algorithm_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.show()