def _removeSeeds(analyzer):
    # Removing seed nodes from the baseline modules
    cleaned_baselines = {}
    for seed_id in analyzer.baselines.keys():
        cleaned_baselines[seed_id] = _removeSeedsFromResultsBasedOnType(
            analyzer.baselines[seed_id], analyzer.baseline_seeds[seed_id]
        )
    analyzer.baselines = cleaned_baselines

    # Removing the seed nodes from the perturbed modules
    for index, row in analyzer.df.iterrows():
        seed_nodes = row["seeds_in_network"]
        row["results"] = _removeSeedsFromResultsBasedOnType(row["results"], seed_nodes)


def _removeSeedsFromResultsBasedOnType(results, seed_nodes):

    # List of sets results (e.g.DOMINO)
    if isinstance(results, list):
        cleaned_modules = []
        for module in results:
            module.difference_update(seed_nodes)
            cleaned_modules.append(module)
        return cleaned_modules

    # Ranked dict results (e.g. RWR)
    elif isinstance(results, dict):
        for seed in seed_nodes:
            results.pop(seed, None)
        return results

    # Single set results (e.g. 1st Neighbors, only one DOMINO module)
    elif isinstance(results, set):
        results.difference_update(seed_nodes)
        return results
