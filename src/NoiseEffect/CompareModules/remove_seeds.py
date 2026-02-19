def _removeSeeds(analyzer):
    # Check algorithm type
    result_type = analyzer.df["result_type"].iloc[0]
    # Removing seed nodes from the baseline modules
    cleaned_baselines = {}
    for seed_id in analyzer.baselines.keys():
        cleaned_baselines[seed_id] = _removeSeedsFromResultsBasedOnType(
            analyzer.baselines[seed_id],
            analyzer.baseline_seeds[seed_id],
            result_type,
        )
    analyzer.baselines = cleaned_baselines

    # Removing the seed nodes from the perturbed modules
    for index, row in analyzer.df.iterrows():
        seed_nodes = row["seeds_in_network"]
        row["results"] = _removeSeedsFromResultsBasedOnType(
            row["results"], seed_nodes, result_type
        )


def _removeSeedsFromResultsBasedOnType(results, seed_nodes, result_type):

    if result_type == "DIAMOnD":
        cleaned_modules = [
            [node, score] for node, score in results if node not in seed_nodes
        ]
        return cleaned_modules

    # List of sets results (e.g.DOMINO)
    elif isinstance(results, list):
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
