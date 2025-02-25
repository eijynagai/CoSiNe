from networkx.algorithms.community import greedy_modularity_communities


def run_greedy_modularity(G):
    """
    Runs greedy modularity community detection (Clauset-Newman-Moore).

    Parameters
    ----------
    G : networkx.Graph
        Undirected input graph.

    Returns
    -------
    dict
        Node-to-community mapping: {node: community_id}.
    """
    # This returns a list (or generator) of sets of nodes, each set is one community.
    communities = list(greedy_modularity_communities(G))

    # Convert from [set1, set2, ...] to {node: community_id}
    community_dict = {}
    for cid, cset in enumerate(communities):
        for node in cset:
            community_dict[node] = cid
    return community_dict
