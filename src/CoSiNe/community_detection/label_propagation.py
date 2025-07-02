import logging

from networkx.algorithms.community import label_propagation_communities


def run_label_propagation(graph, resolution=None):
    """
    Runs Label Propagation community detection.

    Parameters
    ----------
    graph : networkx.Graph
        The positive-only NetworkX graph to cluster.
    resolution : float, optional
        Placeholder to match other run_* signatures. Unused.

    Returns
    -------
    communities : dict
        Mapping from node ID to community label.
    """
    logging.info(
        "Label Propagation: starting on graph with %d nodes and %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    # Compute communities (returns generator of sets)
    comm_sets = label_propagation_communities(graph)

    # Build a node-to-community mapping
    communities = {}
    for cid, comm in enumerate(comm_sets):
        for node in comm:
            communities[node] = cid

    logging.info(
        "Label Propagation: found %d communities",
        len(set(communities.values())),
    )
    return communities
