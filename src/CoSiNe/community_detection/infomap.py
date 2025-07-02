import logging

from infomap import Infomap


def run_infomap(graph, resolution=None):
    """
    Runs the Infomap community detection algorithm.

    Parameters
    ----------
    graph : networkx.Graph
        The positive-only NetworkX graph to cluster.
    resolution : float, optional
        (Unused) Placeholder to match other method signatures.

    Returns
    -------
    communities : dict
        Mapping from node ID to community (module) ID.
    """
    logging.info(
        "Infomap: starting run on graph with %d nodes and %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    # Initialize Infomap
    im = Infomap()

    # Add edges (with weights if present)
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        im.addLink(u, v, weight)

    # Run clustering
    im.run()

    # Extract module assignments
    communities = {}
    for node in im.iterTree():
        # only record leaf nodes
        if node.isLeaf:
            communities[node.node_id] = node.module_id

    logging.info("Infomap: found %d communities", len(set(communities.values())))
    return communities
