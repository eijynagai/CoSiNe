import logging
import networkx as nx
import igraph as ig

def run_walktrap(graph, resolution=None):
    """
    Runs Walktrap community detection via python-igraph.

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
        "Walktrap: starting on graph with %d nodes and %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    # Ensure consistent node ordering
    nodes = list(graph.nodes())
    # Build igraph graph from adjacency
    matrix = nx.to_numpy_array(graph, nodelist=nodes)
    ig_graph = ig.Graph.Adjacency((matrix > 0).tolist())
    # Run Walktrap
    ct = ig_graph.community_walktrap().as_clustering()
    membership = ct.membership
    # Map back to original node labels
    communities = {node: membership[i] for i, node in enumerate(nodes)}
    logging.info("Walktrap: found %d communities", len(ct))
    return communities
    
