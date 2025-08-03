import logging
import networkx as nx
import igraph as ig
import numpy as np

def run_spinglass(graph, resolution=None):
    """
    Runs Spinglass community detection via python-igraph.

    Parameters
    ----------
    graph : networkx.Graph
        The positive-only NetworkX graph to cluster.
    resolution : float, optional
        Placeholder to match other run_* signatures. Unused.

    Returns
    -------
    communities : dict
        Mapping from node ID to community label (or -1 if unassigned).
    """
    logging.info(
        "Spinglass: starting on graph with %d nodes and %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    communities = {n: -1 for n in graph.nodes()}

    for i, comp in enumerate(nx.connected_components(graph)):
        comp_nodes = list(comp)
        logging.info(
            "Spinglass: processing component %d with %d nodes",
            i + 1, len(comp_nodes)
        )
        subgraph = graph.subgraph(comp_nodes)
        matrix = nx.to_numpy_array(subgraph, nodelist=comp_nodes)
        ig_graph = ig.Graph.Adjacency((matrix > 0).tolist())

        try:
            ct = ig_graph.community_spinglass(spins=10)
            membership = ct.membership
            for idx, node in enumerate(comp_nodes):
                communities[node] = membership[idx]
            logging.info(
                "Spinglass: found %d communities in component %d",
                len(ct), i + 1
            )
        except Exception as e:
            logging.error(
                "Spinglass: detection failed on component %d of size %d: %s",
                i + 1, len(comp_nodes), e, exc_info=True
            )
            # nodes remain assigned to -1

    return communities
