from infomap import Infomap


def run_infomap(graph):
    """Runs the Infomap community detection algorithm."""
    im = Infomap()

    # Add edges to the Infomap instance
    for edge in graph.edges():
        im.addLink(edge[0], edge[1])

    # Run Infomap clustering
    im.run()

    # Get community assignments
    communities = {node.node_id: node.module_id for node in im.iterTree()}

    return [communities[node] for node in graph.nodes()]
