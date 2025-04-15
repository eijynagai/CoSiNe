import random

import matplotlib.pyplot as plt
import networkx as nx


def generate_sbm_graph_vis():
    """
    Generate a synthetic SBM graph with two communities.
    Returns a graph with a clear community structure and the 'block' attribute set.
    """
    sizes = [20, 20]  # Two communities of 20 nodes each.
    p_in = 0.8
    p_out = 0.05
    p_matrix = [[p_in, p_out], [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, p_matrix, seed=42)
    # Set all edges to positive weight.
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G


def add_negative_edges_vis(G, num_edges, community_labels):
    """
    Add a specified number of negative edges between communities.
    community_labels: dict mapping node -> community id.
    """
    # Build a list of all possible inter-community pairs.
    inter_pairs = []
    communities = {}
    for node, comm in community_labels.items():
        communities.setdefault(comm, []).append(node)
    comm_ids = list(communities.keys())
    for i in range(len(comm_ids)):
        for j in range(i + 1, len(comm_ids)):
            for u in communities[comm_ids[i]]:
                for v in communities[comm_ids[j]]:
                    inter_pairs.append((u, v))
    random.shuffle(inter_pairs)
    count = 0
    for u, v in inter_pairs:
        if count >= num_edges:
            break
        # Only add an edge if one does not already exist.
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=-1)
            count += 1
    return G


def plot_network(G, title, filename=None):
    """
    Plot the network G with a spring layout.
    Positive edges are drawn in blue (solid) and negative edges in red (dashed).
    Nodes are colored by their community (using the 'block' attribute).
    """
    pos = nx.spring_layout(G, seed=42)

    # Color nodes by their community if available.
    communities = nx.get_node_attributes(G, "block")
    if communities:
        unique_comms = sorted(set(communities.values()))
        # Generate a colormap.
        cmap = plt.cm.Set3
        colors = [cmap(i / len(unique_comms)) for i in range(len(unique_comms))]
        color_map = {comm: colors[i] for i, comm in enumerate(unique_comms)}
        node_colors = [color_map[communities[node]] for node in G.nodes()]
    else:
        node_colors = "lightblue"

    # Draw nodes.
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.9)

    # Separate positive and negative edges.
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) < 0]

    # Draw edges.
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color="blue", width=2)
    nx.draw_networkx_edges(
        G, pos, edgelist=neg_edges, edge_color="red", style="dashed", width=2
    )

    nx.draw_networkx_labels(G, pos, font_color="black", font_size=8)
    plt.title(title, fontsize=16)
    plt.axis("off")
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved network plot to {filename}")
    plt.show()


if __name__ == "__main__":
    # Generate the base graph.
    G_vis = generate_sbm_graph_vis()
    comm_labels = nx.get_node_attributes(G_vis, "block")

    # Calculate maximum possible inter-community edges.
    # For two communities of size 20 each, that's 20*20 = 400.
    max_inter_edges = 20 * 20

    # Define negative edge counts for 10% and 90%.
    neg_10 = int(0.10 * max_inter_edges)  # 10% negative edges: 40 edges.
    neg_90 = int(0.90 * max_inter_edges)  # 90% negative edges: 360 edges.

    # Create two versions of the network.
    G_10 = G_vis.copy()
    G_10 = add_negative_edges_vis(G_10, neg_10, comm_labels)

    G_90 = G_vis.copy()
    G_90 = add_negative_edges_vis(G_90, neg_90, comm_labels)

    # Plot the networks.
    plot_network(
        G_10, "Network with 10% Negative Edges", filename="network_10_negative.png"
    )
    plot_network(
        G_90, "Network with 90% Negative Edges", filename="network_90_negative.png"
    )
