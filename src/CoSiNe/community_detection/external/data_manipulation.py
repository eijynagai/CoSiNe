import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_community(subgraph):
    pos = nx.kamada_kawai_layout(subgraph)
    edge_widths = [
        subgraph[u][v]["weight"]
        / max(subgraph[u][v]["weight"] for u, v in subgraph.edges())
        * 2
        for u, v in subgraph.edges()
    ]
    node_sizes = [subgraph.degree(n) * 10 for n in subgraph.nodes()]
    font_size = 8 if len(subgraph.nodes()) > 100 else 12

    plt.figure(figsize=(12, 8))  # Adjust the figure size
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        labels={node: subgraph.nodes[node]["name"] for node in subgraph.nodes()},
        node_color="skyblue",
        node_size=node_sizes,
        font_size=font_size,
        font_weight="bold",
        edge_color="gray",
        width=edge_widths,
        alpha=0.7,
        edge_cmap=plt.cm.Blues,
    )  # Additional styling
    plt.title("Community Network", fontsize=16, fontweight="bold")
    plt.axis("off")  # Turn off the axis
    plt.show()


def get_nodes_from_partition(partition_filtered):
    """
    Get a list of all nodes from the partition_filtered dictionary.

    Args:
    - partition_filtered (dict): A dictionary with nodes as keys and community IDs as values.

    Returns:
    - List of nodes.
    """
    return list(partition_filtered.keys())


def get_negative_edges(G_negative, partition_filtered):
    """
    Retrieves negative edges related to the nodes in the partition_filtered.

    Args:
    - G_negative (networkx.Graph): Graph containing negative edges.
    - partition_filtered (dict): Dictionary with nodes as keys and community IDs as values.

    Returns:
    - List of negative edges related to the nodes in the partition_filtered.
    """
    filtered_nodes = set(partition_filtered.keys())
    negative_edges = []

    for edge in G_negative.edges():
        if edge[0] in filtered_nodes and edge[1] in filtered_nodes:
            negative_edges.append(edge)

    return negative_edges


def find_negative_edges_between_communities(negative_edges, community_marker_genes):
    """
    Finds edges where negative nodes are found in different communities in community_marker_genes.

    Args:
    - negative_edges (list of tuples): List of negative edges.
    - community_marker_genes (dict): Dictionary mapping communities to sets of marker genes.

    Returns:
    - dict: Dictionary mapping pairs of communities to sets of edges.
    """
    inter_community_negative_edges = {}

    for edge in negative_edges:
        node1, node2 = edge
        node1_community = None
        node2_community = None

        # Identify the communities of the nodes
        for community, genes in community_marker_genes.items():
            if node1 in genes:
                node1_community = community
            if node2 in genes:
                node2_community = community

        # Check if nodes are in different communities
        if (
            node1_community is not None
            and node2_community is not None
            and node1_community != node2_community
        ):
            community_pair = tuple(sorted([node1_community, node2_community]))
            if community_pair not in inter_community_negative_edges:
                inter_community_negative_edges[community_pair] = set()
            inter_community_negative_edges[community_pair].add(edge)

    return inter_community_negative_edges


# Removing disconnected or small communities
def remove_small_communities(partition, min_size=3):
    from modules.metrics_visualization import get_community_sizes

    """
    Remove communities with less than min_size nodes from the partition.
    Args:
    - partition (dict): Dictionary with nodes as keys and community ID as values.
    - min_size (int): Minimum size of communities to be retained.
    Returns:
    - filtered_partition (dict): Partition with small communities removed.
    """
    community_sizes = get_community_sizes(partition)
    small_communities = {
        community for community, size in community_sizes.items() if size < min_size
    }

    filtered_partition = {
        node: community
        for node, community in partition.items()
        if community not in small_communities
    }
    return filtered_partition


def save_communities(partition_filtered, output_dir, filename):
    # Organize genes by community
    genes_by_community = {}
    for gene, community in partition_filtered.items():
        genes_by_community.setdefault(community, []).append(gene)

    # Write genes by community to file
    output_filename = f"{output_dir}/{filename}_GenesByCommunity.txt"
    with open(output_filename, "w") as file:
        for community, genes in genes_by_community.items():
            file.write(
                f"Community {community}, Total Gene: {len(genes)}, Genes: {', '.join(genes)}\n"
            )
    print(f"All genes per community report saved to {output_filename}")


def save_communities2(partition_filtered, output_dir, filename):
    # Organize genes by community
    genes_by_community = {}
    for gene, community in partition_filtered.items():
        genes_by_community.setdefault(community, []).append(gene)
    # Make output with only genes
    output_filename = f"{output_dir}/{filename}.txt"
    with open(output_filename, "w") as file:
        for community, genes in genes_by_community.items():
            file.write(f"Community {community}, {', '.join(genes)}\n")
    print(f"All genes per community report saved to {output_filename}")


def generate_contracted_network(G_positive, G_negative, partition_filtered):
    """
    Reduce the size of networks by creating representation of communities.
    It include the size of nodes connected to each other.

    Args:
    - G_positive (networkx.Graph): Positive graph.
    - G_negative (networkx.Graph): Negative graph.
    - partition_filtered (dict): Partition with small communities removed.

    Returns:

    """
    import networkx as nx

    community_map = partition_filtered

    # Assuming 'G' is your original graph and 'community_map' maps nodes to their communities

    # Create the contracted graph
    contracted_G = nx.Graph()

    # Add nodes for each community with size as an attribute
    for community in set(community_map.values()):
        community_size = sum(
            1 for node in community_map if community_map[node] == community
        )
        contracted_G.add_node(community, size=community_size, label=str(community_size))

    # Add edges between different communities
    for u, v in G_positive.edges():
        u_comm = community_map.get(u, -1)
        v_comm = community_map.get(v, -1)
        if u_comm != v_comm and u_comm != -1 and v_comm != -1:
            contracted_G.add_edge(u_comm, v_comm)

    # Assuming 'contracted_G' is your contracted graph
    # Update the node labels to be the size of the community
    for node in contracted_G.nodes():
        contracted_G.nodes[node]["label"] = contracted_G.nodes[node]["size"]

    return contracted_G


def process_excel_file_for_marker_genes(excel_file_path, term):
    """
    Reads the Excel file and returns a list of unique marker genes associated with the given term.
    The function searches for the term in a predefined range of columns.

    Args:
    - excel_file_path (str): Path to the Excel file.
    - term (str): The term to search for.

    Returns:
    - list: A list of unique marker genes associated with term, or None if term is not found.
    """
    # Load Excel file
    df = pd.read_excel(excel_file_path, engine="openpyxl")

    # Predefined range of columns to search (e.g., columns 3 to 5)
    search_columns = [1, 2, 4, 6]  # Update as needed based on your specific columns

    # Initialize marker genes set
    marker_genes = set()

    # Define the function to process genes
    def process_genes(genes):
        if isinstance(genes, list):
            for gene in genes:
                marker_genes.add(gene)
        else:
            marker_genes.add(genes)

    # Search for the term in the predefined range of columns
    found = False
    for col in search_columns:
        if term in df.iloc[:, col].values:
            df[df.iloc[:, col] == term].iloc[:, 9].dropna().apply(process_genes)
            found = True
            break

    for col in search_columns:
        if term in df.iloc[:, col].values:
            filtered_df = df[df.iloc[:, col] == term]
            print(filtered_df)  # Debug: Print the filtered dataframe
            filtered_df.iloc[:, 9].dropna().apply(process_genes)
            found = True
            break

    if not found:
        print(f"Term '{term}' not found in the predefined columns.")
        return None

    print(f"Number of unique marker genes found: {len(marker_genes)}")
    return list(marker_genes)


# Get the genes from a specific community
def get_genes_from_partition(partition, community_id):
    """
    Extract the genes (nodes) belonging to a specific community from the partition.

    Args:
    - partition (dict): Dictionary with nodes as keys and community ID as values.
    - community_id: The ID of the community whose genes (nodes) are to be extracted.

    Returns:
    - genes (list): List of genes (nodes) belonging to the specified community.
    """
    genes = [node for node, community in partition.items() if community == community_id]
    return genes
