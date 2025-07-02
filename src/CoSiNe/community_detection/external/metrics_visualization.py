import matplotlib.pyplot as plt
import numpy as np
from modules.louvain_signed import LouvainSigned

"""
Module: metrics_visualization.py

Description:
This module provides functions for calculating various metrics related to community analysis in signed networks and plot the results.

Required Libraries:
- numpy
- networkx
- LouvainSigned (custom module)

"""


def calc_entropy(partition):
    cluster_sizes = list(partition.values())
    unique, counts = np.unique(cluster_sizes, return_counts=True)
    cluster_size_distribution = dict(zip(unique, counts))

    sizes = list(cluster_size_distribution.values())
    total_nodes = sum(sizes)
    num_clusters = len(sizes)

    if num_clusters <= 1:
        return 0  # Entropy is zero if there's only one or no cluster

    entropy = -sum((size / total_nodes) * np.log(size / total_nodes) for size in sizes)
    normalized_entropy = entropy / np.log(num_clusters)

    return normalized_entropy


#    print(f"エントロピー: {entropy}")
#    print(f"正規化エントロピー: {normalized_entropy}")


def count_nodes_in_communities(partition):
    community_counts = {}
    for node, community in partition.items():
        if community in community_counts:
            community_counts[community] += 1
        else:
            community_counts[community] = 1

    for community, count in sorted(community_counts.items()):
        print(f"Community {community}: {count} nodes")


def get_community_sizes(partition):
    """
    Extract the number of elements in each community from the partition.

    Args:
    - partition (dict): Dictionary with nodes as keys and community ID as values.

    Returns:
    - community_sizes (dict): Dictionary with community ID as keys and number of elements as values.
    """
    community_sizes = {}
    for community in partition.values():
        community_sizes[community] = community_sizes.get(community, 0) + 1
    return community_sizes


def count_marker_genes_in_communities(marker_genes, gene_community_dict, partition):
    """
    Counts how many marker genes are in each community and collects them.
    """
    community_marker_genes = {}
    community_sizes = get_community_sizes(partition)

    for gene in marker_genes:
        if gene in gene_community_dict:
            community = gene_community_dict[gene]
            if community not in community_marker_genes:
                community_marker_genes[community] = set()
            community_marker_genes[community].add(gene)

    return community_marker_genes, community_sizes


# Assign marker genes to communitites
def calculate_scores(marker_genes, partition):
    """
    Calculate scores based on marker gene occurrences in each community.
    """
    community_scores = {}

    for gene in marker_genes:
        community = partition.get(gene)
        if community is not None:
            if community not in community_scores:
                community_scores[community] = 0
            community_scores[community] += 1

    return community_scores


def analyze_and_plot_communities(
    G_positive, G_negative, directory, filename, res_value, seed, include_scores=True
):
    from utils.data_manipulation import remove_small_communities

    alphas = np.arange(0, 1.05, 0.05)
    results = []

    for alpha in alphas:
        partition = LouvainSigned(G_positive, G_negative).best_partition(
            alpha=alpha, resolution=res_value, seed=seed
        )
        partition_filtered = remove_small_communities(partition)
        communities = set(partition.values())
        filtered_communities = set(partition_filtered.values())
        entropy = calc_entropy(partition)
        n_communities = len(communities)
        n_filtered_communities = len(filtered_communities)

        intra_pos_edges, inter_pos_edges, intra_neg_edges, inter_neg_edges = 0, 0, 0, 0

        for u, v in G_positive.edges():
            if u in partition and v in partition:
                if partition[u] == partition[v]:
                    intra_pos_edges += 1
                else:
                    inter_pos_edges += 1

        for u, v in G_negative.edges():
            if u in partition and v in partition:
                if partition[u] == partition[v]:
                    intra_neg_edges += 1
                else:
                    inter_neg_edges += 1

        total_pos_edges = G_positive.number_of_edges()
        total_neg_edges = G_negative.number_of_edges()

        intra_pos_ratio = (
            intra_pos_edges / total_pos_edges if total_pos_edges != 0 else 0
        )
        inter_pos_ratio = (
            inter_pos_edges / total_pos_edges if total_pos_edges != 0 else 0
        )
        intra_neg_ratio = (
            intra_neg_edges / total_neg_edges if total_neg_edges != 0 else 0
        )
        inter_neg_ratio = (
            inter_neg_edges / total_neg_edges if total_neg_edges != 0 else 0
        )

        results.append(
            {
                "alpha": alpha,
                "intra_pos_ratio": intra_pos_ratio,
                "inter_pos_ratio": inter_pos_ratio,
                "intra_neg_ratio": intra_neg_ratio,
                "inter_neg_ratio": inter_neg_ratio,
                "n_communities": n_communities,
                "n_filtered_communities": n_filtered_communities,
                "entropy": entropy,
            }
        )

    # Extract data for plotting
    alphas = [result["alpha"] for result in results]
    intra_pos_ratios = [result["intra_pos_ratio"] for result in results]
    inter_pos_ratios = [result["inter_pos_ratio"] for result in results]
    intra_neg_ratios = [result["intra_neg_ratio"] for result in results]
    inter_neg_ratios = [result["inter_neg_ratio"] for result in results]
    n_communities_list = [result["n_communities"] for result in results]
    n_filtered_communities_list = [
        result["n_filtered_communities"] for result in results
    ]
    entropies = [result["entropy"] for result in results]

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the ratios if include_scores is True
    if include_scores:
        ax1.plot(alphas, intra_pos_ratios, label="Intra-community Positive Ratio")
        ax1.plot(alphas, inter_pos_ratios, label="Inter-community Positive Ratio")
        ax1.plot(alphas, intra_neg_ratios, label="Intra-community Negative Ratio")
        ax1.plot(alphas, inter_neg_ratios, label="Inter-community Negative Ratio")
        ax1.plot(alphas, entropies, label="Normalized entropy")

    # Plot settings
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Ratio")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Create the second y-axis for the number of communities
    ax2 = ax1.twinx()
    ax1.invert_xaxis()

    bar_width = 0.015
    bar_positions = np.array(alphas) - bar_width / 2
    bar_positions_filtered = np.array(alphas) + bar_width / 2

    ax2.bar(
        bar_positions,
        n_communities_list,
        width=bar_width,
        alpha=0.6,
        color="gray",
        label="Number of Communities",
    )
    ax2.bar(
        bar_positions_filtered,
        n_filtered_communities_list,
        width=bar_width,
        alpha=0.6,
        color="navy",
        label="Filtered Number of Communities",
    )
    ax2.set_ylabel("Number of Communities")
    ax2.legend(loc="upper right")

    # Save the plot to a file
    file_suffix = "_with_scores" if include_scores else "_without_scores"
    filepath = f"{directory}/{filename}_AlphaVarCommIdent{file_suffix}.png"
    plt.savefig(filepath)
    plt.clf()  # Clear the plot
    print(f"Plot saved to {filepath}")

    return results


# -------------------------------------------------------------------------
# Tests
def ensure_list_of_ints(nodes):
    # Check if nodes is a list
    if not isinstance(nodes, list):
        raise ValueError("nodes must be a list")

    # Check each element in the list
    for i, node in enumerate(nodes):
        if not isinstance(node, int):
            try:
                # Attempt to convert to integer
                nodes[i] = int(node)
            except ValueError:
                # Handle the case where conversion is not possible
                raise ValueError(
                    f"Element at index {i} of nodes cannot be converted to an integer"
                )

    return nodes


def ensure_list_format(nodes):
    """Ensure that nodes are in list format."""
    if not isinstance(nodes, list):
        nodes = [nodes] if nodes is not None else []
    return nodes


def plot_weight_distribution(
    weights,
    median_weight,
    mean_weight,
    mode_weight,
    max_weight,
    min_weight,
    directory,
    filename,
    alpha,
):
    """
    Plots the distribution of weights.

    Parameters:
    - weights (list): List of weights to be plotted.
    - median_weight (float): The median value of weights.
    - mean_weight (float): The mean value of weights.
    - mode_weight (float or str): The mode value of weights, or a string indicating no unique mode.
    - max_weight (float): The maximum weight value.
    - min_weight (float): The minimum weight value.
    - output_dir (str): The directory where the plot will be saved.
    - filename (str): The name of the file where the plot will be saved.
    """

    if not weights:
        print("No data to plot.")
        return

    def annotate_stat(stat, name, color, y_pos):
        plt.axvline(x=stat, color=color, linestyle="dashed", linewidth=1)
        plt.annotate(
            f"{name}: {stat:.2f}",
            xy=(stat, 0),
            xytext=(stat, plt.gca().get_ylim()[1] * y_pos),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            color=color,
        )

    plt.hist(weights, bins=30, edgecolor="k", alpha=0.1)

    annotate_stat(median_weight, "Median", "red", 0.6)
    annotate_stat(mean_weight, "Mean", "green", 0.5)
    annotate_stat(max_weight, "Max", "magenta", 0.3)
    annotate_stat(min_weight, "Min", "cyan", 0.2)
    if mode_weight != "No unique mode":
        annotate_stat(mode_weight, "Mode", "blue", 0.4)

    plt.title("Distribution of Weights")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")

    # Save the plot to a file
    filepath = f"{directory}/{filename}_{alpha}_WeightDist.png"
    plt.savefig(filepath)
    plt.clf()  # Clear the plot
    print(f"Plot saved to {filepath}\n")


def plot_community_sizes(partition, directory, filename):
    """
    Plots the distribution of community sizes and saves the plot to a file.

    Args:
    - partition (dict): Dictionary with nodes as keys and community ID as values.
    """
    # Get the community sizes
    community_sizes = get_community_sizes(partition)

    # Sort communities based on their sizes (from largest to smallest)
    sorted_communities = sorted(community_sizes, key=community_sizes.get, reverse=True)
    sorted_sizes = [community_sizes[community] for community in sorted_communities]

    # Create a bar plot with sorted data
    plt.bar(sorted_communities, sorted_sizes)
    plt.xlabel("Community ID")
    plt.ylabel("Number of Elements")
    plt.title("Distribution of Community Sizes")

    # Save the plot to a file
    filepath = f"{directory}/{filename}_CommSizes_min.png"
    plt.savefig(filepath)
    plt.clf()  # Clear the plot after saving to avoid overlap on next plot
    print(f"Plot saved to {filepath}")


def plot_community_sizes2(partition, directory, filename):
    """
    Plots the distribution of the number of communities by community size and saves the plot to a file,
    maintaining the original numerical order for the number of genes.

    Args:
    - partition (dict): Dictionary with nodes as keys and community ID as values.
    """
    # Get the community sizes
    community_sizes = get_community_sizes(partition)

    # Calculate how many communities have each size (frequency of sizes)
    size_counts = {}
    for size in community_sizes.values():
        if size in size_counts:
            size_counts[size] += 1
        else:
            size_counts[size] = 1

    # Prepare data for plotting (maintain original numerical order)
    sizes = list(size_counts.keys())
    counts = [size_counts[size] for size in sizes]

    # Create a bar plot
    plt.bar(sizes, counts)
    plt.xlabel("Number of Genes")
    plt.ylabel("Number of Communities")
    plt.title("Distribution of Communities by Size")

    # Save the plot to a file
    filepath = f"{directory}/{filename}_GeneCommDistribution.png"
    plt.savefig(filepath)
    plt.clf()  # Clear the plot after saving to avoid overlap on next plot
    print(f"Plot saved to {filepath}")


# Plot contracted networks with node sizes representing number of genes
def plot_contracted_network(contracted_G, directory, filename, k=0.5):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import networkx as nx

    # Create a color map
    num_communities = len(contracted_G.nodes())
    color_map = cm.get_cmap("viridis", num_communities)  # Get the colormap function

    # Assign a color to each community node
    node_colors = [color_map(i / num_communities) for i in range(num_communities)]

    # Node sizes proportional to the number of genes
    scaling_factor = 70  # Adjust as needed
    node_sizes = [
        data["size"] * scaling_factor for _, data in contracted_G.nodes(data=True)
    ]

    # Generate layout
    pos = nx.spring_layout(contracted_G, k=k)

    # Draw the network
    plt.figure(figsize=(12, 12))
    nx.draw(
        contracted_G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=True,
        labels=nx.get_node_attributes(contracted_G, "label"),
        font_size=12,
        edge_color="gray",
    )
    plt.title("Contracted Network with Node Sizes Representing Number of Genes")

    # Save the plot to a file
    filepath = f"{directory}/{filename}_ContrNetPlot.png"
    plt.savefig(filepath)
    plt.clf()  # Clear the plot
    print(f"Plot saved to {filepath}")


# Read the sankey file and return the community genes dictionary and partition
def read_sankey_file(file_path):
    community_genes_dict = {}
    partition = {}

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(", ")
            community = int(parts[0].split(" ")[1])
            genes = parts[1:]
            community_genes_dict[community] = genes
            partition[community] = len(genes)

    return community_genes_dict, partition


# Plot the vis_stack_plot (sankey) using the marker genes found in each community
def vis_genes_by_community(
    community_genes_dict, partition, directory, filename, text_width=0.04
):
    # Determine the number of communities and the max number of genes in any community
    num_communities = len(community_genes_dict)
    max_genes_in_community = max(len(genes) for genes in community_genes_dict.values())

    community_sizes = get_community_sizes(partition)

    # Apply scaling factors to adjust figure size
    width_scaling_factor = 2.0  # Adjust as needed
    height_scaling_factor = 0.3  # Adjust as needed
    figsize = (
        num_communities * width_scaling_factor,
        max_genes_in_community * height_scaling_factor,
    )

    fig, ax = plt.subplots(figsize=figsize)

    x_labels = []
    for i, (community, genes) in enumerate(community_genes_dict.items()):
        gene_count = len(genes)
        total_count = community_sizes.get(community, 0)
        x_labels.append(f"c{community} ({gene_count}/{total_count})")

        y_positions = range(len(genes))
        x_positions = [i] * len(genes)
        ax.scatter(x_positions, y_positions, marker="", alpha=0)

        for j, gene in enumerate(genes):
            for k, letter in enumerate(gene):
                ax.text(
                    i + (k - len(gene) / 2) * text_width,
                    j,
                    letter,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xticks(range(len(community_genes_dict)))
    ax.set_xticklabels(x_labels, fontsize=12, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_ylim(-1, max_genes_in_community)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title("Marker Genes by Community", fontsize=14)
    plt.tight_layout()

    # Save the plot to a file
    filepath = f"{directory}/{filename}_MarkGenCommPlot.png"
    plt.savefig(filepath)
    plt.clf()  # Clear the plot
    print(f"Plot saved to {filepath}")
