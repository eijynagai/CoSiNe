# File reading and graph creation
# File reading and graph creation
from statistics import StatisticsError, mean, median, mode
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw
from external.metrics_visualization import plot_weight_distribution


def read_file_and_create_graph(
    file_path: str,
    n_lines: Optional[int] = None,
    cdi_threshold: float = 5.0,
    eei_threshold: float = 1.0,
):
    """
    Reads a file, creates a graph, and tracks weights and node degrees.
    """
    G = nx.Graph()
    weights = []
    node_degrees: dict[str, int] = {}
    threshold = (
        cdi_threshold
        if "CDI" in file_path
        else eei_threshold if "EEI" in file_path else 0.0
    )

    try:
        with open(file_path, "r") as f:
            if n_lines is not None:
                n_lines = int(n_lines)

            for i, line in enumerate(f):
                if i == 0 or (n_lines is not None and i > n_lines):
                    continue  # Skip header or excess lines
                columns = line.strip().split()
                if len(columns) < 5:
                    continue
                node1, node2, weight = columns[2], columns[3], float(columns[4])
                weights.append(weight)
                if weight >= threshold:
                    G.add_edge(node1, node2, weight=weight)
                    node_degrees[node1] = node_degrees.get(node1, 0) + 1
                    node_degrees[node2] = node_degrees.get(node2, 0) + 1

    except FileNotFoundError:
        print(f"Error: File {file_path} could not be opened.")
        return None, [], {}

    return G, weights, node_degrees


def calculate_degree_distribution(node_degrees: dict):
    """
    Calculates the degree distribution of nodes.
    """
    sorted_nodes_by_degree = sorted(
        node_degrees.items(), key=lambda x: x[1], reverse=True
    )
    total_degree = sum(node_degrees.values())
    return sorted_nodes_by_degree, total_degree


def apply_pareto_principle(sorted_nodes_by_degree: list, total_degree: int):
    """
    Applies the Pareto principle to identify top nodes.
    """
    cumulative_degree = 0
    top_nodes = []

    for node, degree in sorted_nodes_by_degree:
        cumulative_degree += degree
        top_nodes.append(node)
        if cumulative_degree / total_degree >= 0.8:
            break

    return top_nodes


def plot_degree_distribution(graph: nx.Graph, directory: str, filename: str):
    """
    Plots the degree distribution and fits a power-law distribution.
    """
    degrees = [degree for _, degree in graph.degree()]
    degree_counts = np.bincount(degrees)
    degree_values = np.nonzero(degree_counts)[0]
    degree_probabilities = degree_counts[degree_values] / sum(degree_counts)

    plt.figure(figsize=(10, 6))
    plt.loglog(degree_values, degree_probabilities, "bo", label="Degree Distribution")

    # Fit to a power-law distribution using the powerlaw package
    fit = powerlaw.Fit(degrees, discrete=True)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin

    # Plot the power-law fit
    fit.power_law.plot_pdf(
        color="r",
        linestyle="--",
        label=f"Power-law fit: alpha={alpha:.2f}, xmin={xmin:.2f}",
    )

    plt.title("Degree Distribution with Power-law Fit")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(f"{directory}/{filename}_degree_distribution.png")
    plt.show()


def calculate_weight_statistics(weights: list):
    """
    Calculates statistical measures of the weights.
    """
    try:
        mode_weight = mode(weights)
    except StatisticsError:
        mode_weight = "No unique mode"

    median_weight = median(weights) if weights else 0
    mean_weight = mean(weights) if weights else 0
    max_weight = max(weights) if weights else 0
    min_weight = min(weights) if weights else 0

    return {
        "median": median_weight,
        "mean": mean_weight,
        "mode": mode_weight,
        "max": max_weight,
        "min": min_weight,
    }


def create_graph_from_file(
    file_path: str,
    directory: str,
    alpha: float,
    filename: str,
    n_lines: Optional[int] = None,
    cdi_threshold: float = 5.0,
    eei_threshold: float = 1.0,
    apply_pareto: bool = True,
):
    """
    Creates a graph from a file and applies the Pareto principle to identify top nodes.

    Parameters:
    - file_path (str): Path to the file.
    - directory (str): Directory where plots will be saved.
    - alpha (float): Alpha value for the LouvainSigned algorithm.
    - filename (str): The name of the file where the plot will be saved.
    - n_lines (int, optional): Number of lines to read. If None, reads the whole file.
    - cdi_threshold (float, optional): Threshold to be used if filename contains "CDI". Default is 5.0.
    - eei_threshold (float, optional): Threshold to be used if filename contains "EEI". Default is 1.0.
    - apply_pareto (bool, optional): Whether to apply the Pareto principle to identify top nodes.

    Returns:
    - G (networkx.Graph): The created subgraph containing top nodes.
    - Dict: Statistical measures of the weights (median, mean, mode, max, min).
    - List: Top 20% nodes contributing to 80% of the connectivity.
    """
    G, weights, node_degrees = read_file_and_create_graph(
        file_path, n_lines, cdi_threshold, eei_threshold
    )

    if G is None:
        return None, {}, []

    weight_stats = calculate_weight_statistics(weights)
    plot_weight_distribution(
        weights,
        weight_stats["median"],
        weight_stats["mean"],
        weight_stats["mode"],
        weight_stats["max"],
        weight_stats["min"],
        directory,
        filename,
        alpha,
    )
    plt.close()

    # Plot degree distribution
    plot_degree_distribution(G, directory, filename)
    plt.close()

    if apply_pareto:
        sorted_nodes_by_degree, total_degree = calculate_degree_distribution(
            node_degrees
        )
        top_nodes = apply_pareto_principle(sorted_nodes_by_degree, total_degree)

        # Create subgraph with top nodes
        subgraph = G.subgraph(top_nodes).copy()
        print(
            f"Graph: Nodes={subgraph.number_of_nodes()}, Edges={subgraph.number_of_edges()}"
        )

        return subgraph, weight_stats, top_nodes
    else:
        print(f"Graph: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")
        return G, weight_stats, []


# Example usage:
# G, stats, top_nodes = create_graph_from_file("path/to/file", "output_dir", 0.5, "output_filename")
# print(f"Top nodes contributing to 80% of connectivity: {top_nodes}")
