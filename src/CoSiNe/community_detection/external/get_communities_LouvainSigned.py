#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
LouvainSigned: Louvain method for signed networks

This script provides an implementation of the Louvain method for detecting communities in gene networks.
It includes functionalities for graph analysis, community detections, and visualization of the results.

Usage:
    Run this script with Python 3.x. Additional library requirements are listed in requirements.txt.

"""

import argparse
import os
import sys

# Ensure the config module is imported correctly
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules')))
import config
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from modules.louvain_signed import LouvainSigned
from modules.network_builder import create_graph_from_file
from sklearn import metrics
from sklearn.metrics import auc, roc_curve
from utils.data_manipulation import *
from utils.metrics_visualization import *


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Community detection for signed networks"
    )
    parser.add_argument("g_pos_path", help="Path to the positive gene network file")
    parser.add_argument("g_neg_path", help="Path to the negative gene network file")
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="Alpha value for the LouvainSigned algorithm",
        default=1.0,
    )
    parser.add_argument(
        "-r",
        "--res",
        type=float,
        help="Resolution value for the LouvainSigned algorithm",
        default=1.0,
    )
    parser.add_argument("-s", "--sample", help="Sample name (any)")

    # Optional arguments
    parser.add_argument(
        "-c", "--cdi_t", type=float, help="CDI threshold for the input files"
    )
    parser.add_argument(
        "-e", "--eei_t", type=float, help="EEI threshold for the input files"
    )
    parser.add_argument(
        "--seed", help="Seed for random number generator", type=int, default=10
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Apply Pareto principle to identify top nodes",
    )
    return parser.parse_args()


def prepare_output_directory():
    directories = [
        config.NETWORKS_GRAPH_FILES_DIR,
        config.NETWORKS_PLOTS_DIR,
        config.PARTITIONS_FILES_DIR,
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return directories


def perform_community_detection(G_positive, G_negative, alpha_value, res_value, seed):
    louvain_instance = LouvainSigned(G_positive, G_negative)
    partition = louvain_instance.best_partition(
        alpha=alpha_value, resolution=res_value, seed=seed
    )
    graph_data = [(u, v, d["weight"]) for u, v, d in G_positive.edges(data=True)]
    graph_data.extend([(u, v, -d["weight"]) for u, v, d in G_negative.edges(data=True)])
    return partition, graph_data


def export_partition_to_txt(partition, filename):
    filepath = os.path.join(config.PARTITIONS_FILES_DIR, f"{filename}.txt")
    with open(filepath, "w") as file:
        for node, community in partition.items():
            file.write(f"{node}, {community}\n")
    print(f"Partition exported to {filepath}")


def visualize_results(partition, sample_name, threshold=2):
    partition_filtered = remove_small_communities(partition, min_size=threshold)
    plot_community_sizes(
        partition, config.PARTITIONS_PLOTS_SANKEY_DIR, f"{sample_name}_all_communities"
    )
    plot_community_sizes(
        partition_filtered,
        config.PARTITIONS_PLOTS_SANKEY_DIR,
        f"{sample_name}_filtered_communities",
    )
    save_communities(partition_filtered, config.PARTITIONS_FILES_DIR, sample_name)
    return partition_filtered


def main():
    args = parse_arguments()
    prepare_output_directory()
    sample = args.sample
    alpha_value = args.alpha
    res_value = args.res
    threshold = 100
    seed = args.seed
    sample_name = f"{sample}_a{alpha_value}_r{res_value}"
    pareto = args.pareto

    print(f"Running LouvainSigned")
    print(f"Sample: {sample}")
    print(f"Alpha value: {alpha_value}")
    print(f"Resolution value: {res_value}")
    print(f"Output directory: {config.PARTITIONS_FILES_DIR}")
    print(f"File name: {sample_name}")
    print(f"Seed: {seed}")
    print(f"Apply pareto: {pareto}")
    print("")

    G_positive, stats_positive, top_nodes_positive = create_graph_from_file(
        args.g_pos_path,
        config.NETWORKS_PLOTS_DIR,
        alpha_value,
        filename=f"{sample_name}_pos",
        apply_pareto=pareto,
    )
    G_negative, stats_negative, top_nodes_negative = create_graph_from_file(
        args.g_neg_path,
        config.NETWORKS_PLOTS_DIR,
        alpha_value,
        filename=f"{sample_name}_neg",
        apply_pareto=pareto,
    )

    if G_positive:
        print(
            f"G_positive: Nodes={G_positive.number_of_nodes()}, Edges={G_positive.number_of_edges()}"
        )
    else:
        print("Failed to create G_positive.")

    if G_negative:
        print(
            f"G_negative: Nodes={G_negative.number_of_nodes()}, Edges={G_negative.number_of_edges()}"
        )
    else:
        print("Failed to create G_negative.")

    # Overviews of the communities through alphas
    analyze_and_plot_communities(
        G_positive,
        G_negative,
        config.NETWORKS_PLOTS_DIR,
        sample_name,
        res_value,
        seed,
        include_scores=True,
    )

    # Perform community detection based on a given alpha value and resolution
    partition, graph_data = perform_community_detection(
        G_positive, G_negative, alpha_value, res_value, seed
    )

    # Export partition to a text file
    filename = f"{sample_name}_partition"
    export_partition_to_txt(partition, filename)

    # Debug: Check the structure of graph_data
    print(f"graph_data (first 10 entries): {graph_data[:10]}")
    print(f"Total number of entries in graph_data: {len(graph_data)}")

    # Save the graph data to a file
    filename = f"{sample_name}_graph"
    filepath = os.path.join(config.NETWORKS_GRAPH_FILES_DIR, f"{filename}.txt")
    try:
        graph_df = pd.DataFrame(graph_data, columns=["id1", "id2", "weight"])
        graph_df.to_csv(filepath, sep="\t", index=False)
    except Exception as e:
        print(f"Error creating DataFrame: {e}")

    # Visualize and filter the results
    partition_filtered = visualize_results(partition, sample_name, threshold)

    # Export filtered partition to a text file
    filename = f"{sample_name}_filtered_partition"
    export_partition_to_txt(partition_filtered, filename)

    try:
        print(
            f"Number of communities after removing clusters smaller than {threshold}: {max(set(partition_filtered.values())) + 1}\n"
        )
    except:
        print(
            f"No communities found after removing clusters smaller than {threshold}\n"
        )

    sys.exit()


if __name__ == "__main__":
    main()
