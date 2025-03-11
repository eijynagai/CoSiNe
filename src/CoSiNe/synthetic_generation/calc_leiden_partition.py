#!/usr/bin/env python3

import argparse
import csv

import igraph as ig
import leidenalg as la


def run_leiden(input_csv: str, output_partition: str):
    """
    Reads an edge list from a CSV, runs Leiden, and writes node->community assignments.

    Parameters
    ----------
    input_csv : str
        Path to the CSV file containing edges. Each row: "u,v" for zero-based node IDs.
    output_partition : str
        Path to the output file listing (node, community) per line.
    """
    edges = []

    # 1) Read edges from CSV
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # Each row is something like ["0", "1"]
            u_str, v_str = row
            u, v = int(u_str), int(v_str)
            edges.append((u, v))

    # 2) Build an igraph Graph
    #    - We need to know how many nodes => max node ID + 1
    max_node_id = max(max(u, v) for (u, v) in edges)
    G = ig.Graph()
    G.add_vertices(max_node_id + 1)  # zero-based IDs from 0..max_node_id
    G.add_edges(edges)

    # 3) Run Leiden with default partition (ModularityVertexPartition)
    partition = la.find_partition(G, la.ModularityVertexPartition)

    # 4) Write out (node, community_id) for each node
    with open(output_partition, "w", encoding="utf-8") as out:
        for node, comm_id in enumerate(partition.membership):
            out.write(f"{node},{comm_id}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Leiden on a cleaned CSV edgelist."
    )
    parser.add_argument(
        "--input", required=True, help="Cleaned CSV edgelist (two columns of node IDs)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="File to save node->community assignments (CSV).",
    )
    args = parser.parse_args()

    run_leiden(args.input, args.output)


if __name__ == "__main__":
    main()
