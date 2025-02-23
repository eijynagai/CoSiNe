# scripts/run_benchmark.py
import networkx as nx

from CoSiNe.benchmarks.benchmark_community_detection import (
    benchmark_community_detection,
)


def main():
    print("📂 Loading dataset...")
    graph = nx.karate_club_graph()  # Example dataset

    print("🚀 Running community detection benchmarks...")
    benchmark_community_detection(graph)


if __name__ == "__main__":
    main()
