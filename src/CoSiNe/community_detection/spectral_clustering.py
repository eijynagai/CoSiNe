import numpy as np
from sklearn.cluster import SpectralClustering


def run_spectral_clustering(G, num_clusters=2):
    """
    Runs spectral clustering on the graph adjacency matrix with scikit-learn.

    Parameters
    ----------
    G : networkx.Graph
    num_clusters : int
        Number of clusters to find.

    Returns
    -------
    dict
        {node: cluster_label}.
    """
    # Build adjacency matrix in the same node order
    nodes = sorted(G.nodes())
    idx_map = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # Construct adjacency matrix (optionally weighted)
    A = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        i = idx_map[u]
        j = idx_map[v]
        w = d.get("weight", 1.0)
        A[i, j] = w
        A[j, i] = w

    # Run spectral clustering
    sc = SpectralClustering(
        n_clusters=num_clusters,
        affinity="precomputed",  # we pass a custom adjacency
        assign_labels="kmeans",
        random_state=42,
    )
    labels = sc.fit_predict(A)

    # Convert labels array -> dict
    community_dict = {}
    for i, node in enumerate(nodes):
        community_dict[node] = labels[i]
    return community_dict
