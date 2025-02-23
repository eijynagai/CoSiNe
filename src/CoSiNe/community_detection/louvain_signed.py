from CoSiNe.community_detection.external.louvain_signed import LouvainSigned


def run_louvain_signed(G_positive, G_negative, alpha=1.0, resolution=1):
    """
    Runs LouvainSigned on the given positive and negative graphs.

    Parameters:
        G_positive (networkx.Graph): Positive edges graph.
        G_negative (networkx.Graph): Negative edges graph.
        alpha (float): Balance parameter.
        resolution (float): Resolution parameter.

    Returns:
        dict: Node partitions (community assignments).
    """
    partition = LouvainSigned(G_positive, G_negative).best_partition(
        alpha=alpha, resolution=resolution
    )
    return partition
