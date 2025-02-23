# tests/test_community_detection.py
import networkx as nx

from CoSiNe.community_detection.infomap import run_infomap
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed


def test_louvain():
    graph = nx.karate_club_graph()
    communities = run_louvain(graph)
    assert len(communities) > 1  # Should detect at least 2 communities


def test_leiden():
    graph = nx.karate_club_graph()
    communities = run_leiden(graph)
    assert len(communities) > 1


def test_infomap():
    graph = nx.karate_club_graph()
    communities = run_infomap(graph)
    assert len(communities) > 1


def test_louvain_signed():
    G_positive = nx.karate_club_graph()
    G_negative = G_positive.copy()  # nx.Graph()
    # G_negative.add_edges_from([(0, 1), (2, 3)])  # Example negative edges
    communities = run_louvain_signed(G_positive, G_negative)
    assert len(set(communities.values())) > 1  # Should detect at least 2 communities
