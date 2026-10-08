"""PageRank, degree, betweenness and closeness centrality."""

import operator

import networkx as _nx

from ._dispatch import GLOBAL, LOCAL, Fallback, accelerate, need_directed, reach

__all__ = [
    "pagerank",
    "degree_centrality",
    "in_degree_centrality",
    "out_degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
]


@accelerate(_nx.pagerank)
def pagerank(
    snap,
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    N = len(G)
    if N == 0:
        return {}
    try:
        import numpy as np
    except ImportError:
        raise Fallback from None
    nodelist = list(G)

    # The three optional vectors are prepared with NumPy exactly as NetworkX
    # prepares them, then handed to Rust as plain lists.
    def vector(mapping):
        return np.array([mapping.get(n, 0) for n in nodelist], dtype=float)

    x = p = dangling_weights = None
    if nstart is not None:
        x = vector(nstart)
        x /= x.sum()
        x = x.tolist()
    if personalization is not None:
        p = vector(personalization)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()
        p = p.tolist()
    if dangling is not None:
        dangling_weights = vector(dangling)
        dangling_weights /= dangling_weights.sum()
        dangling_weights = dangling_weights.tolist()
    try:
        alpha, tol, max_iter = float(alpha), float(tol), max(operator.index(max_iter), 0)
    except (TypeError, ValueError):
        raise Fallback from None

    ranks = snap.pagerank(alpha, max_iter, tol, weight, p, x, dangling_weights)
    if ranks is None:
        raise _nx.PowerIterationFailedConvergence(max_iter)
    return ranks


# NetworkX computes degree centrality in one quick pass, in less time than a
# snapshot takes to build. LOCAL means: use a snapshot if one exists, but do
# not build one for this.
@accelerate(_nx.degree_centrality, tier=LOCAL)
def degree_centrality(snap, G):
    if len(G) <= 1:
        return {n: 1 for n in G}
    return snap.degree_centrality(0)


@accelerate(_nx.in_degree_centrality, tier=LOCAL)
def in_degree_centrality(snap, G):
    need_directed(snap)
    if len(G) <= 1:
        return {n: 1 for n in G}
    return snap.degree_centrality(1)


@accelerate(_nx.out_degree_centrality, tier=LOCAL)
def out_degree_centrality(snap, G):
    need_directed(snap)
    if len(G) <= 1:
        return {n: 1 for n in G}
    return snap.degree_centrality(2)


@accelerate(_nx.betweenness_centrality)
def betweenness_centrality(snap, G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    if k is not None and k != len(G):
        # Sampling draws from Python's random number generator; leave it to
        # NetworkX so seeded results are the same.
        raise Fallback
    return snap.betweenness(weight, bool(normalized), bool(endpoints))


def _closeness_tier(G, u=None, distance=None, wf_improved=True):
    # Closeness of one node needs the distances *to* it.
    return GLOBAL if u is None else reach(u, True)


@accelerate(_nx.closeness_centrality, tier=_closeness_tier)
def closeness_centrality(snap, G, u=None, distance=None, wf_improved=True):
    if u is None:
        return snap.closeness_all(distance, bool(wf_improved))
    return snap.closeness_of(u, distance, bool(wf_improved))
