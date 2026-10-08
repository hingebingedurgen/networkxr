"""The machinery that routes a call to Rust or to NetworkX.

Every accelerated function is written as::

    @accelerate(networkx.some_function)
    def some_function(snap, G, ...same arguments as NetworkX...):
        ...call into Rust via ``snap``...

The decorator returns a function with NetworkX's name, signature and
docstring. When called it gets the Rust snapshot of ``G``, runs the Rust
implementation, and calls the original NetworkX function instead if there is
no snapshot or the implementation raises ``Fallback``.

``Fallback`` means "this input is outside what the Rust path reproduces
exactly". It is raised for multigraphs, graph views, callable weights,
weights that are not plain ints or floats, nodes that are not in the graph,
and so on. NetworkX then produces the result or the error message itself.

When is a snapshot worth building?
----------------------------------
Building a snapshot takes time proportional to the size of the graph. That
is a bargain for an algorithm that reads the whole graph, and a bad deal for
a query NetworkX answers by looking at ten nodes: ``has_path`` between two
neighbours of a million-node graph, say, in a loop that adds an edge between
calls and so invalidates the snapshot every time.

So each function declares how much of the graph a call is likely to read,
its *tier*:

``GLOBAL``
    Reads the whole graph (PageRank, components, spanning trees). The
    snapshot is built at once: NetworkX would spend longer than that anyway.

``reach(source)``
    Reads whatever ``source`` can reach (single-source shortest paths). A
    quick probe on the NetworkX graph checks whether that is a large part of
    the graph. If so, build. If not, see ``LOCAL``.

``LOCAL``
    May read very little (point-to-point queries, searches with a cutoff).
    NetworkX answers these calls and the time it takes is added up, per
    graph. Once that total reaches what a snapshot would cost, the snapshot
    is built and later calls use it. Until the graph changes, which resets
    the total.

Generators (``bfs_edges``, ``all_pairs_shortest_path``) are handled by
:func:`accelerate_stream`: the caller may stop reading after one item, so
they start out reading from NetworkX and switch to the Rust result once the
caller has consumed enough to show it is worth computing.

The effect is that networkxr should not be much slower than NetworkX on any
call pattern, and is much faster whenever there is real work to do.
:func:`set_policy` overrides all of this.
"""

import collections
import functools
import inspect
import itertools
import os
from time import perf_counter

import networkx as _nx

from ._core import Fallback, NegativeCycle, Snapshot, reaches

__all__ = [
    "ACCELERATED",
    "COUNTS",
    "GLOBAL",
    "LOCAL",
    "Fallback",
    "NegativeCycle",
    "accelerate",
    "accelerate_stream",
    "get_policy",
    "reach",
    "set_policy",
    "snapshot",
]

#: name -> accelerated function, for every function networkxr replaces.
ACCELERATED = {}

#: (function name, "rust" or "networkx") -> number of calls answered that way.
#: See networkxr.dispatch_counts().
COUNTS = collections.Counter()

# --- tiers ---------------------------------------------------------------------

GLOBAL = "global"
LOCAL = "local"


def reach(source, reverse=False):
    """The tier of a search that reads what ``source`` can reach, following
    edges backwards if ``reverse``."""
    return ("reach", source, reverse)


# --- policy --------------------------------------------------------------------

_POLICIES = ("adaptive", "eager", "off")
# The environment variable sets the starting policy without editing code.
_policy = os.environ.get("NETWORKXR_POLICY", "adaptive")
if _policy not in _POLICIES:
    raise ImportError(f"NETWORKXR_POLICY must be one of {_POLICIES}, not {_policy!r}")


def set_policy(policy):
    """Choose when networkxr uses Rust.

    ``"adaptive"`` (the default)
        Build a snapshot of a graph when a call, or the calls so far, justify
        its cost. See the module docstring of ``networkxr._dispatch``.
    ``"eager"``
        Build a snapshot on the first accelerated call, whatever it is. Best
        when graphs are large, rarely change and are queried many times.
    ``"off"``
        Never use Rust. Every call goes to NetworkX.

    The environment variable ``NETWORKXR_POLICY`` sets the policy a program
    starts with.
    """
    global _policy
    if policy not in _POLICIES:
        raise ValueError(f"policy must be one of {_POLICIES}, not {policy!r}")
    _policy = policy


def get_policy():
    """Return the current policy. See :func:`set_policy`."""
    return _policy


# --- the cost model ------------------------------------------------------------
#
# Sizes are measured in "units": nodes plus adjacency entries. The rates are
# rough, and only need to be right to within a small factor: they decide
# *when* to build a snapshot, never what is returned.

#: Seconds per unit to build a snapshot and read its weights once. Adjusted
#: at run time from the builds that actually happen.
BUILD_SECONDS_PER_UNIT = 2.0e-7

#: Seconds per unit for one linear-time pass in Rust, including turning the
#: result back into Python objects.
RUST_SECONDS_PER_UNIT = 3.0e-8

#: A search is "large" if it reaches this fraction of the nodes.
REACH_FRACTION = 0.3

# NetworkX gives every graph a dict, ``G.__networkx_cache__``, and empties it
# whenever the graph is changed through its methods. It exists so that
# backends can cache a converted copy of the graph. Everything networkxr
# remembers about a graph lives there, so NetworkX itself discards it at the
# right moments.
_SNAPSHOT = "networkxr"  # a Snapshot, or False if the graph is unsupported
_DEBT = "networkxr:debt"  # seconds NetworkX has spent on LOCAL calls
_SIZE = "networkxr:size"  # the graph's size in units
_EPOCH = "networkxr:epoch"  # an object identifying this state of the graph

# A subclass of Graph or DiGraph is supported only if it leaves all of these
# alone. Overriding a reader could change what the graph *means* (NetworkX
# itself has a private class whose adjacency dict stores the edges that are
# absent). Overriding a writer could change the graph without clearing the
# cache. A subclass that only adds methods of its own is fine.
_MUST_INHERIT = (
    # readers
    "__iter__", "__contains__", "__len__", "__getitem__", "adj", "nodes", "edges", "degree",
    "neighbors", "has_node", "has_edge", "get_edge_data", "nbunch_iter", "is_directed",
    "is_multigraph", "number_of_nodes", "number_of_edges",
    # writers
    "add_node", "add_nodes_from", "remove_node", "remove_nodes_from", "add_edge",
    "add_edges_from", "add_weighted_edges_from", "remove_edge", "remove_edges_from",
    "update", "clear", "clear_edges",
)  # fmt: skip
_MUST_INHERIT_DIRECTED = ("succ", "pred", "successors", "predecessors", "in_edges", "out_edges", "in_degree", "out_degree")

_class_supported = {}


def _supported_class(cls):
    """True if ``cls`` is Graph, DiGraph, or a subclass that behaves like one."""
    try:
        return _class_supported[cls]
    except KeyError:
        pass
    if cls in (_nx.Graph, _nx.DiGraph):
        ok = True
    elif issubclass(cls, _nx.DiGraph):
        names = _MUST_INHERIT + _MUST_INHERIT_DIRECTED
        ok = all(getattr(cls, n, None) is getattr(_nx.DiGraph, n) for n in names)
    elif issubclass(cls, _nx.Graph):
        ok = all(getattr(cls, n, None) is getattr(_nx.Graph, n) for n in _MUST_INHERIT)
    else:
        ok = False
    _class_supported[cls] = ok
    return ok


def _supported(G):
    """True if a snapshot of ``G`` can be built and kept up to date."""
    try:
        if not _supported_class(type(G)):
            return False
        # Graph views keep a reference to the graph they show in `_graph`.
        # Some views share the original's adjacency dicts, and all have a
        # cache of their own that is *not* cleared when the original graph
        # changes, so a snapshot of a view could go stale.
        if hasattr(G, "_graph"):
            return False
        # Custom graph classes may use other mapping types here.
        if type(G._adj) is not dict or G.is_multigraph():
            return False
        return not G.is_directed() or type(G._pred) is dict
    except AttributeError:
        return False


def _lookup(G):
    """Return ``(cache, entry)`` for ``G``.

    ``entry`` is the graph's Snapshot, ``False`` if it cannot have one, or
    ``None`` if it could have one but none has been built.
    """
    # NetworkX sets the cache to None on graphs it is about to modify behind
    # the methods' backs (flow residual networks, for example).
    cache = getattr(G, "__networkx_cache__", None)
    if cache is None:
        return None, False
    entry = cache.get(_SNAPSHOT)
    if entry is None and not _supported(G):
        entry = cache[_SNAPSHOT] = False
    return cache, entry


def _build(G, cache):
    """Build, store and return the snapshot of ``G`` (False if it fails)."""
    global BUILD_SECONDS_PER_UNIT
    start = perf_counter()
    try:
        snap = Snapshot(G._adj, G._pred if G.is_directed() else None)
    except Fallback:
        snap = False
    cache[_SNAPSHOT] = snap
    if snap:
        units = snap.number_of_nodes + 2 * snap.number_of_edges
        if units >= 50_000:
            # Learn this machine's speed from builds big enough to time. The
            # factor 1.5 allows for reading the weights afterwards.
            measured = 1.5 * (perf_counter() - start) / units
            BUILD_SECONDS_PER_UNIT = 0.5 * BUILD_SECONDS_PER_UNIT + 0.5 * measured
    return snap


def snapshot(G):
    """Return the Rust snapshot of ``G``, building it if need be, or None if
    there cannot be one.

    The snapshot holds the graph's structure. Edge attributes are not copied:
    the snapshot keeps references to the edge attribute dicts and reads
    weights from them on every call, so changing an attribute in place
    (``G[u][v]["weight"] = 3``) is always seen.
    """
    cache, entry = _lookup(G)
    if entry is None:
        entry = _build(G, cache)
    return entry or None


def _size(G, cache):
    """The size of ``G`` in units, counted once per state of the graph."""
    size = cache.get(_SIZE)
    if size is None:
        adj = G._adj
        size = cache[_SIZE] = len(adj) + sum(map(len, adj.values()))
    return size


def _worth(G, cache, seconds, rate):
    """True if ``seconds`` of NetworkX's time is at least what ``rate``
    seconds per unit would come to on ``G``."""
    # Counting the edges takes a moment on a big graph, so first compare
    # against the node count alone, which is free and can only be lower.
    if seconds < len(G._adj) * rate:
        return False
    return seconds >= _size(G, cache) * rate


def _decide(G, cache, tier, args, kwargs):
    """With no snapshot yet: build one now (returning it) or not (None)."""
    if _policy == "eager":
        return _build(G, cache)
    if callable(tier):
        try:
            tier = tier(G, *args, **kwargs)
        except TypeError:
            return None  # the arguments do not fit; NetworkX will say so
    if tier is GLOBAL or _worth(G, cache, cache.get(_DEBT, 0.0), BUILD_SECONDS_PER_UNIT):
        return _build(G, cache)
    if tier is LOCAL:
        return None
    _, source, reverse = tier
    adj = G._pred if reverse and G.is_directed() else G._adj
    try:
        large = reaches(adj, source, max(1, int(len(adj) * REACH_FRACTION)))
    except TypeError:
        return None  # an unhashable source; NetworkX will say so
    return _build(G, cache) if large else None


def _accepts(impl):
    """Return a function telling whether ``impl`` can take given arguments.

    ``impl`` has the parameters of the NetworkX function it stands in for,
    as of the NetworkX version it was written against. A later NetworkX may
    add a parameter. A call using one must go to NetworkX, not fail here.
    """
    parameters = list(inspect.signature(impl).parameters.values())[2:]  # skip snap, G
    positional = sum(p.kind is p.POSITIONAL_OR_KEYWORD for p in parameters)
    keywords = frozenset(p.name for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))

    def accepts(args, kwargs):
        return len(args) <= positional and (not kwargs or keywords.issuperset(kwargs))

    return accepts


def accelerate(nx_func, tier=GLOBAL):
    """Decorator: register ``impl`` as the Rust-backed version of ``nx_func``.

    ``tier`` is ``GLOBAL``, ``LOCAL``, or a function taking the call's
    arguments and returning ``GLOBAL``, ``LOCAL`` or ``reach(source)``.
    """

    def decorator(impl):
        name = nx_func.__name__
        via_rust, via_networkx = (name, "rust"), (name, "networkx")
        accepts = _accepts(impl)

        @functools.wraps(nx_func)
        def wrapper(G, *args, **kwargs):
            # Arguments we do not know (including `backend=`, which asks for
            # NetworkX's own dispatch) go straight to NetworkX.
            if _policy == "off" or not accepts(args, kwargs):
                COUNTS[via_networkx] += 1
                return nx_func(G, *args, **kwargs)
            cache, snap = _lookup(G)
            if snap is None:
                snap = _decide(G, cache, tier, args, kwargs)
            if snap:
                try:
                    result = impl(snap, G, *args, **kwargs)
                except Fallback:
                    pass
                else:
                    COUNTS[via_rust] += 1
                    return result
            COUNTS[via_networkx] += 1
            if snap is None:
                # A snapshot was possible but not yet worth it. Record what
                # NetworkX spends, so that enough such calls build one.
                start = perf_counter()
                try:
                    return nx_func(G, *args, **kwargs)
                finally:
                    cache[_DEBT] = cache.get(_DEBT, 0.0) + perf_counter() - start
            return nx_func(G, *args, **kwargs)

        wrapper.__networkxr_impl__ = impl
        ACCELERATED[name] = wrapper
        return wrapper

    return decorator


def accelerate_stream(nx_func):
    """Decorator for NetworkX functions that return a generator the caller
    might only partly consume.

    ``impl(snap, G, ...)`` computes the whole result in Rust and returns a
    function ``resume(k)`` giving an iterator over it from item ``k`` on. The
    items must be exactly those NetworkX yields, in the same order.

    The generator returned to the caller starts by handing out NetworkX's
    items. Once NetworkX has spent as long as the Rust computation would
    take, the caller clearly wants a good part of the result: the whole of
    it is computed in Rust and the rest comes from there, skipping the items
    already given out.
    """

    def decorator(impl):
        name = nx_func.__name__
        via_rust, via_networkx = (name, "rust"), (name, "networkx")
        accepts = _accepts(impl)

        def from_rust(G, cache, epoch, snap, args, kwargs, done):
            """Return an iterator over the Rust result from item ``done`` on,
            or None to stay with NetworkX."""
            if cache.get(_EPOCH) is not epoch:
                return None  # the graph changed while we were iterating
            if snap is None:
                snap = _build(G, cache)
            if not snap:
                return None
            try:
                return impl(snap, G, *args, **kwargs)(done)
            except Fallback:
                return None

        def stream(G, args, kwargs):
            # A generator's body runs when the caller first asks for an item,
            # which may be long after the call. Look at the graph now.
            cache, snap = _lookup(G)
            if snap is False:
                COUNTS[via_networkx] += 1
                yield from nx_func(G, *args, **kwargs)
                return
            # Marks this state of the graph. NetworkX empties the cache when
            # the graph changes, which removes the mark.
            epoch = cache.get(_EPOCH)
            if epoch is None:
                epoch = cache[_EPOCH] = object()

            # When told to be eager, or with a snapshot in hand and a graph
            # so small that computing everything costs next to nothing, go
            # straight to Rust.
            if _policy == "eager" or (
                snap and (snap.number_of_nodes + 2 * snap.number_of_edges) * RUST_SECONDS_PER_UNIT < 2e-5
            ):
                rest = from_rust(G, cache, epoch, snap, args, kwargs, 0)
                if rest is not None:
                    COUNTS[via_rust] += 1
                    yield from rest
                    return

            source = nx_func(G, *args, **kwargs)
            done = 0  # items handed out so far
            spent = 0.0  # seconds NetworkX took to produce them
            block = 1  # items to take from NetworkX next; doubles each time
            may_switch = True
            try:
                while True:
                    start = perf_counter()
                    items = list(itertools.islice(source, block))
                    spent += perf_counter() - start
                    yield from items
                    done += len(items)
                    if len(items) < block:
                        break  # NetworkX's generator is exhausted
                    block = min(2 * block, 1024)
                    if may_switch:
                        rate = RUST_SECONDS_PER_UNIT + (0.0 if snap else BUILD_SECONDS_PER_UNIT)
                        if _worth(G, cache, spent, rate):
                            rest = from_rust(G, cache, epoch, snap, args, kwargs, done)
                            if rest is None:
                                may_switch = False
                            else:
                                COUNTS[via_rust] += 1
                                spent = 0.0
                                source = None  # let go of NetworkX's generator
                                yield from rest
                                return
            finally:
                # Also reached if the caller abandons the generator.
                if spent:
                    COUNTS[via_networkx] += 1
                    if cache.get(_EPOCH) is epoch:
                        cache[_DEBT] = cache.get(_DEBT, 0.0) + spent

        @functools.wraps(nx_func)
        def wrapper(G, *args, **kwargs):
            if _policy == "off" or not accepts(args, kwargs):
                COUNTS[via_networkx] += 1
                return nx_func(G, *args, **kwargs)
            return stream(G, args, kwargs)

        wrapper.__networkxr_impl__ = impl
        ACCELERATED[name] = wrapper
        return wrapper

    return decorator


# --- helpers for the implementations ---------------------------------------------


def as_generator(items):
    """Yield from ``items``. NetworkX returns generators from many functions;
    this keeps the return type the same."""
    yield from items


def from_list(items):
    """A ``resume`` function (see :func:`accelerate_stream`) over a list."""
    return lambda done: itertools.islice(items, done, None)


def chunked(G, snap, fetch):
    """A ``resume`` function over one result per node, computed in parallel
    batches.

    ``fetch(start, stop)`` returns a list of results for node indices
    ``start..stop``. One batch is fetched before this returns, so that a
    ``Fallback`` is raised to the caller and not from inside a generator.
    """
    n = snap.number_of_nodes
    # Bound the memory held at once: each result is roughly one entry per node.
    largest = max(1, min(256, 2_000_000 // max(n, 1)))
    nodes = list(G)
    probe = fetch(0, min(n, 1))

    def resume(done):
        # A caller who has already read `done` results will read more, so
        # start with batches of about that size.
        start, size = done, max(1, min(done, largest))
        if done == 0:
            yield from zip(nodes[:1], probe)
            start = 1
        while start < n:
            stop = min(n, start + size)
            yield from zip(nodes[start:stop], fetch(start, stop))
            start, size = stop, min(2 * size, largest)

    return resume


def need_directed(snap):
    if not snap.directed:
        raise Fallback


def need_undirected(snap):
    if snap.directed:
        raise Fallback


def depth(limit):
    """A ``depth_limit`` argument as a non-negative int or None."""
    if limit is None:
        return None
    if type(limit) is int:
        return max(limit, 0)
    raise Fallback
