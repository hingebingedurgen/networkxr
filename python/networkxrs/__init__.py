"""networkxrs: NetworkX, with the hot algorithms running in Rust.

Change one line::

    import networkxrs as nx

Everything NetworkX provides is available under the same names. The
functions listed by :func:`networkxrs.accelerated` run in Rust and return what
NetworkX returns; everything else is NetworkX itself.

How it works
------------
Graphs are ordinary NetworkX graphs: ``networkxrs.Graph is networkx.Graph``.
The first time an accelerated function is called on a graph, its structure
is copied into a compact Rust snapshot, which is cached on the graph and
discarded when the graph changes. The algorithm runs on the snapshot.

An accelerated function hands the call to NetworkX whenever the input is
outside what the Rust code reproduces exactly: multigraphs, graph views,
callable weights, weights that are not plain ints or floats, and a few
rarely used options. Nothing changes for the caller except the speed.

A snapshot is only built when it will pay for itself, so small queries on
large graphs stay as cheap as they are in NetworkX. :func:`set_policy`
changes that, and :func:`dispatch_counts` shows what ran where.

One addition
------------
:func:`dag_longest_paths` returns *all* longest paths of a DAG. NetworkX has
no equivalent.
"""

import networkx as _nx

# Everything NetworkX exports, under the same names.
from networkx import *  # noqa: F401,F403

from . import _alias
from ._core import __version__ as __networkxrs_version__
from ._dispatch import ACCELERATED as _ACCELERATED
from ._dispatch import COUNTS as _COUNTS
from ._dispatch import get_policy, set_policy  # noqa: F401

# The Rust-backed functions. These imports replace the NetworkX functions of
# the same names brought in by the star import above.
from ._centrality import *  # noqa: F401,F403,E402
from ._components import *  # noqa: F401,F403,E402
from ._dag import *  # noqa: F401,F403,E402
from ._shortest_paths import *  # noqa: F401,F403,E402
from ._traversal import *  # noqa: F401,F403,E402
from ._tree import *  # noqa: F401,F403,E402

_alias.install()

# Code that checks `nx.__version__` is asking about the NetworkX API it can
# rely on, so report NetworkX's version. networkxrs's own version is
# `__networkxrs_version__`.
__version__ = _nx.__version__


def accelerated():
    """Return the sorted names of the functions that run in Rust."""
    return sorted(_ACCELERATED)


def dispatch_counts(reset=False):
    """Return how many calls each accelerated function answered in Rust and
    how many it handed to NetworkX, as ``{name: {"rust": n, "networkx": n}}``.

    Useful for checking that a workload is being accelerated. A call counted
    under ``"rust"`` that raised an exception (no path, say) is not counted.
    """
    counts = {}
    for (name, path), n in sorted(_COUNTS.items()):
        counts.setdefault(name, {"rust": 0, "networkx": 0})[path] = n
    if reset:
        _COUNTS.clear()
    return counts


def __getattr__(name):
    # Called for names not found above: private helpers, lazily loaded
    # submodules, and anything a newer NetworkX adds.
    return getattr(_nx, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_nx)))
