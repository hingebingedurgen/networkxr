"""Run NetworkX's own test-suite against networkxr.

NetworkX ships its tests inside the installed package. They call functions
as ``nx.shortest_path(...)`` after ``import networkx as nx``. :func:`patch`
replaces those attributes on the ``networkx`` module with networkxr's
versions, so the unmodified NetworkX tests exercise the Rust code.

From the command line::

    python -m networkxr.conformance            # the relevant test modules
    python -m networkxr.conformance -x -q      # extra arguments go to pytest

As a pytest plugin, to run any part of NetworkX's suite::

    pytest -p networkxr.conformance --pyargs networkx.algorithms
"""

import sys

import networkx

from ._dispatch import ACCELERATED

#: The NetworkX test packages that cover the accelerated functions.
TEST_TARGETS = [
    "networkx.algorithms.traversal.tests",
    "networkx.algorithms.components.tests",
    "networkx.algorithms.shortest_paths.tests",
    "networkx.algorithms.centrality.tests",
    "networkx.algorithms.link_analysis.tests",
    "networkx.algorithms.tree.tests",
    "networkx.algorithms.tests.test_dag",
    "networkx.algorithms.tests.test_cycles",
]

#: NetworkX tests networkxr is known not to pass, and why. They are skipped
#: whenever this module is loaded as a pytest plugin.
KNOWN_DIFFERENCES = {
    "networkx.algorithms.tests.test_dag::TestDAG::test_topological_sort6": (
        "Adds and removes nodes while iterating over topological_sort and expects "
        "the generator to notice. networkxr computes the whole order when the "
        "function is called, so later changes to the graph are not seen."
    ),
}

_originals = {}


def patch():
    """Replace the accelerated functions on the ``networkx`` module."""
    for name, function in ACCELERATED.items():
        if name not in _originals:
            _originals[name] = getattr(networkx, name)
            setattr(networkx, name, function)


def unpatch():
    """Undo :func:`patch`."""
    while _originals:
        name, function = _originals.popitem()
        setattr(networkx, name, function)


def pytest_configure(config):
    """Hook called by pytest when this module is loaded with ``-p``."""
    patch()


def pytest_collection_modifyitems(config, items):
    """Hook called by pytest with the collected tests: drop the known
    differences and report them as deselected."""
    kept, dropped = [], []
    for item in items:
        cls = f"{item.cls.__name__}::" if getattr(item, "cls", None) else ""
        key = f"{item.module.__name__}::{cls}{item.name}"
        (dropped if key in KNOWN_DIFFERENCES else kept).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


def main(argv=None):
    import pytest

    args = list(sys.argv[1:] if argv is None else argv)
    return pytest.main(["-p", "networkxr.conformance", "--pyargs", *TEST_TARGETS, *args])


if __name__ == "__main__":
    sys.exit(main())
