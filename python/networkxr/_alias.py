"""Make ``networkxr.<submodule>`` resolve to ``networkx.<submodule>``.

Code written for NetworkX imports submodules, for example::

    from networkx.algorithms import bipartite
    import networkx.drawing.nx_pydot

After swapping the package name those lines read ``networkxr.algorithms`` and
``networkxr.drawing.nx_pydot``, which do not exist as files here. The finder
below is asked by Python's import system for every ``networkxr.*`` module
that is not one of networkxr's own files, and answers with the NetworkX
module of the same name.

It has to be asked *first*. Once ``networkxr.algorithms`` is the module
``networkx.algorithms``, Python's normal finder would look for
``networkxr.algorithms.bipartite`` in NetworkX's directory, find the file,
and load a second copy of it under the new name. Two copies of a module mean
two copies of every class in it, which breaks ``isinstance`` checks.
"""

import importlib
import importlib.abc
import importlib.util
import sys

_PREFIX = "networkxr."
_OWN_MODULES = {"conformance"}


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        tail = fullname[len(_PREFIX) :]
        # networkxr's own modules: everything starting with an underscore,
        # plus `conformance`. Returning None passes the question on to the
        # normal finders.
        if tail.startswith("_") or tail.split(".")[0] in _OWN_MODULES:
            return None
        try:
            importlib.import_module("networkx." + tail)
        except ImportError:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        # Hand back the NetworkX module itself, not a copy, so that
        # `networkxr.algorithms is networkx.algorithms`.
        module = sys.modules["networkx." + spec.name[len(_PREFIX) :]]
        # The import system is about to overwrite `__spec__` with the alias
        # spec. Remember the real one so `exec_module` can put it back.
        self._real_spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self._real_spec


def install():
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
