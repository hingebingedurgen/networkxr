#!/usr/bin/env python3
"""Rename the project.

    python tools/rename.py fastnx

Replaces the name ``networkxrs`` everywhere: the Python package, the import
name, the crate names, the environment variable, and every mention in the
docs and tests. Run it from a clean checkout, then rebuild and run the tests.

The name has to be a valid Python identifier, because it is what people will
write after ``import``. Check that it is free on https://pypi.org and
https://crates.io before settling on it.
"""

import pathlib
import sys

OLD = "networkxrs"
SKIP_DIRS = {".git", "target", "dist", "__pycache__", ".pytest_cache", ".venv"}
TEXT_SUFFIXES = {".py", ".rs", ".toml", ".md", ".yml", ".yaml", ".txt", ".sh", ".lock", ""}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    new = sys.argv[1]
    if not new.isidentifier() or new != new.lower():
        sys.exit(f"{new!r} is not a lowercase Python identifier")
    root = pathlib.Path(__file__).resolve().parent.parent

    def wanted(path):
        return not SKIP_DIRS.intersection(path.relative_to(root).parts)

    # File contents first, while the paths are still the old ones.
    changed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not wanted(path) or path.suffix not in TEXT_SUFFIXES:
            continue
        if path.resolve() == pathlib.Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # The crate identifier (networkxr_core) and the environment variable
        # (NETWORKXR_POLICY) are covered by the lower- and upper-case forms.
        updated = text.replace(OLD, new).replace(OLD.upper(), new.upper())
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed += 1

    # Then directories and files with the name in them, deepest first so a
    # parent is not renamed before its children.
    renamed = 0
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if wanted(path) and OLD in path.name:
            path.rename(path.with_name(path.name.replace(OLD, new)))
            renamed += 1

    print(f"{changed} files edited, {renamed} paths renamed: {OLD} -> {new}")
    print("Now rebuild (./build.sh) and run the tests (pytest tests).")


if __name__ == "__main__":
    main()
