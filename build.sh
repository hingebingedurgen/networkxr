#!/bin/sh
# Build an optimised wheel and install it into the current Python.
set -e
cd "$(dirname "$0")"
rm -rf dist
maturin build --release -o dist
pip install --force-reinstall --no-deps dist/*.whl
