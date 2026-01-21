#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

if [[ -x "$repo_root/.venv/bin/python" ]]; then
    py="$repo_root/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    py=python
elif command -v python3 >/dev/null 2>&1; then
    py=python3
else
    echo "Python is not available on PATH." >&2
    exit 1
fi

"$py" -m pip install --upgrade build >/dev/null
"$py" -m build "$repo_root"

wheel=$(ls -t "$repo_root"/dist/impression-*.whl 2>/dev/null | head -n 1 || true)
if [[ -z "$wheel" ]]; then
    echo "No wheel found in $repo_root/dist" >&2
    exit 1
fi

# Ensure manifold3d builds without TBB by forcing serial mode.
CMAKE_ARGS="-DMANIFOLD_PARALLEL=OFF" "$py" -m pip install --upgrade --no-binary=:all: manifold3d

# Install the freshly built wheel.
CMAKE_ARGS="-DMANIFOLD_PARALLEL=OFF" "$py" -m pip install --upgrade --force-reinstall "$wheel"
