#!/usr/bin/env bash
set -euo pipefail

probe_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$probe_root/../.." && pwd)

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

mode=${1:-serial}
case "$mode" in
    serial)
        export CMAKE_ARGS="-DMANIFOLD_PARALLEL=OFF"
        ;;
    parallel)
        unset CMAKE_ARGS
        ;;
    *)
        echo "Usage: probe.sh [serial|parallel]" >&2
        exit 1
        ;;
esac

mkdir -p "$probe_root/dist"

"$py" -m pip install --upgrade pip build >/dev/null
"$py" -m pip wheel --no-deps --no-binary=:all: manifold3d -w "$probe_root/dist"

echo "Built manifold3d wheel in $probe_root/dist"
