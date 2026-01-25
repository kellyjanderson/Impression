#!/usr/bin/env bash
set -euo pipefail

probe_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$probe_root/../.." && pwd)

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    py="$VIRTUAL_ENV/bin/python"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    py="$repo_root/.venv/bin/python"
else
    if command -v python3 >/dev/null 2>&1; then
        sys_py=python3
    elif command -v python >/dev/null 2>&1; then
        sys_py=python
    else
        echo "Python is not available on PATH." >&2
        exit 1
    fi
    tmp_venv=$(mktemp -d "${TMPDIR:-/tmp}/impression-manifold-probe-venv.XXXXXX")
    "$sys_py" -m venv "$tmp_venv"
    py="$tmp_venv/bin/python"
fi

mode=${1:-serial}
case "$mode" in
    serial)
        export CMAKE_ARGS="-DMANIFOLD_PAR=OFF"
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
