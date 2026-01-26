#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
repo_url="${IMPRESSION_REPO_URL:-https://github.com/kellyjanderson/Impression.git}"
install_source="${IMPRESSION_INSTALL_SOURCE:-release}"
release_ref="${IMPRESSION_RELEASE:-}"

venv_path=""
use_local=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--list)
            if ! command -v git >/dev/null 2>&1; then
                echo "git is required to list releases." >&2
                exit 1
            fi
            git ls-remote --tags --sort=-v:refname "$repo_url" \
                | awk '{print $2}' \
                | sed 's#refs/tags/##;s#\\^{}##' \
                | awk 'NF && !seen[$0]++ {print}'
            exit 0
            ;;
        --venv)
            venv_path="$2"
            shift 2
            ;;
        --release)
            release_ref="$2"
            install_source="release"
            shift 2
            ;;
        --local)
            use_local=1
            install_source="local"
            shift 1
            ;;
        *)
            echo "Usage: install_impression.sh [--venv PATH] [--release TAG] [--local] [--list]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$venv_path" ]]; then
    venv_path="$(pwd)/.venv"
fi

release_dir=""
if [[ "$install_source" == "release" ]]; then
    if [[ -z "$release_ref" ]]; then
        if ! command -v git >/dev/null 2>&1; then
            echo "git is required to install a release. Install git or use --local." >&2
            exit 1
        fi
        release_ref="$(git ls-remote --tags --sort=-v:refname "$repo_url" \
            | awk '{print $2}' \
            | sed 's#refs/tags/##;s#\\^{}##' \
            | awk 'NF && !seen[$0]++ {print; exit}')"
        if [[ -z "$release_ref" ]]; then
            echo "No releases found at $repo_url" >&2
            exit 1
        fi
    fi
    if ! command -v git >/dev/null 2>&1; then
        echo "git is required to install a release. Install git or use --local." >&2
        exit 1
    fi
    release_dir="$(mktemp -d)"
    trap '[[ -n "$release_dir" ]] && rm -rf "$release_dir"' EXIT
    git clone --depth 1 --branch "$release_ref" "$repo_url" "$release_dir/impression" >/dev/null 2>&1
    repo_root="$release_dir/impression"
else
    if command -v git >/dev/null 2>&1 && git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        if git -C "$repo_root" diff --quiet && git -C "$repo_root" diff --cached --quiet; then
            git -C "$repo_root" fetch --all --prune
            if ! git -C "$repo_root" pull --ff-only; then
                echo "Warning: could not fast-forward; skipping repo update." >&2
            fi
        else
            echo "Warning: repo has local changes; skipping git pull." >&2
        fi
    fi
fi

python_version="${IMPRESSION_PYTHON_VERSION:-3.13}"

create_venv() {
    if [[ -n "${IMPRESSION_PYTHON:-}" && -x "$IMPRESSION_PYTHON" ]]; then
        "$IMPRESSION_PYTHON" -m venv "$venv_path"
        return 0
    fi

    if command -v uv >/dev/null 2>&1; then
        if uv venv --python "$python_version" "$venv_path"; then
            return 0
        fi
        echo "uv failed to provision Python ${python_version}; falling back to Homebrew." >&2
    fi

    if command -v brew >/dev/null 2>&1; then
        if ! brew list --versions "python@${python_version}" >/dev/null 2>&1; then
            brew install "python@${python_version}"
        fi
        brew_prefix=$(brew --prefix "python@${python_version}")
        if [[ -x "$brew_prefix/bin/python${python_version}" ]]; then
            "$brew_prefix/bin/python${python_version}" -m venv "$venv_path"
            return 0
        fi
    fi

    for candidate in "python${python_version}" "python3.13" "python3.12"; do
        if command -v "$candidate" >/dev/null 2>&1; then
            "$candidate" -m venv "$venv_path"
            return 0
        fi
    done

    echo "Unable to create a Python ${python_version} venv. Install uv or python@${python_version}." >&2
    exit 1
}

if [[ -x "$venv_path/bin/python" ]]; then
    if ! "$venv_path/bin/python" - <<PY
import sys
expected = tuple(int(x) for x in "${python_version}".split("."))
sys.exit(0 if sys.version_info[:2] == expected else 1)
PY
    then
        if [[ "${IMPRESSION_RECREATE_VENV:-}" == "1" ]]; then
            rm -rf "$venv_path"
            create_venv
        else
            echo "Existing venv uses a different Python version. Delete $venv_path or set IMPRESSION_RECREATE_VENV=1." >&2
            exit 1
        fi
    fi
else
    echo "Creating venv at $venv_path (Python ${python_version})"
    create_venv
fi

py="$venv_path/bin/python"

"$py" -m pip install --upgrade pip >/dev/null
"$py" -m pip install --upgrade build scikit-build-core cmake ninja >/dev/null
"$py" -m build "$repo_root"

wheel=$(ls -t "$repo_root"/dist/impression-*.whl 2>/dev/null | head -n 1 || true)
if [[ -z "$wheel" ]]; then
    echo "No wheel found in $repo_root/dist" >&2
    exit 1
fi

# Ensure manifold3d is available. If already installed, skip rebuilding it.
if "$py" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("manifold3d") else 1)
PY
then
    echo "manifold3d already installed; skipping build."
else
    CMAKE_ARGS="-DMANIFOLD_PAR=OFF" "$py" -m pip install --upgrade --no-binary=:all: manifold3d
fi

# Ensure PyVista (viewer) is available in the target venv.
"$py" -m pip install --upgrade pyvista

# Install the freshly built wheel.
CMAKE_ARGS="-DMANIFOLD_PAR=OFF" "$py" -m pip install --upgrade --force-reinstall "$wheel"
