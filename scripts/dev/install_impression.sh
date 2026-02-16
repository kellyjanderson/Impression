#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
repo_url="${IMPRESSION_REPO_URL:-https://github.com/kellyjanderson/Impression.git}"
install_source="${IMPRESSION_INSTALL_SOURCE:-release}"
release_ref="${IMPRESSION_RELEASE:-}"
interactive=0
list_only=0

log() {
    echo "[impression-install] $*"
}

venv_path=""
use_local=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--list)
            list_only=1
            shift 1
            ;;
        -i|--interactive)
            interactive=1
            install_source="release"
            shift 1
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
            echo "Usage: install_impression.sh [--venv PATH] [--release TAG] [--local] [--list] [--interactive]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$venv_path" ]]; then
    venv_path="$(pwd)/.venv"
fi

list_releases() {
    if ! command -v git >/dev/null 2>&1; then
        echo "git is required to list releases." >&2
        exit 1
    fi
    git ls-remote --tags --sort=-v:refname "$repo_url" \
        | awk '{print $2}' \
        | sed 's#refs/tags/##;s#\^{}##' \
        | awk 'NF && !seen[$0]++ {print}'
}

read_project_version() {
    local pyproject="$1/pyproject.toml"
    if [[ ! -f "$pyproject" ]]; then
        return 1
    fi
    awk -F'"' '/^version[[:space:]]*=[[:space:]]*"/ {print $2; exit}' "$pyproject"
}

set_project_version() {
    local root="$1"
    local version="$2"
    local pyproject="$root/pyproject.toml"
    local init_py="$root/src/impression/__init__.py"
    local tmp=""

    if [[ ! -f "$pyproject" || ! -f "$init_py" ]]; then
        return 1
    fi

    tmp="$(mktemp)"
    awk -v v="$version" '
        BEGIN { updated=0 }
        /^version[[:space:]]*=[[:space:]]*"/ {
            print "version = \"" v "\""
            updated=1
            next
        }
        { print }
        END { if (!updated) exit 2 }
    ' "$pyproject" > "$tmp"
    mv "$tmp" "$pyproject"

    tmp="$(mktemp)"
    awk -v v="$version" '
        BEGIN { updated=0 }
        /^__version__[[:space:]]*=[[:space:]]*"/ {
            print "__version__ = \"" v "\""
            updated=1
            next
        }
        { print }
        END { if (!updated) exit 2 }
    ' "$init_py" > "$tmp"
    mv "$tmp" "$init_py"
}

if [[ "${list_only:-0}" == "1" ]]; then
    list_releases
    exit 0
fi

choose_release_interactive() {
    if [[ ! -t 0 ]]; then
        echo "Interactive mode requires a TTY." >&2
        exit 1
    fi
    local releases=()
    if command -v mapfile >/dev/null 2>&1; then
        mapfile -t releases < <(list_releases)
    else
        while IFS= read -r line; do
            releases+=("$line")
        done < <(list_releases)
    fi
    if [[ ${#releases[@]} -eq 0 ]]; then
        echo "No releases found at $repo_url" >&2
        exit 1
    fi
    local index=0
    while true; do
        printf "\\033c"
        echo "Select Impression release (arrow keys, Enter to confirm):"
        echo
        for i in "${!releases[@]}"; do
            if [[ "$i" -eq "$index" ]]; then
                printf "  > %s\\n" "${releases[$i]}"
            else
                printf "    %s\\n" "${releases[$i]}"
            fi
        done
        IFS= read -rsn1 key
        if [[ -z "$key" ]]; then
            break
        fi
        if [[ "$key" == $'\\x1b' ]]; then
            read -rsn2 key
            case "$key" in
                "[A")
                    ((index--))
                    ;;
                "[B")
                    ((index++))
                    ;;
            esac
            if (( index < 0 )); then
                index=0
            fi
            if (( index >= ${#releases[@]} )); then
                index=$(( ${#releases[@]} - 1 ))
            fi
        fi
    done
    printf "%s\\n" "${releases[$index]}"
}

release_dir=""
if [[ "$install_source" == "release" ]]; then
    if [[ -z "$release_ref" ]]; then
        if [[ "$interactive" == "1" ]]; then
            release_ref="$(choose_release_interactive)"
        else
            release_ref="$(list_releases | head -n 1)"
            if [[ -z "$release_ref" ]]; then
                echo "No releases found at $repo_url" >&2
                exit 1
            fi
            # Defensive normalization in case upstream refs include annotated-tag suffixes.
            release_ref="${release_ref%\^\{\}}"
        fi
    fi
    log "Installing Impression release ${release_ref}."
    if ! command -v git >/dev/null 2>&1; then
        echo "git is required to install a release. Install git or use --local." >&2
        exit 1
    fi
    release_dir="$(mktemp -d)"
    trap '[[ -n "$release_dir" ]] && rm -rf "$release_dir"' EXIT
    log "Cloning release into $release_dir/impression"
    if ! git clone --depth 1 --branch "$release_ref" "$repo_url" "$release_dir/impression"; then
        echo "Failed to clone ${repo_url} at ${release_ref}. Check network access or tag name." >&2
        exit 1
    fi
    repo_root="$release_dir/impression"
    release_version="${release_ref#v}"
    project_version="$(read_project_version "$repo_root" || true)"
    if [[ -n "$project_version" && "$project_version" != "$release_version" ]]; then
        mismatch_msg="Release tag ${release_ref} contains project version ${project_version}."
        if [[ "${IMPRESSION_STRICT_TAG_VERSION:-0}" == "1" ]]; then
            echo "${mismatch_msg}" >&2
            echo "Refusing to install mismatched release because IMPRESSION_STRICT_TAG_VERSION=1." >&2
            exit 1
        fi
        log "WARNING: ${mismatch_msg} Normalizing package metadata to ${release_version}."
        if ! set_project_version "$repo_root" "$release_version"; then
            echo "Failed to normalize package version metadata for ${release_ref}." >&2
            exit 1
        fi
        project_version="$(read_project_version "$repo_root" || true)"
        if [[ "$project_version" != "$release_version" ]]; then
            echo "Package version normalization failed (expected ${release_version}, got ${project_version})." >&2
            exit 1
        fi
        if [[ "${IMPRESSION_ALLOW_VERSION_MISMATCH:-0}" == "1" ]]; then
            log "IMPRESSION_ALLOW_VERSION_MISMATCH is set; normalization completed and install will continue."
        else
            log "Version mismatch was auto-corrected for installation."
        fi
    fi
else
    if [[ "$interactive" == "1" ]]; then
        echo "Interactive mode is only available for release installs." >&2
        exit 1
    fi
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
    log "Creating venv at $venv_path (Python ${python_version})"
    create_venv
fi

py="$venv_path/bin/python"

"$py" - <<'PY' > /tmp/impression_installed_version.txt 2>/dev/null || true
import importlib.metadata
try:
    print(importlib.metadata.version("impression"))
except Exception:
    pass
PY
pre_version="$(cat /tmp/impression_installed_version.txt 2>/dev/null | tail -n 1 | tr -d '\r')"

log "Upgrading pip/build tooling"
"$py" -m pip install --upgrade pip
"$py" -m pip install --upgrade build scikit-build-core cmake ninja
"$py" -m pip install --upgrade setuptools wheel
"$py" -m pip install --upgrade packaging
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
    log "manifold3d already installed; skipping build."
else
    log "Building manifold3d (serial mode)"
    CMAKE_ARGS="-DMANIFOLD_PAR=OFF" "$py" -m pip install --upgrade --no-binary=:all: manifold3d
fi

# Ensure PyVista (viewer) is available in the target venv.
log "Ensuring PyVista is installed"
"$py" -m pip install --upgrade pyvista

# Install the freshly built wheel.
log "Installing Impression wheel"
CMAKE_ARGS="-DMANIFOLD_PAR=OFF" "$py" -m pip install --upgrade --force-reinstall "$wheel"

post_version="$("$py" - <<'PY' 2>/dev/null
import importlib.metadata
try:
    print(importlib.metadata.version("impression"))
except Exception:
    pass
PY
)"
post_version="$(printf "%s" "$post_version" | tail -n 1 | tr -d '\r')"

if [[ -z "$pre_version" && -n "$post_version" ]]; then
    echo "Installed Impression ${post_version}."
elif [[ -n "$pre_version" && -n "$post_version" && "$pre_version" != "$post_version" ]]; then
    echo "Upgraded Impression ${pre_version} -> ${post_version}."
elif [[ -n "$post_version" ]]; then
    echo "Impression ${post_version} is already installed."
fi
