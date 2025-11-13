#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -e "$REPO_ROOT"

"$PYTHON_BIN" - <<'PY'
import impression
print("Wrote", __import__('pathlib').Path.home() / '.impression' / 'env')
PY

append_source_line() {
  local rc_file="$1"
  local line='source ~/.impression/env # Impression'
  if [[ -f "$rc_file" ]]; then
    if ! grep -Fq "$line" "$rc_file"; then
      printf '\n# Added by Impression\n%s\n' "$line" >> "$rc_file"
      echo "Appended source line to $rc_file"
    fi
  fi
}

append_source_line "$HOME/.zshrc"
append_source_line "$HOME/.bashrc"
append_source_line "$HOME/.bash_profile"

echo "Development environment ready. Open a new shell or run 'source ~/.impression/env'."
