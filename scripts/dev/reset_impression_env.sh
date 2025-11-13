#!/usr/bin/env bash
set -euo pipefail

IMPRESSION_HOME="$HOME/.impression"
CLI_FOLDER="$HOME/.impression-cli"
ENV_FILE="$IMPRESSION_HOME/env"
LINE='source ~/.impression/env # Impression'

echo "Removing auto-installed CLI folder ($CLI_FOLDER)"
rm -rf "$CLI_FOLDER"
echo "Removing env file ($ENV_FILE)"
rm -f "$ENV_FILE"

clean_rc() {
  local rc_file="$1"
  [[ -f "$rc_file" ]] || return
  if grep -Fq "$LINE" "$rc_file"; then
    tmp="${rc_file}.tmp"
    grep -Fv "$LINE" "$rc_file" > "$tmp"
    mv "$tmp" "$rc_file"
    echo "Removed Impression source line from $rc_file"
  fi
}

clean_rc "$HOME/.zshrc"
clean_rc "$HOME/.bashrc"
clean_rc "$HOME/.bash_profile"

if [[ -n "${IMPRESSION_PY:-}" ]]; then
  unset IMPRESSION_PY
  echo "Unset IMPRESSION_PY for this shell session."
else
  echo "IMPRESSION_PY was not set in this shell session."
fi

GLOBAL_STORAGE_BASES=(
  "$HOME/Library/Application Support/Code/User/globalStorage"
  "$HOME/Library/Application Support/Code - Insiders/User/globalStorage"
  "$HOME/.config/Code/User/globalStorage"
  "$HOME/.config/VSCodium/User/globalStorage"
)

for base in "${GLOBAL_STORAGE_BASES[@]}"; do
  storage="$base/impression.impression-vscode"
  if [[ -d "$storage" ]]; then
    rm -rf "$storage"
    echo "Removed VS Code global storage at $storage"
  fi
done

echo "Reset complete. Restart your terminals/VS Code to ensure IMPRESSION_PY is reloaded."
echo "Reinstall Impression via scripts/dev/setup_dev_env.sh or the VS Code auto-install prompt."
