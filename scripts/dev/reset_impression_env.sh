#!/usr/bin/env bash
set -euo pipefail

IMPRESSION_HOME="$HOME/.impression"
CLI_FOLDER="$HOME/.impression-cli"
ENV_FILE="$IMPRESSION_HOME/env"
LINE='source ~/.impression/env # Impression'

rm -rf "$CLI_FOLDER"
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

echo "Removed ~/.impression-cli and stripped shell configuration." \
"Reinstall Impression or run scripts/dev/setup_dev_env.sh to recreate the environment."
