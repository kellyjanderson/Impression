# VS Code Integration

The repository ships with a starter VS Code extension under `ide/vscode-extension/`. It exposes command-palette helpers that wrap the CLI and installer:

- `Impression: Install Local` – install Impression into the workspace `.venv` (creates it if missing).
- `Impression: Install Global` – install Impression into a global venv at `~/.impression/global-venv`.
- `Impression: Init` – run local install, fetch docs into `./impression-docs`, and print/copy the agent bootstrap prompt.
- `Impression: Preview` – preview the current Python file. This reuses an existing preview window for the same project when available.
- `Impression: Preview in New Window` – preview the current file with `--force-window=true`.
- `Impression: Export STL` – run `impression export` and prompt for an output path.
- `Impression: Run Preview Tests` – execute `scripts/run_preview_tests.py` inside the workspace.

## Install locally

1. `cd ide/vscode-extension`
2. `npm install`
3. `npm run package` (produces `.vsix`)
4. In VS Code run **Extensions: Install from VSIX...** and pick the generated file.

After installation, open the Impression workspace. Each command opens an `Impression` terminal, so you can stop/re-run previews as needed.

## Interpreter detection & auto-install

- The extension looks for `impression.pythonPath` first, then checks `IMPRESSION_PY`, `~/.impression/env`,
  the VS Code Python interpreter setting, common workspace venvs (`.venv`, `venv`), and the CLI shebang.
- The CLI writes `~/.impression/env` (via `IMPRESSION_PY`) the first time it runs. Source that file
  in your shell (`source ~/.impression/env`) so the IDE knows which interpreter owns the CLI.
- If the extension cannot find the CLI, it offers `Install Local` or `Install Global`.
  It prefers `scripts/dev/install_impression.sh` in the workspace when available; otherwise it
  runs `install impression` from your PATH. It also updates `~/.impression/env`.
- Declining the auto-install option opens the [Getting Started guide](../README.md#quickstart)
  with manual clone/install instructions.
- You can also use `scripts/dev/setup_dev_env.sh` to configure a development environment and
  ensure your shell sources `~/.impression/env`.

## Preview modes

Set `impression.previewMode` to `terminal` (default) or `webview`. Webview preview is a placeholder
panel for now, but keeps the door open for an embedded viewer once the UX is ready.
