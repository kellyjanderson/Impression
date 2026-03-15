# Impression VS Code Helpers

A lightweight VS Code extension that shells out to the installer + `impression` CLI so you can install on demand, preview models, and export STL files without leaving the editor.

## Features

- **Impression: Install Local** – installs Impression into the workspace `.venv` and creates it if missing.
- **Impression: Install Global** – installs Impression into `~/.impression/global-venv`.
- **Impression: Init** – runs local install, downloads docs to `./impression-docs`, and prints/copies the agent bootstrap prompt.
- **Impression: Preview** – launches `impression preview` for the active Python file (reuses existing project preview window).
- **Impression: Preview in New Window** – same as Preview but passes `--force-window=true`.
- **Impression: Export STL** – prompts for the model and destination path, then runs `impression export --overwrite`.
- **Impression: Run Preview Tests** – executes `scripts/run_preview_tests.py` using your workspace Python interpreter.

Each command opens a dedicated `Impression` terminal so you can see live CLI logs. Feel free to stop/reuse the terminal like any other VS Code task.

## Interpreter detection

The extension prefers `impression.pythonPath`, then checks `IMPRESSION_PY`, `~/.impression/env`, the VS Code Python interpreter
setting, common workspace venvs, and the CLI shebang. If nothing is found, it offers local/global installer commands.

## Settings

- `impression.pythonPath`: explicitly set the Python interpreter used for Impression commands.
- `impression.previewMode`: choose `terminal` (default) or `webview` (placeholder panel for future embedded previews).

## Getting Started

1. Install project dependencies (`pip install -e .`).
2. In VS Code, run `Extensions: Install from VSIX...` (after packaging) or use `vsce package` inside `ide/vscode-extension/`.
3. Reload VS Code. Press `Cmd+Shift+P` (`Ctrl+Shift+P` on Windows/Linux) and search for “Impression”.

## Packaging

```
cd ide/vscode-extension
npm install
npm run package
```

This produces `impression-vscode-<version>.vsix` that you can share or install. Update the `package.json` metadata (publisher, version) before publishing to the Marketplace.
