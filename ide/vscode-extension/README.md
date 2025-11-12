# Impression VS Code Helpers

A lightweight VS Code extension that shells out to the `impression` CLI so you can preview models, export STL files, and run regression tests without leaving the editor.

## Features

- **Impression: Preview Model** – prompts for a Python module (defaults to the active file) and launches `impression preview` in an integrated terminal.
- **Impression: Export STL** – prompts for the model and destination path, then runs `impression export --overwrite`.
- **Impression: Run Preview Tests** – executes `scripts/run_preview_tests.py` using your workspace Python interpreter.

Each command opens a dedicated `Impression` terminal so you can see live CLI logs. Feel free to stop/reuse the terminal like any other VS Code task.

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
