# VS Code Integration

The repository ships with a starter VS Code extension under `ide/vscode-extension/`. It exposes three commands that wrap the CLI:

- `Impression: Preview Model` – launch `impression preview` (defaults to the active Python file).
- `Impression: Export STL` – run `impression export` and prompt for an output path.
- `Impression: Run Preview Tests` – execute `scripts/run_preview_tests.py` inside the workspace.

## Install locally

1. `cd ide/vscode-extension`
2. `npm install`
3. `npm run package` (produces `.vsix`)
4. In VS Code run **Extensions: Install from VSIX...** and pick the generated file.

After installation, open the Impression workspace. Each command opens an `Impression` terminal, so you can stop/re-run previews as needed. The extension simply leverages the existing CLI, so it works wherever the CLI already runs (including virtual environments).
