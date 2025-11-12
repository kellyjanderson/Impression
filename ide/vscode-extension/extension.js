const vscode = require('vscode');
const path = require('path');

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand('impression.previewModel', () => previewModel()),
    vscode.commands.registerCommand('impression.exportStl', () => exportStl()),
    vscode.commands.registerCommand('impression.runPreviewTests', () => runPreviewTests())
  );
}

function deactivate() {}

async function previewModel() {
  const modelPath = await pickModelFile();
  if (!modelPath) {
    return;
  }
  runInTerminal(`impression preview ${quote(modelPath)}`);
}

async function exportStl() {
  const modelPath = await pickModelFile();
  if (!modelPath) {
    return;
  }
  const defaultOutput = modelPath.replace(/\.py$/, '.stl');
  const outputPath = await vscode.window.showInputBox({
    prompt: 'Output STL path',
    value: defaultOutput,
  });
  if (!outputPath) {
    return;
  }
  runInTerminal(`impression export ${quote(modelPath)} --output ${quote(outputPath)} --overwrite`);
}

function runPreviewTests() {
  const script = path.join(getWorkspaceFolder(), 'scripts', 'run_preview_tests.py');
  runInTerminal(`${quote(process.env.PYTHON || 'python3')} ${quote(script)}`);
}

async function pickModelFile() {
  const active = vscode.window.activeTextEditor?.document?.uri;
  const defaultUri = active && active.fsPath.endsWith('.py') ? active : undefined;
  const [selection] = await vscode.window.showOpenDialog({
    defaultUri,
    canSelectMany: false,
    filters: { Python: ['py'] },
    openLabel: 'Select Impression model',
  }) || [];
  return selection?.fsPath;
}

function runInTerminal(command) {
  const terminal = vscode.window.createTerminal({ name: 'Impression' });
  terminal.show(true);
  terminal.sendText(command);
}

function getWorkspaceFolder() {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    vscode.window.showErrorMessage('Open the Impression workspace before running this command.');
    throw new Error('Workspace not found');
  }
  return folders[0].uri.fsPath;
}

function quote(value) {
  if (!value) {
    return value;
  }
  if (value.includes(' ')) {
    return `"${value}"`;
  }
  return value;
}

module.exports = { activate, deactivate };
