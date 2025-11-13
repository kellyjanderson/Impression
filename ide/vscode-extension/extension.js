const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const os = require('os');
const cp = require('child_process');

const REPO_URL = 'https://github.com/kellyjanderson/Impression.git';
const DOCS_URL = 'https://github.com/kellyjanderson/Impression#getting-started';
const INSTALL_FOLDER = path.join(os.homedir(), '.impression-cli');
const IMPRESSION_HOME = path.join(os.homedir(), '.impression');
const ENV_FILE = path.join(IMPRESSION_HOME, 'env');
const SOURCE_LINE = 'source ~/.impression/env # Impression\n';
const RC_FILES = ['.zshrc', '.bashrc', '.bash_profile'];
const PYTHON_STATE_KEY = 'impression.pythonPath';

let cachedPythonPath = null;
const installerChannel = vscode.window.createOutputChannel('Impression Installer');
let extensionContext;

function activate(context) {
  extensionContext = context;
  hydrateCachedPython();
  context.subscriptions.push(
    vscode.commands.registerCommand('impression.previewModel', () => previewModel()),
    vscode.commands.registerCommand('impression.exportStl', () => exportStl()),
    vscode.commands.registerCommand('impression.runPreviewTests', () => runPreviewTests())
  );
}

function deactivate() {}

async function previewModel() {
  const modelPath = await resolveModelPath();
  if (!modelPath) {
    return;
  }
  const python = await ensurePythonPath();
  if (!python) {
    return;
  }
  runInTerminal(`${quote(python)} -m impression.cli preview ${quote(modelPath)}`);
}

async function exportStl() {
  const modelPath = await resolveModelPath();
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
  const python = await ensurePythonPath();
  if (!python) {
    return;
  }
  runInTerminal(
    `${quote(python)} -m impression.cli export ${quote(modelPath)} --output ${quote(outputPath)} --overwrite`
  );
}

async function runPreviewTests() {
  const python = await ensurePythonPath();
  if (!python) {
    return;
  }
  const script = path.join(getWorkspaceFolder(), 'scripts', 'run_preview_tests.py');
  runInTerminal(`${quote(python)} ${quote(script)}`);
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

async function resolveModelPath() {
  const active = getActiveEditorPath();
  if (active) {
    return active;
  }
  return await pickModelFile();
}

function getActiveEditorPath() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return null;
  }
  const document = editor.document;
  if (document.isUntitled || document.uri.scheme !== 'file') {
    return null;
  }
  if (document.languageId !== 'python' && !document.fileName.endsWith('.py')) {
    return null;
  }
  return document.uri.fsPath;
}

async function ensurePythonPath() {
  if (cachedPythonPath) {
    return cachedPythonPath;
  }

  const resolved = await resolvePythonPath();
  if (resolved) {
    rememberPythonPath(resolved);
    return resolved;
  }

  const selection = await vscode.window.showInformationMessage(
    'The Impression CLI is not available. Install it automatically or view manual instructions.',
    'Install Impression',
    'View Instructions',
    'Cancel'
  );

  if (selection === 'Install Impression') {
    try {
      const python = await installImpression();
      rememberPythonPath(python);
      vscode.window.showInformationMessage('Impression installed. You can now run previews.');
      return python;
    } catch (error) {
      vscode.window.showErrorMessage(`Impression install failed: ${error.message}`);
      return null;
    }
  }

  if (selection === 'View Instructions') {
    vscode.env.openExternal(vscode.Uri.parse(DOCS_URL));
  }

  return null;
}

async function resolvePythonPath() {
  const fromEnv = process.env.IMPRESSION_PY;
  if (fromEnv && (await fileExists(fromEnv))) {
    return fromEnv;
  }

  const fromEnvFile = await pythonFromEnvFile();
  if (fromEnvFile) {
    return fromEnvFile;
  }

  const shebangPython = await pythonFromShebang();
  if (shebangPython) {
    return shebangPython;
  }

  const stored = extensionContext?.globalState.get(PYTHON_STATE_KEY);
  if (stored && (await fileExists(stored))) {
    return stored;
  }

  return null;
}

async function pythonFromShebang() {
  const cmd = process.platform === 'win32' ? 'where' : 'which';
  try {
    const result = cp.execFileSync(cmd, ['impression'], { encoding: 'utf8' }).trim();
    if (!result) {
      return null;
    }
    const impressionBinary = result.split(/\r?\n/)[0];
    const firstLine = fs.readFileSync(impressionBinary, 'utf8').split(/\r?\n/)[0];
    if (firstLine.startsWith('#!')) {
      const interpreter = firstLine.slice(2).trim();
      if (await fileExists(interpreter)) {
        return interpreter;
      }
    }
  } catch (error) {
    return null;
  }
  return null;
}

async function pythonFromEnvFile() {
  try {
    const contents = await fs.promises.readFile(ENV_FILE, 'utf8');
    const match = contents.match(/IMPRESSION_PY="([^"]+)"/);
    if (match && (await fileExists(match[1]))) {
      return match[1];
    }
  } catch (error) {
    if (error.code !== 'ENOENT') {
      installerChannel.appendLine(`Unable to read ${ENV_FILE}: ${error.message}`);
    }
  }
  return null;
}

async function installImpression() {
  return vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: 'Installing Impression…',
      cancellable: false,
    },
    async () => {
      installerChannel.show(true);
      if (!(await pathExists(INSTALL_FOLDER))) {
        await runCommand('git', ['clone', REPO_URL, INSTALL_FOLDER]);
      }
      const pythonBin = getVenvPython(INSTALL_FOLDER);
      if (!(await fileExists(pythonBin))) {
        await runCommand('python3', ['-m', 'venv', path.join(INSTALL_FOLDER, '.venv')]);
      }
      await runCommand(pythonBin, ['-m', 'pip', 'install', '--upgrade', 'pip'], { cwd: INSTALL_FOLDER });
      await runCommand(pythonBin, ['-m', 'pip', 'install', '-e', '.'], { cwd: INSTALL_FOLDER });
      await updateShellIntegration(pythonBin);
      rememberPythonPath(pythonBin);
      promptReloadNotice();
      return pythonBin;
    }
  );
}

function getVenvPython(baseDir) {
  return process.platform === 'win32'
    ? path.join(baseDir, '.venv', 'Scripts', 'python.exe')
    : path.join(baseDir, '.venv', 'bin', 'python');
}

async function fileExists(filePath) {
  if (!filePath) {
    return false;
  }
  try {
    await fs.promises.access(filePath, fs.constants.X_OK);
    return true;
  } catch (error) {
    return false;
  }
}

async function pathExists(filePath) {
  if (!filePath) {
    return false;
  }
  try {
    await fs.promises.access(filePath);
    return true;
  } catch (error) {
    return false;
  }
}

function runCommand(command, args = [], options = {}) {
  installerChannel.appendLine(`$ ${command} ${args.join(' ')}`);
  return new Promise((resolve, reject) => {
    const proc = cp.spawn(command, args, {
      cwd: options.cwd,
      env: options.env ?? process.env,
    });
    proc.stdout.on('data', (data) => installerChannel.append(data.toString()));
    proc.stderr.on('data', (data) => installerChannel.append(data.toString()));
    proc.on('error', (error) => reject(error));
    proc.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${command} exited with code ${code}`));
      }
    });
  });
}

async function updateShellIntegration(pythonPath) {
  try {
    await fs.promises.mkdir(IMPRESSION_HOME, { recursive: true });
    const exportLine = `export IMPRESSION_PY="${pythonPath}"\n`;
    await fs.promises.writeFile(ENV_FILE, exportLine, 'utf8');
    await ensureRcIncludesSource();
  } catch (error) {
    installerChannel.appendLine(`Failed to update ~/.impression/env: ${error.message}`);
  }
}

async function ensureRcIncludesSource() {
  await Promise.all(
    RC_FILES.map(async (file) => {
      const rcPath = path.join(os.homedir(), file);
      try {
        const content = await fs.promises.readFile(rcPath, 'utf8');
        if (content.includes(SOURCE_LINE.trim())) {
          return;
        }
        await fs.promises.appendFile(rcPath, `\n# Added by Impression\n${SOURCE_LINE}`);
      } catch (error) {
        if (error.code !== 'ENOENT') {
          installerChannel.appendLine(`Skipping ${rcPath}: ${error.message}`);
        }
      }
    })
  );
}

function promptReloadNotice() {
  vscode.window
    .showInformationMessage(
      'Impression installed. Reload VS Code to ensure the new interpreter is detected?',
      'Reload Window',
      'Later'
    )
    .then((selection) => {
      if (selection === 'Reload Window') {
        vscode.commands.executeCommand('workbench.action.reloadWindow');
      }
    });
}

function hydrateCachedPython() {
  const envPath = process.env.IMPRESSION_PY;
  if (envPath) {
    cachedPythonPath = envPath;
    return;
  }
  const stored = extensionContext?.globalState.get(PYTHON_STATE_KEY);
  if (stored) {
    cachedPythonPath = stored;
    process.env.IMPRESSION_PY = stored;
  }
}

function rememberPythonPath(pythonPath) {
  cachedPythonPath = pythonPath;
  process.env.IMPRESSION_PY = pythonPath;
  if (extensionContext) {
    extensionContext.globalState.update(PYTHON_STATE_KEY, pythonPath);
  }
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
