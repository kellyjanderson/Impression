const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const os = require('os');
const cp = require('child_process');

const DOCS_URL = 'https://github.com/kellyjanderson/Impression#getting-started';
const IMPRESSION_HOME = path.join(os.homedir(), '.impression');
const GLOBAL_VENV_PATH = path.join(IMPRESSION_HOME, 'global-venv');
const ENV_FILE = path.join(IMPRESSION_HOME, 'env');
const SOURCE_LINE = 'source ~/.impression/env # Impression\n';
const RC_FILES = ['.zshrc', '.zprofile', '.bashrc', '.bash_profile', '.profile'];
const INSTALL_SCRIPT_RELATIVE = path.join('scripts', 'dev', 'install_impression.sh');
const PYTHON_STATE_KEY = 'impression.pythonPathCache';
const PREVIEW_MODE_SETTING = 'previewMode';
const PYTHON_PATH_SETTING = 'pythonPath';
const PREVIEW_MODE_TERMINAL = 'terminal';
const PREVIEW_MODE_WEBVIEW = 'webview';
const AGENT_BOOTSTRAP_PROMPT =
  'I have installed impression a parametric modleing freamework for pytyon. The documentation for impresison can be found in this project under ./impression-docs read through all of the documentation. We will be creating modles using impression. Always referr to the docuements before weriting or editing code to make sure you are useing the tools provided by impression.';
const IMPRESSION_CLI_CHECK =
  'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("impression.cli") else 1)';
const EXECUTE_ACCESS = process.platform === 'win32' ? fs.constants.F_OK : fs.constants.X_OK;

let cachedPythonPath = null;
let previewPanel = null;
const installerChannel = vscode.window.createOutputChannel('Impression Installer');
let extensionContext;

function activate(context) {
  extensionContext = context;
  hydrateCachedPython();
  context.subscriptions.push(
    vscode.commands.registerCommand('impression.installLocal', () => installLocalImpressionCommand()),
    vscode.commands.registerCommand('impression.installGlobal', () => installGlobalImpressionCommand()),
    vscode.commands.registerCommand('impression.init', () => initImpressionProjectCommand()),
    vscode.commands.registerCommand('impression.previewCurrentFile', () => previewCurrentFileCommand()),
    vscode.commands.registerCommand('impression.previewCurrentFileNewWindow', () =>
      previewCurrentFileCommand({ forceWindow: true })
    ),
    vscode.commands.registerCommand('impression.exportStl', () => exportStl()),
    vscode.commands.registerCommand('impression.runPreviewTests', () => runPreviewTests())
  );
}

function deactivate() {}

async function previewCurrentFileCommand(options = {}) {
  const modelPath = await resolveModelPath({ allowPicker: false });
  if (!modelPath) {
    vscode.window.showErrorMessage('Open a Python model file in the editor before previewing.');
    return;
  }
  const python = await ensurePythonPath();
  if (!python) {
    return;
  }
  const forceFlag = options.forceWindow ? ' --force-window=true' : '';
  const command = `${quote(python)} -m impression.cli preview ${quote(modelPath)}${forceFlag}`;
  await launchPreview(command, modelPath, python);
}

async function exportStl() {
  const modelPath = await resolveModelPath({ allowPicker: true });
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
  const command = `${quote(python)} -m impression.cli export ${quote(modelPath)} --output ${quote(
    outputPath
  )} --overwrite`;
  const cwd = getPreferredCwd(modelPath);
  runInTerminal(command, { cwd, env: buildPythonEnv(python) });
}

async function runPreviewTests() {
  const python = await ensurePythonPath();
  if (!python) {
    return;
  }
  const script = path.join(getWorkspaceFolder(), 'scripts', 'run_preview_tests.py');
  const command = `${quote(python)} ${quote(script)}`;
  const cwd = getWorkspaceFolder();
  runInTerminal(command, { cwd, env: buildPythonEnv(python) });
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

function runInTerminal(command, options = {}) {
  const terminal = vscode.window.createTerminal({
    name: options.name ?? 'Impression',
    cwd: options.cwd,
    env: options.env,
  });
  terminal.show(true);
  void executeTerminalCommand(terminal, command);
  return terminal;
}

async function executeTerminalCommand(terminal, command) {
  const shellIntegration = await waitForShellIntegration(terminal, 1500);
  if (shellIntegration && typeof shellIntegration.executeCommand === 'function') {
    shellIntegration.executeCommand(command);
    return;
  }
  terminal.sendText(command, true);
}

async function waitForShellIntegration(terminal, timeoutMs) {
  if (terminal.shellIntegration) {
    return terminal.shellIntegration;
  }
  if (typeof vscode.window.onDidChangeTerminalShellIntegration !== 'function') {
    return null;
  }

  return await new Promise((resolve) => {
    let settled = false;
    const timeout = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      subscription.dispose();
      resolve(terminal.shellIntegration ?? null);
    }, timeoutMs);

    const subscription = vscode.window.onDidChangeTerminalShellIntegration((event) => {
      if (event.terminal !== terminal || settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      subscription.dispose();
      resolve(event.shellIntegration ?? terminal.shellIntegration ?? null);
    });
  });
}

async function resolveModelPath(options = {}) {
  const active = getActiveEditorPath();
  if (active) {
    return active;
  }
  if (options.allowPicker) {
    return await pickModelFile();
  }
  return null;
}

async function launchPreview(command, modelPath, pythonPath) {
  const mode = getPreviewMode();
  const cwd = getPreferredCwd(modelPath);
  const env = buildPythonEnv(pythonPath);
  if (mode === PREVIEW_MODE_WEBVIEW) {
    showPreviewPlaceholder();
  }
  runInTerminal(command, { cwd, env });
}

function showPreviewPlaceholder() {
  if (previewPanel) {
    previewPanel.reveal(vscode.ViewColumn.Beside);
    return;
  }
  previewPanel = vscode.window.createWebviewPanel(
    'impressionPreview',
    'Impression Preview',
    vscode.ViewColumn.Beside,
    { enableScripts: false }
  );
  previewPanel.onDidDispose(() => {
    previewPanel = null;
  });
  previewPanel.webview.html = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Impression Preview</title>
  </head>
  <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px;">
    <h2 style="margin: 0 0 12px;">Webview preview is coming soon.</h2>
    <p style="margin: 0 0 8px;">
      For now, Impression launches previews in a terminal-backed PyVista window.
    </p>
    <p style="margin: 0;">
      You can keep this panel open while we build the embedded viewer.
    </p>
  </body>
</html>`;
}

function getPreferredCwd(modelPath) {
  if (!modelPath) {
    return getWorkspaceFolderPath();
  }
  const workspace = vscode.workspace.getWorkspaceFolder(vscode.Uri.file(modelPath));
  if (workspace) {
    return workspace.uri.fsPath;
  }
  return path.dirname(modelPath);
}

function getWorkspaceFolderPath() {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    return undefined;
  }
  return folders[0].uri.fsPath;
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
    if ((await fileExists(cachedPythonPath)) && (await impressionCliAvailable(cachedPythonPath))) {
      return cachedPythonPath;
    }
    cachedPythonPath = null;
    if (extensionContext) {
      extensionContext.globalState.update(PYTHON_STATE_KEY, undefined);
    }
  }

  const resolved = await resolvePythonPath();
  if (resolved) {
    rememberPythonPath(resolved);
    return resolved;
  }

  const selection = await vscode.window.showInformationMessage(
    'The Impression CLI is not available. Install local to this workspace, install global, or view manual instructions.',
    'Install Local',
    'Install Global',
    'View Instructions',
    'Cancel'
  );

  if (selection === 'Install Local') {
    try {
      const python = await installLocalImpression();
      rememberPythonPath(python);
      vscode.window.showInformationMessage('Impression installed locally. You can now run previews.');
      return python;
    } catch (error) {
      vscode.window.showErrorMessage(`Local Impression install failed: ${error.message}`);
      return null;
    }
  }

  if (selection === 'Install Global') {
    try {
      const python = await installGlobalImpression();
      rememberPythonPath(python);
      vscode.window.showInformationMessage('Impression installed globally. You can now run previews.');
      return python;
    } catch (error) {
      vscode.window.showErrorMessage(`Global Impression install failed: ${error.message}`);
      return null;
    }
  }

  if (selection === 'View Instructions') {
    vscode.env.openExternal(vscode.Uri.parse(DOCS_URL));
  }

  return null;
}

async function resolvePythonPath() {
  const candidates = [];
  const override = getPythonOverride();
  if (override) {
    candidates.push(override);
  }
  const fromEnv = process.env.IMPRESSION_PY;
  if (fromEnv) {
    candidates.push(fromEnv);
  }
  const fromEnvFile = await pythonFromEnvFile();
  if (fromEnvFile) {
    candidates.push(fromEnvFile);
  }
  const fromSettings = await pythonFromSettings();
  if (fromSettings) {
    candidates.push(fromSettings);
  }
  const fromWorkspace = await pythonFromWorkspaceVenv();
  if (fromWorkspace) {
    candidates.push(fromWorkspace);
  }
  const shebangPython = await pythonFromShebang();
  if (shebangPython) {
    candidates.push(shebangPython);
  }
  const stored = extensionContext?.globalState.get(PYTHON_STATE_KEY);
  if (stored) {
    candidates.push(stored);
  }
  const fallback = resolveCommand('python3') || resolveCommand('python');
  if (fallback) {
    candidates.push(fallback);
  }

  const seen = new Set();
  for (const candidate of candidates) {
    const resolved = await validatePythonCandidate(candidate, seen);
    if (resolved) {
      return resolved;
    }
  }

  return null;
}

function getImpressionSettings() {
  return vscode.workspace.getConfiguration('impression');
}

function getPreviewMode() {
  const settings = getImpressionSettings();
  const mode = settings.get(PREVIEW_MODE_SETTING, PREVIEW_MODE_TERMINAL);
  return mode === PREVIEW_MODE_WEBVIEW ? PREVIEW_MODE_WEBVIEW : PREVIEW_MODE_TERMINAL;
}

function getPythonOverride() {
  const settings = getImpressionSettings();
  const override = settings.get(PYTHON_PATH_SETTING);
  return typeof override === 'string' && override.trim() ? override.trim() : null;
}

async function validatePythonCandidate(candidate, seen) {
  const resolved = resolveExecutable(candidate);
  if (!resolved) {
    return null;
  }
  const normalized = path.normalize(resolved);
  if (seen.has(normalized)) {
    return null;
  }
  seen.add(normalized);
  if (!(await fileExists(resolved))) {
    return null;
  }
  if (!(await impressionCliAvailable(resolved))) {
    return null;
  }
  return resolved;
}

async function impressionCliAvailable(pythonPath) {
  try {
    await runCommandSilent(pythonPath, ['-c', IMPRESSION_CLI_CHECK], { env: buildPythonEnv(pythonPath) });
    return true;
  } catch (error) {
    return false;
  }
}

function resolveExecutable(candidate) {
  if (!candidate || typeof candidate !== 'string') {
    return null;
  }
  const trimmed = stripQuotes(expandHome(candidate.trim()));
  if (!trimmed) {
    return null;
  }
  if (path.isAbsolute(trimmed)) {
    return trimmed;
  }
  return resolveCommand(trimmed);
}

function resolveCommand(command) {
  if (!command) {
    return null;
  }
  const cmd = process.platform === 'win32' ? 'where' : 'which';
  try {
    const result = cp.execFileSync(cmd, [command], { encoding: 'utf8' }).trim();
    if (!result) {
      return null;
    }
    return result.split(/\r?\n/)[0];
  } catch (error) {
    return null;
  }
}

async function pythonFromSettings() {
  const config = vscode.workspace.getConfiguration('python');
  const candidates = [
    config.get('defaultInterpreterPath'),
    config.get('interpreterPath'),
    config.get('pythonPath'),
  ]
    .map((value) => (typeof value === 'string' ? expandWorkspaceVariables(value) : null))
    .filter(Boolean);
  for (const candidate of candidates) {
    const resolved = resolveExecutable(candidate);
    if (resolved && (await fileExists(resolved))) {
      return resolved;
    }
  }
  return null;
}

async function pythonFromWorkspaceVenv() {
  const folders = vscode.workspace.workspaceFolders ?? [];
  const venvNames = ['.venv', 'venv', '.virtualenv'];
  for (const folder of folders) {
    for (const venvName of venvNames) {
      const candidate = path.join(
        folder.uri.fsPath,
        venvName,
        process.platform === 'win32' ? 'Scripts' : 'bin',
        process.platform === 'win32' ? 'python.exe' : 'python'
      );
      if (await fileExists(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}

function expandWorkspaceVariables(value) {
  if (!value || typeof value !== 'string') {
    return value;
  }
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    return value;
  }
  const root = folders[0].uri.fsPath;
  return value.replace('${workspaceFolder}', root).replace('${workspaceRoot}', root);
}

function expandHome(value) {
  if (!value || typeof value !== 'string') {
    return value;
  }
  if (value === '~') {
    return os.homedir();
  }
  if (value.startsWith('~/') || value.startsWith('~\\')) {
    return path.join(os.homedir(), value.slice(2));
  }
  return value;
}

function stripQuotes(value) {
  if (!value || typeof value !== 'string') {
    return value;
  }
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    return value.slice(1, -1);
  }
  return value;
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
      const interpreter = parseShebangInterpreter(firstLine);
      const resolved = resolveExecutable(interpreter);
      if (resolved && (await fileExists(resolved))) {
        return resolved;
      }
    }
  } catch (error) {
    return null;
  }
  return null;
}

function parseShebangInterpreter(line) {
  const stripped = line.replace(/^#!\s*/, '').trim();
  if (!stripped) {
    return null;
  }
  const parts = stripped.split(/\s+/);
  if (parts[0].endsWith('env')) {
    return parts[1] || null;
  }
  return parts[0];
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

function getVenvPython(baseDir) {
  return process.platform === 'win32'
    ? path.join(baseDir, '.venv', 'Scripts', 'python.exe')
    : path.join(baseDir, '.venv', 'bin', 'python');
}

function getVenvPythonFromPath(venvPath) {
  return process.platform === 'win32'
    ? path.join(venvPath, 'Scripts', 'python.exe')
    : path.join(venvPath, 'bin', 'python');
}

async function installLocalImpressionCommand() {
  try {
    const python = await installLocalImpression();
    rememberPythonPath(python);
    vscode.window.showInformationMessage(`Impression local install complete (${python}).`);
  } catch (error) {
    vscode.window.showErrorMessage(`Impression local install failed: ${error.message}`);
  }
}

async function installGlobalImpressionCommand() {
  try {
    const python = await installGlobalImpression();
    rememberPythonPath(python);
    vscode.window.showInformationMessage(`Impression global install complete (${python}).`);
  } catch (error) {
    vscode.window.showErrorMessage(`Impression global install failed: ${error.message}`);
  }
}

async function initImpressionProjectCommand() {
  let workspace;
  try {
    workspace = getWorkspaceFolder();
  } catch (error) {
    return;
  }
  try {
    const python = await installLocalImpression();
    rememberPythonPath(python);

    const docsDest = path.join(workspace, 'impression-docs');
    installerChannel.show(true);
    installerChannel.appendLine(`$ ${python} -m impression.cli --get-docs`);
    await runCommand(
      python,
      ['-m', 'impression.cli', '--get-docs'],
      { cwd: workspace, env: buildPythonEnv(python) }
    );
    if (!fs.existsSync(docsDest)) {
      throw new Error(`Docs download did not create ${docsDest}. Check CLI output in "Impression Installer".`);
    }

    await vscode.env.clipboard.writeText(AGENT_BOOTSTRAP_PROMPT);
    runInTerminal(`echo ${shellQuote(AGENT_BOOTSTRAP_PROMPT)}`, {
      name: 'Impression Init',
      cwd: workspace,
      env: buildPythonEnv(python),
    });
    await trySendPromptToAgent(AGENT_BOOTSTRAP_PROMPT);
    vscode.window.showInformationMessage(
      'Impression initialized: local install complete, docs downloaded to ./impression-docs, and agent prompt copied to clipboard.'
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Impression init failed: ${error.message}`);
  }
}

async function installLocalImpression() {
  const workspace = getWorkspaceFolder();
  const venvPath = path.join(workspace, '.venv');
  await installWithInstaller({
    cwd: workspace,
    venvPath,
    title: 'Installing Impression (local)…',
  });
  const python = getVenvPythonFromPath(venvPath);
  if (!(await fileExists(python))) {
    throw new Error(`Expected local interpreter at ${python}, but it was not found.`);
  }
  if (!(await impressionCliAvailable(python))) {
    throw new Error(
      `Installer completed but impression CLI is unavailable in ${venvPath}. Check installer logs in the "Impression Installer" output.`
    );
  }
  return python;
}

async function installGlobalImpression() {
  await installWithInstaller({
    cwd: os.homedir(),
    venvPath: GLOBAL_VENV_PATH,
    title: 'Installing Impression (global)…',
  });
  const python = getVenvPythonFromPath(GLOBAL_VENV_PATH);
  if (!(await fileExists(python))) {
    throw new Error(`Expected global interpreter at ${python}, but it was not found.`);
  }
  if (!(await impressionCliAvailable(python))) {
    throw new Error(
      `Installer completed but impression CLI is unavailable in ${GLOBAL_VENV_PATH}. Check installer logs in the "Impression Installer" output.`
    );
  }
  return python;
}

async function installWithInstaller(options) {
  return vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: options.title ?? 'Installing Impression…',
      cancellable: false,
    },
    async () => {
      installerChannel.show(true);
      const invocation = await resolveInstallerInvocation(options.cwd);
      const args = [...invocation.prefixArgs, '--venv', options.venvPath];
      await runCommand(invocation.command, args, { cwd: options.cwd, env: process.env });
      await updateShellIntegration(getVenvPythonFromPath(options.venvPath));
      promptReloadNotice();
    }
  );
}

async function resolveInstallerInvocation(cwd) {
  const scriptPath = path.join(cwd, INSTALL_SCRIPT_RELATIVE);
  if (await fileExists(scriptPath)) {
    return { command: scriptPath, prefixArgs: [] };
  }

  const installCmd = resolveCommand('install');
  if (installCmd) {
    return { command: installCmd, prefixArgs: ['impression'] };
  }

  throw new Error(
    'Installer not found. Expected scripts/dev/install_impression.sh in the workspace or an `install` command on PATH.'
  );
}

async function fileExists(filePath) {
  if (!filePath) {
    return false;
  }
  try {
    await fs.promises.access(filePath, EXECUTE_ACCESS);
    return true;
  } catch (error) {
    return false;
  }
}

function runCommand(command, args = [], options = {}) {
  const logOutput = options.log !== false;
  if (logOutput) {
    installerChannel.appendLine(`$ ${command} ${args.join(' ')}`);
  }
  return new Promise((resolve, reject) => {
    const proc = cp.spawn(command, args, {
      cwd: options.cwd,
      env: options.env ?? process.env,
    });
    const handleOutput = logOutput ? (data) => installerChannel.append(data.toString()) : () => {};
    proc.stdout.on('data', handleOutput);
    proc.stderr.on('data', handleOutput);
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

function runCommandSilent(command, args = [], options = {}) {
  return runCommand(command, args, { ...options, log: false });
}

function buildPythonEnv(pythonPath) {
  const env = {};
  if (pythonPath) {
    env.IMPRESSION_PY = pythonPath;
  }
  const workspaceSrc = getWorkspaceSourcePath();
  if (workspaceSrc) {
    env.PYTHONPATH = mergePythonPath(process.env.PYTHONPATH, workspaceSrc);
  }
  return env;
}

function mergePythonPath(currentValue, addition) {
  if (!addition) {
    return currentValue;
  }
  const separator = process.platform === 'win32' ? ';' : ':';
  if (!currentValue) {
    return addition;
  }
  const parts = currentValue.split(separator);
  if (parts.includes(addition)) {
    return currentValue;
  }
  return `${addition}${separator}${currentValue}`;
}

function getWorkspaceSourcePath() {
  const folders = vscode.workspace.workspaceFolders ?? [];
  for (const folder of folders) {
    const candidate = path.join(folder.uri.fsPath, 'src', 'impression');
    if (fs.existsSync(candidate)) {
      return path.join(folder.uri.fsPath, 'src');
    }
  }
  return null;
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

  // In multi-root workspaces, prefer the workspace that contains the active editor file.
  const activeUri = vscode.window.activeTextEditor?.document?.uri;
  if (activeUri && activeUri.scheme === 'file') {
    const activeWorkspace = vscode.workspace.getWorkspaceFolder(activeUri);
    if (activeWorkspace) {
      return activeWorkspace.uri.fsPath;
    }
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

function shellQuote(value) {
  if (value == null) {
    return "''";
  }
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

async function trySendPromptToAgent(promptText) {
  try {
    const commands = await vscode.commands.getCommands(true);
    if (!commands.includes('workbench.action.chat.open')) {
      return;
    }
    await vscode.commands.executeCommand('workbench.action.chat.open', { query: promptText });
  } catch (_error) {
    // Best-effort only; prompt is always copied to clipboard and printed to terminal.
  }
}

module.exports = { activate, deactivate };
