# Impression

Impression is a parametric modeling framework for Python that is focused on providing a
comprehensive and consistent modeling interface, fast previews, and watertight STL generation.

## Quickstart

```bash
git clone https://github.com/kellyjanderson/Impression.git
cd Impression
install impression
source .venv/bin/activate
```

After installation you can run `impression --help` from anywhere in that virtual environment.

By default the installer pulls the latest tagged release. To see available releases run:

```bash
scripts/dev/install_impression.sh --list
```

To pick interactively with arrow keys:

```bash
scripts/dev/install_impression.sh --interactive
```

If you want to install a local wheel (to mimic a packaged release), use the helper:

```bash
python3 -m venv .venv
source .venv/bin/activate
scripts/dev/install_impression.sh --local
```

The installer builds a wheel, installs it into the active venv, and forces `manifold3d` to build
in serial mode so Intel TBB is not required.

See **Python Versions and Venvs** below for how the installer chooses a Python version.

The CLI writes `~/.impression/env` with an `IMPRESSION_PY` export that the VS Code
extension (and other tooling) can source. Add the following line to your shell config if you
haven't already:

```bash
source ~/.impression/env
```

## Your First Preview

Every model is a Python module that exposes a `build()` function returning internal meshes.
Use the primitives and helpers in `impression.modeling` (not PyVista objects).

```python
from impression.modeling import make_box, make_cylinder, boolean_union


def build():
    body = make_box(size=(2, 2, 1))
    post = make_cylinder(radius=0.4, height=2.0)
    return boolean_union([body, post])
```

Preview it:

```bash
impression preview path/to/model.py
```

## CLI Highlights

- Preview: `impression preview docs/examples/primitives/box_example.py`
- Export: `impression export docs/examples/csg/union_example.py --output dist/union.stl --overwrite`
- Docs only: `impression --get-docs --docs-dest ./impression-docs`

Full CLI reference: [`docs/cli.md`](docs/cli.md)

## Documentation Map

- [`docs/index.md`](docs/index.md) - documentation portal
- [`docs/modeling/`](docs/modeling/) - primitives, CSG, drawing2d, paths, extrusions, loft, morph, text
- [`docs/examples/`](docs/examples/) - runnable scripts that power the docs
- [`docs/tutorials/`](docs/tutorials/) - guided walkthroughs for new and advanced users
- [`docs/agents.md`](docs/agents.md) - agent bootstrap and feature map
- [`docs/project-plan.md`](docs/project-plan.md) - roadmap and open questions

## Helper Scripts

- `scripts/dev/setup_dev_env.sh` - create/update the repo virtual environment, install the package,
  and append the `source ~/.impression/env` line to your shell configuration files.
- `scripts/dev/reset_impression_env.sh` - remove the auto-installed CLI (`~/.impression-cli`),
  delete `~/.impression/env`, strip the sourcing line from your shell rc files, and clear VS Code
  global state so the extension behaves like a first-time install.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
impression --help
```

The preview window supports orbit/pan/zoom and hot reload (watch mode). For units, configure
`~/.impression/impression.cfg` (JSON) with `millimeters`, `meters`, or `inches`.

## Python Versions and Venvs

Impression relies on PyVista/VTK, which can lag behind the newest CPython releases. The installer
creates a venv with **Python 3.13 by default** to avoid 3.14 wheel gaps.

How it works:

- `install impression` creates a `./.venv` in the current folder.
- It prefers `uv` to provision Python 3.13 automatically.
- If `uv` is missing, it falls back to Homebrew (`python@3.13`).
- Existing venvs are reused only if they match the expected Python version.

Overrides:

- `IMPRESSION_PYTHON_VERSION=3.12` to use a different version.
- `IMPRESSION_PYTHON=/path/to/python3.13` to force a specific interpreter.
- `IMPRESSION_RECREATE_VENV=1` to delete/recreate the venv if the version mismatches.

## Contributing

Work via feature branches and pull requests. Keep changes focused, include documentation with
every new feature, and add example scripts when possible.

## Community & Inclusion

Impression is a collaborative project. We expect respectful communication, inclusive language,
and empathy for contributors of all backgrounds. Harassment or discrimination is not tolerated.
