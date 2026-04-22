# Impression

Impression is a parametric modeling framework for Python that is focused on providing a
comprehensive and consistent modeling interface, fast previews, and watertight STL generation.

## Project DNA

Impression aims for more than functional correctness. We build features, docs, and workflows to
solve real modeling problems, bridge mechanical and organic design, and help users dream bigger
about what they can create.

See: [Project DNA](project/project-dna.md)

## Quickstart

```bash
git clone https://github.com/kellyjanderson/Impression.git
cd Impression
scripts/dev/install_impression.sh
source .venv/bin/activate
```

After installation you can run `impression --help` from anywhere in that virtual environment.

The canonical installer is the repo-local script above. If you keep a local
wrapper such as `install impression` on your machine, it should delegate to
`scripts/dev/install_impression.sh`.

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
- Docs only (matching installed release): `impression --get-docs --docs-dest ./impression-docs`

Full CLI reference: [`docs/cli.md`](docs/cli.md)

## Documentation Map

- [`docs/index.md`](docs/index.md) - documentation portal
- [`docs/modeling/`](docs/modeling/) - primitives, CSG, mesh analysis tools, drawing2d, paths, loft, threading, hinges, text
- [`docs/examples/`](docs/examples/) - runnable scripts that power the docs
- [`docs/tutorials/`](docs/tutorials/) - guided walkthroughs for new and advanced users
- [`docs/agents/`](docs/agents/) - agent usage guide for building with Impression
- [`docs/skills/`](docs/skills/) - installable Codex skills that ship with the docs
- [`project/README.md`](project/README.md) - project workspace for planning, architecture, and records
- [`project/project-dna.md`](project/project-dna.md) - core product values and quality bar
- [`project/planning/README.md`](project/planning/README.md) - roadmap and open questions

## Helper Scripts

- `scripts/dev/setup_dev_env.sh` - create/update the repo virtual environment, install the package,
  and append the `source ~/.impression/env` line to your shell configuration files.
- `scripts/dev/install_impression.sh` - install the latest tagged release or the local repo into a
  target virtual environment.
- `scripts/dev/reset_impression_env.sh` - remove the auto-installed CLI (`~/.impression-cli`),
  delete `~/.impression/env`, strip the sourcing line from your shell rc files, and clear VS Code
  global state so the extension behaves like a first-time install.
- `scripts/dev/run_full_coverage.sh` - run full-repo coverage and write terminal, XML, and HTML
  reports under `project/coverage/`.
- `scripts/dev/run_surface_coverage.sh` - run the surface-body coverage slice and write reports
  under `project/coverage/surface/`.
- `scripts/dev/run_loft_coverage.sh` - run the loft-focused coverage slice over the dedicated loft
  planner, API, and showcase suites, writing reports under `project/coverage/loft/`.

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

- `scripts/dev/install_impression.sh` creates a `./.venv` in the current folder by default.
- It prefers `uv` to provision Python 3.13 automatically.
- If `uv` is missing, it falls back to Homebrew (`python@3.13`).
- Existing venvs are reused only if they match the expected Python version.

Overrides:

- `IMPRESSION_PYTHON_VERSION=3.12` to use a different version.
- `IMPRESSION_PYTHON=/path/to/python3.13` to force a specific interpreter.
- `IMPRESSION_RECREATE_VENV=1` to delete/recreate the venv if the version mismatches.
- `IMPRESSION_MANIFOLD_MODE=auto|source|skip` to control manifold install behavior (`auto` prefers wheels, then falls back to source).

## Contributing

Work via feature branches and pull requests. Keep changes focused, include documentation with
every new feature, and add example scripts when possible.

## Community & Inclusion

Impression is a collaborative project. We expect respectful communication, inclusive language,
and empathy for contributors of all backgrounds. Harassment or discrimination is not tolerated.
