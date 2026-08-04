from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest


def _load_manifest_module(project_root: Path):
    path = project_root / "scripts" / "release" / "artifact_manifest.py"
    spec = importlib.util.spec_from_file_location("impression_release_artifact_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_candidate_artifacts(root: Path, version: str, tag: str) -> Path:
    dist = root / "dist"
    dist.mkdir()
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: impression\n"
        f"Version: {version}\n"
        "Requires-Dist: numpy>=1.26\n"
    ).encode()

    wheel = dist / f"impression-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("impression/__init__.py", f"__version__ = {version!r}\n")
        archive.writestr(f"impression-{version}.dist-info/METADATA", metadata)

    sdist = dist / f"impression-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payloads = {
            f"impression-{version}/PKG-INFO": metadata,
            f"impression-{version}/src/impression/__init__.py": b"",
        }
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    docs = dist / f"impression-docs-{tag}.zip"
    with zipfile.ZipFile(docs, "w") as archive:
        archive.writestr("docs/cli.md", "# CLI\n")
    return dist


def _write_pyproject(root: Path, version: str) -> Path:
    path = root / "pyproject.toml"
    path.write_text(f'[project]\nname = "impression"\nversion = "{version}"\n')
    return path


@pytest.mark.parametrize(
    ("version", "expected"),
    (
        ("1.0.0a3", True),
        ("1.0.0b2", True),
        ("1.0.0rc1", True),
        ("1.0.0", False),
    ),
)
def test_prerelease_classification_follows_pep_440(
    project_root: Path,
    version: str,
    expected: bool,
) -> None:
    manifest_module = _load_manifest_module(project_root)

    assert manifest_module.is_prerelease(version) is expected


def test_qualification_manifest_records_exact_assets_versions_and_hashes(
    project_root: Path,
    tmp_path: Path,
) -> None:
    manifest_module = _load_manifest_module(project_root)
    version = "1.0.0a3"
    tag = f"v{version}"
    pyproject = _write_pyproject(tmp_path, version)
    dist = _write_candidate_artifacts(tmp_path, version, tag)

    manifest = manifest_module.qualify_artifacts(tag, pyproject, dist)

    assert manifest["tag"] == tag
    assert manifest["version"] == version
    assert manifest["prerelease"] is True
    assert {item["type"] for item in manifest["artifacts"]} == {"wheel", "sdist", "docs"}
    for item in manifest["artifacts"]:
        artifact = dist / item["name"]
        assert item["size"] == artifact.stat().st_size
        assert item["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_manifest_verification_selects_only_manifest_assets(
    project_root: Path,
    tmp_path: Path,
) -> None:
    manifest_module = _load_manifest_module(project_root)
    version = "1.0.0a3"
    tag = f"v{version}"
    pyproject = _write_pyproject(tmp_path, version)
    dist = _write_candidate_artifacts(tmp_path, version, tag)
    manifest_path = dist / "qualified-artifacts.json"
    manifest_module.write_manifest_atomically(
        manifest_module.qualify_artifacts(tag, pyproject, dist),
        manifest_path,
    )

    manifest, assets = manifest_module.verify_manifest(
        manifest_path,
        tag=tag,
        pyproject_path=pyproject,
        dist_dir=dist,
    )

    assert {path.name for path in assets} == {
        item["name"] for item in manifest["artifacts"]
    }
    assert manifest_path not in assets


def test_failed_qualification_emits_no_manifest(project_root: Path, tmp_path: Path) -> None:
    manifest_module = _load_manifest_module(project_root)
    version = "1.0.0a3"
    tag = f"v{version}"
    pyproject = _write_pyproject(tmp_path, version)
    dist = _write_candidate_artifacts(tmp_path, version, tag)
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("examples/half_pipe.py", "unsafe")
    manifest_path = dist / "qualified-artifacts.json"

    with pytest.raises(ValueError, match="forbidden experimental payload"):
        manifest_module.write_manifest_atomically(
            manifest_module.qualify_artifacts(tag, pyproject, dist),
            manifest_path,
        )

    assert not manifest_path.exists()


def test_changed_asset_invalidates_qualified_manifest(project_root: Path, tmp_path: Path) -> None:
    manifest_module = _load_manifest_module(project_root)
    version = "1.0.0a3"
    tag = f"v{version}"
    pyproject = _write_pyproject(tmp_path, version)
    dist = _write_candidate_artifacts(tmp_path, version, tag)
    manifest_path = dist / "qualified-artifacts.json"
    manifest_module.write_manifest_atomically(
        manifest_module.qualify_artifacts(tag, pyproject, dist),
        manifest_path,
    )
    docs = dist / f"impression-docs-{tag}.zip"
    with docs.open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="does not match"):
        manifest_module.verify_manifest(
            manifest_path,
            tag=tag,
            pyproject_path=pyproject,
            dist_dir=dist,
        )


def test_release_workflow_gates_build_qualification_and_publication(project_root: Path) -> None:
    workflow = (project_root / ".github" / "workflows" / "release.yml").read_text()

    assert workflow.count("uses: actions/checkout@v4\n        with:\n          lfs: true") == 3
    assert "test:\n    runs-on: macos-14" in workflow
    assert "qualify:\n    needs: test" in workflow
    assert "publish:\n    needs: qualify" in workflow
    assert workflow.index("Run full candidate suite") < workflow.index("Build package artifacts once")
    assert workflow.count("python -m build") == 1
    assert workflow.index("Emit qualified immutable manifest") < workflow.index(
        "Transfer only qualified release inputs"
    )
    assert workflow.index("Verify manifest, tag, version, assets, and hashes") < workflow.index(
        "Publish qualified release assets"
    )
    assert "permissions:\n      contents: write" in workflow
    assert "prerelease: ${{ steps.verify.outputs.prerelease }}" in workflow
    assert "files: ${{ steps.verify.outputs.files }}" in workflow
    assert "Verify published release metadata" in workflow


def test_pr_ci_runs_exact_references_on_the_authoring_platform(project_root: Path) -> None:
    workflow = (project_root / ".github" / "workflows" / "ci.yml").read_text()

    assert "candidate-suite:\n    runs-on: macos-14" in workflow
    assert "Checkout exact reference artifacts" in workflow
    assert "lfs: true" in workflow
    assert "Run full candidate suite\n        run: python -m pytest" in workflow
    assert "build-test:\n    runs-on: ubuntu-latest" in workflow


def test_installed_candidate_smoke_invokes_the_real_cli_app(project_root: Path) -> None:
    smoke = (project_root / "scripts" / "release" / "smoke_installed_candidate.py").read_text()

    assert '"from impression.cli import app; app()"' in smoke
    assert '"-m",\n                "impression.cli"' not in smoke
