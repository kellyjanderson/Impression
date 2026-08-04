#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import tomllib
from typing import Iterable
import zipfile

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QualifiedArtifact:
    name: str
    type: str
    version: str
    size: int
    sha256: str


def version_for_tag(tag: str, pyproject_path: Path) -> Version:
    if not tag.startswith("v") or len(tag) == 1:
        raise ValueError(f"release tag must start with 'v': {tag!r}")
    project_version_text = tomllib.loads(pyproject_path.read_text())["project"]["version"]
    tag_version_text = tag[1:]
    if tag_version_text != project_version_text:
        raise ValueError(
            f"tag version {tag_version_text!r} does not exactly match project version "
            f"{project_version_text!r}"
        )
    return Version(project_version_text)


def is_prerelease(version: str | Version) -> bool:
    return Version(str(version)).is_prerelease


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_distribution_payload(
    path: Path,
    *,
    expected_version: Version,
    artifact_type: str,
) -> None:
    if artifact_type == "wheel":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata_name = next(
                (name for name in names if name.endswith(".dist-info/METADATA")),
                None,
            )
            if metadata_name is None:
                raise ValueError(f"wheel has no METADATA: {path.name}")
            metadata = archive.read(metadata_name).decode("utf-8")
    else:
        with tarfile.open(path) as archive:
            names = archive.getnames()
            metadata_member = next(
                (member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")),
                None,
            )
            if metadata_member is None:
                raise ValueError(f"sdist has no PKG-INFO: {path.name}")
            extracted = archive.extractfile(metadata_member)
            if extracted is None:
                raise ValueError(f"sdist metadata is unreadable: {path.name}")
            metadata = extracted.read().decode("utf-8")

    lowered_names = tuple(name.casefold() for name in names)
    if any("half_pipe" in name or name.endswith("/impression/cad.py") for name in lowered_names):
        raise ValueError(f"forbidden experimental payload in {path.name}")
    if "build123d" in metadata.casefold():
        raise ValueError(f"forbidden build123d dependency in {path.name}")
    if f"Version: {expected_version}" not in metadata:
        raise ValueError(f"metadata version mismatch in {path.name}")


def _assert_docs_payload(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    if not names:
        raise ValueError(f"documentation archive is empty: {path.name}")
    for name in names:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != "docs":
            raise ValueError(f"documentation member is outside docs/: {name!r}")


def qualify_artifacts(tag: str, pyproject_path: Path, dist_dir: Path) -> dict[str, object]:
    version = version_for_tag(tag, pyproject_path)
    files = tuple(
        sorted(
            path
            for path in dist_dir.iterdir()
            if path.is_file() and path.name != "qualified-artifacts.json"
        )
    )
    wheel_paths = tuple(path for path in files if path.suffix == ".whl")
    sdist_paths = tuple(path for path in files if path.name.endswith(".tar.gz"))
    docs_name = f"impression-docs-{tag}.zip"
    docs_paths = tuple(path for path in files if path.name == docs_name)

    if len(wheel_paths) != 1 or len(sdist_paths) != 1 or len(docs_paths) != 1:
        raise ValueError("dist must contain exactly one wheel, one sdist, and the tagged docs ZIP")
    expected_paths = {*wheel_paths, *sdist_paths, *docs_paths}
    unexpected = tuple(path.name for path in files if path not in expected_paths)
    if unexpected:
        raise ValueError(f"unexpected release artifacts: {unexpected!r}")

    wheel_name, wheel_version, _, _ = parse_wheel_filename(wheel_paths[0].name)
    sdist_name, sdist_version = parse_sdist_filename(sdist_paths[0].name)
    if wheel_name != "impression" or sdist_name != "impression":
        raise ValueError("distribution project name is not impression")
    if wheel_version != version or sdist_version != version:
        raise ValueError("distribution filename version does not match tag/project version")

    _assert_distribution_payload(wheel_paths[0], expected_version=version, artifact_type="wheel")
    _assert_distribution_payload(sdist_paths[0], expected_version=version, artifact_type="sdist")
    _assert_docs_payload(docs_paths[0])

    typed_paths = (
        (wheel_paths[0], "wheel"),
        (sdist_paths[0], "sdist"),
        (docs_paths[0], "docs"),
    )
    artifacts = tuple(
        QualifiedArtifact(
            name=path.name,
            type=artifact_type,
            version=str(version),
            size=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path, artifact_type in typed_paths
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tag": tag,
        "version": str(version),
        "prerelease": is_prerelease(version),
        "artifacts": [asdict(artifact) for artifact in artifacts],
    }


def write_manifest_atomically(manifest: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_manifest(
    manifest_path: Path,
    *,
    tag: str,
    pyproject_path: Path,
    dist_dir: Path,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    expected = qualify_artifacts(tag, pyproject_path, dist_dir)
    actual = json.loads(manifest_path.read_text())
    if actual != expected:
        raise ValueError("qualified artifact manifest does not match the supplied artifacts")
    paths = tuple(dist_dir / artifact["name"] for artifact in actual["artifacts"])
    return actual, paths


def _write_github_outputs(path: Path, manifest: dict[str, object], assets: Iterable[Path]) -> None:
    asset_lines = "\n".join(str(asset) for asset in assets)
    with path.open("a") as output:
        output.write(f"prerelease={str(bool(manifest['prerelease'])).lower()}\n")
        output.write("files<<QUALIFIED_ASSETS\n")
        output.write(f"{asset_lines}\n")
        output.write("QUALIFIED_ASSETS\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify qualified release artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--tag", required=True)
        subparser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
        subparser.add_argument("--dist", type=Path, default=Path("dist"))
        subparser.add_argument("--manifest", type=Path, required=True)
        if command == "verify":
            subparser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    if args.command == "create":
        manifest = qualify_artifacts(args.tag, args.pyproject, args.dist)
        write_manifest_atomically(manifest, args.manifest)
        print(args.manifest)
        return 0

    manifest, assets = verify_manifest(
        args.manifest,
        tag=args.tag,
        pyproject_path=args.pyproject,
        dist_dir=args.dist,
    )
    if args.github_output is not None:
        _write_github_outputs(args.github_output, manifest, assets)
    for asset in assets:
        print(asset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
