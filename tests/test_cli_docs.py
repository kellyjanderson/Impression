from __future__ import annotations

import io
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import pytest
import typer

from impression.cli import _extract_docs_archive


def _archive_bytes(entries: list[tuple[str | zipfile.ZipInfo, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return buffer.getvalue()


def _nul_archive_bytes() -> bytes:
    data = _archive_bytes([("repo/docs/badXname.txt", b"unsafe")])
    return data.replace(b"badXname.txt", b"bad\x00name.txt")


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../escape.txt",
        "repo/docs/nested/../../../escape.txt",
        "/docs/escape.txt",
        "C:/docs/escape.txt",
        "C:\\docs\\escape.txt",
        "\\\\server\\share\\docs\\escape.txt",
    ),
)
def test_unsafe_archive_paths_are_rejected_before_clean_or_write(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    destination = tmp_path / "installed-docs"
    destination.mkdir()
    destination_sentinel = destination / "keep.txt"
    destination_sentinel.write_text("keep")
    sibling_sentinel = tmp_path / "escape.txt"
    sibling_sentinel.write_text("outside")
    archive = _archive_bytes(
        [
            ("repo/docs/valid-first.txt", b"must not be written"),
            (unsafe_name, b"unsafe"),
        ]
    )

    with pytest.raises(typer.BadParameter, match="Unsafe docs archive member"):
        _extract_docs_archive(archive, destination, clean=True)

    assert destination_sentinel.read_text() == "keep"
    assert not (destination / "valid-first.txt").exists()
    assert sibling_sentinel.read_text() == "outside"


def test_nul_member_is_rejected_before_mutation(tmp_path: Path) -> None:
    destination = tmp_path / "installed-docs"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep")

    with pytest.raises(typer.BadParameter, match="NUL byte"):
        _extract_docs_archive(_nul_archive_bytes(), destination, clean=True)

    assert sentinel.read_text() == "keep"


def test_symlink_member_is_rejected_before_mutation(tmp_path: Path) -> None:
    link = zipfile.ZipInfo("repo/docs/outside-link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = _archive_bytes(
        [("repo/docs/valid-first.txt", b"must not be written"), (link, b"../escape.txt")]
    )

    with pytest.raises(typer.BadParameter, match="link-like type"):
        _extract_docs_archive(archive, tmp_path / "installed-docs", clean=False)

    assert not (tmp_path / "installed-docs").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_prefix_confusion_is_not_extracted(tmp_path: Path) -> None:
    archive = _archive_bytes(
        [
            ("repo/docs/index.md", b"docs"),
            ("repo/docs-elsewhere/not-docs.md", b"not docs"),
        ]
    )
    destination = tmp_path / "installed-docs"

    _extract_docs_archive(archive, destination, clean=False)

    assert (destination / "index.md").read_bytes() == b"docs"
    assert not (destination / "not-docs.md").exists()


def test_valid_archive_preserves_existing_files_without_clean(tmp_path: Path) -> None:
    destination = tmp_path / "installed-docs"
    destination.mkdir()
    (destination / "existing.md").write_text("existing")
    archive = _archive_bytes(
        [
            ("repo/docs/", b""),
            ("repo/docs/guide/", b""),
            ("repo/docs/guide/index.md", b"new docs"),
        ]
    )

    _extract_docs_archive(archive, destination, clean=False)

    assert (destination / "existing.md").read_text() == "existing"
    assert (destination / "guide" / "index.md").read_bytes() == b"new docs"


def test_valid_archive_replaces_existing_files_with_clean(tmp_path: Path) -> None:
    destination = tmp_path / "installed-docs"
    destination.mkdir()
    (destination / "existing.md").write_text("existing")
    archive = _archive_bytes([("docs/index.md", b"new docs")])

    _extract_docs_archive(archive, destination, clean=True)

    assert not (destination / "existing.md").exists()
    assert (destination / "index.md").read_bytes() == b"new docs"


def test_release_packaged_docs_archive_installs_through_cli_extractor(
    project_root: Path,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "impression-docs-test.zip"
    subprocess.run(
        (
            sys.executable,
            "scripts/release/package_docs_zip.py",
            "--ref",
            "test",
            "--output",
            str(archive_path),
        ),
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    destination = tmp_path / "installed-docs"

    _extract_docs_archive(archive_path.read_bytes(), destination, clean=True)

    assert (destination / "cli.md").is_file()
    assert (destination / "modeling" / "loft.md").is_file()
