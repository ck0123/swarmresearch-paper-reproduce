from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import BinaryIO


DEFAULT_EXCLUDES = {
    ".git",
    ".codex",
    ".claude",
    ".pi",
    "__pycache__",
    ".pytest_cache",
}


def _is_excluded(path: Path, workspace_dir: Path) -> bool:
    relative_parts = path.relative_to(workspace_dir).parts
    return any(part in DEFAULT_EXCLUDES for part in relative_parts)


def create_workspace_archive(workspace_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(workspace_dir.rglob("*")):
            if _is_excluded(path, workspace_dir):
                continue
            archive.add(path, arcname=path.relative_to(workspace_dir))
    return buffer.getvalue()


def extract_workspace_archive(archive_source: bytes | BinaryIO, dest_dir: Path) -> None:
    fileobj: BinaryIO
    if isinstance(archive_source, bytes):
        fileobj = io.BytesIO(archive_source)
    else:
        fileobj = archive_source
        fileobj.seek(0)
    with tarfile.open(fileobj=fileobj, mode="r:gz") as archive:
        archive.extractall(dest_dir)
