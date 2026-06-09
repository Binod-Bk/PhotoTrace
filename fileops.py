"""
PhotoTrace — file operations
============================

Safe MOVE / COPY of selected images to a chosen folder. Deliberately small and
UI-free so it can be unit-tested and reused.

Design rules (matching the project's safety stance):
  * NEVER overwrite an existing file — if the destination name is taken, append
    " (1)", " (2)", ... until a free name is found.
  * NEVER delete anything. Moving is the only "destructive" op, and even that
    just relocates the file; the user deletes manually later.
  * One bad file (permission error, vanished source, ...) must not abort the
    whole batch — collect per-file failures and report them.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def unique_destination(dest_dir: Path, filename: str) -> Path:
    """
    Return a path inside `dest_dir` for `filename` that does not already exist,
    inserting " (1)", " (2)", ... before the extension if needed.
    """
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _transfer(paths, dest_dir: Path, *, move: bool):
    """
    Move or copy each path into dest_dir.

    Returns (succeeded, failed):
      * succeeded -> list of (source_path, final_destination_path)
      * failed    -> list of (source_path, error_message)
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    succeeded: list[tuple[Path, Path]] = []
    failed: list[tuple[Path, str]] = []

    for raw in paths:
        src = Path(raw)
        try:
            if not src.exists():
                raise FileNotFoundError("source file no longer exists")
            # Don't act on a file that's already sitting in the destination.
            if src.resolve().parent == dest_dir.resolve():
                raise ValueError("file is already in the destination folder")

            target = unique_destination(dest_dir, src.name)
            if move:
                shutil.move(str(src), str(target))
            else:
                shutil.copy2(str(src), str(target))  # copy2 preserves metadata
            succeeded.append((src, target))
        except Exception as exc:
            failed.append((src, str(exc)))

    return succeeded, failed


def move_files(paths, dest_dir: Path):
    """Move each path into dest_dir. See _transfer for the return shape."""
    return _transfer(paths, dest_dir, move=True)


def copy_files(paths, dest_dir: Path):
    """Copy each path into dest_dir. See _transfer for the return shape."""
    return _transfer(paths, dest_dir, move=False)
