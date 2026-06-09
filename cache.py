"""
PhotoTrace — persistent embedding cache
=======================================

The expensive part of this app is *detecting faces and computing embeddings*.
We do that exactly once per file (the INDEX phase) and write the results to
disk. Every SEARCH after that just compares cached embeddings — no image
decoding, no face detection — so it's near-instant.

Cache layout (a single pickle file)
-----------------------------------
    {
        "version": 2,
        "files": {
            "<absolute file path>": {
                "mtime":  <float>,   # used to detect changed files
                "size":   <int>,     # used to detect changed files
                "faces":  [
                    {"location": (top, right, bottom, left),
                     "embedding": <128-d numpy array>},
                    ...
                ],
            },
            ...
        },
    }

A file is re-indexed only if its (mtime, size) changed since last time, so
re-running INDEX on a folder skips everything already done.

NOTE: pickle is fine for this stage (it stores numpy arrays natively). Stage 5
graduates this to SQLite. The public functions here (load/save/lookup) are the
seam where that swap will happen.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

CACHE_VERSION = 2


def default_cache_path() -> Path:
    """
    Default cache location: ~/.phototrace/index.pkl

    Kept in the user's home dir (cross-platform via Path.home()) rather than
    inside the scanned folder, so we never write into the user's photo
    directories and read-only source folders still work.
    """
    return Path.home() / ".phototrace" / "index.pkl"


def empty_cache() -> dict:
    """A fresh, empty cache structure."""
    return {"version": CACHE_VERSION, "files": {}}


def load_cache(path: Path) -> dict:
    """
    Load the cache from disk, or return an empty cache if it's missing,
    unreadable, or from an incompatible older version.
    """
    if not path.exists():
        return empty_cache()
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        # Corrupt cache should never crash the app — just start fresh.
        return empty_cache()

    if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
        return empty_cache()
    return data


def save_cache(cache: dict, path: Path) -> None:
    """
    Write the cache to disk atomically (write to a temp file, then replace) so
    an interrupted run can't leave a half-written, corrupt cache.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic on the same filesystem


def file_signature(path: Path) -> tuple[float, int]:
    """(mtime, size) for a file — our cheap 'has this changed?' fingerprint."""
    st = path.stat()
    return st.st_mtime, st.st_size


def is_cached_current(cache: dict, path: Path) -> bool:
    """
    True if `path` is in the cache AND unchanged since it was indexed.
    Any error reading the file's stats means "not current" -> re-index it.
    """
    record = cache["files"].get(str(path))
    if record is None:
        return False
    try:
        mtime, size = file_signature(path)
    except OSError:
        return False
    return record["mtime"] == mtime and record["size"] == size


def store_file(cache: dict, path: Path, faces: list[tuple]) -> None:
    """
    Insert/update the cache entry for one file.

    `faces` is the list of (location, embedding) pairs from engine.detect_faces.
    """
    mtime, size = file_signature(path)
    cache["files"][str(path)] = {
        "mtime": mtime,
        "size": size,
        "faces": [{"location": loc, "embedding": enc} for loc, enc in faces],
    }


def prune_missing(cache: dict, under: Path | None = None) -> int:
    """
    Drop cache entries whose file no longer exists on disk. If `under` is given,
    only prune files inside that directory. Returns how many were removed.
    """
    removed = []
    under_resolved = under.resolve() if under else None
    for path_str in list(cache["files"].keys()):
        p = Path(path_str)
        if under_resolved is not None:
            try:
                p.resolve().relative_to(under_resolved)
            except ValueError:
                continue  # not under the target folder; leave it alone
        if not p.exists():
            removed.append(path_str)
    for path_str in removed:
        del cache["files"][path_str]
    return len(removed)
