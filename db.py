"""
PhotoTrace — SQLite embedding cache (Stage 5)
=============================================

Replaces the Stage 2-4 pickle cache (`cache.py`) with a single local SQLite
database. Still fully offline — just one `.db` file on disk.

Why SQLite now
--------------
  * We no longer load the entire cache into memory to read or update it.
  * Each file's faces are rows we can insert/replace incrementally.
  * We record when each file was indexed (`indexed_at`) and detect changes via
    (mtime, size), so re-indexing only processes new or modified files.

Schema
------
    files(path PK, mtime, size, indexed_at)
    faces(id PK, path -> files.path, top, right, bottom, left, embedding BLOB)

The 128-d embedding is stored as a float64 BLOB (numpy .tobytes()).

The public API (`is_current`, `upsert_file`, `iter_files`, `prune_missing`,
`file_count`, `clear`) is the seam every caller uses, mirroring the old pickle
cache so the CLI and GUI changed only their imports. If a legacy `index.pkl`
sits next to the new `index.db`, it is migrated automatically on first open.
"""

from __future__ import annotations

import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# face_recognition embeddings are float64 vectors of length 128.
EMB_DTYPE = np.float64


def default_db_path() -> Path:
    """Default cache location: ~/.phototrace/index.db (cross-platform)."""
    return Path.home() / ".phototrace" / "index.db"


class FaceCache:
    """SQLite-backed store of per-file face embeddings + locations."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

        # One-time migration from the older pickle cache, if one is sitting
        # beside this db (e.g. ~/.phototrace/index.pkl next to index.db).
        legacy = self.db_path.with_suffix(".pkl")
        if self.file_count() == 0 and legacy.exists():
            self._migrate_pickle(legacy)

    # -- lifecycle ----------------------------------------------------------

    def _create_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path       TEXT PRIMARY KEY,
                mtime      REAL    NOT NULL,
                size       INTEGER NOT NULL,
                indexed_at REAL    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS faces (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                path      TEXT NOT NULL,
                top       INTEGER,
                right     INTEGER,
                bottom    INTEGER,
                "left"    INTEGER,
                embedding BLOB NOT NULL,
                FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_faces_path ON faces(path);
            """
        )
        self.conn.commit()

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.commit()
        self.close()

    # -- writes -------------------------------------------------------------

    def upsert_file(self, path: Path, faces: list[tuple]) -> None:
        """
        Insert or replace one file's record and its faces.

        `faces` is the list of (location, embedding) pairs from
        engine.detect_faces (location = (top, right, bottom, left)).
        """
        st = path.stat()
        p = str(path)
        self.conn.execute("DELETE FROM faces WHERE path = ?", (p,))
        self.conn.execute(
            "INSERT INTO files(path, mtime, size, indexed_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "mtime=excluded.mtime, size=excluded.size, indexed_at=excluded.indexed_at",
            (p, st.st_mtime, st.st_size, time.time()),
        )
        for loc, emb in faces:
            top, right, bottom, left = loc
            self.conn.execute(
                'INSERT INTO faces(path, top, right, bottom, "left", embedding) '
                "VALUES (?, ?, ?, ?, ?, ?)",
                (p, int(top), int(right), int(bottom), int(left),
                 np.asarray(emb, dtype=EMB_DTYPE).tobytes()),
            )

    def clear(self) -> None:
        """Wipe the whole cache (used by --rebuild)."""
        self.conn.execute("DELETE FROM faces")
        self.conn.execute("DELETE FROM files")
        self.conn.commit()

    def prune_missing(self, under: Path | None = None) -> int:
        """
        Delete cache rows whose file no longer exists. If `under` is given, only
        prune files inside that folder. Returns how many files were removed.
        Faces are removed automatically via ON DELETE CASCADE.
        """
        under_resolved = under.resolve() if under else None
        removed = 0
        for path_str in list(self.all_paths()):
            p = Path(path_str)
            if under_resolved is not None:
                try:
                    p.resolve().relative_to(under_resolved)
                except ValueError:
                    continue  # outside the target folder; leave it
            if not p.exists():
                self.conn.execute("DELETE FROM files WHERE path = ?", (path_str,))
                removed += 1
        return removed

    # -- reads --------------------------------------------------------------

    def is_current(self, path: Path) -> bool:
        """True if `path` is cached AND unchanged (same mtime + size)."""
        row = self.conn.execute(
            "SELECT mtime, size FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        if row is None:
            return False
        try:
            st = path.stat()
        except OSError:
            return False
        return row[0] == st.st_mtime and row[1] == st.st_size

    def file_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def all_paths(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT path FROM files").fetchall()}

    def iter_files(self):
        """
        Yield (path, faces) for every cached file, where faces is a list of
        (location, embedding) pairs. Files with zero detected faces yield [].
        """
        faces_by_path: dict[str, list] = defaultdict(list)
        for path, top, right, bottom, left, emb in self.conn.execute(
            'SELECT path, top, right, bottom, "left", embedding FROM faces'
        ):
            faces_by_path[path].append(
                ((top, right, bottom, left), np.frombuffer(emb, dtype=EMB_DTYPE))
            )
        for (path,) in self.conn.execute("SELECT path FROM files ORDER BY path"):
            yield path, faces_by_path.get(path, [])

    # -- migration ----------------------------------------------------------

    def _migrate_pickle(self, pickle_path: Path) -> None:
        """Import a legacy pickle cache once; never fatal if it fails."""
        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            files = data.get("files", {}) if isinstance(data, dict) else {}
            for path, rec in files.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO files(path, mtime, size, indexed_at) "
                    "VALUES (?, ?, ?, ?)",
                    (path, rec.get("mtime", 0.0), rec.get("size", 0), time.time()),
                )
                self.conn.execute("DELETE FROM faces WHERE path = ?", (path,))
                for face in rec.get("faces", []):
                    top, right, bottom, left = face["location"]
                    self.conn.execute(
                        'INSERT INTO faces(path, top, right, bottom, "left", embedding) '
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (path, int(top), int(right), int(bottom), int(left),
                         np.asarray(face["embedding"], dtype=EMB_DTYPE).tobytes()),
                    )
            self.conn.commit()
            sys.stderr.write(
                f"[phototrace] migrated {len(files)} file(s) from "
                f"{pickle_path.name} into SQLite.\n"
            )
        except Exception as exc:  # pragma: no cover - best-effort import
            sys.stderr.write(f"[phototrace] pickle migration skipped: {exc}\n")
