"""
PhotoTrace — STAGE 2: embedding cache + multi-reference (command-line)
=====================================================================

This stage splits the work into two phases, exactly as the architecture
intends:

  INDEX  (slow, run once per folder)
      Walk a folder recursively, detect every face in every image, and cache
      (file_path, face_location, embedding) to disk. Files already in the cache
      and unchanged are skipped, so re-indexing is cheap.

  SEARCH (fast, run as often as you like)
      Take 2-3 reference photos of ONE person, average them into a single
      target signature, and compare against the cached embeddings. No image
      decoding or face detection happens here, so it's near-instant — and
      searching for a *different* person reuses the same index.

Usage
-----
    # 1) Index a folder (do this once; repeat only to pick up new/changed files)
    python phototrace.py index "C:\\Users\\binod\\Pictures"

    # 2) Search using 1-3 reference images of the target person
    python phototrace.py search ref1.jpg ref2.jpg ref3.jpg
    python phototrace.py search ref1.jpg --threshold 0.55 --dir "C:\\Users\\binod\\Pictures"

Options shared by both commands:
    --cache PATH   Use a specific cache file (default: ~/.phototrace/index.pkl)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import engine
import cache as cache_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_images(folder: Path):
    """Yield every supported image path under `folder`, recursively."""
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in engine.SUPPORTED_EXTENSIONS:
            yield path


def resolve_cache_path(args) -> Path:
    return Path(args.cache).expanduser() if args.cache else cache_mod.default_cache_path()


def is_under(path: Path, folder: Path) -> bool:
    """True if `path` is inside `folder` (used to scope a search to one dir)."""
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# INDEX command
# ---------------------------------------------------------------------------

def cmd_index(args) -> int:
    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"Error: target folder not found: {folder}")
        return 1

    cache_path = resolve_cache_path(args)
    cache = cache_mod.empty_cache() if args.rebuild else cache_mod.load_cache(cache_path)

    print("=" * 70)
    print("PhotoTrace — INDEX")
    print("=" * 70)
    print(f"Folder : {folder}")
    print(f"Cache  : {cache_path}")
    if args.rebuild:
        print("Mode   : full rebuild (ignoring existing cache)")
    print("=" * 70)
    print()

    start = time.perf_counter()
    indexed = 0          # files we detected faces in this run
    skipped_cached = 0   # already in cache and unchanged
    errors = 0           # unreadable / corrupt
    total_faces = 0      # faces stored this run

    for image_path in iter_images(folder):
        # Skip files we've already indexed and that haven't changed.
        if cache_mod.is_cached_current(cache, image_path):
            skipped_cached += 1
            continue

        try:
            faces = engine.detect_faces(image_path)
        except Exception as exc:
            # Never let one bad file kill the whole index run.
            errors += 1
            print(f"  [skip] {image_path}  ({exc})")
            continue

        cache_mod.store_file(cache, image_path, faces)
        indexed += 1
        total_faces += len(faces)
        print(f"  [index] {len(faces)} face(s)  {image_path}")

    # Forget files that were deleted from disk since last index.
    pruned = cache_mod.prune_missing(cache, under=folder)

    cache_mod.save_cache(cache, cache_path)
    elapsed = time.perf_counter() - start

    print()
    print("=" * 70)
    print("INDEX SUMMARY")
    print("=" * 70)
    print(f"Newly indexed      : {indexed}  ({total_faces} faces)")
    print(f"Skipped (cached)   : {skipped_cached}")
    print(f"Errors (unreadable): {errors}")
    print(f"Pruned (deleted)   : {pruned}")
    print(f"Total files cached : {len(cache['files'])}")
    print(f"Time taken         : {elapsed:.2f} s")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# SEARCH command
# ---------------------------------------------------------------------------

def cmd_search(args) -> int:
    cache_path = resolve_cache_path(args)
    cache = cache_mod.load_cache(cache_path)
    if not cache["files"]:
        print(f"Error: cache is empty. Run 'index' first.\n  cache: {cache_path}")
        return 1

    scope_dir = Path(args.dir).expanduser() if args.dir else None
    threshold = args.threshold

    print("=" * 70)
    print("PhotoTrace — SEARCH")
    print("=" * 70)
    print(f"References     : {', '.join(args.references)}")
    print(f"Cache          : {cache_path}")
    if scope_dir:
        print(f"Limited to     : {scope_dir}")
    print(f"Distance thresh: {threshold}  (lower = stricter; match when distance <= {threshold})")
    print("=" * 70)
    print()

    # --- Build ONE target signature by averaging the reference photos. ---
    print("Encoding reference image(s)...")
    ref_encodings = []
    for ref in args.references:
        ref_path = Path(ref).expanduser()
        if not ref_path.is_file():
            print(f"  ! Skipping (not found): {ref_path}")
            continue
        try:
            enc = engine.encode_reference(ref_path)
        except Exception as exc:
            print(f"  ! Skipping (unreadable): {ref_path}  ({exc})")
            continue
        if enc is None:
            print(f"  ! Skipping (no face found): {ref_path}")
            continue
        ref_encodings.append(enc)
        print(f"  + encoded {ref_path.name}")

    if not ref_encodings:
        print("\nError: none of the reference images had a usable face.")
        return 1

    target = engine.average_encodings(ref_encodings)
    print(f"Averaged {len(ref_encodings)} reference(s) into one target signature.\n")

    # --- Compare against the cache. This is the fast part. ---
    start = time.perf_counter()
    searched = 0
    matches = []  # (path, best_distance)

    for path_str, record in cache["files"].items():
        path = Path(path_str)
        if scope_dir and not is_under(path, scope_dir):
            continue
        if not path.exists():
            continue  # file was moved/deleted since indexing
        if not record["faces"]:
            continue
        searched += 1

        encodings = [f["embedding"] for f in record["faces"]]
        distances = engine.face_distance(encodings, target)
        best = float(min(distances))
        if best <= threshold:
            matches.append((path_str, best))

    elapsed = time.perf_counter() - start

    print("=" * 70)
    print("SEARCH SUMMARY")
    print("=" * 70)
    print(f"Files searched : {searched}")
    print(f"Matches found  : {len(matches)}")
    print(f"Time taken     : {elapsed:.4f} s   <- from cache, no detection")
    print("=" * 70)

    if matches:
        print("\nMatches (best first):")
        for path_str, dist in sorted(matches, key=lambda m: m[1]):
            print(f"  {dist:.3f}  {path_str}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phototrace",
        description="PhotoTrace Stage 2 — index a folder once, then search it fast.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Detect + cache faces for a folder.")
    p_index.add_argument("folder", help="Folder to index (scanned recursively).")
    p_index.add_argument("--cache", help="Cache file path (default: ~/.phototrace/index.pkl).")
    p_index.add_argument("--rebuild", action="store_true",
                         help="Ignore the existing cache and re-index everything.")
    p_index.set_defaults(func=cmd_index)

    # search
    p_search = sub.add_parser("search", help="Find a person using 1-3 reference photos.")
    p_search.add_argument("references", nargs="+",
                          help="1-3 reference images of the target person.")
    p_search.add_argument("--threshold", type=float, default=0.6,
                          help="Face-distance threshold; lower = stricter. Default: 0.6")
    p_search.add_argument("--dir", help="Limit the search to files under this folder.")
    p_search.add_argument("--cache", help="Cache file path (default: ~/.phototrace/index.pkl).")
    p_search.set_defaults(func=cmd_search)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
