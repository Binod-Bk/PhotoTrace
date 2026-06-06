"""
PhotoTrace — STAGE 1: Prove the matching (command-line, no UI, no cache)
=======================================================================

Goal of this stage
-------------------
Given ONE reference image of a person and a target folder, walk the folder
recursively, detect every face in every image, and report which images contain
the reference person (including group photos — a match if ANY face matches).

This stage deliberately has:
  * NO cache (every run re-scans everything)
  * NO UI
  * NO multi-reference averaging (that arrives in Stage 2)

The point is to validate that the recognition is accurate enough before we
build anything on top of it. Tune the threshold (printed at the top of every
run) until the matches look right to you.

Usage
-----
    python stage1_match.py REFERENCE_IMAGE TARGET_FOLDER [--threshold 0.6]

Example
-------
    python stage1_match.py me.jpg "C:\\Users\\binod\\Pictures" --threshold 0.55

About the threshold
-------------------
We use face *distance* (lower = more similar), not a similarity score.
  * 0.6  -> the library's default. Reasonable starting point.
  * 0.5  -> stricter: fewer false matches, but may miss some real ones.
  * 0.45 -> very strict.
A face is considered a match when its distance to the reference is
<= threshold. Lower the threshold if you get false matches; raise it if real
photos are being missed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import face_recognition

# Image formats we will attempt to read. Anything else is skipped.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Recognition engine (isolated behind two small functions).
#
# Everything that talks to `face_recognition` lives here. The rest of the
# script only deals with paths, distances, and printing. When we swap the
# engine for InsightFace later, only this section changes.
# ---------------------------------------------------------------------------

def get_face_encodings(image_path: Path) -> list:
    """
    Detect all faces in an image and return their encodings (embeddings).

    Returns a list of 128-dimensional numpy vectors, one per face found.
    Returns an empty list if the image has no detectable faces.

    Raises on unreadable / corrupt images — the caller is responsible for
    catching that and continuing.
    """
    # load_image_file uses Pillow under the hood, so it supports .webp too.
    image = face_recognition.load_image_file(str(image_path))

    # First locate faces, then compute an encoding for each.
    # model="hog" is CPU-friendly and the default; "cnn" is more accurate but
    # needs a GPU to be practical. HOG is the right call for the MVP.
    face_locations = face_recognition.face_locations(image, model="hog")
    encodings = face_recognition.face_encodings(image, face_locations)
    return encodings


def reference_encoding(image_path: Path):
    """
    Load the reference image and return a single encoding for the person.

    If the reference has no face, we cannot proceed — return None.
    If it has multiple faces, we use the first and warn (the reference is
    supposed to be a clear photo of ONE person).
    """
    encodings = get_face_encodings(image_path)

    if len(encodings) == 0:
        return None

    if len(encodings) > 1:
        print(
            f"  ! Warning: reference image has {len(encodings)} faces; "
            f"using the first one. Use a photo with only the target person "
            f"for best results.\n"
        )

    return encodings[0]


# ---------------------------------------------------------------------------
# Folder walking + matching
# ---------------------------------------------------------------------------

def iter_images(folder: Path):
    """Yield every supported image path under `folder`, recursively."""
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PhotoTrace Stage 1 — find a person across a folder of images."
    )
    parser.add_argument("reference", help="Path to ONE reference image of the person.")
    parser.add_argument("folder", help="Target folder to search (scanned recursively).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Face-distance threshold; lower = stricter. Default: 0.6",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    target_folder = Path(args.folder).expanduser()
    threshold = args.threshold

    # --- Validate inputs up front so we fail fast with a clear message. ---
    if not reference_path.is_file():
        print(f"Error: reference image not found: {reference_path}")
        return 1
    if not target_folder.is_dir():
        print(f"Error: target folder not found: {target_folder}")
        return 1

    # --- Banner: print the threshold prominently so it's easy to tune. ---
    print("=" * 70)
    print("PhotoTrace — Stage 1 (matching proof)")
    print("=" * 70)
    print(f"Reference image : {reference_path}")
    print(f"Target folder   : {target_folder}")
    print(f"Distance thresh : {threshold}  (lower = stricter; a face matches "
          f"when distance <= {threshold})")
    print("=" * 70)
    print()

    # --- Build the reference encoding. ---
    print("Encoding reference image...")
    try:
        target_encoding = reference_encoding(reference_path)
    except Exception as exc:
        print(f"Error: could not read reference image: {exc}")
        return 1

    if target_encoding is None:
        print("Error: no face found in the reference image. "
              "Use a clear, front-facing photo of the person.")
        return 1
    print("Reference encoded.\n")

    # --- Scan the folder. ---
    start_time = time.perf_counter()
    scanned = 0           # images we successfully opened and processed
    skipped = 0           # images we could not process (corrupt, perm, etc.)
    no_face = 0           # images that opened fine but had no faces
    matches = []          # list of (path, best_distance)

    print("Scanning images...\n")
    for image_path in iter_images(target_folder):
        try:
            encodings = get_face_encodings(image_path)
        except Exception as exc:
            # Corrupt image, permission error, unsupported content — skip it
            # and keep going. We never let one bad file kill the whole run.
            skipped += 1
            print(f"  [skip] {image_path}  ({exc})")
            continue

        scanned += 1

        if len(encodings) == 0:
            no_face += 1
            continue

        # Compare the reference against EVERY face in this image. The image
        # matches if the closest face is within the threshold — this is what
        # makes group photos work.
        distances = face_recognition.face_distance(encodings, target_encoding)
        best_distance = float(min(distances))

        if best_distance <= threshold:
            matches.append((image_path, best_distance))
            print(f"  [MATCH] {best_distance:.3f}  {image_path}")

    elapsed = time.perf_counter() - start_time

    # --- Summary. ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Images scanned     : {scanned}")
    print(f"  with no faces    : {no_face}")
    print(f"Images skipped     : {skipped}  (corrupt / unreadable)")
    print(f"Matches found      : {len(matches)}")
    print(f"Time taken         : {elapsed:.2f} s")
    print("=" * 70)

    if matches:
        # Sort best (smallest distance) first for an easy-to-read list.
        print("\nMatches (best first):")
        for path, dist in sorted(matches, key=lambda m: m[1]):
            print(f"  {dist:.3f}  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
