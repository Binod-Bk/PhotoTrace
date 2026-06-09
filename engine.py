"""
PhotoTrace — recognition engine
================================

This module is the ONLY place that talks to the underlying face-recognition
library (`face_recognition`, which is dlib-based). Everything else in the app
deals in plain data: file paths, (top, right, bottom, left) face locations, and
128-d numpy embeddings.

Keeping the engine isolated like this means we can later swap `face_recognition`
for InsightFace (or anything else) by rewriting only this file — the cache, the
search logic, and the future UI never import the recognition library directly.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import face_recognition
from PIL import Image

# Image formats we will attempt to read. Anything else is skipped.
# .avif support depends on Pillow being built with libavif (Pillow >= 11.3
# bundles it). If your Pillow lacks AVIF support, those files are simply
# skipped as unreadable rather than crashing the run.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

# Detection model: "hog" is CPU-friendly and the default; "cnn" is more accurate
# but needs a GPU to be practical. HOG is the right call for the MVP.
_DETECTION_MODEL = "hog"

# --- Speed knobs (CPU-only, no GPU needed) ---------------------------------
# Detection time scales with pixel count, so we run the detector on a downscaled
# copy of the image (originals on disk are never modified). MAX_DETECT_DIM is the
# longest edge, in pixels, used for *detection*. Faces are mapped back to the
# original coordinates and the embedding is computed from the full-resolution
# image, so match accuracy and the on-thumbnail highlight box stay correct.
MAX_DETECT_DIM = 800

# Upsampling helps find *small* faces but is expensive; 0 is fast and fine for
# normally-sized faces in photos/screenshots.
_DETECT_UPSAMPLE = 0


def detect_faces(image_path: Path) -> list[tuple[tuple, np.ndarray]]:
    """
    Detect every face in an image.

    Returns a list of (location, embedding) pairs, one per face found:
      * location  -> (top, right, bottom, left) pixel box, ORIGINAL coordinates
      * embedding -> 128-d numpy vector

    Returns an empty list if the image has no detectable faces.
    Raises on unreadable / corrupt images — the caller decides how to handle it.

    Speed: faces are *detected* on a downscaled copy (MAX_DETECT_DIM) with no
    upsampling, then *encoded* from the full-resolution image for accuracy.
    """
    # Pillow handles .webp / .avif; the original file is only read, never written.
    img = Image.open(image_path).convert("RGB")
    original = np.asarray(img)
    h, w = original.shape[:2]

    # Downscale only for the (expensive) detection step.
    scale = min(1.0, MAX_DETECT_DIM / max(h, w)) if max(h, w) else 1.0
    if scale < 1.0:
        small = np.asarray(
            img.resize((max(1, int(w * scale)), max(1, int(h * scale)))))
    else:
        small = original

    locs_small = face_recognition.face_locations(
        small, number_of_times_to_upsample=_DETECT_UPSAMPLE, model=_DETECTION_MODEL)
    if not locs_small:
        return []

    # Map boxes back to original-resolution coordinates (clamped to bounds).
    inv = 1.0 / scale
    locs_orig = []
    for top, right, bottom, left in locs_small:
        t = max(0, min(int(top * inv), h))
        b = max(0, min(int(bottom * inv), h))
        l = max(0, min(int(left * inv), w))
        r = max(0, min(int(right * inv), w))
        locs_orig.append((t, r, b, l))

    # Encode from the full-resolution image for the best embedding quality.
    encodings = face_recognition.face_encodings(original, locs_orig)
    return list(zip(locs_orig, encodings))


# ---------------------------------------------------------------------------
# Parallel detection (uses the CPU cores you already have; no GPU)
# ---------------------------------------------------------------------------

def default_workers() -> int:
    """
    How many detection processes to run at once. Defaults to half the logical
    CPUs (each worker loads its own ~400 MB copy of dlib, so we stay modest to
    avoid exhausting RAM). Override with the PHOTOTRACE_WORKERS env var.
    """
    env = os.environ.get("PHOTOTRACE_WORKERS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, (os.cpu_count() or 2) // 2)


def _detect_one(path_str: str):
    """Top-level (picklable) worker: detect faces for one file in a subprocess."""
    try:
        return path_str, detect_faces(Path(path_str)), None
    except Exception as exc:
        return path_str, None, str(exc)


def detect_many(paths, workers: int | None = None):
    """
    Detect faces across many files in parallel, yielding results as they finish:
        (path_str, faces_or_None, error_message_or_None)

    Order is NOT preserved (results stream back as workers complete). The caller
    writes results to the cache, so no DB handles cross process boundaries.
    """
    paths = [str(p) for p in paths]
    workers = workers or default_workers()

    if workers <= 1 or len(paths) <= 1:
        for p in paths:               # sequential fast-path / tiny batches
            yield _detect_one(p)
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_detect_one, p) for p in paths]
        for fut in as_completed(futures):
            yield fut.result()


def encode_reference(image_path: Path) -> np.ndarray | None:
    """
    Compute a single embedding for a reference photo of the target person.

    Uses the first face found. Returns None if the image has no face (the
    caller should warn and skip that reference).
    """
    faces = detect_faces(image_path)
    if not faces:
        return None
    # faces[i] is (location, embedding); we want the embedding of the first.
    return faces[0][1]


def average_encodings(encodings: list[np.ndarray]) -> np.ndarray:
    """
    Average several reference embeddings into ONE target signature.

    Averaging 2-3 photos of the same person smooths out pose/lighting noise and
    gives a more robust signature than any single photo. Empty input is a
    programming error and raises.
    """
    if not encodings:
        raise ValueError("average_encodings() needs at least one embedding")
    return np.mean(np.asarray(encodings), axis=0)


def face_distance(known_encodings: list[np.ndarray], target: np.ndarray) -> np.ndarray:
    """
    Distance from `target` to each embedding in `known_encodings`
    (lower = more similar). Thin pass-through so callers never import the
    recognition library directly.
    """
    return face_recognition.face_distance(known_encodings, target)


def distance_to_confidence(distance: float, threshold: float = 0.6) -> float:
    """
    Turn a raw face *distance* (lower = better) into a friendly 0..1 confidence
    (higher = better) for display in the UI.

    Calibrated against the threshold so it reads intuitively:
      * distance 0          -> ~1.0  (≈100%)
      * distance == threshold -> 0.5  (50% — right at the match boundary)
      * distance well beyond  -> 0.0
    This is for human readability only; matching still uses the raw distance.
    """
    if distance > threshold:
        span = (1.0 - threshold)
        linear = (1.0 - distance) / (span * 2.0)
        return max(0.0, linear)
    # Inside the match region: ease the curve up toward 1.0.
    span = threshold
    linear = 1.0 - (distance / (span * 2.0))
    return linear + (1.0 - linear) * math.pow((linear - 0.5) * 2.0, 0.2)
