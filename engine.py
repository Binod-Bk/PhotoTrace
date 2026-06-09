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
from pathlib import Path

import numpy as np
import face_recognition

# Image formats we will attempt to read. Anything else is skipped.
# .avif support depends on Pillow being built with libavif (Pillow >= 11.3
# bundles it). If your Pillow lacks AVIF support, those files are simply
# skipped as unreadable rather than crashing the run.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

# Detection model: "hog" is CPU-friendly and the default; "cnn" is more accurate
# but needs a GPU to be practical. HOG is the right call for the MVP.
_DETECTION_MODEL = "hog"


def detect_faces(image_path: Path) -> list[tuple[tuple, np.ndarray]]:
    """
    Detect every face in an image.

    Returns a list of (location, embedding) pairs, one per face found:
      * location  -> (top, right, bottom, left) pixel box
      * embedding -> 128-d numpy vector

    Returns an empty list if the image has no detectable faces.
    Raises on unreadable / corrupt images — the caller decides how to handle it.
    """
    # load_image_file uses Pillow under the hood, so it supports .webp / .avif.
    image = face_recognition.load_image_file(str(image_path))
    locations = face_recognition.face_locations(image, model=_DETECTION_MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    return list(zip(locations, encodings))


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
