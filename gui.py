"""
PhotoTrace — STAGE 3: minimal PyQt6 UI
======================================

A small desktop window over the Stage 2 engine + cache:

  * Pick 1-3 reference images of ONE person.
  * Pick a target folder.
  * INDEX it (slow, runs in a background thread with a live progress bar).
  * SEARCH (fast) and see matches in a scrollable thumbnail grid, each with a
    confidence score and a checkbox.

Heavy work (face detection during indexing, and reference encoding during
search) runs in QThread workers so the window never freezes. Thumbnails are
loaded through Pillow, so .webp / .avif render even though Qt can't always
decode them natively.

File operations (move/copy), select-all, and the live confidence slider arrive
in Stage 4 — the checkboxes here are the groundwork for that.

Run:
    python gui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)
from PIL import Image

import engine
import cache as cache_mod

THUMB_SIZE = 170        # px, square thumbnails in the results grid
REF_THUMB_SIZE = 84     # px, small thumbnails for chosen references
GRID_COLUMNS = 4        # results per row
MAX_REFERENCES = 3


# ---------------------------------------------------------------------------
# Thumbnail loading (via Pillow so every supported format renders)
# ---------------------------------------------------------------------------

def load_thumbnail(path: Path, size: int) -> QPixmap | None:
    """Load `path` and return a QPixmap no larger than size x size, or None."""
    try:
        im = Image.open(path)
        im.draft("RGB", (size, size))        # speeds up large JPEG decoding
        im = im.convert("RGB")
        im.thumbnail((size, size))
        data = im.tobytes("raw", "RGB")
        qimg = QImage(data, im.width, im.height, im.width * 3,
                      QImage.Format.Format_RGB888)
        # fromImage copies into the pixmap's own storage, so it's safe after
        # `data` goes out of scope.
        return QPixmap.fromImage(qimg.copy())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class IndexWorker(QThread):
    """Indexes a folder in the background, emitting progress as it goes."""
    progress = pyqtSignal(int, int, str)   # done, total, current path
    done = pyqtSignal(dict)                 # summary stats
    failed = pyqtSignal(str)

    def __init__(self, folder: Path, cache_path: Path):
        super().__init__()
        self.folder = folder
        self.cache_path = cache_path

    def run(self):
        try:
            cache = cache_mod.load_cache(self.cache_path)
            images = [p for p in self.folder.rglob("*")
                      if p.is_file() and p.suffix.lower() in engine.SUPPORTED_EXTENSIONS]
            total = len(images)
            indexed = skipped = errors = faces_total = 0

            for i, img in enumerate(images, 1):
                if cache_mod.is_cached_current(cache, img):
                    skipped += 1
                else:
                    try:
                        faces = engine.detect_faces(img)
                        cache_mod.store_file(cache, img, faces)
                        indexed += 1
                        faces_total += len(faces)
                    except Exception:
                        errors += 1
                self.progress.emit(i, total, str(img))

            pruned = cache_mod.prune_missing(cache, under=self.folder)
            cache_mod.save_cache(cache, self.cache_path)
            self.done.emit({
                "indexed": indexed, "skipped": skipped, "errors": errors,
                "faces": faces_total, "pruned": pruned,
                "total_cached": len(cache["files"]),
            })
        except Exception as exc:
            self.failed.emit(str(exc))


class SearchWorker(QThread):
    """Encodes references + scans the cache in the background."""
    done = pyqtSignal(list, float)          # [(path, distance)], elapsed seconds
    failed = pyqtSignal(str)

    def __init__(self, references: list[Path], cache_path: Path, threshold: float):
        super().__init__()
        self.references = references
        self.cache_path = cache_path
        self.threshold = threshold

    def run(self):
        import time
        try:
            cache = cache_mod.load_cache(self.cache_path)
            if not cache["files"]:
                self.failed.emit("The cache is empty. Index a folder first.")
                return

            ref_encodings = []
            for ref in self.references:
                enc = engine.encode_reference(ref)
                if enc is not None:
                    ref_encodings.append(enc)
            if not ref_encodings:
                self.failed.emit("No usable face found in the reference image(s).")
                return

            target = engine.average_encodings(ref_encodings)

            start = time.perf_counter()
            matches = []
            for path_str, record in cache["files"].items():
                if not record["faces"] or not Path(path_str).exists():
                    continue
                encodings = [f["embedding"] for f in record["faces"]]
                best = float(min(engine.face_distance(encodings, target)))
                if best <= self.threshold:
                    matches.append((path_str, best))
            elapsed = time.perf_counter() - start

            matches.sort(key=lambda m: m[1])
            self.done.emit(matches, elapsed)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PhotoTraceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoTrace")
        self.resize(900, 720)

        self.reference_paths: list[Path] = []
        self.target_folder: Path | None = None
        self.cache_path = cache_mod.default_cache_path()
        self.result_checkboxes: list[tuple[str, QCheckBox]] = []
        self._worker = None  # keep a reference so threads aren't GC'd mid-run

        self._build_ui()

    # -- UI construction ----------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        # --- References row ---
        ref_box = QVBoxLayout()
        ref_header = QHBoxLayout()
        ref_header.addWidget(QLabel("<b>1. Reference photos</b> (1–3 of the same person)"))
        ref_header.addStretch()
        self.add_ref_btn = QPushButton("Add reference…")
        self.add_ref_btn.clicked.connect(self.add_references)
        self.clear_ref_btn = QPushButton("Clear")
        self.clear_ref_btn.clicked.connect(self.clear_references)
        ref_header.addWidget(self.add_ref_btn)
        ref_header.addWidget(self.clear_ref_btn)
        ref_box.addLayout(ref_header)

        self.ref_list = QListWidget()
        self.ref_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.ref_list.setIconSize(QSize(REF_THUMB_SIZE, REF_THUMB_SIZE))
        self.ref_list.setFixedHeight(REF_THUMB_SIZE + 40)
        self.ref_list.setMovement(QListWidget.Movement.Static)
        ref_box.addWidget(self.ref_list)
        root.addLayout(ref_box)

        # --- Target folder row ---
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("<b>2. Folder</b>"))
        self.folder_label = QLabel("<i>no folder selected</i>")
        self.folder_label.setMinimumWidth(360)
        folder_row.addWidget(self.folder_label, stretch=1)
        self.pick_folder_btn = QPushButton("Choose folder…")
        self.pick_folder_btn.clicked.connect(self.pick_folder)
        folder_row.addWidget(self.pick_folder_btn)
        root.addLayout(folder_row)

        # --- Actions row ---
        action_row = QHBoxLayout()
        self.index_btn = QPushButton("Index folder")
        self.index_btn.clicked.connect(self.start_index)
        action_row.addWidget(self.index_btn)

        action_row.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.10, 1.00)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.60)
        self.threshold_spin.setToolTip("Lower = stricter. Live slider comes in Stage 4.")
        action_row.addWidget(self.threshold_spin)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.start_search)
        action_row.addWidget(self.search_btn)
        action_row.addStretch()
        root.addLayout(action_row)

        # --- Progress + status ---
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.status = QLabel(f"Ready. Cache: {self.cache_path}")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        # --- Results grid (scrollable) ---
        self.results_host = QWidget()
        self.results_grid = QGridLayout(self.results_host)
        self.results_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.results_host)
        root.addWidget(scroll, stretch=1)

        self.setCentralWidget(central)

    # -- Reference handling -------------------------------------------------

    def add_references(self):
        remaining = MAX_REFERENCES - len(self.reference_paths)
        if remaining <= 0:
            QMessageBox.information(self, "PhotoTrace",
                                    f"You can use at most {MAX_REFERENCES} references.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose reference image(s)", "",
            "Images (*.jpg *.jpeg *.png *.webp *.avif)")
        for f in files[:remaining]:
            p = Path(f)
            self.reference_paths.append(p)
            item = QListWidgetItem(p.name)
            pix = load_thumbnail(p, REF_THUMB_SIZE)
            if pix is not None:
                item.setIcon(QIcon(pix))
            self.ref_list.addItem(item)
        self._refresh_status()

    def clear_references(self):
        self.reference_paths.clear()
        self.ref_list.clear()
        self._refresh_status()

    # -- Folder handling ----------------------------------------------------

    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder to search")
        if folder:
            self.target_folder = Path(folder)
            self.folder_label.setText(str(self.target_folder))
            self._refresh_status()

    def _refresh_status(self):
        self.status.setText(
            f"{len(self.reference_paths)} reference(s) · "
            f"folder: {self.target_folder or '—'} · cache: {self.cache_path}")

    # -- Indexing -----------------------------------------------------------

    def start_index(self):
        if not self.target_folder:
            QMessageBox.warning(self, "PhotoTrace", "Choose a folder to index first.")
            return
        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.setText("Indexing… detecting faces (first run can be slow).")

        self._worker = IndexWorker(self.target_folder, self.cache_path)
        self._worker.progress.connect(self._on_index_progress)
        self._worker.done.connect(self._on_index_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_index_progress(self, done: int, total: int, path: str):
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.status.setText(f"Indexing {done}/{total}: {Path(path).name}")

    def _on_index_done(self, stats: dict):
        self.progress.setVisible(False)
        self._set_busy(False)
        self.status.setText(
            f"Index complete — {stats['indexed']} new ({stats['faces']} faces), "
            f"{stats['skipped']} cached, {stats['errors']} unreadable, "
            f"{stats['pruned']} pruned. Total cached: {stats['total_cached']}.")

    # -- Searching ----------------------------------------------------------

    def start_search(self):
        if not self.reference_paths:
            QMessageBox.warning(self, "PhotoTrace", "Add at least one reference photo.")
            return
        self._set_busy(True)
        self.status.setText("Searching…")
        self._clear_results()

        self._worker = SearchWorker(
            list(self.reference_paths), self.cache_path, self.threshold_spin.value())
        self._worker.done.connect(self._on_search_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_search_done(self, matches: list, elapsed: float):
        self._set_busy(False)
        threshold = self.threshold_spin.value()
        self.status.setText(
            f"{len(matches)} match(es) in {elapsed*1000:.0f} ms "
            f"(threshold {threshold:.2f}).")
        self._populate_results(matches, threshold)

    # -- Results grid -------------------------------------------------------

    def _clear_results(self):
        self.result_checkboxes.clear()
        while self.results_grid.count():
            item = self.results_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _populate_results(self, matches: list, threshold: float):
        self._clear_results()
        for idx, (path_str, distance) in enumerate(matches):
            card = self._make_result_card(path_str, distance, threshold)
            self.results_grid.addWidget(card, idx // GRID_COLUMNS, idx % GRID_COLUMNS)

    def _make_result_card(self, path_str: str, distance: float, threshold: float) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        v = QVBoxLayout(card)

        thumb = QLabel()
        thumb.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = load_thumbnail(Path(path_str), THUMB_SIZE)
        thumb.setPixmap(pix) if pix else thumb.setText("(no preview)")
        v.addWidget(thumb, alignment=Qt.AlignmentFlag.AlignCenter)

        conf = engine.distance_to_confidence(distance, threshold) * 100
        info = QLabel(f"{conf:.0f}% &nbsp;·&nbsp; dist {distance:.3f}")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(info)

        check = QCheckBox(Path(path_str).name)
        check.setToolTip(path_str)
        v.addWidget(check)
        self.result_checkboxes.append((path_str, check))

        return card

    # -- Shared helpers -----------------------------------------------------

    def _on_worker_failed(self, message: str):
        self.progress.setVisible(False)
        self._set_busy(False)
        QMessageBox.critical(self, "PhotoTrace", message)
        self.status.setText(f"Error: {message}")

    def _set_busy(self, busy: bool):
        for w in (self.index_btn, self.search_btn, self.add_ref_btn,
                  self.clear_ref_btn, self.pick_folder_btn):
            w.setEnabled(not busy)


def main():
    app = QApplication(sys.argv)
    window = PhotoTraceWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
