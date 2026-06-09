"""
PhotoTrace — STAGE 4: file operations + live confidence slider
==============================================================

Builds on the Stage 3 window:

  * Per-image selection, plus Select all / Deselect all.
  * MOVE or COPY the selected images to a folder you choose. Move asks for
    confirmation first (it relocates files); copy does not.
  * A live confidence slider that re-filters the displayed results instantly —
    no re-search needed, because a search returns every candidate up to a cap
    and the slider just changes which ones are shown.

Still: no delete (moving to a folder is the safe alternative), heavy work runs
in background threads, and thumbnails load through Pillow so .webp/.avif render.

Run:
    python gui.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
)
from PIL import Image

import engine
import cache as cache_mod
import fileops

THUMB_SIZE = 170        # px, square thumbnails in the results grid
REF_THUMB_SIZE = 84     # px, small thumbnails for chosen references
GRID_COLUMNS = 4        # results per row
MAX_REFERENCES = 3

# A search returns every image whose best face distance is <= this cap; the
# slider then filters within that set. Keeping a cap bounds how much we hold.
SEARCH_CAP = 0.80
SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT = 30, 80, 60   # represent 0.30..0.80


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
    """
    Encodes references + scans the cache in the background. Returns EVERY image
    whose best distance is <= SEARCH_CAP (sorted best-first) so the UI slider
    can re-filter live without re-searching.
    """
    done = pyqtSignal(list, float)          # [(path, distance)], elapsed seconds
    failed = pyqtSignal(str)

    def __init__(self, references: list[Path], cache_path: Path):
        super().__init__()
        self.references = references
        self.cache_path = cache_path

    def run(self):
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
            candidates = []
            for path_str, record in cache["files"].items():
                if not record["faces"] or not Path(path_str).exists():
                    continue
                encodings = [f["embedding"] for f in record["faces"]]
                best = float(min(engine.face_distance(encodings, target)))
                if best <= SEARCH_CAP:
                    candidates.append((path_str, best))
            elapsed = time.perf_counter() - start

            candidates.sort(key=lambda m: m[1])
            self.done.emit(candidates, elapsed)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PhotoTraceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoTrace")
        self.resize(940, 760)

        self.reference_paths: list[Path] = []
        self.target_folder: Path | None = None
        self.cache_path = cache_mod.default_cache_path()

        self.all_candidates: list[tuple[str, float]] = []   # full result set (<= cap)
        self.visible_checks: list[tuple[str, QCheckBox]] = []  # currently shown cards
        self.selected_paths: set[str] = set()               # survives re-filtering
        self._thumb_cache: dict[str, QPixmap] = {}          # path -> grid pixmap
        self._worker = None

        self._build_ui()

    # -- UI construction ----------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        # --- References row ---
        ref_header = QHBoxLayout()
        ref_header.addWidget(QLabel("<b>1. Reference photos</b> (1–3 of the same person)"))
        ref_header.addStretch()
        self.add_ref_btn = QPushButton("Add reference…")
        self.add_ref_btn.clicked.connect(self.add_references)
        self.clear_ref_btn = QPushButton("Clear")
        self.clear_ref_btn.clicked.connect(self.clear_references)
        ref_header.addWidget(self.add_ref_btn)
        ref_header.addWidget(self.clear_ref_btn)
        root.addLayout(ref_header)

        self.ref_list = QListWidget()
        self.ref_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.ref_list.setIconSize(QSize(REF_THUMB_SIZE, REF_THUMB_SIZE))
        self.ref_list.setFixedHeight(REF_THUMB_SIZE + 40)
        self.ref_list.setMovement(QListWidget.Movement.Static)
        root.addWidget(self.ref_list)

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
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.start_search)
        action_row.addWidget(self.index_btn)
        action_row.addWidget(self.search_btn)
        action_row.addStretch()
        root.addLayout(action_row)

        # --- Live confidence slider ---
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Confidence threshold:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self.slider.setValue(SLIDER_DEFAULT)
        self.slider.setToolTip("Drag to re-filter results live. Lower = stricter.")
        self.slider.valueChanged.connect(self._on_slider_changed)
        filter_row.addWidget(self.slider, stretch=1)
        self.slider_label = QLabel(f"{self._threshold():.2f}")
        filter_row.addWidget(self.slider_label)
        root.addLayout(filter_row)

        # --- Progress + status ---
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.status = QLabel(f"Ready. Cache: {self.cache_path}")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        # --- Selection / file-ops toolbar ---
        ops_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Select all")
        self.select_all_btn.clicked.connect(lambda: self._set_all_selected(True))
        self.deselect_all_btn = QPushButton("Deselect all")
        self.deselect_all_btn.clicked.connect(lambda: self._set_all_selected(False))
        ops_row.addWidget(self.select_all_btn)
        ops_row.addWidget(self.deselect_all_btn)
        ops_row.addStretch()
        self.copy_btn = QPushButton("Copy selected…")
        self.copy_btn.clicked.connect(self.copy_selected)
        self.move_btn = QPushButton("Move selected…")
        self.move_btn.clicked.connect(self.move_selected)
        ops_row.addWidget(self.copy_btn)
        ops_row.addWidget(self.move_btn)
        root.addLayout(ops_row)

        # --- Results grid (scrollable) ---
        self.results_host = QWidget()
        self.results_grid = QGridLayout(self.results_host)
        self.results_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.results_host)
        root.addWidget(scroll, stretch=1)

        self.setCentralWidget(central)

    # -- Small helpers ------------------------------------------------------

    def _threshold(self) -> float:
        return self.slider.value() / 100.0

    def _thumb(self, path_str: str) -> QPixmap | None:
        """Cached grid thumbnail so live re-filtering never re-reads from disk."""
        if path_str not in self._thumb_cache:
            self._thumb_cache[path_str] = load_thumbnail(Path(path_str), THUMB_SIZE)
        return self._thumb_cache[path_str]

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
        self.selected_paths.clear()
        self._clear_grid()

        self._worker = SearchWorker(list(self.reference_paths), self.cache_path)
        self._worker.done.connect(self._on_search_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_search_done(self, candidates: list, elapsed: float):
        self._set_busy(False)
        self.all_candidates = candidates
        self._last_search_ms = elapsed * 1000
        self._apply_filter()

    # -- Live filter + results grid -----------------------------------------

    def _on_slider_changed(self):
        self.slider_label.setText(f"{self._threshold():.2f}")
        if self.all_candidates:
            self._apply_filter()

    def _apply_filter(self):
        """Show only candidates within the current threshold. Cheap: reuses
        cached thumbnails and just rebuilds the grid widgets."""
        threshold = self._threshold()
        shown = [(p, d) for (p, d) in self.all_candidates if d <= threshold]

        self._clear_grid()
        for idx, (path_str, distance) in enumerate(shown):
            card = self._make_result_card(path_str, distance, threshold)
            self.results_grid.addWidget(card, idx // GRID_COLUMNS, idx % GRID_COLUMNS)

        ms = getattr(self, "_last_search_ms", 0.0)
        self.status.setText(
            f"Showing {len(shown)} of {len(self.all_candidates)} candidate(s) "
            f"at threshold {threshold:.2f} · last search {ms:.0f} ms.")

    def _clear_grid(self):
        self.visible_checks.clear()
        while self.results_grid.count():
            item = self.results_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _make_result_card(self, path_str: str, distance: float, threshold: float) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        v = QVBoxLayout(card)

        thumb = QLabel()
        thumb.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = self._thumb(path_str)
        thumb.setPixmap(pix) if pix else thumb.setText("(no preview)")
        v.addWidget(thumb, alignment=Qt.AlignmentFlag.AlignCenter)

        conf = engine.distance_to_confidence(distance, threshold) * 100
        info = QLabel(f"{conf:.0f}% &nbsp;·&nbsp; dist {distance:.3f}")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(info)

        check = QCheckBox(Path(path_str).name)
        check.setToolTip(path_str)
        check.setChecked(path_str in self.selected_paths)   # restore selection
        check.toggled.connect(lambda on, p=path_str: self._on_check(p, on))
        v.addWidget(check)
        self.visible_checks.append((path_str, check))

        return card

    def _on_check(self, path_str: str, on: bool):
        if on:
            self.selected_paths.add(path_str)
        else:
            self.selected_paths.discard(path_str)

    def _set_all_selected(self, on: bool):
        for path_str, check in self.visible_checks:
            check.setChecked(on)   # toggled signal keeps selected_paths in sync

    # -- File operations ----------------------------------------------------

    def _selected_existing(self) -> list[str]:
        """Selected paths that still exist on disk (and are currently shown)."""
        shown = {p for p, _ in self.visible_checks}
        return [p for p in shown if p in self.selected_paths and Path(p).exists()]

    def copy_selected(self):
        self._do_transfer(move=False)

    def move_selected(self):
        self._do_transfer(move=True)

    def _do_transfer(self, *, move: bool):
        selected = self._selected_existing()
        if not selected:
            QMessageBox.information(self, "PhotoTrace", "No images selected.")
            return

        verb = "Move" if move else "Copy"
        dest = QFileDialog.getExistingDirectory(self, f"{verb} {len(selected)} image(s) to…")
        if not dest:
            return
        dest_dir = Path(dest)

        # Confirmation is required before a MOVE (it relocates the originals).
        if move:
            reply = QMessageBox.question(
                self, "Confirm move",
                f"Move {len(selected)} image(s) to:\n{dest_dir}\n\n"
                f"The original files will be relocated. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        if move:
            succeeded, failed = fileops.move_files(selected, dest_dir)
        else:
            succeeded, failed = fileops.copy_files(selected, dest_dir)

        # After a move, the originals are gone — drop them from the result set
        # and the current selection so the grid reflects reality.
        if move:
            moved = {str(src) for src, _ in succeeded}
            self.all_candidates = [(p, d) for (p, d) in self.all_candidates if p not in moved]
            self.selected_paths -= moved
            self._apply_filter()

        msg = f"{verb}d {len(succeeded)} image(s) to:\n{dest_dir}"
        if failed:
            msg += f"\n\n{len(failed)} failed:\n" + "\n".join(
                f"  • {Path(p).name}: {e}" for p, e in failed[:10])
        QMessageBox.information(self, "PhotoTrace", msg)

    # -- Shared helpers -----------------------------------------------------

    def _on_worker_failed(self, message: str):
        self.progress.setVisible(False)
        self._set_busy(False)
        QMessageBox.critical(self, "PhotoTrace", message)
        self.status.setText(f"Error: {message}")

    def _set_busy(self, busy: bool):
        for w in (self.index_btn, self.search_btn, self.add_ref_btn,
                  self.clear_ref_btn, self.pick_folder_btn,
                  self.copy_btn, self.move_btn):
            w.setEnabled(not busy)


def main():
    app = QApplication(sys.argv)
    window = PhotoTraceWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
