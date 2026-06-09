"""
PhotoTrace — desktop UI (PyQt6)
===============================

The full local app over the recognition engine (`engine.py`), the SQLite cache
(`db.py`), and file operations (`fileops.py`):

  * Pick 1-3 reference photos of ONE person and choose a folder.
  * INDEX it (slow, once) — runs in a background thread with a progress bar.
  * SEARCH (fast) — matches appear in a scrollable thumbnail grid, each with a
    confidence score, a checkbox, an "Open" button (default editor), and a green
    box around the matched face so group photos are easy to verify.
  * A live confidence slider re-filters the shown results instantly (no
    re-search: a search collects every candidate up to a cap, the slider just
    changes which are displayed).
  * Select all / Deselect all, then MOVE or COPY the selected images to a folder.
    Move asks for confirmation first; there is no delete.

Heavy work runs in QThread workers so the window never freezes. Thumbnails load
through Pillow so .webp/.avif render even where Qt can't decode them natively.

Run:
    python gui.py
"""

from __future__ import annotations

import sys
import time
import webbrowser
from pathlib import Path
from string import Template

import numpy as np
from PyQt6.QtCore import Qt, QThread, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QVBoxLayout,
    QWidget,
)
from PIL import Image

import engine
import fileops
from db import FaceCache, default_db_path

THUMB_SIZE = 170        # px, square thumbnails in the results grid
REF_THUMB_SIZE = 84     # px, small thumbnails for chosen references
GRID_COLUMNS = 4        # results per row
MAX_REFERENCES = 3

# A search returns every image whose best face distance is <= this cap; the
# slider then filters within that set. Keeping a cap bounds how much we hold.
SEARCH_CAP = 0.80
SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT = 30, 80, 50   # represent 0.30..0.80
                                                      # default 0.50 = stricter,
                                                      # fewer false positives

# Two themes (dark / light) sharing one indigo/violet accent. The stylesheet is
# built once per palette from a template, then applied to the whole app via
# QApplication.setStyleSheet — toggling at runtime simply swaps which one is set.
# Primary actions use a dynamic `primary="true"` property; footer links use
# `link="true"`; result cards use objectName "card" (+ a "selected" property).
REPO_URL = "https://github.com/Binod-Bk/PhotoTrace"
ACCENT = "#6366F1"

_QSS = Template("""
* { font-family: "Segoe UI", "Inter", sans-serif; font-size: 13px; color: ${text}; }
QMainWindow, QWidget { background-color: ${bg}; }

QLabel { color: ${text}; background: transparent; }
QLabel#appTitle { font-size: 24px; font-weight: 800; color: ${text}; }
QLabel#sectionHeader { font-size: 14px; font-weight: 700; color: ${text}; }
QLabel#muted { color: ${muted}; }

QPushButton {
    background-color: ${btn}; color: ${text};
    border: 1px solid ${border}; border-radius: 8px; padding: 7px 14px;
}
QPushButton:hover { background-color: ${btn_hover}; }
QPushButton:pressed { background-color: ${btn_pressed}; }
QPushButton:disabled { color: ${muted}; }

QPushButton[primary="true"] {
    background-color: ${accent}; border: 1px solid ${accent};
    color: #FFFFFF; font-weight: 600; padding: 8px 20px;
}
QPushButton[primary="true"]:hover { background-color: ${accent_hover}; border-color: ${accent_hover}; }
QPushButton[primary="true"]:pressed { background-color: ${accent_pressed}; }

QPushButton[link="true"] {
    background: transparent; border: none; color: ${accent};
    padding: 4px 6px; font-weight: 600;
}
QPushButton[link="true"]:hover { color: ${accent_hover}; }

QFrame#card { background-color: ${card}; border: 1px solid ${border}; border-radius: 12px; }
QFrame#card[selected="true"] { border: 2px solid ${accent}; background-color: ${card_sel}; }
QFrame#footer { border-top: 1px solid ${border}; }

QListWidget { background-color: ${card}; border: 1px solid ${border}; border-radius: 10px; padding: 6px; }
QListWidget::item { color: ${text}; }
QListWidget::item:selected { background: transparent; }

QSlider::groove:horizontal { height: 6px; background: ${border}; border-radius: 3px; }
QSlider::sub-page:horizontal { background: ${accent}; border-radius: 3px; }
QSlider::handle:horizontal {
    background: ${handle}; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
}

QProgressBar {
    background: ${card}; border: 1px solid ${border}; border-radius: 8px;
    text-align: center; color: ${text}; height: 18px;
}
QProgressBar::chunk { background-color: ${accent}; border-radius: 7px; }

QCheckBox { color: ${text}; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid ${border}; background: ${indicator};
}
QCheckBox::indicator:checked { background: ${accent}; border-color: ${accent}; }

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: ${scroll}; border-radius: 6px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: ${scroll_hover}; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QDialog { background-color: ${bg}; }
QToolTip { background: ${tooltip}; color: ${text}; border: 1px solid ${border}; padding: 4px; }
""")

_DARK = dict(
    bg="#1A1A20", card="#24242C", card_sel="#2A2A38", border="#32323D",
    text="#E5E7EB", muted="#8A8D99", accent=ACCENT, accent_hover="#7C7FF2",
    accent_pressed="#4F46E5", btn="#2B2B35", btn_hover="#34343F",
    btn_pressed="#26262E", indicator="#2B2B35", handle="#FFFFFF",
    scroll="#3A3A46", scroll_hover="#4B4B58", tooltip="#2B2B35",
)
_LIGHT = dict(
    bg="#F4F5F7", card="#FFFFFF", card_sel="#EEF0FE", border="#E2E4E9",
    text="#1F2430", muted="#6B7280", accent=ACCENT, accent_hover="#7C7FF2",
    accent_pressed="#4F46E5", btn="#FFFFFF", btn_hover="#F0F1F4",
    btn_pressed="#E6E8EC", indicator="#FFFFFF", handle=ACCENT,
    scroll="#C9CCD4", scroll_hover="#B3B7C2", tooltip="#FFFFFF",
)
THEME_DARK = _QSS.substitute(_DARK)
THEME_LIGHT = _QSS.substitute(_LIGHT)
THEME = THEME_DARK   # default; also used by headless tests


def _confidence_pill_style(conf: float) -> str:
    """Inline style for the confidence badge — green / amber / slate by tier."""
    if conf >= 80:
        rgb = "52, 211, 153"       # green
    elif conf >= 65:
        rgb = "251, 191, 36"       # amber
    else:
        rgb = "148, 163, 184"      # slate
    return (f"background: rgba({rgb}, 0.16); color: rgb({rgb}); "
            f"font-weight: 700; border-radius: 9px; padding: 2px 10px;")


# ---------------------------------------------------------------------------
# Thumbnail loading (via Pillow so every supported format renders)
# ---------------------------------------------------------------------------

def load_thumbnail(path: Path, size: int, highlight: tuple | None = None) -> QPixmap | None:
    """
    Load `path` and return a QPixmap no larger than size x size, or None.

    If `highlight` is a (top, right, bottom, left) box in ORIGINAL image
    coordinates, draw a green rectangle around it (scaled to the thumbnail) so
    the user can see which face matched.
    """
    try:
        im = Image.open(path)
        orig_w, orig_h = im.size
        im.draft("RGB", (size, size))        # speeds up large JPEG decoding
        im = im.convert("RGB")
        im.thumbnail((size, size))
        data = im.tobytes("raw", "RGB")
        qimg = QImage(data, im.width, im.height, im.width * 3,
                      QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())

        if highlight is not None and orig_w and orig_h:
            top, right, bottom, left = highlight
            sx, sy = pix.width() / orig_w, pix.height() / orig_h
            painter = QPainter(pix)
            pen = QPen(QColor(40, 200, 90))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(int(left * sx), int(top * sy),
                             int((right - left) * sx), int((bottom - top) * sy))
            painter.end()
        return pix
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class IndexWorker(QThread):
    """Indexes one or more folders in the background, emitting progress."""
    scanning = pyqtSignal()                 # walking folders (no count yet)
    counted = pyqtSignal(int)               # total images found
    progress = pyqtSignal(int, int, str)    # done, total, current path
    done = pyqtSignal(dict)                 # summary stats
    failed = pyqtSignal(str)

    def __init__(self, folders: list[Path], cache_path: Path):
        super().__init__()
        self.folders = folders
        self.cache_path = cache_path

    def run(self):
        try:
            cache = FaceCache(self.cache_path)

            # Phase 1: walk every folder and gather a de-duplicated file list.
            # On huge trees this can take a while, so signal a busy state first.
            self.scanning.emit()
            seen: set[str] = set()
            images: list[Path] = []
            for folder in self.folders:
                for p in folder.rglob("*"):
                    key = str(p)
                    if (key not in seen and p.is_file()
                            and p.suffix.lower() in engine.SUPPORTED_EXTENSIONS):
                        seen.add(key)
                        images.append(p)
            # Skip files already cached & unchanged so we only do real work.
            skipped = 0
            to_detect: list[Path] = []
            for img in images:
                if cache.is_current(img):
                    skipped += 1
                else:
                    to_detect.append(img)

            total = len(to_detect)
            self.counted.emit(total)

            # Phase 2: detect faces in parallel across CPU cores; write results
            # to the cache here in the worker thread as each one streams back.
            indexed = errors = faces_total = 0
            done = 0
            for path_str, faces, err in engine.detect_many(to_detect):
                done += 1
                if err is None:
                    cache.upsert_file(Path(path_str), faces)
                    indexed += 1
                    faces_total += len(faces)
                else:
                    errors += 1
                self.progress.emit(done, total, path_str)

            # Global prune only drops files that no longer exist on disk, so it's
            # safe across multiple folders (other folders' files still exist).
            pruned = cache.prune_missing()
            cache.commit()
            total_cached = cache.file_count()
            cache.close()
            self.done.emit({
                "indexed": indexed, "skipped": skipped, "errors": errors,
                "faces": faces_total, "pruned": pruned,
                "total_cached": total_cached,
            })
        except Exception as exc:
            self.failed.emit(str(exc))


class SearchWorker(QThread):
    """
    Encodes references + scans the cache in the background. Returns EVERY image
    whose best distance is <= SEARCH_CAP (sorted best-first) so the UI slider
    can re-filter live without re-searching.
    """
    done = pyqtSignal(list, float)          # [(path, distance, location)], elapsed
    failed = pyqtSignal(str)

    def __init__(self, references: list[Path], cache_path: Path):
        super().__init__()
        self.references = references
        self.cache_path = cache_path

    def run(self):
        try:
            cache = FaceCache(self.cache_path)
            if cache.file_count() == 0:
                cache.close()
                self.failed.emit("The cache is empty. Index a folder first.")
                return

            ref_encodings = []
            for ref in self.references:
                enc = engine.encode_reference(ref)
                if enc is not None:
                    ref_encodings.append(enc)
            if not ref_encodings:
                cache.close()
                self.failed.emit("No usable face found in the reference image(s).")
                return

            target = engine.average_encodings(ref_encodings)

            start = time.perf_counter()
            candidates = []
            for path_str, faces in cache.iter_files():
                if not faces or not Path(path_str).exists():
                    continue
                encodings = [emb for _loc, emb in faces]
                distances = engine.face_distance(encodings, target)
                best_idx = int(np.argmin(distances))
                best = float(distances[best_idx])
                if best <= SEARCH_CAP:
                    # keep the location of the matched face for highlighting
                    candidates.append((path_str, best, faces[best_idx][0]))
            elapsed = time.perf_counter() - start
            cache.close()

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
        self.target_folders: list[Path] = []
        self.cache_path = default_db_path()

        self.all_candidates: list[tuple] = []   # (path, distance, location), <= cap
        self.visible_checks: list[tuple[str, QCheckBox]] = []  # currently shown cards
        self.selected_paths: set[str] = set()               # survives re-filtering
        self._thumb_cache: dict[str, QPixmap] = {}          # path -> grid pixmap
        self._worker = None
        self._dark = True                                   # current theme

        self._build_ui()

    # -- UI construction ----------------------------------------------------

    def _section_header(self, text: str, hint: str = "") -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel(text)
        title.setObjectName("sectionHeader")
        row.addWidget(title)
        if hint:
            h = QLabel(hint)
            h.setObjectName("muted")
            row.addWidget(h)
        row.addStretch()
        return row

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 16, 22, 12)
        root.setSpacing(10)

        # === FIXED TOP: theme toggle + centered title + subtitle ===========
        topbar = QHBoxLayout()
        self.theme_btn = QPushButton("☀  Light")
        self.theme_btn.setFixedWidth(110)
        self.theme_btn.setToolTip("Switch between dark and light themes")
        self.theme_btn.clicked.connect(self._toggle_theme)
        topbar.addWidget(self.theme_btn)
        topbar.addStretch(1)
        title = QLabel("PhotoTrace")
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        topbar.addWidget(title)
        topbar.addStretch(1)
        right_spacer = QWidget()           # balances the toggle so the title centers
        right_spacer.setFixedWidth(110)
        topbar.addWidget(right_spacer)
        root.addLayout(topbar)

        subtitle = QLabel("Find every photo of a person — locally, offline.")
        subtitle.setObjectName("muted")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)

        # === SCROLLABLE MIDDLE: setup + actions + results ==================
        # Everything here scrolls together, so when you scroll down to browse
        # results, the setup controls move out of the way and the results use
        # the whole viewport. Slider/ops/footer stay fixed below.
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        # --- 1. References ---
        ref_header = self._section_header("1  ·  Reference photos",
                                          "1–3 clear photos of the same person")
        self.add_ref_btn = QPushButton("＋  Add reference…")
        self.add_ref_btn.clicked.connect(self.add_references)
        self.clear_ref_btn = QPushButton("Clear")
        self.clear_ref_btn.clicked.connect(self.clear_references)
        ref_header.addWidget(self.add_ref_btn)
        ref_header.addWidget(self.clear_ref_btn)
        col.addLayout(ref_header)

        self.ref_list = QListWidget()
        self.ref_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.ref_list.setIconSize(QSize(REF_THUMB_SIZE, REF_THUMB_SIZE))
        self.ref_list.setFixedHeight(REF_THUMB_SIZE + 44)
        self.ref_list.setSpacing(8)
        self.ref_list.setMovement(QListWidget.Movement.Static)
        col.addWidget(self.ref_list)

        # --- 2. Folders (one or more) ---
        folder_header = self._section_header("2  ·  Folders to search",
                                             "add one or more folders")
        self.add_folder_btn = QPushButton("＋  Add folder…")
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.remove_folder_btn = QPushButton("Remove")
        self.remove_folder_btn.clicked.connect(self.remove_selected_folders)
        self.clear_folders_btn = QPushButton("Clear")
        self.clear_folders_btn.clicked.connect(self.clear_folders)
        folder_header.addWidget(self.add_folder_btn)
        folder_header.addWidget(self.remove_folder_btn)
        folder_header.addWidget(self.clear_folders_btn)
        col.addLayout(folder_header)

        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self.folder_list.setFixedHeight(72)
        self.folder_list.setToolTip("Tip: choose your photo folders, not whole "
                                    "drives — scanning C:\\ wastes time on system images.")
        col.addWidget(self.folder_list)

        # --- 3. Index, its progress, Search, its progress (stacked) ---
        self.index_btn = QPushButton("Index folder")
        self.index_btn.setProperty("primary", True)
        self.index_btn.setToolTip("Analyze the folders once (slow). Required before searching.")
        self.index_btn.clicked.connect(self.start_index)
        col.addWidget(self.index_btn)

        self.index_progress = QProgressBar()
        self.index_progress.setVisible(False)
        col.addWidget(self.index_progress)

        self.search_btn = QPushButton("🔍  Search")
        self.search_btn.setProperty("primary", True)
        self.search_btn.clicked.connect(self.start_search)
        col.addWidget(self.search_btn)

        self.search_progress = QProgressBar()
        self.search_progress.setVisible(False)
        col.addWidget(self.search_progress)

        # --- Results grid (the part that gets the whole window on scroll) ---
        self.results_host = QWidget()
        self.results_grid = QGridLayout(self.results_host)
        self.results_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_grid.setSpacing(14)
        self.results_grid.setContentsMargins(0, 4, 0, 0)
        col.addWidget(self.results_host)
        col.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        root.addWidget(scroll, stretch=1)

        # === FIXED BOTTOM: status, file-ops, confidence slider (last) ======
        self.status = QLabel("Ready — add reference photos and folders to begin.")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        ops_row = QHBoxLayout()
        ops_row.setSpacing(8)
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

        # Confidence slider lives last — it re-filters results live, so it's
        # always reachable here while you scroll the results above.
        filter_row = QHBoxLayout()
        slabel = QLabel("Confidence threshold")
        slabel.setObjectName("muted")
        filter_row.addWidget(slabel)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self.slider.setValue(SLIDER_DEFAULT)
        self.slider.setToolTip("Drag to re-filter results live. Lower = stricter.")
        self.slider.valueChanged.connect(self._on_slider_changed)
        filter_row.addWidget(self.slider, stretch=1)
        self.slider_label = QLabel(f"{self._threshold():.2f}")
        self.slider_label.setMinimumWidth(34)
        filter_row.addWidget(self.slider_label)
        root.addLayout(filter_row)

        # === FIXED FOOTER: guide + source-code links =======================
        footer = QFrame()
        footer.setObjectName("footer")
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(2, 8, 2, 2)
        guide_btn = QPushButton("Need a Guide?")
        guide_btn.setProperty("link", True)
        guide_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        guide_btn.clicked.connect(self.show_guide)
        footer_row.addWidget(guide_btn)
        footer_row.addStretch()
        src_btn = QPushButton("Source code  ↗")
        src_btn.setProperty("link", True)
        src_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        src_btn.setToolTip(REPO_URL)
        src_btn.clicked.connect(lambda: webbrowser.open(REPO_URL))
        footer_row.addWidget(src_btn)
        root.addWidget(footer)

        self.setCentralWidget(central)
        self.setMinimumSize(820, 640)

    # -- Theme + footer actions --------------------------------------------

    def _toggle_theme(self):
        self._dark = not self._dark
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(THEME_DARK if self._dark else THEME_LIGHT)
        # Show the mode you'd switch TO next.
        self.theme_btn.setText("☀  Light" if self._dark else "🌙  Dark")

    def show_guide(self):
        """Pop a friendly step-by-step walkthrough."""
        dlg = QDialog(self)
        dlg.setWindowTitle("How to use PhotoTrace")
        dlg.setMinimumWidth(460)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(12)

        heading = QLabel("How to use PhotoTrace")
        heading.setObjectName("appTitle")
        lay.addWidget(heading)

        steps = QLabel(
            "<ol style='margin-left:-18px; line-height:150%;'>"
            "<li><b>Add reference photos</b> — pick 1–3 clear, front-facing "
            "photos of the person you're looking for.</li>"
            "<li><b>Add folders</b> — one or more folders of photos to search "
            "(subfolders included). Tip: pick your photo folders, not whole "
            "drives like C:\\ — that wastes time on system images.</li>"
            "<li><b>Index folder</b> — analyzes every photo once. The first run "
            "is slow (you'll see a progress bar); after that it's cached and "
            "instant.</li>"
            "<li><b>Search</b> — matches appear as thumbnails with a confidence "
            "score and a green box on the matched face (great for group photos).</li>"
            "<li><b>Confidence slider</b> — drag to loosen or tighten matches "
            "live; no need to search again.</li>"
            "<li><b>Select &amp; act</b> — tick the photos you want, then "
            "<b>Copy</b> or <b>Move</b> them to a folder. <b>Open</b> views a "
            "photo. There is no delete — moving is the safe alternative.</li>"
            "</ol>"
        )
        steps.setWordWrap(True)
        steps.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(steps)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Got it")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setProperty("primary", True)
        buttons.accepted.connect(dlg.accept)
        lay.addWidget(buttons)
        dlg.exec()

    # -- Small helpers ------------------------------------------------------

    def _threshold(self) -> float:
        return self.slider.value() / 100.0

    def _thumb(self, path_str: str, highlight: tuple | None = None) -> QPixmap | None:
        """Cached grid thumbnail (with matched-face box) so live re-filtering
        never re-reads from disk."""
        if path_str not in self._thumb_cache:
            self._thumb_cache[path_str] = load_thumbnail(
                Path(path_str), THUMB_SIZE, highlight=highlight)
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

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add a folder to search")
        if not folder:
            return
        p = Path(folder)
        if p in self.target_folders:
            return
        self.target_folders.append(p)
        self.folder_list.addItem(QListWidgetItem(str(p)))
        self._refresh_status()

    def remove_selected_folders(self):
        for item in self.folder_list.selectedItems():
            row = self.folder_list.row(item)
            self.folder_list.takeItem(row)
            try:
                self.target_folders.remove(Path(item.text()))
            except ValueError:
                pass
        self._refresh_status()

    def clear_folders(self):
        self.target_folders.clear()
        self.folder_list.clear()
        self._refresh_status()

    def _refresh_status(self):
        n = len(self.target_folders)
        folders = f"{n} folder(s)" if n else "no folders"
        self._status(
            f"{len(self.reference_paths)} reference(s) · {folders} selected.")

    # -- Status line --------------------------------------------------------

    def _status(self, text: str, success: bool = False):
        """Set the status text; green when reporting a completed result."""
        self.status.setText(text)
        # Inline style overrides the muted QSS; cleared (=="") reverts to muted.
        self.status.setStyleSheet(
            "color: #16A34A; font-weight: 600;" if success else "")

    # -- Busy indicator -----------------------------------------------------

    @staticmethod
    def _show_busy(bar: QProgressBar):
        bar.setRange(0, 0)        # 0,0 = animated 'busy' mode
        bar.setVisible(True)

    @staticmethod
    def _hide_progress(bar: QProgressBar):
        bar.setVisible(False)
        bar.setRange(0, 1)

    # -- Indexing -----------------------------------------------------------

    def start_index(self):
        if not self.target_folders:
            QMessageBox.warning(self, "PhotoTrace", "Add at least one folder to index.")
            return
        self._set_busy(True)
        self._show_busy(self.index_progress)        # animated under the Index button
        self._status("⏳  Scanning folders… please wait.")

        self._worker = IndexWorker(list(self.target_folders), self.cache_path)
        self._worker.scanning.connect(
            lambda: self._status("⏳  Scanning folders for images… please wait."))
        self._worker.counted.connect(self._on_index_counted)
        self._worker.progress.connect(self._on_index_progress)
        self._worker.done.connect(self._on_index_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_index_counted(self, total: int):
        # Switch from the animated busy bar to a real 0..total progress bar.
        self.index_progress.setRange(0, max(total, 1))
        self.index_progress.setValue(0)
        if total == 0:
            self._status("✓ Everything already indexed — nothing new to do.",
                         success=True)
        else:
            self._status(
                f"⏳  {total} new/changed image(s) to process. Detecting faces…")

    def _on_index_progress(self, done: int, total: int, path: str):
        self.index_progress.setValue(done)
        self._status(f"⏳  Indexing {done}/{total}: {Path(path).name}")

    def _on_index_done(self, stats: dict):
        self._hide_progress(self.index_progress)
        self._set_busy(False)
        self._status(
            f"✓ Index complete — {stats['indexed']} new ({stats['faces']} faces), "
            f"{stats['skipped']} cached, {stats['errors']} unreadable, "
            f"{stats['pruned']} pruned. Total cached: {stats['total_cached']}.",
            success=True)

    # -- Searching ----------------------------------------------------------

    def start_search(self):
        if not self.reference_paths:
            QMessageBox.warning(self, "PhotoTrace", "Add at least one reference photo.")
            return
        self._set_busy(True)
        self._show_busy(self.search_progress)       # animated under the Search button
        self._status("⏳  Searching… please wait.")
        self.selected_paths.clear()
        self._clear_grid()

        self._worker = SearchWorker(list(self.reference_paths), self.cache_path)
        self._worker.done.connect(self._on_search_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_search_done(self, candidates: list, elapsed: float):
        self._hide_progress(self.search_progress)
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
        shown = [c for c in self.all_candidates if c[1] <= threshold]

        self._clear_grid()
        for idx, (path_str, distance, location) in enumerate(shown):
            card = self._make_result_card(path_str, distance, location, threshold)
            self.results_grid.addWidget(card, idx // GRID_COLUMNS, idx % GRID_COLUMNS)

        ms = getattr(self, "_last_search_ms", 0.0)
        self._status(
            f"✓ Showing {len(shown)} of {len(self.all_candidates)} candidate(s) "
            f"at threshold {threshold:.2f} · last search {ms:.0f} ms.",
            success=len(shown) > 0)

    def _clear_grid(self):
        self.visible_checks.clear()
        while self.results_grid.count():
            item = self.results_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _make_result_card(self, path_str: str, distance: float,
                          location: tuple, threshold: float) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        card.setProperty("selected", path_str in self.selected_paths)
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 12, 12, 10)
        v.setSpacing(8)

        thumb = QLabel()
        thumb.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = self._thumb(path_str, highlight=location)   # green box on matched face
        thumb.setPixmap(pix) if pix else thumb.setText("(no preview)")
        v.addWidget(thumb, alignment=Qt.AlignmentFlag.AlignCenter)

        # Confidence badge (coloured by tier) + muted distance, on one row.
        conf = engine.distance_to_confidence(distance, threshold) * 100
        meta = QHBoxLayout()
        badge = QLabel(f"{conf:.0f}% match")
        badge.setStyleSheet(_confidence_pill_style(conf))
        meta.addWidget(badge)
        meta.addStretch()
        dist_lbl = QLabel(f"dist {distance:.3f}")
        dist_lbl.setObjectName("muted")
        meta.addWidget(dist_lbl)
        v.addLayout(meta)

        # Filename (elided/muted) + selection checkbox + Open.
        bottom = QHBoxLayout()
        check = QCheckBox(self._elide(Path(path_str).name, 20))
        check.setToolTip(path_str)
        check.setChecked(path_str in self.selected_paths)   # restore selection
        check.toggled.connect(lambda on, p=path_str, c=card: self._on_check(p, on, c))
        bottom.addWidget(check, stretch=1)
        open_btn = QPushButton("Open")
        open_btn.setToolTip("Open in the system's default editor")
        open_btn.clicked.connect(lambda _=False, p=path_str: fileops.open_in_editor(p))
        bottom.addWidget(open_btn)
        v.addLayout(bottom)

        self.visible_checks.append((path_str, check))
        return card

    @staticmethod
    def _elide(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def _on_check(self, path_str: str, on: bool, card: QWidget | None = None):
        if on:
            self.selected_paths.add(path_str)
        else:
            self.selected_paths.discard(path_str)
        if card is not None:   # repaint the accent selection border
            card.setProperty("selected", on)
            card.style().unpolish(card)
            card.style().polish(card)

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
            self.all_candidates = [c for c in self.all_candidates if c[0] not in moved]
            self.selected_paths -= moved
            self._apply_filter()

        msg = f"{verb}d {len(succeeded)} image(s) to:\n{dest_dir}"
        if failed:
            msg += f"\n\n{len(failed)} failed:\n" + "\n".join(
                f"  • {Path(p).name}: {e}" for p, e in failed[:10])
        QMessageBox.information(self, "PhotoTrace", msg)

    # -- Shared helpers -----------------------------------------------------

    def _on_worker_failed(self, message: str):
        self._hide_progress(self.index_progress)
        self._hide_progress(self.search_progress)
        self._set_busy(False)
        QMessageBox.critical(self, "PhotoTrace", message)
        self._status(f"Error: {message}")

    def _set_busy(self, busy: bool):
        for w in (self.index_btn, self.search_btn, self.add_ref_btn,
                  self.clear_ref_btn, self.add_folder_btn, self.remove_folder_btn,
                  self.clear_folders_btn, self.copy_btn, self.move_btn):
            w.setEnabled(not busy)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(THEME)
    window = PhotoTraceWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Required so multiprocessing worker processes (used for parallel indexing)
    # don't relaunch the whole GUI — essential once frozen into an .exe.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
