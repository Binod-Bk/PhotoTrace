# PhotoTrace

Find every photo containing a specific person across a folder — including group
photos — using local, offline face recognition. No network, no cloud.

> **Status: Stage 5 (complete)** — full local desktop app. SQLite-backed cache,
> matched-face highlighting, open-in-default-editor, plus everything from the
> earlier stages: multi-reference search, move/copy, live confidence slider.

## Install (Windows + Python 3.13)

The only tricky dependency is `dlib`. There is no official prebuilt `dlib`
wheel for Windows/Python 3.13, and building from source needs Visual Studio
C++ build tools. We sidestep that with the prebuilt **`dlib-bin`** package.

Run these **in order** (a virtual environment is recommended but optional):

```powershell
# (optional but recommended) create + activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1. prebuilt dlib engine (provides the importable `dlib` module)
python -m pip install dlib-bin

# 2. face_recognition WITHOUT deps, so pip doesn't try to build source dlib
python -m pip install face_recognition==1.3.0 --no-deps

# 3. the remaining real dependencies
python -m pip install face_recognition_models numpy Pillow "setuptools<81"
```

Why `setuptools<81`: `face_recognition_models` loads its model files via
`pkg_resources`, which was **removed** in setuptools 81. Python 3.13 doesn't
bundle setuptools at all, so we install an older one explicitly. You'll see a
harmless `pkg_resources is deprecated` warning — that's expected.

### Linux / macOS

The same commands work. On most Linux/macOS + Python combos the plain
`pip install dlib` also compiles cleanly (cmake + a C++ compiler required), but
`dlib-bin` is the simplest path everywhere it has a matching wheel.

### Verify the install

```powershell
python -c "import face_recognition; print('engine ready')"
```

## Build a standalone .exe (Windows)

To produce a single `PhotoTrace.exe` that runs without Python installed:

```powershell
python -m pip install pyinstaller
python -m PyInstaller PhotoTrace.spec
```

The exe lands in `dist\PhotoTrace.exe` (~150 MB — it bundles Python, Qt, dlib,
and the face models). It's a windowed app: double-click to run. The build
recipe `PhotoTrace.spec` handles the two things PyInstaller can't auto-detect —
the dlib model data files and Pillow's AVIF/WebP plugins.

> The `.exe` is too large for the git repo (GitHub caps files at 100 MB);
> distribute it via a GitHub **Release** instead. `build/` and `dist/` are
> git-ignored.

## Usage — desktop app

Either double-click `dist\PhotoTrace.exe`, or run from source:

```powershell
python gui.py
```

1. **Add reference…** — pick 1–3 photos of the same person (they're averaged).
2. **Choose folder…** — the folder to search.
3. **Index folder** — one-time, runs in the background with a live progress bar.
4. **Search** — shows matches in a scrollable thumbnail grid. Each result has a
   confidence score, a checkbox, an **Open** button (opens the image in the
   system's default editor), and a **green box drawn around the matched face**
   so you can verify before acting — important for group photos.
5. **Confidence slider** — drag to re-filter the shown results instantly (no
   re-search; a search collects every candidate up to a cap and the slider just
   changes which are displayed).
6. **Select all / Deselect all**, then **Copy selected…** or **Move selected…**
   to relocate them to a folder you choose. Move asks for confirmation first;
   existing names are never overwritten (`(1)`, `(2)`… is appended). There is
   **no delete** — moving to a folder is the safe alternative.

Indexing and searching run on background threads, so the window stays
responsive. Thumbnails load through Pillow, so `.webp` / `.avif` previews work.

## Usage — command line

The GUI and CLI share the same SQLite cache, so you can index from one and
search from the other.

Two phases: **index** a folder once (slow), then **search** it as often as you
like (instant). Searching for a different person reuses the same index.

```powershell
# 1) Index a folder (recursive). Repeat later only to pick up new/changed files.
python phototrace.py index "C:\Users\binod\Pictures"

# 2) Search with 1-3 reference photos of ONE person (they get averaged together).
python phototrace.py search ref1.jpg ref2.jpg ref3.jpg
python phototrace.py search ref1.jpg --threshold 0.55 --dir "C:\Users\binod\Pictures"
```

- Supported formats: `.jpg .jpeg .png .webp .avif` (AVIF needs Pillow >= 11.3,
  which bundles libavif; older Pillow skips them), scanned recursively.
- `--threshold` — face *distance* cutoff (lower = stricter). Default `0.6`.
- `--dir` — limit a search to one folder within the index.
- `--cache PATH` — use a specific cache file (default: `~/.phototrace/index.db`).
- `index --rebuild` — ignore the existing cache and re-index everything.

**Why two phases:** face detection + embedding is the slow part. The index does
it once and caches `(file_path, face_location, embedding)` to disk. Re-indexing
skips unchanged files (tracked by mod/size), and every search just compares
cached vectors — typically a few **milliseconds**.

## Indexing speed (CPU-only)

Indexing is the expensive step. Three CPU-only optimizations (no GPU needed)
keep it fast, and originals are never modified:

- **Downscale for detection** — faces are *detected* on a downscaled copy
  (`engine.MAX_DETECT_DIM`, default 800 px long edge), then *encoded* from the
  full-resolution image. Detection time scales with pixels, so large photos and
  screenshots get much faster with no loss of match accuracy.
- **No upsampling** — the detector skips the expensive small-face upsample pass.
- **Parallel detection** — files are detected across multiple CPU cores. Defaults
  to half your logical cores (each worker loads its own copy of dlib, so this is
  kept modest to protect RAM). Override with the `PHOTOTRACE_WORKERS` env var,
  e.g. `set PHOTOTRACE_WORKERS=2` on a low-memory machine.

**Tip:** index your actual photo folders (Pictures, camera, Downloads) — not
whole drives like `C:\`, which are full of faceless system/app images.

## The cache (SQLite)

The cache is a single local SQLite file at `~/.phototrace/index.db` (override
with `--cache`). It has two tables: `files` (path, mtime, size, indexed_at) and
`faces` (a row per detected face: location box + 128-d embedding BLOB). Tracking
`mtime`/`size` lets re-indexing process only new or changed files. Nothing
leaves your machine. A pre-existing `index.pkl` from earlier versions is
migrated into SQLite automatically the first time the new code opens it.

## Module layout

| File              | Responsibility                                              |
|-------------------|-------------------------------------------------------------|
| `engine.py`       | All face-recognition calls (swap here for InsightFace later)|
| `db.py`           | SQLite embedding cache (`FaceCache`); migrates old pickle   |
| `phototrace.py`   | CLI: `index` and `search` commands                          |
| `fileops.py`      | Safe move/copy + open-in-editor (no overwrite, no delete)   |
| `gui.py`          | PyQt6 desktop UI                                            |
| `stage1_match.py` | Stage 1 single-file proof (kept for reference)              |

Concerns are separated so the recognition engine, the cache, file operations,
and the UI can each change independently. In particular, swapping `engine.py`
for a different recognition library (e.g. InsightFace) requires no changes
elsewhere.

### Tuning the threshold

We compare faces by **distance** (lower = more similar). A face matches when
its distance to the reference is `<= threshold`.

| threshold | behaviour                                  |
|-----------|--------------------------------------------|
| 0.6       | library default; good starting point       |
| 0.5       | stricter: fewer false matches, may miss some |
| 0.45      | very strict                                |

If you get false matches, lower it. If real photos are missed, raise it.
The active threshold is printed at the top of every run so it's easy to tune.
