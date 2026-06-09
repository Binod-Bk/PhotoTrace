# PhotoTrace

Find every photo containing a specific person across a folder — including group
photos — using local, offline face recognition. No network, no cloud.

> **Status: Stage 3** — minimal PyQt6 desktop UI on top of the Stage 2 engine:
> pick references, index a folder, and browse matches in a thumbnail grid.

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

## Usage (Stage 3 — desktop app)

```powershell
python gui.py
```

1. **Add reference…** — pick 1–3 photos of the same person (they're averaged).
2. **Choose folder…** — the folder to search.
3. **Index folder** — one-time, runs in the background with a live progress bar.
4. **Search** — shows matches in a scrollable thumbnail grid, each with a
   confidence score and a checkbox.

Indexing and searching run on background threads, so the window stays
responsive. Thumbnails load through Pillow, so `.webp` / `.avif` previews work.
(Select-all, move/copy, and a live confidence slider arrive in Stage 4.)

## Usage (command-line, Stage 2)

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
- `--cache PATH` — use a specific cache file (default: `~/.phototrace/index.pkl`).
- `index --rebuild` — ignore the existing cache and re-index everything.

**Why two phases:** face detection + embedding is the slow part. The index does
it once and caches `(file_path, face_location, embedding)` to disk. Re-indexing
skips unchanged files (tracked by mod/size), and every search just compares
cached vectors — typically a few **milliseconds**.

### Module layout

| File              | Responsibility                                              |
|-------------------|-------------------------------------------------------------|
| `engine.py`       | All face-recognition calls (swap here for InsightFace later)|
| `cache.py`        | Persistent embedding cache (pickle now; SQLite in Stage 5)  |
| `phototrace.py`   | CLI: `index` and `search` commands                          |
| `gui.py`          | PyQt6 desktop UI (Stage 3+)                                  |
| `stage1_match.py` | Stage 1 single-file proof (kept for reference)              |

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
