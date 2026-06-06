# PhotoTrace

Find every photo containing a specific person across a folder — including group
photos — using local, offline face recognition. No network, no cloud.

> **Status: Stage 1** — command-line proof of the matching. No UI, no cache yet.

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

## Usage (Stage 1)

```powershell
python stage1_match.py REFERENCE_IMAGE TARGET_FOLDER [--threshold 0.6]
```

- `REFERENCE_IMAGE` — one clear, front-facing photo of the target person.
- `TARGET_FOLDER` — scanned recursively for `.jpg .jpeg .png .webp`.
- `--threshold` — face *distance* cutoff (lower = stricter). Default `0.6`.

Example:

```powershell
python stage1_match.py me.jpg "C:\Users\binod\Pictures" --threshold 0.55
```

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
