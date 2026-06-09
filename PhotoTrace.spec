# PyInstaller build recipe for the PhotoTrace desktop app (face_recognition engine).
#
# Produces a single windowed PhotoTrace.exe under dist/.
# Build with:   python -m PyInstaller PhotoTrace.spec
#
# The non-obvious parts PyInstaller can't auto-detect:
#   * face_recognition_models ships its dlib .dat model files as package data,
#     loaded at runtime via pkg_resources -> collect_data_files + pkg_resources.
#   * Pillow's AVIF/WebP support lives in compiled plugins -> collect PIL binaries
#     and submodules so .avif/.webp thumbnails still decode in the frozen app.

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules,
)

datas = collect_data_files("face_recognition_models")
binaries = collect_dynamic_libs("PIL") + collect_dynamic_libs("dlib")
hiddenimports = ["pkg_resources"] + collect_submodules("PIL")

a = Analysis(
    ["gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PhotoTrace",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,          # windowed app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
