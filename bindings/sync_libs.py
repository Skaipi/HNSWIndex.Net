from pathlib import Path
import shutil, sys

ROOT = Path(__file__).resolve().parent
art = ROOT / "artifacts" / "native"
pkg = ROOT / "python" / "hnsw" / "_libs"
pkg.mkdir(parents=True, exist_ok=True)

architectures = [
    ("win-x64", "HNSW.Native.dll"),
    ("win-arm64", "HNSW.Native.dll"),
    ("linux-x64", "libHNSW.Native.so"),
    ("linux-arm64", "libHNSW.Native.so"),
    ("osx-x64", "libHNSW.Native.dylib"),
    ("osx-arm64", "libHNSW.Native.dylib"),
]

missing = []
for rid, fname in architectures:
    src = art / rid / fname
    dst_dir = pkg / rid
    dst = dst_dir / fname
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        missing.append(str(src))

if missing:
    print(
        "NOTE: missing some builds (ok if you didn't target them):",
        *missing,
        sep="\n- ",
    )
