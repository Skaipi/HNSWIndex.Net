dotnet publish bindings/HNSWIndex.Native -c Release -r $(python - <<'PY'
import platform, sys
sysname=platform.system(); arch=platform.machine().lower()
print({"Windows":"win-x64","Linux":"linux-x64","Darwin":"osx-arm64" if arch in ("arm64","aarch64") else "osx-x64"}[sysname])
PY
)
# python scripts/sync_libs.py
# python scripts/prune_libs.py <your-rid>
python -m build
pip install --force-reinstall dist/hnswindex-*.whl
python - <<'PY'
from hnswindex import Index
import numpy as np
idx = Index(128)
vid = idx.add(np.array(np.ones(128, dtype=np.float32)))
print("ids", vid, "ok")
PY