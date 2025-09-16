import ctypes as ct
import platform
import sys
from importlib.resources import files, as_file
from pathlib import Path

import numpy as np


def get_runtime_id():
    sysname = platform.system()
    arch = platform.machine().lower()
    if sysname == "Windows":
        return "win-arm64" if "arm" in arch else "win-x64"
    if sysname == "Linux":
        return "linux-arm64" if arch in ("aarch64", "arm64") else "linux-x64"
    if sysname == "Darwin":
        return "osx-arm64" if arch in ("arm64", "aarch64") else "osx-x64"
    raise RuntimeError(f"Unsupported platform: {sysname} {arch}")


def get_lib_filename():
    if sys.platform.startswith("win"):
        return "HNSWIndex.Native.dll"
    if sys.platform == "darwin":
        return "HNSWIndex.Native.dylib"
    return "HNSWIndex.Native.so"


def load_lib():
    rid = get_runtime_id()
    _base_path = Path(__file__).resolve().parent
    _lib_path = _base_path / "artifacts" / "native" / rid / get_lib_filename()
    if not _lib_path.exists():
        raise FileNotFoundError(f"Native library missing {_lib_path}")
    return ct.CDLL(str(_lib_path))


# Application Binary Interface
lib = load_lib()
lib.hnsw_create.restype = ct.c_void_p
lib.hnsw_create.argtypes = [ct.c_int]

lib.hnsw_free.restype = None
lib.hnsw_free.argtypes = [ct.c_void_p]

lib.hnsw_add.restype = ct.c_int
lib.hnsw_add.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.POINTER(ct.c_float)),
    ct.c_int,
    ct.c_int,
    ct.POINTER(ct.c_int),
]

lib.hnsw_remove.restype = None
lib.hnsw_remove.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int), ct.c_int]

lib.hnsw_knn_query.restype = ct.c_int
lib.hnsw_knn_query.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_float),
    ct.c_int,
    ct.c_int,
    ct.POINTER(ct.c_int),
    ct.POINTER(ct.c_float),
]

lib.hnsw_serialize.restype = ct.c_int
lib.hnsw_serialize.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int]

lib.hnsw_deserialize.restype = ct.c_void_p
lib.hnsw_deserialize.argtypes = [ct.c_char_p, ct.c_int]

lib.hnsw_get_last_error_utf8.restype = ct.c_int
lib.hnsw_get_last_error_utf8.argtypes = [ct.c_void_p, ct.c_int]


def _last_error():
    n = lib.hnsw_get_last_error_utf8(None, 0)
    if n <= 0:
        return ""
    buf = ct.create_string_buffer(n + 1)
    lib.hnsw_get_last_error_utf8(buf, len(buf))
    return buf.value.decode("utf-8")


def _as_f32(x):
    a = np.asarray(x, dtype=np.float32)
    return a if a.flags["C_CONTIGUOUS"] else np.ascontiguousarray(a)


def _as_2d_f32(x, dim_expected=None):
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError("expected a 2D array of shape (n, dim) or a 1D vector")
    if dim_expected is not None and a.shape[1] != dim_expected:
        raise ValueError(f"expected dim={dim_expected}, got {a.shape[1]}")
    return a if a.flags["C_CONTIGUOUS"] else np.ascontiguousarray(a)


def _row_ptrs_f32(a2d: np.ndarray):
    n, _ = a2d.shape
    T = ct.POINTER(ct.c_float) * n
    return T(*(a2d[i].ctypes.data_as(ct.POINTER(ct.c_float)) for i in range(n)))


def _row_ptrs_i32(a2d: np.ndarray):
    n, _ = a2d.shape
    T = ct.POINTER(ct.c_int) * n
    return T(*(a2d[i].ctypes.data_as(ct.POINTER(ct.c_int)) for i in range(n)))


class Index:
    def __init__(self, dim: int):
        h = lib.hnsw_create(dim)
        if not h:
            raise RuntimeError("hnsw_create failed: " + _last_error())
        self._h = h
        self.dim = dim

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            lib.hnsw_free(h)
            self._h = None

    def set_collection_size(self, init_size):
        raise NotImplementedError()

    def set_max_edges(self, max_conn):
        raise NotImplementedError()

    def set_max_candidates(self, max_candidates):
        raise NotImplementedError()

    def set_distribution_rate(self, dist_rate):
        raise NotImplementedError()

    def set_random_seed(self, random_seed):
        raise NotImplementedError()

    def set_min_nn(self, min_nn):
        raise NotImplementedError()

    def set_zero_layer_base(self, zero_layer_base):
        raise NotImplementedError()

    # batch add float32 vectors
    def add(self, vecs) -> np.ndarray:
        a = _as_2d_f32(vecs, self.dim)
        n, d = a.shape
        ptrs = _row_ptrs_f32(a)
        out_ids = np.empty(n, dtype=np.int32)
        rc = lib.hnsw_add(
            self._h,
            ptrs,
            int(n),
            int(d),
            out_ids.ctypes.data_as(ct.POINTER(ct.c_int)),
        )
        if rc < 0:
            raise RuntimeError(_last_error())
        return out_ids[:rc].copy()

    # Batch remove by ids array
    def remove(self, ids) -> None:
        arr = np.asarray(ids, dtype=np.int32).ravel()
        if arr.size == 0:
            return
        lib.hnsw_remove(
            self._h,
            arr.ctypes.data_as(ct.POINTER(ct.c_int)),
            int(arr.size),
        )
        err = _last_error()
        if err:
            raise RuntimeError(err)

    def knn(self, query, k: int):
        q = _as_f32(query)
        if q.ndim != 1 or q.size != self.dim:
            raise ValueError(f"expected 1D vector of length {self.dim}")
        ids = np.empty(k, dtype=np.int32)
        dists = np.empty(k, dtype=np.float32)
        n = lib.hnsw_knn_query(
            self._h,
            q.ctypes.data_as(ct.POINTER(ct.c_float)),
            q.size,
            k,
            ids.ctypes.data_as(ct.POINTER(ct.c_int)),
            dists.ctypes.data_as(ct.POINTER(ct.c_float)),
        )
        if n < 0:
            raise RuntimeError(_last_error())
        return ids[:n].copy(), dists[:n].copy()
