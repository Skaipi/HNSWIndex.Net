"""
ctypes bindings for a HNSWIndex.NET.
"""

import ctypes as ct
import platform
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import numpy.typing as npt


def _get_runtime_id():
    sysname = platform.system()
    arch = platform.machine().lower()
    if sysname == "Windows":
        return "win-arm64" if "arm" in arch else "win-x64"
    if sysname == "Linux":
        return "linux-arm64" if arch in ("aarch64", "arm64") else "linux-x64"
    if sysname == "Darwin":
        return "osx-arm64" if arch in ("arm64", "aarch64") else "osx-x64"
    raise RuntimeError(f"Unsupported platform: {sysname} {arch}")


def _get_lib_filename():
    if sys.platform.startswith("win"):
        return "HNSWIndex.Native.dll"
    if sys.platform == "darwin":
        return "HNSWIndex.Native.dylib"
    return "HNSWIndex.Native.so"


def _load_lib():
    rid = _get_runtime_id()
    _base_path = Path(__file__).resolve().parent
    _lib_path = _base_path / "artifacts" / "native" / rid / _get_lib_filename()
    if not _lib_path.exists():
        raise FileNotFoundError(f"Native library missing {_lib_path}")
    return ct.CDLL(str(_lib_path))


# Application Binary Interface
lib = _load_lib()
lib.hnsw_create.restype = ct.c_void_p
lib.hnsw_create.argtypes = [ct.c_char_p]

lib.hnsw_free.restype = None
lib.hnsw_free.argtypes = [ct.c_void_p]

lib.hnsw_add.restype = ct.c_int
lib.hnsw_add.argtypes = [
    ct.c_void_p,  # handle
    ct.POINTER(ct.c_float),  # vectors
    ct.c_int,  # count
    ct.c_int,  # dim
    ct.POINTER(ct.c_int),  # outIds
]

lib.hnsw_remove.restype = ct.c_int
lib.hnsw_remove.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int), ct.c_int]

lib.hnsw_knn_query.restype = ct.c_int
lib.hnsw_knn_query.argtypes = [
    ct.c_void_p,  # handle
    ct.POINTER(ct.c_float),  # vectors
    ct.c_int,  # count
    ct.c_int,  # dim
    ct.c_int,  # k
    ct.POINTER(ct.c_int),  # outIds
    ct.POINTER(ct.c_float),  # outDists
]

lib.hnsw_range_query.restype = ct.c_int
lib.hnsw_range_query.argtypes = [
    ct.c_void_p,  # handle
    ct.POINTER(ct.c_float),  # vectors
    ct.c_int,  # count
    ct.c_int,  # dim
    ct.c_float,  # range
    ct.POINTER(ct.c_void_p),  # outIds
    ct.POINTER(ct.c_void_p),  # outDists
    ct.POINTER(ct.c_int),  # counts
]

lib.hnsw_free_results.restype = None
lib.hnsw_free_results.argtypes = [
    ct.POINTER(ct.c_void_p),
    ct.POINTER(ct.c_void_p),
    ct.c_int,
]

lib.hnsw_set_collection_size.restype = ct.c_int
lib.hnsw_set_collection_size.argtypes = [ct.c_int]

lib.hnsw_set_max_edges.restype = ct.c_int
lib.hnsw_set_max_edges.argtypes = [ct.c_int]

lib.hnsw_set_max_candidates.restype = ct.c_int
lib.hnsw_set_max_candidates.argtypes = [ct.c_int]

lib.hnsw_set_distribution_rate.restype = ct.c_int
lib.hnsw_set_distribution_rate.argtypes = [ct.c_float]

lib.hnsw_set_random_seed.restype = ct.c_int
lib.hnsw_set_random_seed.argtypes = [ct.c_int]

lib.hnsw_set_min_nn.restype = ct.c_int
lib.hnsw_set_min_nn.argtypes = [ct.c_int]

lib.hnsw_set_allow_removals.restype = ct.c_int
lib.hnsw_set_allow_removals.argtypes = [ct.c_bool]

lib.hnsw_get_last_error_utf8.restype = ct.c_int
lib.hnsw_get_last_error_utf8.argtypes = [ct.c_void_p, ct.c_int]


def _last_error():
    n = lib.hnsw_get_last_error_utf8(None, 0)
    if n <= 0:
        return ""
    buf = ct.create_string_buffer(n + 1)
    lib.hnsw_get_last_error_utf8(buf, len(buf))
    return buf.value.decode("utf-8")


def _as_2d_f32(x: npt.ArrayLike, dim_expected=None):
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError("expected a 2D array of shape (n, dim) or a 1D vector")
    if dim_expected is not None and a.shape[1] != dim_expected:
        raise ValueError(f"expected dim={dim_expected}, got {a.shape[1]}")
    return a if a.flags["C_CONTIGUOUS"] else np.ascontiguousarray(a)


class Index:
    """
    HNSW Index class for efficient nearest neighbor querying in high dimensional spaces.
    Supported metrics:
     - "sq_euclid"
     - "cosine"
     - "ucosine"

    Examples
    --------
    >>> # Squared Euclidean distance by default
    >>> index = Index(dim=128, metric="sq_euclid")
    >>> vectors = np.random.rand(2_000, 128)
    >>> index.set_collection_size(2000)
    >>> ids = index.add(vectors)
    >>> results = index.knn_query(vectors, k=1)
    """

    def __init__(self, dim: int, metric="sq_euclid"):
        self.dim = dim
        self.metric = metric
        self._initialized = False
        self._h = None

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            lib.hnsw_free(h)
            self._h = None

    def __initialize(self):
        h = lib.hnsw_create(self.metric.encode("utf-8"))
        if not h:
            raise RuntimeError("hnsw_create failed: " + _last_error())
        self._h = h
        self._initialized = True

    def set_collection_size(self, init_size: int):
        """
        Set expected number of elements in index to improve efficiency.

        All parameter setters will throw if used on initialized index.
        """
        status = lib.hnsw_set_collection_size(init_size)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_max_edges(self, max_conn: int):
        """
        Set maximum number of connections per node. This may improve search quality at the cost of memory usage.

        All parameter setters will throw if used on initialized index.
        """
        status = lib.hnsw_set_max_edges(max_conn)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_max_candidates(self, max_candidates: int):
        """
        Set the size of candidate lists considered during graph construction. This may affect construction time.

        All parameter setters will throw if used on initialized index.
        """
        status = lib.hnsw_set_max_candidates(max_candidates)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_distribution_rate(self, dist_rate: float):
        """
        Set distribution rate for promoting elements to higher layers of the graph.

        All parameter setters will throw if used on initialized index.
        """
        status = lib.hnsw_set_distribution_rate(dist_rate)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_random_seed(self, random_seed: int):
        """
        Set the PRNG seed used by the native algorithm.

        All parameter setters will throw if used on initialized index.
        """
        status = lib.hnsw_set_random_seed(random_seed)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_min_nn(self, min_nn: int):
        """
        Set minimum number of elements retrieved by query.
        If k is less than min_nn parameter then k best elements are returned and remaining elements are discarded.

        All parameter setters will throw if used on initialized index.
        """
        status = lib.hnsw_set_min_nn(min_nn)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_allow_removals(self, allowRemovals: bool):
        """
        Set flag which enables removal from hnsw structure.
        By default set to `True`.
        Setting this to `False` may improve memory footprint and construction time.
        """
        status = lib.hnsw_set_allow_removals(allowRemovals)
        if status < 0:
            raise RuntimeError(_last_error())

    def add(self, vecs: npt.ArrayLike) -> npt.NDArray[np.int32]:
        """
        Batch add vectors to hnsw index.
        Each vector should be represented as list of floating point values
        """
        if not self._initialized:
            self.__initialize()
        a = _as_2d_f32(vecs, self.dim)
        n, d = a.shape
        out_ids = np.empty(n, dtype=np.int32)
        rc = lib.hnsw_add(
            self._h,
            a.ctypes.data_as(ct.POINTER(ct.c_float)),
            int(n),
            int(d),
            out_ids.ctypes.data_as(ct.POINTER(ct.c_int)),
        )
        if rc < 0:
            raise RuntimeError(_last_error())
        return out_ids[:rc].copy()

    def remove(self, ids: npt.ArrayLike) -> None:
        """
        Batch remove elements from hnsw index.
        """
        arr = np.asarray(ids, dtype=np.int32).ravel()
        if arr.size == 0:
            return
        result = lib.hnsw_remove(
            self._h,
            arr.ctypes.data_as(ct.POINTER(ct.c_int)),
            int(arr.size),
        )
        if result < 0:
            raise RuntimeError(_last_error())

    def knn_query(
        self, queries: npt.ArrayLike, k: int
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """
        Perform batch knn query for provided list of query vectors.
        """
        q = _as_2d_f32(queries, self.dim)
        n = int(q.shape[0])
        ids = np.empty((n, k), dtype=np.int32)
        dists = np.empty((n, k), dtype=np.float32)
        status = lib.hnsw_knn_query(
            self._h,
            q.ctypes.data_as(ct.POINTER(ct.c_float)),
            n,
            self.dim,
            k,
            ids.ctypes.data_as(ct.POINTER(ct.c_int)),
            dists.ctypes.data_as(ct.POINTER(ct.c_float)),
        )
        if status < 0:
            raise RuntimeError(_last_error())
        return ids.copy(), dists.copy()

    def range_query(
        self, queries: npt.ArrayLike, query_range: float
    ) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.float32]]]:
        """
        Perform batch range query for provided list of query vectors.
        """
        q = _as_2d_f32(queries, self.dim)
        n = int(q.shape[0])

        ids_pp = (ct.c_void_p * n)()
        dists_pp = (ct.c_void_p * n)()
        counts = (ct.c_int * n)()

        status = lib.hnsw_range_query(
            self._h,
            q.ctypes.data_as(ct.POINTER(ct.c_float)),
            n,
            self.dim,
            query_range,
            ids_pp,
            dists_pp,
            counts,
        )

        if status < 0:
            raise RuntimeError(_last_error())

        ids, dists = [], []
        try:
            for i in range(n):
                m = counts[i]
                if m == 0:
                    ids.append(np.empty(0, dtype=np.int32))
                    dists.append(np.empty(0, dtype=np.float32))
                    continue
                i_ids = ct.cast(ids_pp[i], ct.POINTER(ct.c_int))
                i_dists = ct.cast(dists_pp[i], ct.POINTER(ct.c_float))
                ids.append(np.ctypeslib.as_array(i_ids, shape=(m,)).copy())
                dists.append(np.ctypeslib.as_array(i_dists, shape=(m,)).copy())
        finally:
            # Free allocated results
            lib.hnsw_free_results(ids_pp, dists_pp, n)

        return ids, dists
