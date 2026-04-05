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

lib.hnsw_set_remove_max_candidates.restype = ct.c_int
lib.hnsw_set_remove_max_candidates.argtypes = [ct.c_int]

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
    Python binding for a native HNSW index.

    This class wraps the native ``HNSWIndex.Native`` library and exposes
    batch insertion, removal, k-nearest-neighbor search, and radius search
    for ``float32`` vectors of fixed dimensionality.

    Parameters
    ----------
    dim : int
        Dimensionality of all vectors stored in the index.
    metric : {"sq_euclid", "cosine", "ucosine"}, default="sq_euclid"
        Distance metric used by the native index.

    Notes
    -----
    The native index is created lazily on first insertion. Configuration
    setters must therefore be called before the first operation that
    initializes the underlying native structure.

    Examples
    --------
    >>> index = Index(dim=128, metric="sq_euclid")
    >>> index.set_collection_size(2000)
    >>> x = np.random.rand(2000, 128).astype(np.float32)
    >>> ids = index.add(x)
    >>> nn_ids, nn_dists = index.knn_query(x[:10], k=5)
    """

    def __init__(self, dim: int, metric="sq_euclid"):
        """
        Create an uninitialized index wrapper.

        Parameters
        ----------
        dim : int
            Dimensionality of vectors accepted by this index.
        metric : {"sq_euclid", "cosine", "ucosine"}, default="sq_euclid"
            Distance metric used for indexing and querying.
        """
        self.dim = dim
        self.metric = metric
        self._initialized = False
        self._h = None

    def __del__(self):
        if self._h:
            lib.hnsw_free(self._h)
            self._h = None

    def __initialize(self):
        h = lib.hnsw_create(self.metric.encode("utf-8"))
        if not h:
            raise RuntimeError("hnsw_create failed: " + _last_error())
        self._h = h
        self._initialized = True

    def set_collection_size(self, init_size: int):
        """
        Set the expected number of elements in the index.

        Providing this estimate allows the native implementation to allocate
        internal storage more efficiently before construction begins.

        Parameters
        ----------
        init_size : int
            Expected number of vectors to be inserted.

        Raises
        ------
        RuntimeError
            If the native library rejects the value or the index has already
            been initialized.

        Notes
        -----
        This setter must be called before the native index is initialized.
        """
        status = lib.hnsw_set_collection_size(init_size)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_max_edges(self, max_conn: int):
        """
        Set the maximum number of outgoing connections per node.

        Larger values generally improve recall at the cost of higher memory
        use and slower construction.

        Parameters
        ----------
        max_conn : int
            Maximum number of graph neighbors maintained for each node.

        Raises
        ------
        RuntimeError
            If the native library rejects the value or the index has already
            been initialized.

        Notes
        -----
        This setter must be called before the native index is initialized.
        """
        status = lib.hnsw_set_max_edges(max_conn)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_max_candidates(self, max_candidates: int):
        """
        Set the candidate-list size used during graph construction.

        Larger values typically improve index quality but increase build time.

        Parameters
        ----------
        max_candidates : int
            Number of candidate neighbors examined while inserting elements.

        Raises
        ------
        RuntimeError
            If the native library rejects the value or the index has already
            been initialized.

        Notes
        -----
        This setter must be called before the native index is initialized.
        """
        status = lib.hnsw_set_max_candidates(max_candidates)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_remove_max_candidates(self, rem_max_candidates: int):
        """
        Set the candidate-list size used when repairing the graph after removal.

        Parameters
        ----------
        rem_max_candidates : int
            Number of candidates considered when reconnecting the graph after
            deleting nodes.

        Raises
        ------
        RuntimeError
            If the native library rejects the value or the index has already
            been initialized.

        Notes
        -----
        This setter must be called before the native index is initialized.
        """
        status = lib.hnsw_set_remove_max_candidates(rem_max_candidates)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_distribution_rate(self, dist_rate: float):
        """
        Set the layer-promotion rate used by the HNSW hierarchy.

        This value controls how aggressively elements are promoted to upper
        layers of the graph.

        Parameters
        ----------
        dist_rate : float
            Distribution parameter for level assignment in the native index.

        Raises
        ------
        RuntimeError
            If the native library rejects the value or the index has already
            been initialized.

        Notes
        -----
        This setter must be called before the native index is initialized.
        """
        status = lib.hnsw_set_distribution_rate(dist_rate)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_random_seed(self, random_seed: int):
        """
        Set the random seed used by the native implementation.

        Parameters
        ----------
        random_seed : int
            Seed for randomized parts of index construction.

        Raises
        ------
        RuntimeError
            If the native library rejects the value or the index has already
            been initialized.

        Notes
        -----
        This setter must be called before the native index is initialized.
        """
        status = lib.hnsw_set_random_seed(random_seed)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_min_nn(self, min_nn: int):
        """
        Set the minimum internal neighbor count used by the native search code.

        Parameters
        ----------
        min_nn : int
            Minimum number of neighbors the native implementation considers
            internally during querying.

        Raises
        ------
        RuntimeError
            If the native library rejects the value or the index has already
            been initialized.

        Notes
        -----
        This setter must be called before the native index is initialized.
        """
        status = lib.hnsw_set_min_nn(min_nn)
        if status < 0:
            raise RuntimeError(_last_error())

    def set_allow_removals(self, allow_removals: bool):
        """
        Enable or disable support for removing indexed elements.

        Disabling removals may reduce memory use and improve construction
        performance if deletions are never needed.

        Parameters
        ----------
        allowRemovals : bool
            Whether the index should support element removal.

        Raises
        ------
        RuntimeError
            If the native library rejects the value or the index has already
            been initialized.

        Notes
        -----
        This setter must be called before the native index is initialized.
        """
        status = lib.hnsw_set_allow_removals(allow_removals)
        if status < 0:
            raise RuntimeError(_last_error())

    def add(self, vecs: npt.ArrayLike) -> npt.NDArray[np.int32]:
        """
        Add an array of vectors to the index.

        Parameters
        ----------
        vecs : array-like of shape (n_vectors, dim) or (dim,)
            Input vectors. Values are converted to contiguous ``float32``.

        Returns
        -------
        ndarray of shape (n_vectors,), dtype=int32
            Native integer identifiers assigned to the inserted vectors.

        Raises
        ------
        ValueError
            If the input cannot be interpreted as a 2D array of shape
            ``(n_vectors, self.dim)``.
        RuntimeError
            If insertion fails in the native library.

        Notes
        -----
        Calling this method initializes the underlying native index if it has
        not been created yet.
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
        Remove an array of elements by identifier.

        Parameters
        ----------
        ids : array-like of int
            Integer identifiers previously returned by ``add``.

        Raises
        ------
        RuntimeError
            If removal fails in the native library.

        Notes
        -----
        An empty input is ignored.

        This method assumes the native index has already been initialized.
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
        Perform batched k-nearest-neighbor search.

        Parameters
        ----------
        queries : array-like of shape (n_queries, dim) or (dim,)
            Query vectors. Values are converted to contiguous ``float32``.
        k : int
            Number of nearest neighbors to return for each query.

        Returns
        -------
        ids : ndarray of shape (n_queries, k), dtype=int32
            Neighbor identifiers for each query.
        dists : ndarray of shape (n_queries, k), dtype=float32
            Distances corresponding to ``ids``.

        Raises
        ------
        ValueError
            If the query array does not have shape ``(n_queries, self.dim)``.
        RuntimeError
            If the query fails in the native library.

        Notes
        -----
        This method assumes the native index has already been initialized and
        populated with data.
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
        Perform batched radius search.

        Parameters
        ----------
        queries : array-like of shape (n_queries, dim) or (dim,)
            Query vectors. Values are converted to contiguous ``float32``.
        query_range : float
            Maximum distance threshold. All indexed elements within this radius
            are returned for each query.

        Returns
        -------
        ids : list of ndarray
            ``ids[i]`` contains the identifiers of all neighbors within
            ``query_range`` for query ``i``.
        dists : list of ndarray
            ``dists[i]`` contains the corresponding distances for query ``i``.

        Raises
        ------
        ValueError
            If the query array does not have shape ``(n_queries, self.dim)``.
        RuntimeError
            If the query fails in the native library.

        Notes
        -----
        Result counts may differ between queries, so results are returned as
        per-query arrays rather than a single 2D array.

        This method assumes the native index has already been initialized and
        populated with data.
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
