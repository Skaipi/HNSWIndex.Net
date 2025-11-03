import numpy as np
from hnswindex import Index

DIM = 128


def test_default_recall():
    vectors = np.random.rand(2_000, DIM)
    index = Index(DIM)

    ids = index.add(vectors)
    result_ids = index.knn_query(vectors, 1)[0][:, 0]
    recall = (ids == result_ids).sum() / len(ids)

    assert recall > 0.85


def test_low_connectivity():
    vectors = np.random.rand(2_000, DIM)
    index = Index(DIM)
    index.set_max_edges(1)

    ids = index.add(vectors)
    result_ids = index.knn_query(vectors, 1)[0][:, 0]
    recall = (ids == result_ids).sum() / len(ids)

    assert recall < 0.1


def test_low_candidates_set():
    vectors = np.random.rand(2_000, DIM)
    index = Index(DIM)
    index.set_max_candidates(1)

    ids = index.add(vectors)
    result = index.knn_query(vectors, 1)
    result_ids = index.knn_query(vectors, 1)[0][:, 0]
    recall = (ids == result_ids).sum() / len(ids)

    assert recall < 0.5


def test_resize():
    vectors = np.random.rand(2_000, DIM)
    index = Index(DIM)
    index.set_collection_size(100)

    ids = index.add(vectors)
    result_ids = index.knn_query(vectors, 1)[0][:, 0]
    recall = (ids == result_ids).sum() / len(ids)

    assert recall > 0.85


def test_min_nn():
    vectors = np.random.rand(2_000, DIM)
    index = Index(DIM)

    ids = index.add(vectors)
    result_ids = index.knn_query(vectors, 1)[0][:, 0]
    default_recall = (ids == result_ids).sum() / len(ids)

    index = Index(DIM)
    index.set_min_nn(1)
    ids = index.add(vectors)
    result_ids = index.knn_query(vectors, 1)[0][:, 0]
    recall = (ids == result_ids).sum() / len(ids)

    assert recall < default_recall
