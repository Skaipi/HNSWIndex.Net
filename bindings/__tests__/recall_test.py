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


def test_removal():
    vectors = np.random.rand(2_000, DIM)
    index = Index(DIM)

    ids = index.add(vectors)
    result_ids = index.knn_query(vectors, 1)[0][:, 0]
    add_recall = (ids == result_ids).sum() / len(ids)

    remove_ids = ids[:1000]
    remain_ids = ids[1000:]
    remain_vectors = vectors[1000:]

    index.remove(remove_ids)
    result_ids = index.knn_query(remain_vectors, 1)[0][:, 0]
    remove_recall = (remain_ids == result_ids).sum() / len(remain_ids)

    assert remove_recall > add_recall


def test_resize():
    vectors = np.random.rand(2_000, DIM)
    index = Index(DIM)
    index.set_collection_size(100)

    ids = index.add(vectors)
    result_ids = index.knn_query(vectors, 1)[0][:, 0]
    recall = (ids == result_ids).sum() / len(ids)

    assert recall > 0.85
