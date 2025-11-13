import numpy as np
from hnswindex import Index

DIM = 128


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
    result_ids = index.knn_query(vectors, 1)[0][:, 0]
    recall = (ids == result_ids).sum() / len(ids)

    assert recall < 0.6


def test_disabled_removals():
    vectors = np.random.rand(2_000, DIM)
    index = Index(DIM)
    index.set_allow_removals(False)

    ids = index.add(vectors)
    result_ids = index.knn_query(vectors, 1)[0][:, 0]
    recall = (ids == result_ids).sum() / len(ids)

    assert recall > 0.85


def test_random_seed():
    vectors = np.random.rand(2_000, DIM)
    index_one = Index(DIM)
    index_one.set_random_seed(1337)

    # Add "single threaded"
    ids = []
    for v in vectors:
        ids.append(index_one.add([v])[0])
    result_ids = index_one.knn_query(vectors, 1)[0][:, 0]
    recall_one = (ids == result_ids).sum() / len(ids)

    index_two = Index(DIM)
    index_two.set_random_seed(1337)

    ids = []
    for v in vectors:
        ids.append(index_two.add([v])[0])
    result_ids = index_two.knn_query(vectors, 1)[0][:, 0]
    recall_two = (ids == result_ids).sum() / len(ids)

    assert recall_one == recall_two
