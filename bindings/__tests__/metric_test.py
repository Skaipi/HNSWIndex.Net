import numpy as np
from hnswindex import Index

DIM = 128


def squared_euclidean(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    diff = x - y
    return float(np.dot(diff, diff))


def cosine_distance(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0.0 or ny == 0.0:
        raise ValueError("Cosine distance is undefined for zero vectors.")
    cos_sim = float(np.dot(x, y) / (nx * ny))
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    return 1.0 - cos_sim


def cosine_distance_unit(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    dot = float(np.dot(x, y))
    dot = float(np.clip(dot, -1.0, 1.0))
    return 1.0 - dot


def test_squared_euclidean_metric():
    vectors = np.random.rand(100, DIM)
    index = Index(DIM, metric="sq_euclid")

    vectors_map = {}
    ids = index.add(vectors)
    for i, _id in enumerate(ids):
        vectors_map[_id] = vectors[i]
    res_ids, dists = index.knn_query(vectors, k=2)

    # compare distances to second closest
    local_dists = np.zeros(100)
    for i, res_id in enumerate(res_ids):
        vector_id = res_id[1]
        vector = vectors_map[vector_id]
        local_dists[i] = squared_euclidean(vectors[i], vector)

    mask = np.isclose(dists[:, 1], local_dists, rtol=0.0, atol=10e-6)
    assert np.all(mask)


def test_cosine_metric():
    vectors = np.random.rand(100, DIM)
    index = Index(DIM, metric="cosine")

    vectors_map = {}
    ids = index.add(vectors)
    for i, _id in enumerate(ids):
        vectors_map[_id] = vectors[i]
    res_ids, dists = index.knn_query(vectors, k=2)

    # compare distances to second closest
    local_dists = np.zeros(100)
    for i, res_id in enumerate(res_ids):
        vector_id = res_id[1]
        vector = vectors_map[vector_id]
        local_dists[i] = cosine_distance(vectors[i], vector)

    mask = np.isclose(dists[:, 1], local_dists, rtol=0.0, atol=10e-6)
    assert np.all(mask)


def test_unit_cosine_metric():
    vectors = np.random.rand(100, DIM)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-12, None)
    index = Index(DIM, metric="ucosine")

    vectors_map = {}
    ids = index.add(vectors)
    for i, _id in enumerate(ids):
        vectors_map[_id] = vectors[i]
    res_ids, dists = index.knn_query(vectors, k=2)

    # compare distances to second closest
    local_dists = np.zeros(100)
    for i, res_id in enumerate(res_ids):
        vector_id = res_id[1]
        vector = vectors_map[vector_id]
        local_dists[i] = cosine_distance(vectors[i], vector)

    mask = np.isclose(dists[:, 1], local_dists, rtol=0.0, atol=10e-6)
    assert np.all(mask)
