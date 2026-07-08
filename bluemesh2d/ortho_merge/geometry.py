"""
Small, dependency-free geometric/topological helpers shared across the
``ortho_merge`` modules.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def build_edges_from_tria(tria: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (edge_nodes, edge_faces) from 0-based triangles.

    Edges are numbered by first occurrence in face-then-edge traversal order,
    each edge as (min_node, max_node); an edge's two faces are in traversal
    order (right face = -1 for boundary), with any further faces of a
    non-manifold edge ignored.

    Parameters
    ----------
    tria : (T,3) int array, 0-based.

    Returns
    -------
    edge_nodes : (E,2) int64
    edge_faces : (E,2) int64 (right face = -1 for boundary)
    """
    tria = np.asarray(tria, dtype=np.int64)
    if tria.ndim != 2 or tria.shape[1] != 3:
        raise ValueError("tria must be an array of shape (T,3) with 0-based indices")

    n_faces = tria.shape[0]
    if n_faces == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    # Half-edges in traversal order: (a,b), (b,c), (c,a) per face.
    he_a = tria[:, [0, 1, 2]].ravel()
    he_b = tria[:, [1, 2, 0]].ravel()
    he_f = np.repeat(np.arange(n_faces, dtype=np.int64), 3)
    lo = np.minimum(he_a, he_b)
    hi = np.maximum(he_a, he_b)
    key = lo * np.int64(max(int(tria.max(initial=-1)) + 2, 1)) + hi

    _, first_pos, inverse = np.unique(key, return_index=True, return_inverse=True)
    insertion = np.argsort(first_pos, kind="stable")
    rank = np.empty(insertion.size, dtype=np.int64)
    rank[insertion] = np.arange(insertion.size)
    edge_id = rank[inverse]

    edge_nodes = np.column_stack([lo[first_pos][insertion], hi[first_pos][insertion]])
    edge_faces = np.full((edge_nodes.shape[0], 2), -1, dtype=np.int64)
    pos = np.argsort(edge_id, kind="stable")
    sid = edge_id[pos]
    is_first = np.r_[True, sid[1:] != sid[:-1]]
    starts = np.flatnonzero(is_first)
    edge_faces[sid[starts], 0] = he_f[pos[starts]]
    seconds = starts + 1
    seconds = seconds[seconds < sid.size]
    seconds = seconds[~is_first[seconds]]
    edge_faces[sid[seconds], 1] = he_f[pos[seconds]]

    return edge_nodes, edge_faces
