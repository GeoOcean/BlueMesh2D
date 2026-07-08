import numpy as np
from shapely.geometry import LinearRing, Point, Polygon


def read_poly_from_dat(dat_path, delimiter=None):
    """
    Equivalent Python implementation of the MATLAB contour reader:
    Reads polygon(s) from a .dat file where 'NaN NaN' separates contours,
    automatically closes each contour, and builds the global node/edge arrays.

    Parameters
    ----------
    dat_path : str
        Path to the .dat file.
    delimiter : str, optional
        Delimiter for numpy.loadtxt (default: auto-detect).

    Returns
    -------
    node : ndarray of shape (N, 2)
        Node coordinates (x, y)
    edge : ndarray of shape (M, 2)
        Edge connectivity (zero-based indices)
    """

    # --- Load file
    p0 = np.loadtxt(dat_path, delimiter=delimiter)
    if p0.shape[1] < 2:
        raise ValueError("The .dat file must contain at least two columns: x y")

    # --- Find NaN separators
    isnan = np.isnan(p0[:, 0])
    s = np.where(isnan)[0]
    s = np.concatenate(([0], s, [len(p0)]))

    node = []
    edge = []
    cont = 0

    # --- Loop over polygons
    for i in range(len(s) - 1):
        p = p0[s[i] : s[i + 1], :]
        p = p[~np.isnan(p[:, 0])]  # remove NaN rows
        if len(p) == 0:
            continue

        n = len(p)
        # Close the polygon by connecting last point to first
        c = np.column_stack([np.arange(0, n), np.arange(1, n + 1)])
        c[-1, 1] = 0  # last edge closes to first

        # Apply offset to edge indices
        c = c + cont

        # Append
        node.append(p)
        edge.append(c)

        cont += n  # offset for next polygon

    # --- Concatenate all nodes and edges
    node = np.vstack(node)
    edge = np.vstack(edge).astype(int)

    return node, edge


def _split_edges_at_discontinuity(edges):
    """
    Split edges whenever two consecutive rows do NOT share a vertex.
    Works directly on the provided order (no reordering).
    Returns list of (n_i, 2) edge arrays.
    """
    if edges is None or edges.size == 0:
        return []
    edges = np.asarray(edges, dtype=int)
    n = len(edges)
    if n == 1:
        return [edges]
    
    chunks = []
    start = 0
    
    for i in range(1, n):
        a, b = int(edges[i, 0]), int(edges[i, 1])
        a_prev, b_prev = int(edges[i - 1, 0]), int(edges[i - 1, 1])
        # Check if current edge shares a vertex with previous
        shared = (a == a_prev or a == b_prev or b == a_prev or b == b_prev)
        
        if not shared:
            # Discontinuity: cut here
            chunks.append(edges[start:i])
            start = i
    
    # Add the last chunk
    chunks.append(edges[start:])
    return chunks


def _ordered_edges_to_chains(edges):
    """
    First order edges into chains (consecutive share a vertex within each component),
    then split the ordered result at discontinuities.
    """
    if edges is None or edges.size == 0:
        return []
    edges = np.asarray(edges, dtype=int)
    n = len(edges)
    
    # Build adjacency
    adj = {}
    for i in range(n):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        if u not in adj:
            adj[u] = []
        adj[u].append((i, v))
        if v not in adj:
            adj[v] = []
        adj[v].append((i, u))
    
    used = set()
    ordered_chains = []

    def extend_forward(last_v, lst, used_set):
        while True:
            cands = [(ei, other) for ei, other in adj[last_v] if ei not in used_set]
            if not cands:
                break
            ei, other = cands[0]
            lst.append((int(edges[ei, 0]), int(edges[ei, 1])))
            used_set.add(ei)
            last_v = other

    def extend_backward(first_v, lst, used_set):
        while True:
            cands = [(ei, other) for ei, other in adj[first_v] if ei not in used_set]
            if not cands:
                break
            ei, other = cands[0]
            lst.insert(0, (int(edges[ei, 0]), int(edges[ei, 1])))
            used_set.add(ei)
            first_v = other

    # Build chains per connected component
    for start_i in range(n):
        if start_i in used:
            continue
        a0, b0 = int(edges[start_i, 0]), int(edges[start_i, 1])
        chain = [(a0, b0)]
        used.add(start_i)
        extend_forward(b0, chain, used)
        extend_backward(a0, chain, used)
        ordered_chains.append(np.array(chain, dtype=int))

    if not ordered_chains:
        return []
    
    # Concatenate all chains into one ordered array
    flat = np.vstack(ordered_chains)
    
    # Split at discontinuities (consecutive rows without shared vertex)
    return _split_edges_at_discontinuity(flat)


def _chain_edges_to_nodelist(edge_arr):
    """Ordered (m, 2) edges -> 1D array of node indices."""
    if edge_arr is None or len(edge_arr) == 0:
        return np.array([], dtype=int)
    edge_arr = np.asarray(edge_arr, dtype=int)
    out = [int(edge_arr[0, 0]), int(edge_arr[0, 1])]
    for i in range(1, len(edge_arr)):
        a, b = int(edge_arr[i, 0]), int(edge_arr[i, 1])
        if a == out[-1]:
            out.append(b)
        elif b == out[-1]:
            out.append(a)
        elif a == out[0]:
            out.insert(0, b)
        elif b == out[0]:
            out.insert(0, a)
        else:
            out.append(a)
            out.append(b)
    return np.array(out, dtype=int)


def identify_boundary(vert, tria, z, zlim=0.0, Manual_open_boundary=None):
    """
    Identify open and land boundaries from a triangulated mesh.
    Edges are classified by depth vs zlim and optional Manual_open_boundary.
    Contours are ordered and split at discontinuities (consecutive edges
    without a shared vertex). Returns a dict.

    Parameters
    ----------
    vert : (N, 2) array
        Node coordinates (x, y).
    tria : (M, 3) array
        Triangle connectivity (node indices).
    z : (N,) array
        Elevation at nodes.
    zlim : float
        Elevation threshold (zmean > zlim -> open).
    Manual_open_boundary : shapely Polygon, optional
        If provided, edges whose midpoint lies inside are classified as open.

    Returns
    -------
    dict with keys:
        edge_tag : (K, 3) array, (node1, node2, tag), tag=1 open, tag=2 land.
        edge_open : (L, 2) array, flat open boundary edges.
        edge_land : (P, 2) array, flat land boundary edges.
        open_contours : list of 1D arrays of node indices (one per contour).
        land_contours : list of 1D arrays of node indices (one per contour).
    """
    edges = np.vstack([tria[:, [0, 1]], tria[:, [1, 2]], tria[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges_sorted, counts = np.unique(edges, axis=0, return_counts=True)
    edge_free = edges_sorted[counts == 1]
    if edge_free.size == 0:
        return {
            "edge_tag": np.empty((0, 3), dtype=int),
            "edge_open": np.empty((0, 2), dtype=int),
            "edge_land": np.empty((0, 2), dtype=int),
            "open_contours": [],
            "land_contours": [],
        }

    edge_open_list = []
    edge_land_list = []
    for (a, b) in edge_free:
        zmean = 0.5 * (z[a] + z[b])
        mid = (vert[a] + vert[b]) / 2.0
        in_manual = (
            Manual_open_boundary.contains(Point(mid))
            if Manual_open_boundary is not None
            else False
        )
        if zmean > zlim or in_manual:
            edge_open_list.append([a, b])
        else:
            edge_land_list.append([a, b])

    edge_open = (
        np.array(edge_open_list, dtype=int)
        if edge_open_list
        else np.empty((0, 2), dtype=int)
    )
    edge_land = (
        np.array(edge_land_list, dtype=int)
        if edge_land_list
        else np.empty((0, 2), dtype=int)
    )

    open_chains = _ordered_edges_to_chains(edge_open)
    land_chains = _ordered_edges_to_chains(edge_land)

    open_contours = [_chain_edges_to_nodelist(c) for c in open_chains]
    land_contours = [_chain_edges_to_nodelist(c) for c in land_chains]

    edge_open_flat = (
        np.vstack(open_chains) if open_chains else np.empty((0, 2), dtype=int)
    )
    edge_land_flat = (
        np.vstack(land_chains) if land_chains else np.empty((0, 2), dtype=int)
    )

    tag_open = np.ones((edge_open_flat.shape[0], 1), dtype=int)
    tag_land = np.full((edge_land_flat.shape[0], 1), 2, dtype=int)
    edge_tag_parts = []
    if edge_open_flat.shape[0] > 0:
        edge_tag_parts.append(np.hstack([edge_open_flat, tag_open]))
    if edge_land_flat.shape[0] > 0:
        edge_tag_parts.append(np.hstack([edge_land_flat, tag_land]))
    edge_tag = (
        np.vstack(edge_tag_parts) if edge_tag_parts else np.empty((0, 3), dtype=int)
    )

    return {
        "edge_tag": edge_tag,
        "edge_open": edge_open_flat,
        "edge_land": edge_land_flat,
        "open_contours": open_contours,
        "land_contours": land_contours,
    }