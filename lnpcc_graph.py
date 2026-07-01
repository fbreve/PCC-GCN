# -*- coding: utf-8 -*-
"""
Geração de grafo k-NN guiado por rótulos para LN-PCC.

Convenção:
  slabel: -1 = não rotulado, 0..C-1 = classes.
"""

import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors



def build_knn_graph_with_labels(data, slabel, k_nn=10, disttype="euclidean", n_jobs=None):
    """
    Versão k-NN que prioriza vizinhos com o mesmo rótulo para nós rotulados.

    Parâmetros
    ----------
    data : array-like, shape (n_nodes, n_features)
        Matriz de atributos dos nós.
    slabel : array-like, shape (n_nodes,)
        -1 = não rotulado, 0..C-1 = classes.
    k_nn : int
        Número de vizinhos desejado.
    disttype : str
        Métrica de distância para o k-NN (ex.: "euclidean", "cosine").
    n_jobs : int, optional
        Número de CPUs para Scikit-Learn.
    """
    X = np.asarray(data, dtype=np.float64)
    slabel = np.asarray(slabel, dtype=np.int64)
    n_nodes = X.shape[0]
    k = int(k_nn)

    neib_qt = np.zeros(n_nodes, dtype=np.int64)
    neib_list = np.zeros((n_nodes, k), dtype=np.int64)

    # 1) Nós não rotulados: k-NN padrão
    idx_unl = np.where(slabel == -1)[0]
    if idx_unl.size > 0:
        # Numerical Guard: Ensure features are valid before passing to C++ libraries
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Scikit-learn KNN can crash on Windows if threads are over-subscribed.
        # Ensure we use n_jobs=1 internally if not specified otherwise.
        nj = n_jobs if n_jobs is not None else 1
        
        print(f" [KNN] Fitting NearestNeighbors on {X.shape[0]} nodes (n_jobs={nj})...", flush=True)
        nn_all = NearestNeighbors(
            n_neighbors=k_nn + 1,
            metric='cosine',
            algorithm='auto',
            n_jobs=nj,
        ).fit(X)
        
        print(f" [KNN] Querying neighbors for {idx_unl.shape[0]} unlabeled nodes...", flush=True)
        K_all = nn_all.kneighbors(X[idx_unl, :], return_distance=False)

        K_all = K_all[:, 1:k + 1]  # remove self
        neib_qt[idx_unl] = k
        neib_list[idx_unl, :] = K_all.astype(np.int64)

    # 2) Nós rotulados: prioriza vizinhos com o mesmo rótulo
    labels = np.unique(slabel[slabel >= 0])
    for c in labels:
        idx_c = np.where(slabel == c)[0]
        if idx_c.size == 0:
            continue

        same_inds = np.where(slabel == c)[0]
        other_inds = np.where(slabel != c)[0]

        X_same = X[same_inds, :]
        X_other = X[other_inds, :]

        k_same_max = min(k + 1, X_same.shape[0])
        if X_same.shape[0] > 0:
            nn_same = NearestNeighbors(
                n_neighbors=k_same_max,
                metric=disttype,
                n_jobs=n_jobs,
            ).fit(X_same)
            Idx_same_local = nn_same.kneighbors(X[idx_c, :], return_distance=False)
            Idx_same_global = same_inds[Idx_same_local]
        else:
            Idx_same_global = np.empty((idx_c.size, 0), dtype=np.int64)

        for t, node in enumerate(idx_c):
            row = Idx_same_global[t, :]
            row = row[row != node]
            row = np.unique(row)

            if row.size >= k:
                neigh = row[:k]
            else:
                need = k - row.size
                extra = np.empty(0, dtype=np.int64)
                if X_other.shape[0] > 0 and need > 0:
                    k_other = min(need, X_other.shape[0])
                    nn_other = NearestNeighbors(
                        n_neighbors=k_other,
                        metric=disttype,
                        n_jobs=n_jobs,
                    ).fit(X_other)
                    Idx_other_local = nn_other.kneighbors(
                        X[node, :].reshape(1, -1),
                        return_distance=False,
                    )[0]
                    extra = other_inds[Idx_other_local]
                neigh = np.concatenate([row, extra])

            neib_qt[node] = neigh.size
            neib_list[node, :neigh.size] = neigh.astype(np.int64)

    # 3) Simetriza e remove duplicatas via sparse matrix (vetorizado)
    nnz = int(neib_qt.sum())
    if nnz > 0:
        # Vectorized construction of row/column indices
        # neib_list has shape (n_nodes, k), we only want neib_qt[:i] from row i
        mask = np.arange(neib_list.shape[1]) < neib_qt[:, None]
        rows_cat = np.repeat(np.arange(n_nodes), neib_qt)
        cols_cat = neib_list[mask]
        
        ones = np.ones(len(rows_cat), dtype=np.int8)
        adj = csr_matrix((ones, (rows_cat, cols_cat)), shape=(n_nodes, n_nodes), dtype=np.int8)
        adj = adj + adj.T
        adj.setdiag(0)
        adj.eliminate_zeros()
        adj.data[:] = 1  # binarize
    else:
        from scipy.sparse import csr_matrix as _csr
        adj = _csr((n_nodes, n_nodes), dtype=np.int8)

    neib_qt = np.diff(adj.indptr).astype(np.int64)
    max_deg = int(neib_qt.max()) if neib_qt.max() > 0 else 0

    neib_list = np.full((n_nodes, max(max_deg, 1)), -1, dtype=np.int64)
    # Vectorized fill: use CSR indptr to scatter each row's neighbors at once.
    degrees = (adj.indptr[1:] - adj.indptr[:-1]).astype(np.int64)
    row_idx = np.repeat(np.arange(n_nodes, dtype=np.int64), degrees)
    col_idx = np.concatenate(
        [np.arange(d, dtype=np.int64) for d in degrees]
    ) if n_nodes > 0 else np.empty(0, dtype=np.int64)
    if len(row_idx) > 0:
        neib_list[row_idx, col_idx] = adj.indices.astype(np.int64)

    return neib_list, neib_qt

def build_knn_graph(data, k_nn=10, metric="minkowski", p=2, n_jobs=None):
    """
    Build a symmetric k-NN graph from a feature matrix.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Input feature matrix. Each row is a data item, each column an attribute.
    k_nn : int, optional (default=10)
        Each node is connected to its k nearest neighbors.
    metric : str or callable, optional (default="minkowski")
        Distance metric passed to sklearn.neighbors.NearestNeighbors.
        Common choices:
            - "minkowski" with p=2: Euclidean distance.
            - "minkowski" with p=1: Manhattan distance.
            - Any metric supported by NearestNeighbors.
    p : int, optional (default=2)
        Power parameter for the Minkowski metric. Ignored if metric does not
        use this parameter.

    Returns
    -------
    neib_list : ndarray, shape (n_nodes, max_deg), dtype=int64
        Matrix of neighbors indices for each node. Row i contains the neighbors
        of node i. Columns beyond neib_qt[i] are undefined.
    neib_qt : ndarray, shape (n_nodes,), dtype=int64
        Degree (number of neighbors) of each node.

    Notes
    -----
    - Uses sklearn.neighbors.NearestNeighbors, which chooses the most efficient
      method to find the neighbors (usually not brute force).
    - The graph is symmetrized via a sparse adjacency matrix (vectorized),
      avoiding Python-level loops.
    - Self-loops are never included.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array of shape (n_samples, n_features).")

    n_nodes = data.shape[0]

    if not isinstance(k_nn, int) or k_nn < 1:
        raise ValueError("k_nn must be a positive integer.")

    if k_nn >= n_nodes:
        raise ValueError(
            f"k_nn ({k_nn}) must be smaller than the number of samples ({n_nodes})."
        )

    nbrs = NearestNeighbors(
        n_neighbors=k_nn + 1,  # +1 because the query point itself is returned
        algorithm="auto",
        metric=metric,
        p=p,
        n_jobs=n_jobs,
    ).fit(data)

    # indices[i, 0] is the query node itself; drop it
    indices = nbrs.kneighbors(data, return_distance=False)[:, 1:]  # (n_nodes, k_nn)

    # --- vectorized symmetrization via sparse matrix ---
    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), k_nn)
    cols = indices.ravel().astype(np.int64)

    # Build directed adjacency, then symmetrize: A + A^T > 0
    ones = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((ones, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.int8)
    adj = adj + adj.T                       # symmetric
    adj.setdiag(0)                          # remove self-loops (safety)
    adj.eliminate_zeros()

    # Convert back to dense neighbor lists
    neib_qt = np.diff(adj.indptr).astype(np.int64)   # degree of each node
    max_deg = int(neib_qt.max())

    neib_list = np.full((n_nodes, max_deg), -1, dtype=np.int64)
    # Vectorized fill: use CSR indptr to scatter each row's neighbors at once.
    degrees = (adj.indptr[1:] - adj.indptr[:-1]).astype(np.int64)
    row_idx = np.repeat(np.arange(n_nodes, dtype=np.int64), degrees)
    col_idx = np.concatenate(
        [np.arange(d, dtype=np.int64) for d in degrees]
    ) if n_nodes > 0 else np.empty(0, dtype=np.int64)
    neib_list[row_idx, col_idx] = adj.indices.astype(np.int64)

    return neib_list, neib_qt

def build_graph_from_edge_index(num_nodes, edge_index):
    """
    Build a graph from a pre-defined edge list (edge_index format).

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph.
    edge_index : array-like, shape (2, num_edges)
        Edge list as used by PyTorch Geometric:
        edge_index[0, e] = source node of edge e
        edge_index[1, e] = target node of edge e

    Returns
    -------
    neib_list : ndarray, shape (n_nodes, max_deg), dtype=int64
        Matrix of neighbors indices for each node.
    neib_qt : ndarray, shape (n_nodes,), dtype=int64
        Degree (number of neighbors) of each node.

    Notes
    -----
    - Assumes an undirected graph and adds reciprocal connections
      by construction (i.e., both (u,v) and (v,u)).
    - Symmetrization and deduplication are done via a sparse adjacency
      matrix, avoiding Python-level loops over edges.
    - Self-loops present in edge_index are silently removed.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    if num_nodes == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64)

    src = edge_index[0]
    dst = edge_index[1]

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])

    ones = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((ones, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.int8)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1

    neib_qt = np.diff(adj.indptr).astype(np.int64)
    max_deg = int(neib_qt.max()) if num_nodes > 0 else 0

    neib_list = np.full((num_nodes, max(max_deg, 1)), -1, dtype=np.int64)
    # Vectorized fill: use CSR indptr to scatter each row's neighbors at once.
    degrees2 = (adj.indptr[1:] - adj.indptr[:-1]).astype(np.int64)
    row_idx2 = np.repeat(np.arange(num_nodes, dtype=np.int64), degrees2)
    col_idx2 = np.concatenate(
        [np.arange(d, dtype=np.int64) for d in degrees2]
    ) if num_nodes > 0 else np.empty(0, dtype=np.int64)
    if len(row_idx2) > 0:
        neib_list[row_idx2, col_idx2] = adj.indices.astype(np.int64)

    return neib_list, neib_qt

def build_augmented_graph(
    neib_list_edge, neib_qt_edge,
    neib_list_knn,  neib_qt_knn,
    labels_ref, train_mask,
    strategy: str,          # "s" (same) | "d" (difference) | "p" (pure) | "u" (original only)
):
    """
    Merges the original graph (edge_index) with a feature-based k-NN graph.

    Strategies:
    's' (Same):       Add kNN edge only if both ends have the same label.
    'd' (Difference): Add kNN edge unless both ends have different labels.
    'p' (Pure):       Add all kNN edges (ignoring labels).
    'u' (Unused):     Use ORIGINAL graph only (ignores kNN).
    """
    from scipy.sparse import csr_matrix

    n_nodes = neib_list_edge.shape[0]

    # ── 1. Vectorized construction of edge graph ──────────────────────────────
    if neib_qt_edge.sum() > 0:
        mask_e = np.arange(neib_list_edge.shape[1]) < neib_qt_edge[:, None]
        re = np.repeat(np.arange(n_nodes, dtype=np.int64), neib_qt_edge)
        ce = neib_list_edge[mask_e].astype(np.int64)
    else:
        re = np.empty(0, dtype=np.int64)
        ce = np.empty(0, dtype=np.int64)

    if strategy.lower() == 'u':
        # Strategy 'u' uses ONLY the original graph
        all_r2 = np.concatenate([re, ce])
        all_c2 = np.concatenate([ce, re])
    else:
        # ── 2. Vectorized kNN edge filtering by strategy ──────────────────────
        # Flatten all valid kNN edges (excluding padding -1) in one pass,
        # then apply the strategy mask in bulk via numpy indexing.
        if neib_qt_knn.sum() > 0:
            mask_k = (np.arange(neib_list_knn.shape[1]) < neib_qt_knn[:, None]) \
                   & (neib_list_knn >= 0)
            all_i = np.repeat(np.arange(n_nodes, dtype=np.int64),
                               mask_k.sum(axis=1))
            all_j = neib_list_knn[mask_k].astype(np.int64)

            s = strategy.lower()
            if s == 'p':
                edge_mask = np.ones(len(all_i), dtype=bool)
            elif s == 's':
                both_labeled = train_mask[all_i] & train_mask[all_j]
                edge_mask = both_labeled & (labels_ref[all_i] == labels_ref[all_j])
            elif s == 'd':
                both_labeled = train_mask[all_i] & train_mask[all_j]
                edge_mask = ~(both_labeled & (labels_ref[all_i] != labels_ref[all_j]))
            else:
                edge_mask = np.ones(len(all_i), dtype=bool)  # default: pure

            rk = all_i[edge_mask]
            ck = all_j[edge_mask]

            # Symmetrize: include both directions
            all_r2 = np.concatenate([re, rk, ck, ce])
            all_c2 = np.concatenate([ce, ck, rk, re])
        else:
            all_r2 = np.concatenate([re, ce])
            all_c2 = np.concatenate([ce, re])

    ones = np.ones(len(all_r2), dtype=np.int8)
    adj = csr_matrix((ones, (all_r2, all_c2)), shape=(n_nodes, n_nodes), dtype=np.int8)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1  # binarize

    neib_qt_out = np.diff(adj.indptr).astype(np.int64)
    max_d = int(neib_qt_out.max()) if neib_qt_out.max() > 0 else 1

    nl = np.full((n_nodes, max_d), -1, dtype=np.int64)
    # Vectorized fill: use CSR indptr to scatter each row's neighbors at once.
    degrees3 = (adj.indptr[1:] - adj.indptr[:-1]).astype(np.int64)
    row_idx3 = np.repeat(np.arange(n_nodes, dtype=np.int64), degrees3)
    col_idx3 = np.concatenate(
        [np.arange(d, dtype=np.int64) for d in degrees3]
    ) if n_nodes > 0 else np.empty(0, dtype=np.int64)
    if len(row_idx3) > 0:
        nl[row_idx3, col_idx3] = adj.indices.astype(np.int64)

    return nl, neib_qt_out
