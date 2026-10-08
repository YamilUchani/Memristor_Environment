"""
neurolab/crossbar/nodal_solver.py
=================================
Resolvedor Nodal de Kirchhoff (MNA - Modified Nodal Analysis)
para Crossbar Memristivo N×M.

Modela exactamente la red de resistencias de wordlines (filas)
y bitlines (columnas) acopladas mediante las conductancias G[i,j].
"""

import numpy as np


import scipy.sparse as sp
import scipy.sparse.linalg as spla

def solve_crossbar_nodal(
    G: np.ndarray,
    V_rows: np.ndarray,
    R_wire: float = 1e-3,
    R_sense: float = 1e-12,
    R_driver: float = 1e-12,
) -> np.ndarray:
    G = np.asarray(G, dtype=float)
    V_rows = np.asarray(V_rows, dtype=float)
    N, M = G.shape

    if N == 0 or M == 0:
        return np.zeros(M)

    if R_wire < 1e-11 and R_sense < 1e-11 and R_driver < 1e-11:
        return G.T @ V_rows

    g_wire = 1.0 / max(R_wire, 1e-12)
    g_sense = 1.0 / max(R_sense, 1e-12)
    g_drv = 1.0 / max(R_driver, 1e-12)

    n_nodes = 2 * N * M

    def idx_R(i: int, j: int) -> int:
        return i * M + j

    def idx_C(i: int, j: int) -> int:
        return N * M + i * M + j

    # Construir sistema sparse (Listas COO)
    row_idx = []
    col_idx = []
    data_val = []
    
    def add_val(r, c, val):
        row_idx.append(r)
        col_idx.append(c)
        data_val.append(val)

    b = np.zeros(n_nodes)

    for i in range(N):
        for j in range(M):
            k_r = idx_R(i, j)
            k_c = idx_C(i, j)
            g_cell = float(G[i, j])

            add_val(k_r, k_r, g_cell)
            add_val(k_r, k_c, -g_cell)

            if j > 0:
                add_val(k_r, k_r, g_wire)
                add_val(k_r, idx_R(i, j - 1), -g_wire)
            else:
                add_val(k_r, k_r, g_drv)
                b[k_r] += g_drv * V_rows[i]

            if j < M - 1:
                add_val(k_r, k_r, g_wire)
                add_val(k_r, idx_R(i, j + 1), -g_wire)

            add_val(k_c, k_c, g_cell)
            add_val(k_c, k_r, -g_cell)

            if i > 0:
                add_val(k_c, k_c, g_wire)
                add_val(k_c, idx_C(i - 1, j), -g_wire)

            if i < N - 1:
                add_val(k_c, k_c, g_wire)
                add_val(k_c, idx_C(i + 1, j), -g_wire)
            else:
                add_val(k_c, k_c, g_sense)

    A_sparse = sp.coo_matrix((data_val, (row_idx, col_idx)), shape=(n_nodes, n_nodes)).tocsc()

    try:
        V_nodes = spla.spsolve(A_sparse, b)
    except Exception:
        # Fallback a densa si falla el sparse
        V_nodes = np.linalg.lstsq(A_sparse.toarray(), b, rcond=None)[0]

    I_out = np.zeros(M)
    for j in range(M):
        V_bottom = V_nodes[idx_C(N - 1, j)]
        I_out[j] = g_sense * V_bottom

    return I_out
