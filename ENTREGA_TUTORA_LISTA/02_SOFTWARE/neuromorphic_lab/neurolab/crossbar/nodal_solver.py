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

class CrossbarNodalFactorization:
    """Factorized MNA network for repeated reads with unchanged conductances."""

    def __init__(
        self,
        G: np.ndarray,
        R_wire: float = 1e-3,
        R_sense: float = 1e-12,
        R_driver: float = 1e-12,
    ):
        self.G = np.asarray(G, dtype=float).copy()
        self.N, self.M = self.G.shape
        self._ideal = (
            R_wire < 1e-11 and R_sense < 1e-11 and R_driver < 1e-11
        )
        if self.N == 0 or self.M == 0 or self._ideal:
            self._factor = None
            self._dense_matrix = None
            self._g_sense = 0.0
            return

        g_wire = 1.0 / max(R_wire, 1e-12)
        self._g_sense = 1.0 / max(R_sense, 1e-12)
        g_drv = 1.0 / max(R_driver, 1e-12)
        n_nodes = 2 * self.N * self.M
        row_idx = []
        col_idx = []
        data_val = []

        def idx_R(i, j):
            return i * self.M + j

        def idx_C(i, j):
            return self.N * self.M + i * self.M + j

        def add_val(r, c, val):
            row_idx.append(r)
            col_idx.append(c)
            data_val.append(val)

        for i in range(self.N):
            for j in range(self.M):
                k_r = idx_R(i, j)
                k_c = idx_C(i, j)
                g_cell = float(self.G[i, j])

                add_val(k_r, k_r, g_cell)
                add_val(k_r, k_c, -g_cell)

                if j > 0:
                    add_val(k_r, k_r, g_wire)
                    add_val(k_r, idx_R(i, j - 1), -g_wire)
                else:
                    add_val(k_r, k_r, g_drv)

                if j < self.M - 1:
                    add_val(k_r, k_r, g_wire)
                    add_val(k_r, idx_R(i, j + 1), -g_wire)

                add_val(k_c, k_c, g_cell)
                add_val(k_c, k_r, -g_cell)

                if i > 0:
                    add_val(k_c, k_c, g_wire)
                    add_val(k_c, idx_C(i - 1, j), -g_wire)

                if i < self.N - 1:
                    add_val(k_c, k_c, g_wire)
                    add_val(k_c, idx_C(i + 1, j), -g_wire)
                else:
                    add_val(k_c, k_c, self._g_sense)

        matrix = sp.coo_matrix(
            (data_val, (row_idx, col_idx)),
            shape=(n_nodes, n_nodes),
        ).tocsc()
        try:
            self._factor = spla.splu(matrix)
            self._dense_matrix = None
        except RuntimeError:
            self._factor = None
            self._dense_matrix = matrix.toarray()
        self._g_driver = g_drv

    def solve(self, V_rows: np.ndarray) -> np.ndarray:
        V_rows = np.asarray(V_rows, dtype=float)
        if V_rows.shape != (self.N,):
            raise ValueError(
                f"V_rows shape {V_rows.shape} != expected {(self.N,)}"
            )
        if self.N == 0 or self.M == 0:
            return np.zeros(self.M)
        if self._ideal:
            return self.G.T @ V_rows

        b = np.zeros(2 * self.N * self.M)
        b[np.arange(self.N) * self.M] = self._g_driver * V_rows
        if self._factor is not None:
            V_nodes = self._factor.solve(b)
        else:
            V_nodes = np.linalg.lstsq(self._dense_matrix, b, rcond=None)[0]
        bottom_nodes = self.N * self.M + (self.N - 1) * self.M + np.arange(self.M)
        return self._g_sense * V_nodes[bottom_nodes]


def factor_crossbar_nodal(
    G: np.ndarray,
    R_wire: float = 1e-3,
    R_sense: float = 1e-12,
    R_driver: float = 1e-12,
) -> CrossbarNodalFactorization:
    return CrossbarNodalFactorization(
        G=G,
        R_wire=R_wire,
        R_sense=R_sense,
        R_driver=R_driver,
    )


def solve_crossbar_nodal(
    G: np.ndarray,
    V_rows: np.ndarray,
    R_wire: float = 1e-3,
    R_sense: float = 1e-12,
    R_driver: float = 1e-12,
) -> np.ndarray:
    return factor_crossbar_nodal(
        G=G,
        R_wire=R_wire,
        R_sense=R_sense,
        R_driver=R_driver,
    ).solve(V_rows)
