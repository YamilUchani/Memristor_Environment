"""
neurolab/crossbar/line_resistance.py
=====================================
Modelo de resistencias de línea (H y V) con conservación de carga KCL.

Las interconexiones metálicas tienen resistencia finita que
produce caídas de voltaje a lo largo de filas y columnas.
"""

import numpy as np
from neurolab.crossbar.nodal_solver import solve_crossbar_nodal


class LineResistanceModel:
    """
    Modelo de resistencias de línea para crossbar.
    
    Parámetros
    ----------
    R_H : float
        Resistencia por segmento horizontal (Ω).
    R_V : float
        Resistencia por segmento vertical (Ω).
    max_iter : int
        Iteraciones para convergencia.
    tol : float
        Tolerancia de convergencia.
    """
    
    def __init__(self, R_H=0.0, R_V=0.0, max_iter=15, tol=1e-6):
        self.R_H = float(R_H)
        self.R_V = float(R_V)
        self.max_iter = max_iter
        self.tol = tol
    
    def _compute_ir_drop(self, V_rows: np.ndarray, G: np.ndarray, I_cell: np.ndarray):
        """
        Calcula la caída acumulada de tensión por tramos usando corrientes de rama reales.
        
        Corriente de tramo horizontal entre col k y k+1 en la fila i:
            I_tramo_H[i,k] = sum_{m=k+1}^{M-1} I_cell[i,m]
        Corriente de tramo vertical entre fila k y k+1 en la columna j:
            I_tramo_V[k,j] = sum_{m=k+1}^{N-1} I_cell[m,j]
        """
        N, M = G.shape
        drop_H = np.zeros((N, M))
        drop_V = np.zeros((N, M))
        
        for i in range(N):
            for j in range(M):
                # Caída acumulada en fila i hasta columna j
                drop_h_sum = 0.0
                for k in range(j):
                    I_tramo_H_k = sum(I_cell[i, m] for m in range(k, M))
                    drop_h_sum += I_tramo_H_k * self.R_H
                
                # Caída acumulada en columna j desde fila i
                drop_v_sum = 0.0
                for k in range(i, N - 1):
                    I_tramo_V_k = sum(I_cell[m, j] for m in range(0, k + 1))
                    drop_v_sum += I_tramo_V_k * self.R_V
                
                drop_H[i, j] = drop_h_sum
                drop_V[i, j] = drop_v_sum
        
        return drop_H, drop_V

    def solve(self, G_matrix: np.ndarray, V_rows: np.ndarray):
        """
        Resuelve el sistema con resistencias de línea usando MNA nodal solver.
        
        Parameters
        ----------
        G_matrix : ndarray (N, M)
        V_rows : ndarray (N,)
        
        Returns
        -------
        I_cols : ndarray (M,)
            Corrientes de salida por columna.
        V_eff : ndarray (N, M)
            Voltajes efectivos en cada celda.
        """
        G = np.asarray(G_matrix, dtype=float)
        V = np.asarray(V_rows, dtype=float)
        N, M = G.shape

        if self.R_H <= 1e-12 and self.R_V <= 1e-12:
            I_cols = G.T @ V
            V_eff = np.tile(V[:, None], (1, M))
            return I_cols, V_eff

        # Usar el resolvedor nodal exacto MNA
        R_wire = max(self.R_H, self.R_V)
        I_cols = solve_crossbar_nodal(G, V, R_wire=R_wire)

        # Calcular voltajes efectivos nodales V_eff
        V_eff = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                drop_h = sum(
                    sum(G[i, m] * V[i] for m in range(k, M)) * self.R_H
                    for k in range(j)
                )
                V_eff[i, j] = max(0.0, V[i] - drop_h)

        return I_cols, V_eff

    def compute_error(self, G_matrix: np.ndarray, V_rows: np.ndarray) -> np.ndarray:
        """Calcula error relativo porcentual por IR drop."""
        I_ideal = G_matrix.T @ V_rows
        I_real, _ = self.solve(G_matrix, V_rows)
        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.abs(I_real - I_ideal) / np.abs(I_ideal + 1e-15) * 100
        return np.nan_to_num(error)
