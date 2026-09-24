"""
neurolab/crossbar/line_resistance.py
=====================================
Modelo de resistencias de línea (H y V).

Las interconexiones metálicas tienen resistencia finita que
produce caídas de voltaje a lo largo de filas y columnas.

Modelo:
    V_ij_efectivo = V_i - I_ij · R_line
"""

import numpy as np


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
    
    def __init__(self, R_H=0.0, R_V=0.0, max_iter=10, tol=1e-6):
        self.R_H = R_H
        self.R_V = R_V
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, G_matrix, V_rows):
        """
        Resuelve el sistema con resistencias de línea.
        
        Usa iteración de punto fijo:
        1. Calcular corrientes con V ideal
        2. Actualizar V con caída de línea
        3. Repetir hasta convergencia
        
        Parameters
        ----------
        G_matrix : ndarray (n_rows, n_cols)
        V_rows : ndarray (n_rows,)
        
        Returns
        -------
        I_cols : ndarray (n_cols,)
            Corrientes con caída de línea.
        V_eff : ndarray (n_rows, n_cols)
            Voltajes efectivos en cada celda.
        """
        n_rows, n_cols = G_matrix.shape
        V = np.asarray(V_rows, dtype=float)
        
        # Inicializar
        I_cols = G_matrix.T @ V
        V_eff = np.tile(V[:, None], (1, n_cols))
        
        for iteration in range(self.max_iter):
            # Calcular caída de voltaje
            V_new = np.zeros((n_rows, n_cols))
            
            for i in range(n_rows):
                for j in range(n_cols):
                    # Caída horizontal: acumulada desde col 0
                    drop_H = sum(
                        I_cols[j] * self.R_H * k
                        for k in range(j + 1)
                    )
                    # Caída vertical: acumulada desde fila 0
                    drop_V = sum(
                        I_cols[j] * self.R_V * k
                        for k in range(i + 1)
                    )
                    V_new[i, j] = V[i] - drop_H - drop_V
            
            # Recalcular corrientes
            I_new = np.zeros(n_cols)
            for j in range(n_cols):
                I_new[j] = np.sum(G_matrix[:, j] * V_new[:, j])
            
            # Verificar convergencia
            if np.allclose(I_new, I_cols, rtol=self.tol):
                break
            
            I_cols = I_new
            V_eff = V_new
        
        return I_cols, V_eff
    
    def compute_error(self, G_matrix, V_rows):
        """
        Calcula error relativo por resistencias de línea.
        
        Returns
        -------
        errors : ndarray (n_cols,)
        """
        I_ideal = G_matrix.T @ V_rows
        I_real, _ = self.solve(G_matrix, V_rows)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.abs(I_real - I_ideal) / np.abs(I_ideal) * 100
        return np.nan_to_num(error)
