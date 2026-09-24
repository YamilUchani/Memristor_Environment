"""
neurolab/crossbar/sneak_paths.py
=================================
Modelo detallado de sneak paths en crossbar.

Los sneak paths son corrientes parásitas que fluyen por celdas
no seleccionadas, degradando la precisión de lectura.

Modelo físico:
    I_total = I_objetivo + I_sneak
"""

import numpy as np


class SneakPathModel:
    """
    Modelo de sneak paths para crossbar memristivo.
    
    En un crossbar N×M, al leer una columna j con voltaje V_i
    en filas, la corriente parásita fluye por caminos alternativos
    a través de celdas no seleccionadas.
    """
    
    def __init__(self, alpha=0.1):
        """
        Parameters
        ----------
        alpha : float
            Factor de acoplamiento de sneak paths [0, 1].
            - 0: sin sneak paths (ideal)
            - 0.1: sneak paths moderados
            - 0.5: sneak paths severos
        """
        self.alpha = alpha
    
    def compute_sneak_current(self, G_matrix, V_rows, target_col):
        """
        Calcula corriente de sneak path para una columna objetivo.
        
        Parameters
        ----------
        G_matrix : ndarray (n_rows, n_cols)
            Matriz de conductancias.
        V_rows : ndarray (n_rows,)
            Voltajes de fila.
        target_col : int
            Columna objetivo.
        
        Returns
        -------
        I_sneak : float
            Corriente parásita (A).
        """
        n_rows, n_cols = G_matrix.shape
        
        # Corriente que fluye por caminos no deseados
        I_sneak = 0.0
        for i in range(n_rows):
            for j in range(n_cols):
                if j != target_col:
                    I_sneak += self.alpha * G_matrix[i, j] * V_rows[i]
        
        return I_sneak
    
    def compute_all_sneak_currents(self, G_matrix, V_rows):
        """
        Calcula sneak paths para todas las columnas.
        
        Returns
        -------
        I_sneak_all : ndarray (n_cols,)
        """
        n_cols = G_matrix.shape[1]
        I_sneak_all = np.zeros(n_cols)
        for j in range(n_cols):
            I_sneak_all[j] = self.compute_sneak_current(
                G_matrix, V_rows, j
            )
        return I_sneak_all
    
    def read_error(self, G_matrix, V_rows):
        """
        Calcula el error relativo por sneak paths.
        
        Returns
        -------
        errors : ndarray (n_cols,)
            Error relativo porcentual por columna.
        """
        I_ideal = G_matrix.T @ V_rows
        I_sneak = self.compute_all_sneak_currents(G_matrix, V_rows)
        I_total = I_ideal + I_sneak
        
        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.abs(I_total - I_ideal) / np.abs(I_ideal) * 100
        return np.nan_to_num(error)
