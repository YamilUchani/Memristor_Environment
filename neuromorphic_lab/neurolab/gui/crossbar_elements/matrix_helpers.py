"""
neurolab.gui.crossbar_elements.matrix_helpers
=============================================
Funciones auxiliares para trabajar con matrices de elementos en la interfaz gráfica.
"""

import numpy as np
from typing import Dict, Any


def get_G_matrix(elements: Dict[str, Any], n_rows: int = 4, n_cols: int = 4) -> np.ndarray:
    """Extrae la matriz de conductancias G (en Siemens) de los elementos M_ij."""
    G = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            key = f'M{i+1}{j+1}'
            if key in elements:
                G[i, j] = float(elements[key].params.get('G', elements[key].params.get('G_11', 69.4e-6)))

    return G


def get_V_vector(elements: Dict[str, Any], n_rows: int = 4) -> np.ndarray:
    """Extrae el vector de voltajes V (en Volts) de los elementos S_i."""
    V = np.zeros(n_rows)
    for i in range(n_rows):
        key = f'S{i+1}'
        if key in elements:
            V[i] = float(elements[key].params.get('V_out', 0.0))
    return V


def set_G_matrix(elements: Dict[str, Any], G_matrix: np.ndarray, R_on: float = 100.0, R_off: float = 16000.0) -> None:
    """Aplica una matriz de conductancias G a los elementos M_ij actualizando x0."""
    n_rows, n_cols = G_matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            key = f'M{i+1}{j+1}'
            if key in elements:
                g_val = float(G_matrix[i, j])
                g_val = max(g_val, 1e-12)
                R = 1.0 / g_val
                x = (R_off - R) / (R_off - R_on)
                x = float(np.clip(x, 0.0, 1.0))
                elements[key].params['x'] = x
                elements[key].params['x0'] = x
                elements[key].params['G'] = g_val
                elements[key].params['G_11'] = g_val
                if hasattr(elements[key], 'on_params_changed'):
                    elements[key].on_params_changed()



def compute_currents(G_matrix: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Calcula las corrientes de salida I = G^T · V."""
    return G_matrix.T @ V


def matrix_to_heatmap_string(G_matrix: np.ndarray) -> str:
    """Convierte la matriz G en conductancias (μS) a texto formateado."""
    n_rows, n_cols = G_matrix.shape
    G_uS = G_matrix * 1e6
    texto = "        " + "  ".join([f"Col {j+1:>6}" for j in range(n_cols)]) + "\n"
    for i in range(n_rows):
        texto += f"Fila {i+1}: "
        for j in range(n_cols):
            texto += f"{G_uS[i, j]:>8.2f}  "
        texto += "\n"
    return texto
