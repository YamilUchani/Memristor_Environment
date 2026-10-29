"""
neurolab/crossbar/programming.py
=================================
Helpers para esquemas de programación.

Funciones puras que retornan la matriz de voltajes
sin modificar el crossbar.
"""

import numpy as np


def build_V_matrix_row(n_rows, n_cols, i_row, V_program):
    """Matriz de voltajes para programación por fila."""
    V = np.zeros((n_rows, n_cols))
    V[i_row, :] = V_program
    return V


def build_V_matrix_1T1R(n_rows, n_cols, i, j, V_program):
    """Matriz de voltajes para 1T1R."""
    V = np.zeros((n_rows, n_cols))
    V[i, j] = V_program
    return V


def build_V_rowscols_V2(n_rows, n_cols, i, j, V_program):
    """Retorna (V_rows, V_cols) para esquema V/2."""
    V_rows = np.zeros(n_rows)
    V_cols = np.zeros(n_cols)
    if i is not None and 0 <= i < n_rows:
        V_rows[i] = V_program / 2.0
    if j is not None and 0 <= j < n_cols:
        V_cols[j] = -V_program / 2.0
    return V_rows, V_cols


def build_V_matrix_V2(n_rows, n_cols, i, j, V_program):
    """Matriz de voltajes para V/2 (V_cell = V_row - V_col)."""
    V_rows, V_cols = build_V_rowscols_V2(n_rows, n_cols, i, j, V_program)
    return V_rows[:, None] - V_cols[None, :]


def build_V_matrix_V3(n_rows, n_cols, i, j, V_program):
    """Matriz de voltajes para V/3."""
    V = np.zeros((n_rows, n_cols))
    V_third = V_program / 3.0
    for r in range(n_rows):
        for c in range(n_cols):
            if r == i and c == j:
                V[r, c] = V_program
            elif r == i:
                V[r, c] = 2.0 * V_third
            elif c == j:
                V[r, c] = -V_third
    return V


def classify_cells(n_rows, n_cols, i_target, j_target):
    """
    Clasifica las celdas según su rol en programación V/2 o V/3.
    
    Returns
    -------
    roles : ndarray (n_rows, n_cols) dtype=object
        'target' | 'half' | 'idle'
    """
    roles = np.full((n_rows, n_cols), 'idle', dtype=object)
    if i_target is None or j_target is None or i_target < 0 or j_target < 0:
        return roles

    for r in range(n_rows):
        for c in range(n_cols):
            if r == i_target and c == j_target:
                roles[r, c] = 'target'
            elif r == i_target or c == j_target:
                roles[r, c] = 'half'
    return roles
