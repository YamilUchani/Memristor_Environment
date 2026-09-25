"""
neurolab/crossbar/nodal_solver.py
=================================
Resolvedor Nodal de Kirchhoff (MNA - Modified Nodal Analysis)
para Crossbar Memristivo N×M.

Modela exactamente la red de resistencias de wordlines (filas)
y bitlines (columnas) acopladas mediante las conductancias G[i,j].
"""

import numpy as np


def solve_crossbar_nodal(
    G: np.ndarray,
    V_rows: np.ndarray,
    R_wire: float = 1e-3,
    R_sense: float = 1e-12,
    R_driver: float = 1e-12,
) -> np.ndarray:
    """
    Resuelve el sistema nodal de Kirchhoff (MNA) para un crossbar N×M.

    Topología del circuito:
      - Nodografía de Fila (Wordline): V_R[i, j] para i ∈ [0, N-1], j ∈ [0, M-1]
      - Nodografía de Columna (Bitline): V_C[i, j] para i ∈ [0, N-1], j ∈ [0, M-1]
      - Celda memristiva (i, j): conecta V_R[i, j] con V_C[i, j] con conductancia G[i, j].
      - Resistencia de alambre horizontal (Wordline): R_H = R_wire entre V_R[i, j] y V_R[i, j+1].
      - Resistencia de alambre vertical (Bitline): R_V = R_wire entre V_C[i, j] y V_C[i+1, j].
      - Driver de Fila: inyecta V_rows[i] en V_R[i, 0] a través de R_driver.
      - Sense Amplificador / Tierra: conecta V_C[N-1, j] a 0V a través de R_sense.

    Parameters
    ----------
    G : ndarray (N, M)
        Matriz de conductancias memristivas (Siemens).
    V_rows : ndarray (N,)
        Vector de voltajes aplicados a las filas.
    R_wire : float
        Resistencia por segmento de alambre de interconexión (Ω).
    R_sense : float
        Resistencia de lectura al final de las columnas (Ω).
    R_driver : float
        Resistencia interna del driver de fila (Ω).

    Returns
    -------
    I_out : ndarray (M,)
        Corrientes de salida por columna (Amperios).
    """
    G = np.asarray(G, dtype=float)
    V_rows = np.asarray(V_rows, dtype=float)
    N, M = G.shape

    # Caso trivial o 0D
    if N == 0 or M == 0:
        return np.zeros(M)

    # Si R_wire es virtualmente cero (< 1e-11), usar la solución ideal instantánea
    if R_wire < 1e-11 and R_sense < 1e-11 and R_driver < 1e-11:
        return G.T @ V_rows

    g_wire = 1.0 / max(R_wire, 1e-12)
    g_sense = 1.0 / max(R_sense, 1e-12)
    g_drv = 1.0 / max(R_driver, 1e-12)

    # Número total de incógnitas nodales: N*M (Wordlines) + N*M (Bitlines) = 2*N*M
    n_nodes = 2 * N * M

    def idx_R(i: int, j: int) -> int:
        """Índice del nodo de Wordline en la celda (i, j)."""
        return i * M + j

    def idx_C(i: int, j: int) -> int:
        """Índice del nodo de Bitline en la celda (i, j)."""
        return N * M + i * M + j

    A = np.zeros((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    for i in range(N):
        for j in range(M):
            k_r = idx_R(i, j)
            k_c = idx_C(i, j)
            g_cell = float(G[i, j])

            # -----------------------------------------------------------------
            # 1. KCL en Nodo de Wordline V_R[i, j]
            # -----------------------------------------------------------------
            # Corriente a través de la celda memristiva (hacia Bitline)
            A[k_r, k_r] += g_cell
            A[k_r, k_c] -= g_cell

            # Acoplamiento a la izquierda (j-1)
            if j > 0:
                A[k_r, k_r] += g_wire
                A[k_r, idx_R(i, j - 1)] -= g_wire
            else:
                # Entrada de la Wordline (columna 0): conectada al driver V_rows[i]
                A[k_r, k_r] += g_drv
                b[k_r] += g_drv * V_rows[i]

            # Acoplamiento a la derecha (j+1)
            if j < M - 1:
                A[k_r, k_r] += g_wire
                A[k_r, idx_R(i, j + 1)] -= g_wire

            # -----------------------------------------------------------------
            # 2. KCL en Nodo de Bitline V_C[i, j]
            # -----------------------------------------------------------------
            # Corriente que entra de la celda memristiva (desde Wordline)
            A[k_c, k_c] += g_cell
            A[k_c, k_r] -= g_cell

            # Acoplamiento hacia arriba (i-1)
            if i > 0:
                A[k_c, k_c] += g_wire
                A[k_c, idx_C(i - 1, j)] -= g_wire

            # Acoplamiento hacia abajo (i+1)
            if i < N - 1:
                A[k_c, k_c] += g_wire
                A[k_c, idx_C(i + 1, j)] -= g_wire
            else:
                # Salida de la Bitline (fila N-1): conectada a masa/sense (0V)
                A[k_c, k_c] += g_sense
                # b[k_c] += g_sense * 0.0

    # Resolver el sistema lineal A * V_nodes = b
    try:
        V_nodes = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        V_nodes = np.linalg.lstsq(A, b, rcond=None)[0]

    # Extraer corrientes de salida en el extremo inferior de las columnas (fila N-1)
    I_out = np.zeros(M)
    for j in range(M):
        k_c_bottom = idx_C(N - 1, j)
        V_bottom = V_nodes[k_c_bottom]
        I_out[j] = g_sense * V_bottom

    return I_out
