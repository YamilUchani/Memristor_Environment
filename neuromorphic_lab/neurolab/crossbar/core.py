"""
neurolab/crossbar/core.py
=========================
Clase principal unificada del crossbar memristivo.

Soporta:
- Modos ideales y con no-idealidades (sneak paths, resistencias de línea)
- Operaciones 1x1, NxM
- Estocasticidad D2D / C2C
- Modos de programación (ROW, 1T1R, V/2, V/3)
"""

from typing import Optional
import numpy as np
from neurolab.devices import MemristorStrukov, MemristorYakopcic, YakopcicVirginConfig
from neurolab.configs import StrukovConfig
from .configs import CrossbarConfig, ProgrammingMode
from .line_resistance import LineResistanceModel


class Crossbar:
    """
    Crossbar memristivo N×M unificado.
    """

    def __init__(self, config: CrossbarConfig = None, **kwargs):
        """Inicializa el crossbar."""
        if config is None:
            config = CrossbarConfig(**kwargs)
        self.cfg = config
        self.config = config  # Alias para compatibilidad

        self.n_rows = config.n_rows
        self.n_cols = config.n_cols
        self.rng = np.random.default_rng(config.seed)

        # --- D2D factors (uno por celda, fijo) ---
        self._d2d_factors = np.ones((self.n_rows, self.n_cols))
        self._init_d2d_factors()

        # --- Crear celdas ---
        self.cells = [[None for _ in range(self.n_cols)]
                      for _ in range(self.n_rows)]
        self._init_cells()

        # --- Modelo de línea ---
        self.line_model = LineResistanceModel(
            R_H=getattr(config, 'R_line_H', getattr(config, 'line_resistance', 0.0)),
            R_V=getattr(config, 'R_line_V', getattr(config, 'line_resistance', 0.0)),
        )

        # --- Estado ---
        self.programming_history = []
        self.last_operation = None
        self.V_rows = np.zeros(self.n_rows)
        self.V_cols = np.zeros(self.n_cols)
        self.I_out = np.zeros(self.n_cols)

    @property
    def V_applied(self) -> np.ndarray:
        return self.V_rows

    @V_applied.setter
    def V_applied(self, val: np.ndarray):
        self.V_rows = np.asarray(val, dtype=float)

    @property
    def memristors(self) -> np.ndarray:
        """Propiedad de compatibilidad que retorna array 2D de celdas."""
        arr = np.empty((self.n_rows, self.n_cols), dtype=object)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                arr[i, j] = self.cells[i][j]
        return arr

    def _init_d2d_factors(self):
        """Inicializa los factores estocásticos D2D."""
        if getattr(self.cfg, 'enable_d2d', False) or getattr(self.cfg, 'd2d_enabled', False):
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    f = 1.0 + (self.cfg.G_sigma / max(1e-12, self.cfg.G_mean)) * self.rng.standard_normal()
                    self._d2d_factors[i, j] = float(np.clip(f, 0.5, 1.5))

    def _init_cells(self):
        """Inicializa cada celda."""
        use_yakopcic = getattr(self.cfg, 'use_yakopcic', False)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if use_yakopcic:
                    G_0 = float(self.cfg.G_mean * self._d2d_factors[i, j])
                    G_0 = float(np.clip(G_0, self.cfg.G_min, self.cfg.G_max))
                    cfg = YakopcicVirginConfig(G_initial=G_0)
                    self.cells[i][j] = MemristorYakopcic(cfg, G_0=G_0)
                else:
                    cfg = StrukovConfig(
                        RON=getattr(self.cfg, 'R_on', getattr(self.cfg, 'R_ON', 100.0)),
                        ROFF=getattr(self.cfg, 'R_off', getattr(self.cfg, 'R_OFF', 16000.0)),
                        x0=getattr(self.cfg, 'x0', 0.1),
                        D=getattr(self.cfg, 'D', 10e-9),
                        mu_v=getattr(self.cfg, 'mu_v', 1e-14),
                        clip_x=getattr(self.cfg, 'clip_x', True),
                    )
                    m = MemristorStrukov(
                        cfg,
                        enable_volatile=getattr(self.cfg, 'enable_volatile', False),
                        tau_relax=getattr(self.cfg, 'tau_relax', 0.5),
                        x_eq=getattr(self.cfg, 'x_eq', 0.05)
                    )
                    if getattr(self.cfg, 'enable_d2d', False) or getattr(self.cfg, 'd2d_enabled', False):
                        G_0 = float(self.cfg.G_mean * self._d2d_factors[i, j])
                        self._set_cell_conductance(m, G_0)
                    self.cells[i][j] = m

    def _set_cell_conductance(self, cell, G: float) -> None:
        """Establece la conductancia de una celda memristiva."""
        if hasattr(cell, 'set_conductance') and callable(getattr(cell, 'set_conductance')):
            cell.set_conductance(G)
        elif hasattr(cell, '_x') or hasattr(cell, 'x'):
            R = 1.0 / max(1e-15, float(G))
            r_on = getattr(cell, 'r_on', getattr(getattr(cell, 'electrical', None), 'r_on', 100.0))
            r_off = getattr(cell, 'r_off', getattr(getattr(cell, 'electrical', None), 'r_off', 16000.0))
            if abs(r_off - r_on) > 1e-12:
                x = (r_off - R) / (r_off - r_on)
            else:
                x = 0.5
            cell.x = float(np.clip(x, 0.0, 1.0))
        else:
            cell.G = G

    # ================================================================
    # PROPIEDADES DE CONDUCTANCIA
    # ================================================================

    @property
    def G_matrix(self) -> np.ndarray:
        """Matriz de conductancias (S)."""
        G = np.zeros((self.n_rows, self.n_cols))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                G[i, j] = self.cells[i][j].conductance
        return G

    @property
    def R_matrix(self) -> np.ndarray:
        """Matriz de resistencias (Ω)."""
        with np.errstate(divide='ignore'):
            return 1.0 / self.G_matrix

    @property
    def G_matrix_uS(self) -> np.ndarray:
        """Matriz de conductancias (μS)."""
        return self.G_matrix * 1e6

    # ================================================================
    # MANIPULACIÓN DE CONDUCTANCIAS (COMPATIBILIDAD)
    # ================================================================

    def set_conductance(self, i: int, j: int, G: float) -> None:
        """Establece la conductancia de una celda específica."""
        cell = self.cells[i][j]
        self._set_cell_conductance(cell, G)

    def set_uniform_conductance(self, G: float) -> None:
        """Establece la misma conductancia en todas las celdas."""
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self.set_conductance(i, j, G)

    def get_conductance(self, i: int, j: int) -> float:
        """Devuelve la conductancia de una celda específica."""
        return float(self.cells[i][j].conductance)

    def set_matrix_conductance(self, G_matrix: np.ndarray) -> None:
        """Establece la matriz completa de conductancias."""
        G_mat = np.asarray(G_matrix, dtype=float)
        if G_mat.shape != (self.n_rows, self.n_cols):
            raise ValueError(f"G_matrix debe tener dimensiones ({self.n_rows}, {self.n_cols})")
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self.set_conductance(i, j, G_mat[i, j])

    # ================================================================
    # LECTURA
    # ================================================================

    def read_ideal(self, V_rows: np.ndarray) -> np.ndarray:
        """Lectura IDEAL: I = G^T · V."""
        V = np.asarray(V_rows, dtype=float)
        self.V_rows = V.copy()
        self.V_cols = np.zeros(self.n_cols)
        self.I_out = self.G_matrix.T @ V
        return self.I_out

    def read_with_sneak_paths(self, V_rows: np.ndarray = None) -> np.ndarray:
        """Lectura con Sneak Paths (modelo de vecinos)."""
        if V_rows is None:
            V_rows = self.V_rows
        G = self.G_matrix
        V = np.asarray(V_rows, dtype=float)
        self.V_rows = V.copy()
        self.V_cols = np.zeros(self.n_cols)
        I_target = G.T @ V

        alpha = getattr(self.cfg, 'R_sneak_factor', 0.05)
        V_mean = float(np.mean(np.abs(V))) if len(V) > 0 else 0.0
        I_sneak = np.zeros(self.n_cols)
        for j in range(self.n_cols):
            for i in range(self.n_rows):
                neighbors = int(i > 0) + int(i < self.n_rows - 1)
                I_sneak[j] += alpha * G[i, j] * V_mean * neighbors
        self.I_out = I_target + I_sneak
        return self.I_out

    def read_with_line_resistance(self, V_rows: np.ndarray = None) -> np.ndarray:
        """Lectura con resistencias de línea (IR drop)."""
        if V_rows is None:
            V_rows = self.V_rows
        self.V_rows = np.asarray(V_rows, dtype=float).copy()
        self.V_cols = np.zeros(self.n_cols)
        I_cols, _ = self.line_model.solve(self.G_matrix, V_rows)
        self.I_out = I_cols
        return self.I_out

    def read(self, V_rows: np.ndarray = None) -> np.ndarray:
        """Lectura con no idealidades activas según config."""
        if V_rows is None:
            V_rows = self.V_rows
        if getattr(self.cfg, 'enable_sneak_paths', False):
            return self.read_with_sneak_paths(V_rows)
        elif getattr(self.cfg, 'enable_line_resistance', False):
            return self.read_with_line_resistance(V_rows)
        else:
            return self.read_ideal(V_rows)

    def read_currents(self) -> np.ndarray:
        """Alias de compatibilidad para lectura."""
        return self.read(self.V_applied)

    def read_single(self, V_in: float) -> float:
        """Lectura para 1×1. Retorna un escalar."""
        if self.n_rows != 1 or self.n_cols != 1:
            raise ValueError("read_single solo es válido para crossbar 1×1")
        G = self.cells[0][0].conductance
        return float(G * V_in)

    def read_objective_only(self, V_rows: np.ndarray = None) -> np.ndarray:
        """Solo la corriente objetivo (sin parásitas)."""
        if V_rows is None:
            V_rows = self.V_rows
        V = np.asarray(V_rows, dtype=float)
        return self.G_matrix.T @ V

    def reference_currents(self) -> np.ndarray:
        """Corriente de referencia ideal sin parásitos."""
        return self.read_objective_only(self.V_applied)

    def sneak_error_pct(self, V_rows: np.ndarray = None) -> float:
        """Calcula el porcentaje de error por sneak paths respecto a la corriente ideal."""
        if V_rows is None:
            V_rows = self.V_rows
        I_ideal = self.read_objective_only(V_rows)
        I_total = self.read_with_sneak_paths(V_rows)
        norm = np.linalg.norm(I_ideal)
        if norm < 1e-15:
            return 0.0
        return float(np.linalg.norm(I_total - I_ideal) / norm * 100.0)

    def _effective_voltage(self, V_row: float, position: int) -> float:
        """Voltaje efectivo en la posición `position` de la fila considerando IR drop."""
        R_line = getattr(self.cfg, 'line_resistance', getattr(self.cfg, 'R_line_H', 0.0))
        if R_line <= 0:
            return V_row
        R_acc = R_line * position
        I_avg = float(np.mean(np.abs(self.G_matrix)) * abs(V_row))
        V_drop = I_avg * R_acc
        return V_row - V_drop

    def read_voltages(self, V_rows: np.ndarray) -> np.ndarray:
        """Retorna voltajes efectivos en cada celda."""
        V = np.asarray(V_rows, dtype=float)
        self.V_rows = V.copy()
        return self.V_rows[:, None] - self.V_cols[None, :]

    def apply_voltages(self, V: np.ndarray) -> None:
        """Aplica voltajes a las filas."""
        V = np.atleast_1d(V)
        if len(V) != self.n_rows:
            raise ValueError(f"V debe tener {self.n_rows} elementos")
        self.V_applied = np.asarray(V, dtype=float)

    def update_memristors(self, dt: float) -> None:
        """Actualiza el estado de todos los memristores según V_applied."""
        self._apply_voltages(self.V_rows[:, None] - self.V_cols[None, :], dt)

    # ================================================================
    # PROGRAMACIÓN
    # ================================================================

    def _apply_voltages(self, V_matrix: np.ndarray, dt: float):
        """Aplica matriz de voltajes a las celdas y actualiza G."""
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if abs(V_matrix[i, j]) > 1e-9:
                    self.cells[i][j].update(V_matrix[i, j], dt)

    def program_row(self, i: int, V_program: float = None, dt: float = None):
        """Programa toda la fila i (V_col = 0)."""
        if V_program is None:
            V_program = self.cfg.V_program
        if dt is None:
            dt = getattr(self.cfg, 'dt_physics', getattr(self.cfg, 'dt_pulse', 1e-3))

        self.V_rows = np.zeros(self.n_rows)
        self.V_cols = np.zeros(self.n_cols)
        if 0 <= i < self.n_rows:
            self.V_rows[i] = V_program
        V_matrix = self.V_rows[:, None] - self.V_cols[None, :]

        self._apply_voltages(V_matrix, dt)
        self.last_operation = ('program_row', i)
        self.programming_history.append({
            'mode': 'row', 'row': i, 'V': V_program, 'dt': dt,
        })

    def program_1T1R(self, i: int, j: int, V_program: float = None, dt: float = None):
        """Modo 1T1R: programa SOLO la celda (i,j)."""
        if V_program is None:
            V_program = self.cfg.V_program
        if dt is None:
            dt = getattr(self.cfg, 'dt_physics', getattr(self.cfg, 'dt_pulse', 1e-3))

        self.V_rows = np.zeros(self.n_rows)
        self.V_cols = np.zeros(self.n_cols)
        if 0 <= i < self.n_rows:
            self.V_rows[i] = V_program
        V_matrix = np.zeros((self.n_rows, self.n_cols))
        V_matrix[i, j] = V_program

        self._apply_voltages(V_matrix, dt)
        self.last_operation = ('program_1T1R', i, j)
        self.programming_history.append({
            'mode': '1T1R', 'cell': (i, j), 'V': V_program, 'dt': dt,
        })

    def program_V2(self, i_target: int, j_target: int, V_program: float = None, dt: float = None):
        """Modo V/2: programa celda (i,j) con cruz de half-selected."""
        if V_program is None:
            V_program = self.cfg.V_program
        if dt is None:
            dt = getattr(self.cfg, 'dt_physics', getattr(self.cfg, 'dt_pulse', 1e-3))

        self.V_rows = np.zeros(self.n_rows)
        self.V_cols = np.zeros(self.n_cols)
        if 0 <= i_target < self.n_rows:
            self.V_rows[i_target] = V_program / 2.0
        if 0 <= j_target < self.n_cols:
            self.V_cols[j_target] = -V_program / 2.0

        V_matrix = self.V_rows[:, None] - self.V_cols[None, :]

        self._apply_voltages(V_matrix, dt)
        self.last_operation = ('program_V2', i_target, j_target)
        self.programming_history.append({
            'mode': 'V2', 'cell': (i_target, j_target),
            'V': V_program, 'dt': dt,
        })

    def program_V3(self, i_target: int, j_target: int, V_program: float = None, dt: float = None):
        """Modo V/3: programa con tres niveles de tensión."""
        if V_program is None:
            V_program = self.cfg.V_program
        if dt is None:
            dt = getattr(self.cfg, 'dt_physics', getattr(self.cfg, 'dt_pulse', 1e-3))

        V_third = V_program / 3.0
        V_matrix = np.zeros((self.n_rows, self.n_cols))

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if r == i_target and c == j_target:
                    V_matrix[r, c] = V_program
                elif r == i_target:
                    V_matrix[r, c] = 2.0 * V_third
                elif c == j_target:
                    V_matrix[r, c] = -V_third

        self._apply_voltages(V_matrix, dt)
        self.last_operation = ('program_V3', i_target, j_target)
        self.programming_history.append({
            'mode': 'V3', 'cell': (i_target, j_target),
            'V': V_program, 'dt': dt,
        })

    def program(self, i: int, j: int = None, **kwargs):
        """Programa según el modo configurado."""
        mode = getattr(self.cfg, 'mode', getattr(self.cfg, 'programming_mode', 'V2'))

        if mode == ProgrammingMode.ROW or mode == 'ROW' or mode == 'row':
            self.program_row(i, **kwargs)
        elif mode == ProgrammingMode.ONE_T_ONE_R or mode == '1T1R':
            if j is None:
                raise ValueError("Modo 1T1R requiere (i, j)")
            self.program_1T1R(i, j, **kwargs)
        elif mode == ProgrammingMode.V2 or mode == 'V2':
            if j is None:
                raise ValueError("Modo V/2 requiere (i, j)")
            self.program_V2(i, j, **kwargs)
        elif mode == ProgrammingMode.V3 or mode == 'V3':
            if j is None:
                raise ValueError("Modo V/3 requiere (i, j)")
            self.program_V3(i, j, **kwargs)
        else:
            raise ValueError(f"Modo desconocido: {mode}")

    # ================================================================
    # RESET / REINICIO
    # ================================================================

    def reset(self):
        """Reinicia el crossbar a su estado D2D inicial."""
        self.rng = np.random.default_rng(self.cfg.seed)
        self._init_d2d_factors()
        self._init_cells()
        self.programming_history = []
        self.last_operation = None

    # ================================================================
    # VISUALIZACIÓN Y RESUMEN
    # ================================================================

    def print_state(self, title: str = "ESTADO DEL CROSSBAR", unit: str = 'μS'):
        """Imprime la matriz de conductancias."""
        G = self.G_matrix * 1e6 if unit == 'μS' else self.G_matrix

        print("\n" + "=" * (10 + 9 * self.n_cols))
        print(f"  {title}")
        print("=" * (10 + 9 * self.n_cols))
        print("        " + "".join(f"  COL {j+1}  " for j in range(self.n_cols)))

        for i in range(self.n_rows):
            row_str = f"FILA {i+1}: "
            for j in range(self.n_cols):
                row_str += f"{G[i, j]:7.2f} "
            print(row_str)
        print("=" * (10 + 9 * self.n_cols))

    def summary(self) -> dict:
        """Resumen estadístico."""
        G = self.G_matrix_uS
        stats = {
            'mean': float(G.mean()),
            'std': float(G.std()),
            'min': float(G.min()),
            'max': float(G.max()),
            'cv_percent': float(G.std() / max(1e-12, G.mean()) * 100),
            'shape': (self.n_rows, self.n_cols),
            'n_cells': self.n_rows * self.n_cols,
        }
        return stats

    def print_summary(self):
        """Imprime resumen."""
        s = self.summary()
        print(f"\n  Media:  {s['mean']:.3f} uS")
        print(f"  Std:    {s['std']:.3f} uS")
        print(f"  Min:    {s['min']:.3f} uS")
        print(f"  Max:    {s['max']:.3f} uS")
        print(f"  CV:     {s['cv_percent']:.2f} %")
        print(f"  Celdas: {s['n_cells']} ({s['shape'][0]}x{s['shape'][1]})")
