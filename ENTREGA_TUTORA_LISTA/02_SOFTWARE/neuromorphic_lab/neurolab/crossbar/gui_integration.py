"""
neurolab/crossbar/gui_integration.py
=====================================
Helper para integración con GUI (Tkinter / PyQt / PySide6 / etc.).

Proporciona métodos listos para conectar botones, spinboxes y
visualización.
"""

import numpy as np
from .core import Crossbar
from .configs import CrossbarConfig, ProgrammingMode
from .programming import classify_cells


# --- Esquema de colores para la GUI ---
COLORS = {
    'target':        '#ef4444',  # 🔴 Rojo: celda objetivo
    'half':          '#eab308',  # 🟡 Amarillo: half-selected
    'idle':          '#1e293b',  # ⚪ Gris: no afectada
    'read':          '#06b6d4',  # 🔵 Cyan: lectura
    'write':         '#ec4899',  # 🩷 Rosa: escritura
    'border_idle':   '#475569',
    'border_active': '#fbbf24',
}

# --- Umbral de voltaje para lectura vs escritura ---
V_READ_THRESHOLD = 0.49  # V


class CrossbarGUIHelper:
    """
    Helper para conectar un Crossbar con una GUI.
    
    Uso típico:
        helper = CrossbarGUIHelper(crossbar)
        
        # En cada frame:
        state = helper.get_display_state(V_rows)
        # state['cells'][i][j] = {text, color, border, role}
    """
    
    def __init__(self, crossbar: Crossbar):
        self.cb = crossbar
    
    # ================================================================
    # ESTADO PARA RENDERIZADO
    # ================================================================
    
    def get_display_state(self, V_rows, V_cols=None, mode: str = 'read',
                          i_target: int = None, j_target: int = None):
        """
        Genera el estado de visualización para la GUI.
        
        Parameters
        ----------
        V_rows : ndarray (n_rows,)
            Voltajes de fila.
        V_cols : ndarray (n_cols,), optional
            Voltajes de columna.
        mode : str
            'read' | 'program_row' | 'program_V2' | 'program_V3'
        i_target, j_target : int, optional
            Coordenadas de la celda objetivo (para V/2 y V/3).
        """
        G_uS = self.cb.G_matrix * 1e6
        V_r = np.asarray(V_rows, dtype=float)
        if V_cols is None:
            V_c = np.zeros(self.cb.n_cols)
        else:
            V_c = np.asarray(V_cols, dtype=float)
        
        # Calcular matriz de voltajes
        if mode == 'program_V2' and i_target is not None and j_target is not None:
            V_r_prog, V_c_prog = np.zeros(self.cb.n_rows), np.zeros(self.cb.n_cols)
            V_r_prog[i_target] = V_r[i_target] if i_target < len(V_r) else 1.0
            V_c_prog[j_target] = -V_r_prog[i_target]
            V_matrix = V_r_prog[:, None] - V_c_prog[None, :]
        else:
            V_matrix = self._compute_V_matrix(V_r, mode, i_target, j_target)
        
        # Clasificar roles
        roles = self._classify_roles(mode, i_target, j_target)
        
        # Construir estado
        cells_state = []
        for i in range(self.cb.n_rows):
            row_state = []
            for j in range(self.cb.n_cols):
                role = roles[i, j]
                V_cell = V_matrix[i, j]
                
                # Determinar color y badge
                if mode == 'read':
                    v_eff = V_r[i] - V_c[j]
                    if abs(v_eff) > V_READ_THRESHOLD:
                        badge = f"⚡ E: {v_eff:+.2f}V"
                        color = COLORS['write']
                        role = 'write'
                    else:
                        badge = f"📖 L: {v_eff:+.2f}V"
                        color = COLORS['read']
                        role = 'read'
                elif role == 'target':
                    badge = f"🔴 Full V={V_cell:+.1f}V"
                    color = COLORS['target']
                elif role == 'half':
                    badge = f"🟡 Half V={V_cell:+.1f}V"
                    color = COLORS['half']
                else:
                    badge = f"⚪ Vnet={V_cell:+.1f}V"
                    color = COLORS['idle']
                
                row_state.append({
                    'G_uS': float(G_uS[i, j]),
                    'V_cell': float(V_cell),
                    'role': role,
                    'color': color,
                    'badge': badge,
                    'text': f"{G_uS[i, j]:.1f} μS",
                })
            cells_state.append(row_state)
        
        # Corrientes leídas
        I_cols = self.cb.read(V_r)
        
        return {
            'cells': cells_state,
            'currents': I_cols,
            'currents_uA': I_cols * 1e6,
            'V_rows': V_r.tolist(),
            'V_cols': V_c.tolist(),
        }
    
    def _compute_V_matrix(self, V_rows, mode, i_target, j_target):
        """Calcula la matriz de voltajes efectivos."""
        n_rows, n_cols = self.cb.n_rows, self.cb.n_cols
        V = np.zeros((n_rows, n_cols))
        
        if mode == 'read':
            for i in range(n_rows):
                V[i, :] = V_rows[i]
        
        elif mode == 'program_row' and i_target is not None:
            V[i_target, :] = V_rows[i_target]
        
        elif mode == 'program_V2' and i_target is not None:
            V_half = V_rows[i_target] / 2
            for r in range(n_rows):
                for c in range(n_cols):
                    if r == i_target and c == j_target:
                        V[r, c] = V_rows[i_target]
                    elif r == i_target or c == j_target:
                        V[r, c] = V_half
        
        elif mode == 'program_V3' and i_target is not None:
            V_third = V_rows[i_target] / 3
            for r in range(n_rows):
                for c in range(n_cols):
                    if r == i_target and c == j_target:
                        V[r, c] = V_rows[i_target]
                    elif r == i_target:
                        V[r, c] = 2 * V_third
                    elif c == j_target:
                        V[r, c] = -V_third
        
        return V
    
    def _classify_roles(self, mode, i_target, j_target):
        """Clasifica cada celda: target/half/idle/read/write."""
        n_rows, n_cols = self.cb.n_rows, self.cb.n_cols
        roles = np.full((n_rows, n_cols), 'idle', dtype=object)
        
        if mode in ('program_V2', 'program_V3') and i_target is not None:
            roles = classify_cells(n_rows, n_cols, i_target, j_target)
        
        return roles
    
    # ================================================================
    # ACCIONES PARA BOTONES
    # ================================================================
    
    def action_read(self, V_rows):
        """Acción: leer con voltajes dados (no destructivo)."""
        V_read = np.clip(V_rows, 0, V_READ_THRESHOLD)
        return self.cb.read(V_read)
    
    def action_write_row(self, i_row, V_program=None, dt=None):
        """Acción: escribir fila completa."""
        self.cb.program_row(i_row, V_program=V_program, dt=dt)
    
    def action_write_cell_V2(self, i, j, V_program=None, dt=None):
        """Acción: escribir celda con V/2."""
        self.cb.program_V2(i, j, V_program=V_program, dt=dt)
    
    def action_write_cell_1T1R(self, i, j, V_program=None, dt=None):
        """Acción: escribir celda con 1T1R."""
        self.cb.program_1T1R(i, j, V_program=V_program, dt=dt)
    
    def action_ltp_pulse(self, i, j, V_pulse=2.0, dt=1e-3):
        """Acción: pulso LTP (+V)."""
        self.cb.program_V2(i, j, V_program=V_pulse, dt=dt)
    
    def action_ltd_pulse(self, i, j, V_pulse=-2.0, dt=1e-3):
        """Acción: pulso LTD (−V)."""
        self.cb.program_V2(i, j, V_program=-abs(V_pulse), dt=dt)
    
    def action_reset(self):
        """Acción: reset del crossbar."""
        self.cb.reset()
    
    # ================================================================
    # UTILIDADES
    # ================================================================
    
    def get_cell_conductance(self, i, j) -> float:
        """Retorna G de la celda (i,j) en μS."""
        return float(self.cb.cells[i][j].conductance * 1e6)
    
    def get_cell_resistance(self, i, j) -> float:
        """Retorna R de la celda (i,j) en kΩ."""
        return float(self.cb.cells[i][j].resistance / 1e3)
    
    def get_matrix_summary(self) -> dict:
        """Resumen de la matriz."""
        return self.cb.summary()
