"""
neurolab/crossbar/addressing.py
================================
Decodificadores de dirección y drivers de línea.

Modela explícitamente:
- AddressDecoder: convierte dirección (int) en vector one-hot
- RowDriver: aplica V a la fila activa
- ColumnDriver: aplica V a la columna activa
- ReadConfig / WriteConfig: configuraciones explícitas de operación
"""

import numpy as np
from dataclasses import dataclass


# =====================================================================
# DECODIFICADOR DE DIRECCIONES
# =====================================================================

class AddressDecoder:
    """
    Decodificador de direcciones N → 2^N líneas.
    
    En un chip real, esto es una red de compuertas AND que
    convierte un número binario en una señal one-hot.
    
    Ejemplo:
        decoder = AddressDecoder(n_lines=4)
        one_hot = decoder.decode(address=1)  # → [0, 1, 0, 0]
    """
    
    def __init__(self, n_lines: int):
        """
        Parameters
        ----------
        n_lines : int
            Número de líneas de salida (filas o columnas).
        """
        self.n_lines = n_lines
        self.n_bits = int(np.ceil(np.log2(max(n_lines, 1))))
        self.last_address = -1
        self.last_one_hot = np.zeros(n_lines)
    
    def decode(self, address: int) -> np.ndarray:
        """
        Convierte dirección en vector one-hot.
        
        Parameters
        ----------
        address : int
            Índice de la línea a activar (0-based).
        
        Returns
        -------
        one_hot : ndarray (n_lines,)
            Vector con 1.0 en la posición activa, 0.0 en el resto.
        """
        if address < 0 or address >= self.n_lines:
            raise ValueError(
                f"Dirección {address} fuera de rango [0, {self.n_lines-1}]"
            )
        
        one_hot = np.zeros(self.n_lines)
        one_hot[address] = 1.0
        
        self.last_address = address
        self.last_one_hot = one_hot
        
        return one_hot
    
    def address_to_binary(self, address: int) -> str:
        """Convierte dirección a string binario (para visualización)."""
        if address < 0 or address >= self.n_lines:
            return "N/A"
        return format(address, f'0{self.n_bits}b')


# =====================================================================
# DRIVER DE FILA
# =====================================================================

class RowDriver:
    """
    Driver de línea de palabra (Word Line).
    
    Aplica voltaje a la fila activa.
    Las filas inactivas reciben V_inactive (típicamente 0V).
    """
    
    def __init__(self, n_rows: int):
        self.n_rows = n_rows
        self.V_active = 0.0      # Voltaje en la fila activa
        self.V_inactive = 0.0    # Voltaje en las filas inactivas
        self.active_row = -1     # Fila activa actual
    
    def drive(self, one_hot: np.ndarray,
              V_active: float = None) -> np.ndarray:
        """
        Aplica voltajes a las filas.
        
        Parameters
        ----------
        one_hot : ndarray (n_rows,)
            Vector one-hot del decodificador.
        V_active : float, optional
            Voltaje a aplicar a la fila activa. Si None, usa self.V_active.
        
        Returns
        -------
        V_rows : ndarray (n_rows,)
            Voltaje de cada fila.
        """
        if V_active is None:
            V_active = self.V_active
        
        V_rows = np.full(self.n_rows, self.V_inactive)
        active_idx = np.where(one_hot > 0)[0]
        if len(active_idx) > 0:
            V_rows[active_idx[0]] = V_active
            self.active_row = int(active_idx[0])
        else:
            self.active_row = -1
        
        return V_rows
    
    def drive_all(self, V_array: np.ndarray) -> np.ndarray:
        """Aplica voltajes individuales a cada fila (modo lectura con sensores)."""
        return np.asarray(V_array, dtype=float)
    
    def reset(self):
        """Desactiva todas las filas."""
        self.V_active = 0.0
        self.V_inactive = 0.0
        self.active_row = -1


# =====================================================================
# DRIVER DE COLUMNA
# =====================================================================

class ColumnDriver:
    """
    Driver de línea de bit (Bit Line).
    
    Similar a RowDriver pero para columnas.
    En modo V/2, aplica voltaje NEGATIVO a la columna activa.
    """
    
    def __init__(self, n_cols: int):
        self.n_cols = n_cols
        self.V_active = 0.0
        self.V_inactive = 0.0
        self.active_col = -1
    
    def drive(self, one_hot: np.ndarray,
              V_active: float = None) -> np.ndarray:
        """Aplica voltajes a las columnas."""
        if V_active is None:
            V_active = self.V_active
        
        V_cols = np.full(self.n_cols, self.V_inactive)
        active_idx = np.where(one_hot > 0)[0]
        if len(active_idx) > 0:
            V_cols[active_idx[0]] = V_active
            self.active_col = int(active_idx[0])
        else:
            self.active_col = -1
        
        return V_cols
    
    def drive_all(self, V_array: np.ndarray) -> np.ndarray:
        """Aplica voltajes individuales a cada columna."""
        return np.asarray(V_array, dtype=float)
    
    def reset(self):
        """Desactiva todas las columnas."""
        self.V_active = 0.0
        self.V_inactive = 0.0
        self.active_col = -1


# =====================================================================
# CONFIGURACIONES EXPLÍCITAS
# =====================================================================

@dataclass
class ReadConfig:
    """Configuración explícita del modo lectura."""
    V_read: float = 0.20      # Voltaje de lectura (V)
    V_th: float = 0.50        # Umbral de programación (V)
    t_settle: float = 1e-3    # Tiempo de establecimiento (s)
    V_col: float = 0.0        # Voltaje de columnas en lectura (V)
    
    def is_safe(self) -> bool:
        """Verifica que V_read < V_th (lectura no destructiva)."""
        return self.V_read < self.V_th
    
    def describe(self) -> str:
        return (f"📖 Lectura: V_read={self.V_read:.2f}V, "
                f"V_col={self.V_col:.2f}V, "
                f"V_th={self.V_th:.2f}V")


@dataclass
class WriteConfig:
    """Configuración explícita del modo escritura."""
    # --- Voltajes ---
    V_program: float = 2.00    # Voltaje completo (target)
    V_th: float = 0.50         # Umbral de programación
    # --- Esquema ---
    scheme: str = 'V2'         # 'row' | '1T1R' | 'V2' | 'V3'
    # --- Pulsos ---
    n_pulses: int = 1          # Número de pulsos
    t_pulse: float = 1e-3      # Duración de cada pulso (s)
    # --- C2C/D2D ---
    c2c_enabled: bool = False
    c2c_sigma: float = 0.02
    
    def V_row_active(self) -> float:
        """Voltaje de fila activa según esquema."""
        if self.scheme == 'V2':
            return +self.V_program / 2.0
        elif self.scheme == 'V3':
            return +2.0 * self.V_program / 3.0
        else:
            return self.V_program
    
    def V_col_active(self) -> float:
        """Voltaje de columna activa según esquema."""
        if self.scheme == 'V2':
            return -self.V_program / 2.0
        elif self.scheme == 'V3':
            return -self.V_program / 3.0
        else:
            return 0.0
    
    def describe(self) -> str:
        return (f"⚡ Escritura [{self.scheme}]: "
                f"V_program={self.V_program:.2f}V, "
                f"V_row={self.V_row_active():+.2f}V, "
                f"V_col={self.V_col_active():+.2f}V, "
                f"{self.n_pulses} pulso(s) × {self.t_pulse*1e3:.1f}ms")
