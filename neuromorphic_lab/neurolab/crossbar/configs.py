"""
neurolab/crossbar/configs.py
============================
Configuración completa del crossbar memristivo.
"""

from dataclasses import dataclass, field
from enum import Enum


class ProgrammingMode(Enum):
    """Modos de programación disponibles."""
    ROW = "row"             # Escritura por fila (sensor → fila)
    ONE_T_ONE_R = "1T1R"    # Aislamiento perfecto
    V2 = "V2"               # Esquema V/2 (cruz)
    V3 = "V3"               # Esquema V/3 (cruz suave)


@dataclass
class CrossbarConfig:
    """
    Configuración completa del crossbar.
    
    Todos los parámetros en unidades SI (Ω, S, V, s).
    """
    
    # --- Dimensiones ---
    n_rows: int = 4
    n_cols: int = 4
    
    # --- Parámetros eléctricos base ---
    R_on: float = 100.0          # Ω
    R_off: float = 16000.0       # Ω
    x0: float = 0.10             # adim.
    D: float = 10e-9             # m
    mu_v: float = 1e-14          # m²/(V·s)
    clip_x: bool = True
    line_resistance: float = 0.0 # Ω
    
    # --- Volatilidad ---
    enable_volatile: bool = False
    tau_relax: float = 0.5       # s
    x_eq: float = 0.05
    
    # --- Conductancias (S) ---
    G_mean: float = 70.0e-6      # Conductancia media inicial D2D (70 μS)
    G_sigma: float = 8.0e-6      # Desviación D2D (8 μS)
    G_min: float = 1.0e-6        # Límite inferior
    G_max: float = 500.0e-6      # Límite superior
    
    # --- Voltajes (V) ---
    V_read: float = 0.20         # Voltaje de lectura (no destructivo)
    V_th: float = 0.50           # Umbral de programación
    V_program: float = 2.00      # Voltaje de programación
    V_reset: float = -2.00       # Voltaje de reset
    
    # --- Tiempos (s) ---
    dt_pulse: float = 1e-3       # Duración del pulso
    dt_physics: float = 0.02     # Paso de integración
    
    # --- Modos de programación ---
    mode: ProgrammingMode = ProgrammingMode.ROW
    programming_mode: str = 'V2'  # Compatibilidad con código anterior ('1T1R', 'V2', 'V3')
    
    # --- No idealidades y Variabilidad ---
    d2d_enabled: bool = True
    enable_d2d: bool = True      # Variabilidad estática D2D
    d2d_sigma: float = 0.08      # Desviación estocástica D2D
    d2d_range: tuple = (0.02e-6, 1.0e-3)

    c2c_enabled: bool = False
    enable_c2c: bool = False     # Variabilidad dinámica C2C
    c2c_sigma: float = 0.05
    c2c_range: tuple = (0.9, 1.1)

    enable_sneak_paths: bool = False   # 🔒 DESACTIVADO por defecto
    enable_line_resistance: bool = False
    
    # --- Resistencias de línea (Ω por segmento) ---
    R_line_H: float = 0.0
    R_line_V: float = 0.0
    
    # --- Sneak paths ---
    R_sneak_factor: float = 0.1  # Fracción de G que contribuye a sneak
    
    # --- Reproducibilidad ---
    seed: int = 42

    def __post_init__(self):
        """Validación de parámetros."""
        if self.n_rows < 1 or self.n_cols < 1:
            raise ValueError("Dimensiones deben ser >= 1")
        if self.G_mean <= 0:
            raise ValueError("G_mean debe ser > 0")
        if self.V_th <= 0:
            raise ValueError("V_th debe ser > 0")
        if not (self.V_read < self.V_th <= abs(self.V_program)):
            raise ValueError(f"Voltajes incoherentes: V_read ({self.V_read}) debe ser < V_th ({self.V_th}) <= |V_program| ({abs(self.V_program)})")
