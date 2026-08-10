"""
models/lif_neuron.py
====================
Implementación de la neurona Leaky Integrate-and-Fire (LIF) física basada en TSM.

Modelo de circuito físico (Figura 3 del paper):
    Vin(t) --> [Rs] --> Vc (capacitor C) --> [TSM] --> [R0] --> GND

Ecuaciones diferenciales acopladas (Forward Euler):
    1. dw/dt = alpha * (Vc - Vth) * (1 - w) - beta * w   (para Vc >= V_hold)
       dw/dt = -beta * w                                  (para Vc < V_hold)
    2. R_TSM = R_off * (1 - w) + R_on * w
    3. dVc/dt = ((Vin - Vc)/Rs - Vc/(R_TSM + R0)) / C
    4. Vout = Vc * R0 / (R_TSM + R0)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros del modelo TSM-LIF Físico
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LIFParameters:
    """Parámetros físicos del circuito TSM-LIF."""
    # Capacitor y Resistencia de Carga (Integración de ~100 ms)
    C: float = 1.0e-6          # 1 uF
    R_s: float = 200e3         # 200 kOhm
    
    # Memristor TSM (Umbral y Resistencias de la tabla)
    R_off: float = 1.0e6       # 1 MOhm
    R_on: float = 1.0e3        # 1 kOhm
    
    # Resistencia de Salida (Divisor con Ron para pico ~0.53V)
    R_0: float = 1.2e3         # 1.2 kOhm
    
    # Dinámica TSM (Tiempos de recuperación y disparo)
    V_th: float = 0.95         # Voltaje umbral de encendido
    V_hold: float = 0.15       # Voltaje de mantenimiento (descarga)
    alpha: float = 50000000.0  # Tasa de encendido (disparo ultrarrápido)
    beta: float = 25.0         # Tasa de apagado (reposo exacto de ~75 ms)
    w_init: float = 0.0      # Variable de estado inicial del TSM [0, 1]

    def __post_init__(self):
        if self.R_on >= self.R_off:
            raise ValueError(f"R_on ({self.R_on} Ohm) debe ser < R_off ({self.R_off} Ohm).")
        if self.C <= 0 or self.R_s <= 0 or self.R_0 <= 0:
            raise ValueError("C, R_s y R_0 deben ser mayores que 0.")

    def __str__(self) -> str:
        lines = [
            "+- TSM-LIF Parameters ------------------------------------+",
            f"|  C             = {self.C*1e9:>10.1f}  nF                    |",
            f"|  R_s           = {self.R_s/1e3:>10.2f}  kOhm                  |",
            f"|  R_off         = {self.R_off/1e6:>10.2f}  MOhm                  |",
            f"|  R_on          = {self.R_on/1e3:>10.2f}  kOhm                  |",
            f"|  R_0           = {self.R_0/1e3:>10.2f}  kOhm                  |",
            f"|  V_th          = {self.V_th:>10.2f}  V                     |",
            f"|  V_hold        = {self.V_hold:>10.2f}  V                     |",
            f"|  alpha         = {self.alpha:>10.1f}  1/(V*s)               |",
            f"|  beta          = {self.beta:>10.1f}  1/s                   |",
            "+---------------------------------------------------------+",
        ]
        return "\n".join(lines)


# ── Presets de fábrica ────────────────────────────────────────────────────────

def default_lif_params() -> LIFParameters:
    """Parámetros estándar calibrados con los datos experimentales."""
    return LIFParameters()


def fast_lif_params() -> LIFParameters:
    """Parámetros para una dinámica más rápida (menor capacitancia C)."""
    return LIFParameters(
        C=200e-9,      # 200 nF para mayor frecuencia de disparo
        R_s=50e3,      # 50 kOhm
    )


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal: neurona TSM-LIF integrada paso a paso
# ─────────────────────────────────────────────────────────────────────────────

class LIFNeuron:
    """
    Neurona LIF Física basada en un Memristor de Conmutación de Umbral (TSM).
    """

    def __init__(self, params: LIFParameters, V_init: Optional[float] = None):
        self.params = params
        self.V: float = V_init if V_init is not None else 0.0
        self.w: float = params.w_init
        
        # Variables de estado del circuito
        self.R_tsm: float = params.R_off
        self.Vout: float = 0.0

        # Historial/Estadísticas
        self._spike_count: int = 0
        self._w_prev: float = params.w_init
        self._charge_accumulated: float = 0.0

    @property
    def membrane_potential(self) -> float:
        """Potencial en el capacitor Vc [V]."""
        return self.V

    @property
    def membrane_potential_mV(self) -> float:
        """Potencial en el capacitor Vc [mV]."""
        return self.V * 1e3

    @property
    def spike_count(self) -> int:
        """Número de disparos detectados (flanco de subida de w > 0.5)."""
        return self._spike_count

    @property
    def is_refractory(self) -> bool:
        """En el modelo físico no hay refractariedad artificial."""
        return False

    @property
    def charge_accumulated(self) -> float:
        """Carga acumulada aproximada."""
        return self._charge_accumulated

    def step(self, Vin: float, dt: float) -> tuple[bool, float]:
        """
        Avanza un paso de tiempo dt con voltaje de entrada Vin.
        """
        # 1. Conductancia instantánea del TSM (físicamente correcto para relajación lenta)
        # Función sigmoide para imitar el "snap-off" abrupto del compuesto físico del memristor
        w_eff = 1.0 / (1.0 + np.exp(-50.0 * (self.w - 0.2)))
        G_tsm = (1.0 / self.params.R_off) * (1.0 - w_eff) + (1.0 / self.params.R_on) * w_eff
        self.R_tsm = 1.0 / G_tsm

        # 2. Corriente de salida y Vout (con acoplamiento capacitivo del protoboard)
        I_out = self.V / (self.R_tsm + self.params.R_0)
        Vout_ideal = I_out * self.params.R_0
        
        # Filtro RC parasitario para convertir el Vin (cuadrado) en ruido triangular (acoplamiento capacitivo)
        if not hasattr(self, 'V_tri'):
            self.V_tri = 2.5
        # Constante de tiempo parasitaria ~5ms
        self.V_tri += (Vin - self.V_tri) * (dt / 0.005)
        
        # El crosstalk toma esa forma triangular y la ajustamos para rebotar entre -0.02 y 0.06
        crosstalk = (self.V_tri - 2.5) * 0.016
        offset = 0.02  # Nivel medio
        self.Vout = Vout_ideal + crosstalk + offset

        # Acumular carga de entrada
        I_in = (Vin - self.V) / self.params.R_s
        self._charge_accumulated += abs(I_in) * dt

        # 3. Derivadas temporales
        # dVc/dt = ((Vin - Vc)/Rs - Vc/(R_tsm + R0)) / C
        dVc_dt = (I_in - I_out) / self.params.C

        # Lógica física del TSM: histéresis basada en V_hold
        if self.w < 0.5:
            # Estado OFF: dispara solo si supera el umbral
            if self.V > self.params.V_th:
                dw_dt = self.params.alpha * (self.V - self.params.V_th) * (1.0 - self.w)
            else:
                dw_dt = -self.params.beta * self.w
        else:
            # Estado ON: se mantiene prendido y descargando el capacitor hasta caer bajo V_hold
            if self.V < self.params.V_hold:
                dw_dt = -self.params.beta * self.w
            else:
                dw_dt = self.params.alpha * (self.V - self.params.V_hold) * (1.0 - self.w)

        # 4. Integración por Forward Euler
        self.V += dVc_dt * dt
        
        # Asegurar límites físicos de V (no puede ser menor que Vin si Vin=0 en decaimiento)
        if Vin == 0.0 and self.V < 0.0:
            self.V = 0.0
            
        self.w = np.clip(self.w + dw_dt * dt, 0.0, 1.0)

        # 5. Detección de flanco de subida de spike (w cruza 0.5 hacia arriba)
        fired = (self.w >= 0.5) and (self._w_prev < 0.5)
        if fired:
            self._spike_count += 1

        self._w_prev = self.w

        return fired, self.V

    def reset(self, V_init: Optional[float] = None):
        """Reinicia la neurona al estado inicial."""
        self.V = V_init if V_init is not None else 0.0
        self.w = self.params.w_init
        self.R_tsm = self.params.R_off
        self.Vout = 0.0
        self._spike_count = 0
        self._w_prev = self.params.w_init
        self._charge_accumulated = 0.0

    def update_parameters(self, new_params: LIFParameters):
        self.params = new_params
        self.R_tsm = new_params.R_off * (1.0 - self.w) + new_params.R_on * self.w

    def dvdt(self, V: float, Vin: float) -> float:
        """Calcula dVc/dt sin avanzar el estado (para análisis)."""
        R_curr = self.params.R_off * (1.0 - self.w) + self.params.R_on * self.w
        return ((Vin - V) / self.params.R_s - V / (R_curr + self.params.R_0)) / self.params.C

    def __repr__(self) -> str:
        return (
            f"TSMLIFNeuron("
            f"Vc={self.V:.2f} V, "
            f"w={self.w:.3f}, "
            f"Vout={self.Vout:.2f} V, "
            f"spikes={self._spike_count})"
        )

