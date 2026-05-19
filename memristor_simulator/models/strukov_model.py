"""
models/strukov_model.py
=======================
Implementación del modelo físico de Strukov et al. (2008).

Ecuaciones fundamentales (Ecs. 5–7 del paper):
    V(t) = [R_on · x(t) + R_off · (1 − x(t))] · I(t)
    dx/dt = (μ_v · R_on / D²) · I(t)  · f(x)
    M(q) = R_off · [1 − (μ_v·R_on/D²) · q(t)]   (memristancia analítica)

La función ventana f(x) previene que x salga de [0, 1]:
    f(x) = 1 − (2x − 1)^{2p}    (Biolek simplificada, p = 5 en esta impl.)
"""

import numpy as np
from ..config.parameters import StrukovParameters


# ─────────────────────────────────────────────────────────────────────────────
# Funciones puras del modelo (sin estado — usables en RK4, tests, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_resistance(x: float, r_on: float, r_off: float) -> float:
    """
    R(x) = R_on · x + R_off · (1 − x)    [Ec. 5 del paper]

    Parámetros
    ----------
    x : float
        Variable de estado normalizada w/D ∈ [0, 1].
    r_on : float
        Resistencia ON del dispositivo (Ω).
    r_off : float
        Resistencia OFF del dispositivo (Ω).

    Retorna
    -------
    float : Resistencia instantánea (Ω).
    """
    return r_on * x + r_off * (1.0 - x)


def dxdt_strukov(current: float, r_on: float, d: float, mu_v: float) -> float:
    """
    dx/dt = (μ_v · R_on / D²) · I(t)    [Ec. 6 del paper]

    Parámetros
    ----------
    current : float
        Corriente instantánea (A).
    r_on : float
        Resistencia ON (Ω).
    d : float
        Espesor de la película (m).
    mu_v : float
        Movilidad iónica media (m²/V·s).

    Retorna
    -------
    float : Tasa de cambio de la variable de estado (s⁻¹).
    """
    return (mu_v * r_on / d**2) * current


def window_biolek(x: float, p: int = 5) -> float:
    """
    Función ventana de Biolek simplificada:
        f(x) = 1 − (2x − 1)^{2p}

    Garantiza que dx/dt → 0 cuando x → 0 o x → 1, evitando saturación
    física de la variable de estado fuera del intervalo [0, 1].

    Parámetros
    ----------
    x : float
        Variable de estado normalizada ∈ [0, 1].
    p : int
        Parámetro de forma (mayor p → transición más abrupta en bordes).

    Retorna
    -------
    float ∈ [0, 1]
    """
    return 1.0 - (2.0 * x - 1.0) ** (2 * p)


def memristance_analytical(q: float, r_on: float, r_off: float,
                            mu_v: float, d: float) -> float:
    """
    Memristancia analítica M(q) del paper (Ec. 8 simplificada):
        M(q) = R_off · [1 − (μ_v·R_on/D²) · q]

    Solo válida en el régimen lineal (sin función ventana).

    Parámetros
    ----------
    q : float
        Carga eléctrica acumulada (C = A·s).
    """
    return r_off * (1.0 - (mu_v * r_on / d**2) * q)


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal: memristor con estado interno
# ─────────────────────────────────────────────────────────────────────────────

class StrukovMemristor:
    """
    Memristor Strukov (2008) con estado interno integrable paso a paso.

    El estado x = w/D representa la fracción de la película dopada.
    La integración usa el método de Euler explícito; para mayor precisión
    véase utils.integrators.rk4_step() que opera sobre las funciones puras.

    Ejemplo de uso::

        from memristor_simulator.config.parameters import fig2b_params
        from memristor_simulator.models.strukov_model import StrukovMemristor
        import numpy as np

        params = fig2b_params()
        device = StrukovMemristor(params)

        dt = 1e-4
        for t in np.arange(0, 4.0, dt):
            v = np.sin(2 * np.pi * 0.5 * t)
            i = device.step(v, dt)
    """

    def __init__(self, params: StrukovParameters):
        self.params = params
        self.x: float = params.w_init          # Variable de estado x = w/D
        self.resistance: float = 0.0
        self.temperature: float = params.T_amb  # Para módulo térmico
        self._q_accumulated: float = 0.0        # Carga acumulada (A·s)
        self._recalculate_resistance()

    # ── Propiedades de solo lectura ───────────────────────────────────────

    @property
    def memristance(self) -> float:
        """Memristancia actual M(q) ≡ R(x) (Ω)."""
        return self.resistance

    @property
    def state(self) -> float:
        """Variable de estado normalizada x = w/D ∈ [0, 1]."""
        return self.x

    @property
    def charge_accumulated(self) -> float:
        """Carga eléctrica total acumulada (C)."""
        return self._q_accumulated

    # ── Integración ───────────────────────────────────────────────────────

    def step(self, voltage: float, dt: float) -> float:
        """
        Avanza un paso de tiempo Δt bajo voltaje aplicado v(t).

        Algoritmo
        ---------
        1. Calcular corriente: I = V / R(x)
        2. Calcular dx/dt con función ventana opcional
        3. Actualizar x con integración de Euler (clip a [0, 1])
        4. Actualizar temperatura si módulo térmico activo
        5. Recalcular R(x)

        Parámetros
        ----------
        voltage : float
            Voltaje aplicado v(t) en voltios.
        dt : float
            Paso de tiempo en segundos.

        Retorna
        -------
        float : Corriente I(t) en amperios.
        """
        # 1. Corriente (protección anti-div/0)
        r = max(self.resistance, 1.0)
        current = voltage / r

        # 2. Tasa de cambio de estado
        dxdt = dxdt_strukov(current, self.params.R_on, self.params.D, self.params.mu_v)

        # Aplicar función ventana de Biolek (drift no lineal en bordes)
        if self.params.enable_nonlinear_drift:
            dxdt *= window_biolek(self.x, p=5)

        # 3. Integración Euler + hard-switching
        x_new = self.x + dxdt * dt
        if self.params.enable_hard_switching:
            x_new = np.clip(x_new, 0.0, 1.0)
        else:
            x_new = np.clip(x_new, 0.0, 1.0)  # siempre garantizamos físico
        self.x = x_new

        # Acumular carga
        self._q_accumulated += abs(current) * dt

        # 4. Módulo térmico (Joule Heating)
        if self.params.enable_thermal:
            power = (voltage ** 2) / r
            heat_loss = (self.temperature - self.params.T_amb) / self.params.R_th
            dT = (power - heat_loss) / self.params.C_th
            self.temperature += dT * dt

        # 5. Recalcular resistencia
        self._recalculate_resistance()

        return current

    # ── Métodos internos ──────────────────────────────────────────────────

    def _recalculate_resistance(self):
        """Actualiza self.resistance con R(x) y corrección térmica."""
        r = calculate_resistance(self.x, self.params.R_on, self.params.R_off)
        if self.params.enable_thermal:
            r *= (1.0 + self.params.alpha_T * (self.temperature - self.params.T_amb))
        self.resistance = r

    def reset(self, x0: float = None):
        """Reinicia el estado del dispositivo."""
        self.x = x0 if x0 is not None else self.params.w_init
        self.temperature = self.params.T_amb
        self._q_accumulated = 0.0
        self._recalculate_resistance()

    def calculate_state_derivative(self, voltage: float) -> float:
        """dx/dt para un voltaje dado (sin avanzar el estado)."""
        r = max(self.resistance, 1.0)
        current = voltage / r
        dxdt = dxdt_strukov(current, self.params.R_on, self.params.D, self.params.mu_v)
        if self.params.enable_nonlinear_drift:
            dxdt *= window_biolek(self.x)
        return dxdt

    def update_parameters(self, new_params: StrukovParameters):
        """Actualiza parámetros en caliente (usado por variabilidad C2C)."""
        self.params = new_params
        self._recalculate_resistance()

    def __repr__(self) -> str:
        return (f"StrukovMemristor(x={self.x:.4f}, "
                f"R={self.resistance:.1f} Ohm, "
                f"T={self.temperature:.2f} K)")
