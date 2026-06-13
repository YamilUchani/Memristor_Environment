"""
models/lif_neuron.py
====================
Implementación de la neurona Leaky Integrate-and-Fire (LIF).

Modelo matemático (Lapicque, 1907 — formulación moderna):

    τ_m · dV/dt = -(V(t) - E_L) + R_m · I(t)        [Ec. principal]

    Si V(t) ≥ V_th  →  emitir spike  →  V ← V_reset  [Regla de disparo]

    Durante t_ref ms post-spike:  V se mantiene en V_reset [Refractario]

Discretización por Forward Euler (mismo método que strukov_model.py):

    V(t + Δt) = V(t) + (Δt / τ_m) · [-(V(t) - E_L) + R_m · I(t)]

Referencia principal:
    Gerstner & Kistler (2002) — Spiking Neuron Models, Cambridge UP.
    Capítulo 4.1: The Leaky Integrate-and-Fire Model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros del modelo LIF
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LIFParameters:
    """
    Parámetros biofísicos de la neurona LIF.

    Todos los valores en el Sistema Internacional (SI).

    Atributos
    ---------
    tau_m : float
        Constante de tiempo de membrana [s]. Determina la velocidad de
        integración. τ_m = R_m · C_m.
        Valor típico cortical: 10–20 ms.
    E_L : float
        Potencial de reposo (leak reversal potential) [V].
        Hacia donde decae V cuando no hay input. Típico: -70 mV.
    R_m : float
        Resistencia de membrana [Ω]. Amplifica la corriente de entrada.
        Típico: 10 MΩ (neuronas corticales in vitro).
    V_th : float
        Umbral de disparo [V]. Si V ≥ V_th → spike.
        Típico: -50 mV (20 mV sobre el reposo).
    V_reset : float
        Potencial de reset post-spike [V]. Ligeramente bajo el reposo.
        Típico: -65 mV.
    t_ref : float
        Período refractario absoluto [s]. Durante este tiempo V se
        fija en V_reset y no puede disparar.
        Típico: 2 ms.
    C_m : float
        Capacitancia de membrana [F]. Usada para verificar que τ_m = R_m·C_m.
        Valor derivado: C_m = τ_m / R_m.
        Típico: 200 pF.
    """
    # ── Parámetros eléctricos fundamentales ──────────────────────────────────
    tau_m:   float = 20e-3     # s      (20 ms)
    E_L:     float = -70e-3    # V      (-70 mV — reposo)
    R_m:     float = 10e6      # Ω      (10 MΩ)
    V_th:    float = -50e-3    # V      (-50 mV — umbral)
    V_reset: float = -65e-3    # V      (-65 mV — reset)
    t_ref:   float = 2e-3      # s      (2 ms — refractario)

    # ── Capacitancia (derivada de tau_m y R_m) ────────────────────────────────
    @property
    def C_m(self) -> float:
        """Capacitancia de membrana implícita: C_m = τ_m / R_m  [F]."""
        return self.tau_m / self.R_m

    # ── Validación de coherencia física ───────────────────────────────────────
    def __post_init__(self):
        if self.V_reset >= self.V_th:
            raise ValueError(
                f"V_reset ({self.V_reset:.3f} V) debe ser < V_th ({self.V_th:.3f} V)."
            )
        if self.E_L >= self.V_th:
            raise ValueError(
                f"E_L ({self.E_L:.3f} V) debe ser < V_th ({self.V_th:.3f} V)."
            )
        if self.tau_m <= 0 or self.t_ref < 0 or self.R_m <= 0:
            raise ValueError("tau_m, R_m deben ser > 0. t_ref debe ser ≥ 0.")

    def __str__(self) -> str:
        lines = [
            "+- LIFParameters ----------------------------------------+",
            f"|  tau_m         = {self.tau_m*1e3:>10.2f}  ms                    |",
            f"|  E_L           = {self.E_L*1e3:>10.2f}  mV  (reposo)           |",
            f"|  V_th          = {self.V_th*1e3:>10.2f}  mV  (umbral)           |",
            f"|  V_reset       = {self.V_reset*1e3:>10.2f}  mV  (post-spike)        |",
            f"|  R_m           = {self.R_m/1e6:>10.2f}  MOhm                  |",
            f"|  C_m           = {self.C_m*1e12:>10.2f}  pF                    |",
            f"|  t_ref         = {self.t_ref*1e3:>10.2f}  ms  (refractario)       |",
            "+---------------------------------------------------------+",
        ]
        return "\n".join(lines)


# ── Presets de fábrica ────────────────────────────────────────────────────────

def default_lif_params() -> LIFParameters:
    """
    Parámetros LIF estándar para neuronas corticales de capa 5 (in vitro).
    Basado en valores típicos de la literatura (Gerstner & Kistler, 2002).
    """
    return LIFParameters(
        tau_m   = 20e-3,    # 20 ms
        E_L     = -70e-3,   # -70 mV
        R_m     = 10e6,     # 10 MΩ
        V_th    = -50e-3,   # -50 mV
        V_reset = -65e-3,   # -65 mV
        t_ref   = 2e-3,     # 2 ms
    )


def fast_lif_params() -> LIFParameters:
    """
    Parámetros para una neurona más rápida y excitable.
    Útil para verificar comportamiento con alta frecuencia de disparo.
    """
    return LIFParameters(
        tau_m   = 10e-3,    # 10 ms — más rápida
        E_L     = -70e-3,
        R_m     = 10e6,
        V_th    = -55e-3,   # -55 mV — umbral más bajo (más excitable)
        V_reset = -65e-3,
        t_ref   = 1e-3,     # 1 ms — refractario corto
    )


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal: neurona LIF con estado interno integrable paso a paso
# ─────────────────────────────────────────────────────────────────────────────

class LIFNeuron:
    """
    Neurona Leaky Integrate-and-Fire con estado interno integrable paso a paso.

    La integración temporal usa Forward Euler (primer orden), equivalente al
    método ya aplicado en StrukovMemristor para garantizar coherencia numérica.

    Capacidades
    -----------
    - Recibir corriente externa I(t).
    - Acumular carga en la capacitancia de membrana C_m.
    - Incrementar el potencial de membrana V(t).
    - Comparar V contra el umbral V_th.
    - Generar un spike cuando V ≥ V_th.
    - Reiniciar el potencial a V_reset después del disparo.
    - Respetar el período refractario absoluto t_ref.

    Ejemplo de uso::

        from memristor_simulator.models.lif_neuron import LIFNeuron, default_lif_params

        neuron = LIFNeuron(default_lif_params())
        dt = 0.1e-3  # 0.1 ms

        spikes = []
        for t in np.arange(0, 0.5, dt):
            I_in = 3e-9  # 3 nA — corriente constante de entrada
            fired, V = neuron.step(I_in, dt)
            if fired:
                spikes.append(t)
    """

    def __init__(self, params: LIFParameters, V_init: Optional[float] = None):
        """
        Inicializa la neurona LIF.

        Parámetros
        ----------
        params : LIFParameters
            Parámetros biofísicos del modelo.
        V_init : float, opcional
            Potencial inicial [V]. Por defecto = E_L (reposo).
        """
        self.params = params
        self.V: float = V_init if V_init is not None else params.E_L

        # ── Estado interno ─────────────────────────────────────────────────
        self._t_since_spike: float = params.t_ref + 1.0  # Arranca fuera del refractario
        self._spike_count:   int   = 0
        self._charge_accumulated: float = 0.0            # C = A·s

    # ── Propiedades de solo lectura ───────────────────────────────────────────

    @property
    def membrane_potential(self) -> float:
        """Potencial de membrana actual V(t) [V]."""
        return self.V

    @property
    def membrane_potential_mV(self) -> float:
        """Potencial de membrana actual V(t) [mV]."""
        return self.V * 1e3

    @property
    def spike_count(self) -> int:
        """Número total de spikes emitidos desde la creación o último reset."""
        return self._spike_count

    @property
    def is_refractory(self) -> bool:
        """True si la neurona está en período refractario (no puede disparar)."""
        return self._t_since_spike < self.params.t_ref

    @property
    def charge_accumulated(self) -> float:
        """Carga eléctrica total integrada desde la creación [C]."""
        return self._charge_accumulated

    # ── Integración principal ─────────────────────────────────────────────────

    def step(self, current: float, dt: float) -> tuple[bool, float]:
        """
        Avanza un paso de tiempo Δt bajo corriente de entrada I(t).

        Algoritmo
        ---------
        1. Si en período refractario: mantener V = V_reset, retornar False.
        2. Integrar la ecuación diferencial por Forward Euler:
               V(t+Δt) = V(t) + (Δt/τ_m)·[-(V(t) - E_L) + R_m·I(t)]
        3. Comparar V contra V_th:
               Si V ≥ V_th → registrar spike, V ← V_reset, iniciar refractario.

        Parámetros
        ----------
        current : float
            Corriente de entrada I(t) [A]. Puede ser positiva (excitadora)
            o negativa (inhibidora).
        dt : float
            Paso de tiempo [s]. Debe satisfacer dt << τ_m para estabilidad
            numérica (recomendado: dt ≤ τ_m / 100).

        Retorna
        -------
        fired : bool
            True si la neurona disparó un spike en este paso.
        V : float
            Potencial de membrana al final del paso [V].
        """
        # 1. Acumular carga (independiente del estado refractario)
        self._charge_accumulated += abs(current) * dt
        self._t_since_spike += dt

        # 2. Período refractario: neurona "bloqueada"
        if self._t_since_spike < self.params.t_ref:
            self.V = self.params.V_reset
            return False, self.V

        # 3. Integración Forward Euler de la ecuación LIF
        #    dV/dt = (1/τ_m) · [-(V - E_L) + R_m · I(t)]
        p = self.params
        dV = (dt / p.tau_m) * (-(self.V - p.E_L) + p.R_m * current)
        self.V += dV

        # 4. Condición de disparo
        if self.V >= p.V_th:
            self._spike_count += 1
            self.V = p.V_reset
            self._t_since_spike = 0.0
            return True, self.V

        return False, self.V

    # ── Control de estado ─────────────────────────────────────────────────────

    def reset(self, V_init: Optional[float] = None):
        """
        Reinicia la neurona al estado inicial.

        Parámetros
        ----------
        V_init : float, opcional
            Potencial inicial [V]. Por defecto = E_L (reposo).
        """
        self.V = V_init if V_init is not None else self.params.E_L
        self._t_since_spike = self.params.t_ref + 1.0
        self._spike_count = 0
        self._charge_accumulated = 0.0

    def update_parameters(self, new_params: LIFParameters):
        """Actualiza parámetros en caliente (para estudios paramétricos)."""
        self.params = new_params

    # ── Ecuación diferencial sin avanzar el estado (para RK4, análisis) ──────

    def dvdt(self, V: float, current: float) -> float:
        """
        Calcula dV/dt para un voltaje y corriente dados, sin avanzar el estado.

        Útil para integración de orden superior (RK4) o análisis de
        trayectorias en el plano de fase.

        Parámetros
        ----------
        V : float
            Potencial de membrana [V].
        current : float
            Corriente de entrada [A].

        Retorna
        -------
        float : dV/dt [V/s]
        """
        p = self.params
        return (-(V - p.E_L) + p.R_m * current) / p.tau_m

    # ── Representación ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"LIFNeuron("
            f"V={self.V*1e3:.2f} mV, "
            f"spikes={self._spike_count}, "
            f"refractory={self.is_refractory})"
        )
