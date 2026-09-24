"""
simulations/iv_characterization.py
====================================
Caracterización I-V del memristor Strukov (2008).

Genera las Figuras 2b y 2c del paper:
  - Fig 2b: voltaje senoidal v₀·sin(ω₀t), R_off/R_on = 160
  - Fig 2c: voltaje 6v₀·sin²(ω₀t), R_off/R_on = 380

También soporta barridos con onda triangular para la validación
metodológica del Taller de Grado I.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ..config.parameters import StrukovParameters, fig2b_params, fig2c_params
from ..config.simulation_settings import (SimulationSettings, WaveformType,
                                          settings_fig2b, settings_fig2c)
from ..models.strukov_model import StrukovMemristor
from ..utils.integrators import generate_waveform_scalar


# ─────────────────────────────────────────────────────────────────────────────
# Resultado de simulación
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Resultado completo de una simulación I-V del memristor.

    Atributos
    ---------
    time : np.ndarray
        Instantes de tiempo (s).
    voltage : np.ndarray
        Voltaje aplicado v(t) (V).
    current : np.ndarray
        Corriente resultante I(t) (A).
    resistance : np.ndarray
        Resistencia instantánea R(x, t) (Ω).
    state_variable : np.ndarray
        Variable de estado normalizada x = w/D ∈ [0, 1].
    params : StrukovParameters
        Parámetros del dispositivo utilizados.
    settings : SimulationSettings
        Configuración de la simulación.
    label : str
        Etiqueta descriptiva (e.g., "Fig 2b").
    """
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    resistance: np.ndarray
    state_variable: np.ndarray
    params: StrukovParameters = field(default_factory=StrukovParameters)
    settings: SimulationSettings = field(default_factory=SimulationSettings)
    label: str = ""

    @property
    def current_mA(self) -> np.ndarray:
        """Corriente en miliamperios."""
        return self.current * 1e3

    @property
    def flux(self) -> np.ndarray:
        """Flujo magnético acumulado Φ = ∫V dt (V·s)."""
        dt = self.settings.dt
        return np.cumsum(self.voltage) * dt

    @property
    def charge(self) -> np.ndarray:
        """Carga acumulada q = ∫I dt (C)."""
        dt = self.settings.dt
        return np.cumsum(self.current) * dt

    def as_dict(self) -> dict:
        """Convierte a diccionario para exportación."""
        return {
            "time": self.time,
            "voltage": self.voltage,
            "current": self.current,
            "resistance": self.resistance,
            "state_variable": self.state_variable,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Motor de simulación I-V
# ─────────────────────────────────────────────────────────────────────────────

class IVCharacterization:
    """
    Runner de simulación I-V para el memristor Strukov (2008).

    Ejecuta la integración numérica paso a paso:
        1. Genera la forma de onda de voltaje
        2. Aplica el voltaje al dispositivo
        3. Registra todos los estados

    Uso::

        from memristor_simulator.config.parameters import fig2b_params
        from memristor_simulator.config.simulation_settings import settings_fig2b
        from memristor_simulator.simulations.iv_characterization import IVCharacterization

        sim = IVCharacterization(fig2b_params(), settings_fig2b())
        result = sim.run()
    """

    def __init__(self, params: StrukovParameters,
                 settings: SimulationSettings,
                 label: str = ""):
        self.params = params
        self.settings = settings
        self.label = label

    def run(self, verbose: bool = True) -> SimulationResult:
        """
        Ejecuta la simulación I-V completa.

        Parámetros
        ----------
        verbose : bool
            Si True, imprime progreso y estadísticas al finalizar.

        Retorna
        -------
        SimulationResult : Resultado completo de la simulación.
        """
        p = self.params
        s = self.settings
        dt = s.dt
        n_steps = s.n_steps

        if verbose:
            print(f"\n  Simulando {self.label or 'I-V'} ...")
            print(f"    dt = {dt:.1e} s | N = {n_steps} | "
                  f"f = {s.frequency} Hz | A = {s.amplitude} V")

        # Inicializar dispositivo
        device = StrukovMemristor(p)

        # Arrays de resultado (pre-alocados para eficiencia)
        t_arr = np.zeros(n_steps)
        v_arr = np.zeros(n_steps)
        i_arr = np.zeros(n_steps)
        r_arr = np.zeros(n_steps)
        x_arr = np.zeros(n_steps)

        # Loop de integración
        for k in range(n_steps):
            t = k * dt
            v = generate_waveform_scalar(t, s.waveform, s.amplitude, s.frequency)
            i = device.step(v, dt)

            t_arr[k] = t
            v_arr[k] = v
            i_arr[k] = i
            r_arr[k] = device.resistance
            x_arr[k] = device.x

        result = SimulationResult(
            time=t_arr,
            voltage=v_arr,
            current=i_arr,
            resistance=r_arr,
            state_variable=x_arr,
            params=self.params,
            settings=self.settings,
            label=self.label,
        )

        if verbose:
            self._print_stats(result)

        return result

    @staticmethod
    def _print_stats(r: SimulationResult):
        """Imprime estadísticas de la simulación."""
        print(f"    [OK] Completado:")
        print(f"      V: [{r.voltage.min():.3f}, {r.voltage.max():.3f}] V")
        print(f"      I: [{r.current_mA.min():.3f}, {r.current_mA.max():.3f}] mA")
        print(f"      x: [{r.state_variable.min():.4f}, {r.state_variable.max():.4f}]")
        print(f"      R: [{r.resistance.min():.0f}, {r.resistance.max():.0f}] Ohm")


# ─────────────────────────────────────────────────────────────────────────────
# Funciones de conveniencia para reproducir el paper directamente
# ─────────────────────────────────────────────────────────────────────────────

def run_figure_2b(verbose: bool = True) -> SimulationResult:
    """
    Reproduce la Figura 2b del paper Strukov (2008).

    Configuración:
      - Voltaje senoidal v₀·sin(ω₀t), v₀ = 1 V, f₀ = 0.5 Hz
      - R_off/R_on = 160 (R_off = 16 kΩ, R_on = 100 Ω)
      - 4 ciclos completos, dt = 0.1 ms
    """
    if verbose:
        print("\n" + "="*60)
        print("  FIGURA 2b — Senoidal, R_off/R_on = 160")
        print("="*60)
    sim = IVCharacterization(fig2b_params(), settings_fig2b(), label="Fig 2b")
    return sim.run(verbose=verbose)


def run_figure_2c(verbose: bool = True) -> SimulationResult:
    """
    Reproduce la Figura 2c del paper Strukov (2008).

    Configuración:
      - Voltaje 4·sin²(ω₀t) — unipolar, múltiples lazos
      - R_off/R_on = 380 (R_off = 38 kΩ, R_on = 100 Ω)
      - 4 ciclos completos, dt = 0.1 ms
    """
    if verbose:
        print("\n" + "="*60)
        print("  FIGURA 2c — sin², R_off/R_on = 380")
        print("="*60)
    sim = IVCharacterization(fig2c_params(), settings_fig2c(), label="Fig 2c")
    return sim.run(verbose=verbose)


def run_triangular_sweep(params: StrukovParameters = None,
                          frequency: float = 0.5,
                          amplitude: float = 1.0,
                          n_cycles: int = 4,
                          verbose: bool = True) -> SimulationResult:
    """
    Barrido I-V con onda triangular (validación metodológica Taller de Grado I).

    La onda triangular es preferida en laboratorio por su derivada constante,
    lo que produce una histéresis I-V más fácil de analizar morfológicamente.
    """
    p = params or fig2b_params()
    s = SimulationSettings(
        waveform=WaveformType.TRIANGLE,
        amplitude=amplitude,
        frequency=frequency,
        n_cycles=n_cycles,
        dt=1e-4,
    )
    if verbose:
        print("\n" + "="*60)
        print(f"  BARRIDO TRIANGULAR — f={frequency} Hz, A={amplitude} V")
        print("="*60)
    sim = IVCharacterization(p, s, label=f"Triangular {frequency} Hz")
    return sim.run(verbose=verbose)
