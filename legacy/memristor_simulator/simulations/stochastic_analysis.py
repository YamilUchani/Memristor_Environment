"""
simulations/stochastic_analysis.py
=====================================
Análisis estadístico de variabilidad ciclo-a-ciclo (C2C).

Objetivo 2 del Taller de Grado I (80%):
  - Simula N ciclos con variabilidad C2C en R_on y R_off
  - Calcula métricas estadísticas: media, std, R², coeficiente de variacion
  - Genera distribuciones de parámetros y corriente máxima
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from ..config.parameters import StrukovParameters, fig2b_params
from ..config.simulation_settings import SimulationSettings, WaveformType
from ..models.stochastic_model import StochasticMemristor, C2CConfig
from ..utils.integrators import generate_waveform_scalar
from ..validation.metrics import calculate_r_squared
from .iv_characterization import SimulationResult


@dataclass
class StochasticResult:
    """Resultado del análisis estadístico C2C."""
    cycles: List[SimulationResult] = field(default_factory=list)
    r_on_samples: List[float] = field(default_factory=list)
    r_off_samples: List[float] = field(default_factory=list)
    i_max_samples: List[float] = field(default_factory=list)

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)

    @property
    def r_on_mean(self) -> float:
        return float(np.mean(self.r_on_samples)) if self.r_on_samples else 0.0

    @property
    def r_on_std(self) -> float:
        return float(np.std(self.r_on_samples)) if self.r_on_samples else 0.0

    @property
    def r_on_cv(self) -> float:
        """Coeficiente de variacion real (sigma/mu)."""
        return self.r_on_std / self.r_on_mean if self.r_on_mean > 0 else 0.0

    @property
    def i_max_mean_mA(self) -> float:
        return float(np.mean(self.i_max_samples) * 1e3) if self.i_max_samples else 0.0

    @property
    def i_max_std_mA(self) -> float:
        return float(np.std(self.i_max_samples) * 1e3) if self.i_max_samples else 0.0

    def compute_r_squared(self) -> float:
        """R^2 de la corriente simulada vs media de todos los ciclos."""
        if len(self.cycles) < 2:
            return float("nan")
        i_mean = np.mean([c.current for c in self.cycles], axis=0)
        r2_list = []
        for c in self.cycles:
            r2 = calculate_r_squared(i_mean, c.current)
            r2_list.append(r2)
        return float(np.mean(r2_list))


class StochasticAnalysis:
    """
    Análisis de variabilidad C2C del memristor Strukov.

    Simula N ciclos independientes con distribuciones log-normales
    para R_on y R_off, permitiendo cuantificar:
      - Márgenes de error de corriente (sigma_I)
      - Distribución de resistencias
      - Coeficiente de determinacion R^2

    Uso::

        sa = StochasticAnalysis(fig2b_params(), n_cycles=50)
        result = sa.run()
        sa.print_stats(result)
    """

    def __init__(self, params: StrukovParameters = None,
                 c2c_config: C2CConfig = None,
                 n_cycles: int = 30,
                 frequency: float = 0.5,
                 amplitude: float = 1.0,
                 signal_cycles: int = 3):
        self.params = params or fig2b_params()
        self.c2c = c2c_config or C2CConfig(enabled=True, r_on_cv=0.15, seed=42)
        self.n_cycles = n_cycles
        self.frequency = frequency
        self.amplitude = amplitude
        self.signal_cycles = signal_cycles  # periodos por cada ciclo C2C

    def run(self, verbose: bool = True) -> StochasticResult:
        """
        Ejecuta el análisis estadístico C2C completo.

        Cada ciclo es una simulacion independiente de 1 ciclo completo
        con parametros re-muestreados desde su distribucion.
        """
        if verbose:
            print("\n" + "="*60)
            print(f"  ANALISIS ESTOCASTICO C2C — {self.n_cycles} ciclos")
            print(f"  R_on CV = {self.c2c.r_on_cv*100:.0f}%  |  "
                  f"R_off CV = {self.c2c.r_off_cv*100:.0f}%")
            print("="*60)

        duration = self.signal_cycles / self.frequency  # Ej: 3 periodos = 6 s
        dt = 1e-4
        n_steps = int(duration / dt)

        stoch_result = StochasticResult()
        np.random.seed(self.c2c.seed)

        for cyc in range(self.n_cycles):
            # Re-muestrear parametros para este ciclo
            from ..models.stochastic_model import sample_r_on, sample_r_off
            import copy

            new_params = copy.deepcopy(self.params)
            r_on_s = sample_r_on(self.params.R_on, self.c2c.r_on_cv,
                                  self.c2c.use_lognormal)
            r_off_s = sample_r_off(self.params.R_off, self.c2c.r_off_cv,
                                    r_on_s, self.c2c.use_lognormal)
            new_params.R_on = r_on_s
            new_params.R_off = r_off_s

            device = StochasticMemristor(new_params, c2c_config=C2CConfig(enabled=False))

            t_arr = np.zeros(n_steps)
            v_arr = np.zeros(n_steps)
            i_arr = np.zeros(n_steps)
            r_arr = np.zeros(n_steps)
            x_arr = np.zeros(n_steps)

            for k in range(n_steps):
                t = k * dt
                v = generate_waveform_scalar(t, WaveformType.SINE,
                                              self.amplitude, self.frequency)
                i = device.step(v, dt)
                t_arr[k] = t
                v_arr[k] = v
                i_arr[k] = i
                r_arr[k] = device.resistance
                x_arr[k] = device.x

            s = SimulationSettings(waveform=WaveformType.SINE,
                                   amplitude=self.amplitude,
                                   frequency=self.frequency,
                                   dt=dt, duration=duration)
            cycle_result = SimulationResult(
                time=t_arr, voltage=v_arr, current=i_arr,
                resistance=r_arr, state_variable=x_arr,
                params=new_params, settings=s,
                label=f"Ciclo {cyc+1}"
            )
            stoch_result.cycles.append(cycle_result)
            stoch_result.r_on_samples.append(r_on_s)
            stoch_result.r_off_samples.append(r_off_s)
            stoch_result.i_max_samples.append(float(np.abs(i_arr).max()))

        if verbose:
            self.print_stats(stoch_result)

        return stoch_result

    @staticmethod
    def print_stats(result: StochasticResult):
        """Imprime resumen estadístico."""
        r2 = result.compute_r_squared()
        print(f"\n  Estadísticas C2C ({result.n_cycles} ciclos):")
        print(f"    R_on media    : {result.r_on_mean:.2f} Ohm")
        print(f"    R_on std      : {result.r_on_std:.2f} Ohm")
        print(f"    R_on CV real  : {result.r_on_cv*100:.2f} %")
        print(f"    I_max media   : {result.i_max_mean_mA:.3f} mA")
        print(f"    I_max std     : {result.i_max_std_mA:.3f} mA")
        print(f"    R^2 ciclos    : {r2:.4f}")
        print()
