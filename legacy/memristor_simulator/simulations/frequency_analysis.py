"""
simulations/frequency_analysis.py
====================================
Análisis del colapso de histéresis I-V con la frecuencia.

Teorema del paper (pág. 81):
    "Any symmetrical alternating-current voltage bias results in double-loop
    i-v hysteresis that collapses to a straight line for high frequencies."
"""

import numpy as np
from typing import Dict, Tuple
from ..config.parameters import StrukovParameters, fig2b_params
from ..config.simulation_settings import SimulationSettings, WaveformType
from ..models.strukov_model import StrukovMemristor
from ..utils.integrators import generate_waveform_scalar
from ..validation.metrics import hysteresis_area
from .iv_characterization import SimulationResult


class FrequencyAnalysis:
    """
    Barrido de frecuencias para demostrar el colapso de histéresis I-V.
    Simula el mismo dispositivo a w0, 2w0, 5w0, 10w0.
    """

    DEFAULT_MULTIPLIERS = {"w0": 1, "2w0": 2, "5w0": 5, "10w0": 10}

    def __init__(self, params: StrukovParameters = None,
                 base_frequency: float = 0.5,
                 amplitude: float = 1.0,
                 n_cycles: int = 3):
        self.params = params or fig2b_params()
        self.base_frequency = base_frequency
        self.amplitude = amplitude
        self.n_cycles = n_cycles

    def _run_single(self, frequency: float, label: str,
                    verbose: bool = True) -> SimulationResult:
        duration = self.n_cycles / frequency
        dt = max(1e-5, duration / 5000)
        n_steps = int(duration / dt)

        device = StrukovMemristor(self.params)
        device.reset()

        t_arr = np.zeros(n_steps)
        v_arr = np.zeros(n_steps)
        i_arr = np.zeros(n_steps)
        r_arr = np.zeros(n_steps)
        x_arr = np.zeros(n_steps)

        for k in range(n_steps):
            t = k * dt
            v = generate_waveform_scalar(t, WaveformType.SINE,
                                          self.amplitude, frequency)
            i = device.step(v, dt)
            t_arr[k] = t
            v_arr[k] = v
            i_arr[k] = i
            r_arr[k] = device.resistance
            x_arr[k] = device.x

        delta_x = x_arr.max() - x_arr.min()
        if verbose:
            print(f"    {label:6s} = {frequency:5.1f} Hz  "
                  f"dx = {delta_x:.5f}  "
                  f"area = {hysteresis_area(v_arr, i_arr):.2e} V*A")

        s = SimulationSettings(waveform=WaveformType.SINE,
                               amplitude=self.amplitude,
                               frequency=frequency, dt=dt, duration=duration)
        return SimulationResult(time=t_arr, voltage=v_arr, current=i_arr,
                                resistance=r_arr, state_variable=x_arr,
                                params=self.params, settings=s, label=label)

    def run_sweep(self, multipliers: Dict[str, int] = None,
                  verbose: bool = True) -> Dict[str, SimulationResult]:
        """Ejecuta el barrido completo de frecuencias."""
        mults = multipliers or self.DEFAULT_MULTIPLIERS

        if verbose:
            print("\n" + "="*60)
            print("  BARRIDO DE FRECUENCIAS — Colapso de Histéresis")
            print(f"  f0 = {self.base_frequency} Hz | A = {self.amplitude} V")
            print("="*60)

        results = {}
        for label, mult in mults.items():
            results[label] = self._run_single(
                self.base_frequency * mult, label, verbose)

        if verbose:
            self.print_collapse_report(results)
        return results

    def print_collapse_report(self, results: Dict[str, SimulationResult]):
        """Imprime tabla de colapso de histéresis."""
        print(f"\n  {'Label':<8} {'f (Hz)':<10} {'dx':<12} {'Area (V*A)':<15} {'Estado'}")
        print("  " + "-"*65)
        areas = []
        for label, r in results.items():
            dx = r.state_variable.max() - r.state_variable.min()
            area = hysteresis_area(r.voltage, r.current)
            areas.append(area)
            estado = ("Histeresis clara" if dx > 0.05 else
                      "Transicion" if dx > 0.01 else "COLAPSADO OK")
            print(f"  {label:<8} {r.settings.frequency:<10.2f} "
                  f"{dx:<12.6f} {area:<15.2e} {estado}")
        if len(areas) >= 2 and areas[0] > 1e-20:
            pct = (1.0 - areas[-1] / areas[0]) * 100
            print(f"\n  Reduccion total de area: {pct:.1f}%\n")

    def get_low_high_pair(self, results) -> Tuple[SimulationResult, SimulationResult]:
        """Retorna el par (baja frecuencia, alta frecuencia)."""
        keys = list(results.keys())
        return results[keys[0]], results[keys[-1]]
