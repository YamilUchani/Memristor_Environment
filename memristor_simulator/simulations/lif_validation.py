"""
simulations/lif_validation.py
==============================
Simulación de validación del modelo físico de la neurona TSM-LIF.

Genera cuatro experimentos que demuestran cada capacidad física:
  EXP-1: Rampa de voltaje    → Vc aumenta con Vin hasta conmutar.
  EXP-2: Umbral (threshold)  → Vin subumbral (0.5V) vs sobreumbral (5V).
  EXP-3: Spike físico único  → subida, disparo, descarga rápida y lenta.
  EXP-4: Tren de spikes      → disparos periódicos bajo Vin constante de 5V.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ..models.lif_neuron import LIFNeuron, LIFParameters, default_lif_params


# ─────────────────────────────────────────────────────────────────────────────
# Estructura de resultado por experimento
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LIFExperimentResult:
    """
    Resultado de un experimento de validación TSM-LIF.
    """
    label:       str
    time:        np.ndarray
    voltage:     np.ndarray     # Vc [V]
    current:     np.ndarray     # Vin [V] (usado para mantener compatibilidad de nombres)
    vout:        np.ndarray     # Vout [V]
    w:           np.ndarray     # w [0, 1]
    spike_times: np.ndarray
    params:      LIFParameters

    @property
    def voltage_mV(self) -> np.ndarray:
        """Potencial Vc en mV (usado por graficador antiguo o adaptado)."""
        return self.voltage * 1e3

    @property
    def current_nA(self) -> np.ndarray:
        """En el modelo físico, devuelve Vin en voltios."""
        return self.current

    @property
    def n_spikes(self) -> int:
        return len(self.spike_times)

    @property
    def mean_firing_rate_Hz(self) -> float:
        T = self.time[-1] - self.time[0]
        return self.n_spikes / T if T > 0 else 0.0


@dataclass
class LIFValidationResults:
    ramp:      LIFExperimentResult
    threshold: LIFExperimentResult
    single:    LIFExperimentResult
    train:     LIFExperimentResult
    comparison: LIFExperimentResult


# ─────────────────────────────────────────────────────────────────────────────
# Motor de validación
# ─────────────────────────────────────────────────────────────────────────────

def _run_simulation(neuron: LIFNeuron,
                    vin_array: np.ndarray,
                    dt: float,
                    label: str) -> LIFExperimentResult:
    n = len(vin_array)
    neuron.reset()

    t_arr  = np.zeros(n)
    v_arr  = np.zeros(n)
    vout_arr = np.zeros(n)
    w_arr  = np.zeros(n)
    spikes = []

    for k in range(n):
        t = k * dt
        fired, V = neuron.step(vin_array[k], dt)
        t_arr[k] = t
        v_arr[k] = V
        vout_arr[k] = neuron.Vout
        w_arr[k] = neuron.w
        if fired:
            spikes.append(t)

    return LIFExperimentResult(
        label       = label,
        time        = t_arr,
        voltage     = v_arr,
        current     = vin_array.copy(),  # Vin
        vout        = vout_arr,
        w           = w_arr,
        spike_times = np.array(spikes),
        params      = neuron.params,
    )


class LIFValidation:
    """
    Batería de cuatro experimentos de validación física del modelo TSM-LIF.
    """

    def __init__(self,
                 params: Optional[LIFParameters] = None,
                 dt: float = 0.05e-3):
        self.params = params or default_lif_params()
        self.dt = dt

    # ── EXP-1: Rampa de voltaje ─────────────────────────────────────────────

    def _exp1_ramp(self) -> LIFExperimentResult:
        """Rampa lineal de Vin: de 0 a 5 V en 500 ms."""
        T = 0.50
        n = int(T / self.dt)
        vin = np.linspace(0.0, 5.0, n)
        neuron = LIFNeuron(self.params)
        return _run_simulation(neuron, vin, self.dt, "EXP-1: Rampa de voltaje (0 -> 5 V)")

    # ── EXP-2: Umbral (threshold) ─────────────────────────────────────────────

    def _exp2_threshold(self) -> LIFExperimentResult:
        """Vin subumbral (0.5V) y Vin sobreumbral (5V)."""
        T_total = 0.50
        T_sub   = 0.20
        T_pause = 0.05
        
        n_total = int(T_total / self.dt)
        n_sub   = int(T_sub / self.dt)
        n_pause = int(T_pause / self.dt)

        vin = np.zeros(n_total)
        vin[:n_sub] = 0.5                      # 0.5 V - no conmutará
        vin[n_sub:n_sub+n_pause] = 0.0         # pausa
        vin[n_sub+n_pause:] = 5.0              # 5.0 V - provocará conmutación

        neuron = LIFNeuron(self.params)
        return _run_simulation(neuron, vin, self.dt, "EXP-2: Vin subumbral (0.5 V) vs sobreumbral (5 V)")

    # ── EXP-3: Spike único ────────────────────────────────────────────────────

    def _exp3_single_spike(self) -> LIFExperimentResult:
        """Pulso corto de Vin para generar exactamente un disparo físico."""
        T = 0.30
        T_pulse = 0.120   # lo suficientemente largo para conmutar una vez
        
        n = int(T / self.dt)
        n_pulse = int(T_pulse / self.dt)

        vin = np.zeros(n)
        vin[:n_pulse] = 5.0

        neuron = LIFNeuron(self.params)
        return _run_simulation(neuron, vin, self.dt, "EXP-3: Spike fisico unico")

    # ── EXP-4: Tren de spikes ─────────────────────────────────────────────────
    def _exp4_spike_train(self) -> LIFExperimentResult:
        """Vin constante sostenido de 5 V en 1.5 segundos para observar el tren de disparos."""
        T = 1.50
        n = int(T / self.dt)
        vin = np.full(n, 5.0)
        neuron = LIFNeuron(self.params)
        return _run_simulation(neuron, vin, self.dt, "EXP-4: Tren de spikes (Vin = 5 V constante)")

    # ── EXP-5: Comparación con Datos Experimentales ─────────────────────────
    def _exp5_experimental_comparison(self) -> LIFExperimentResult:
        """EXP-5 — Comparación directa con datos experimentales (onda cuadrada a 100 Hz)."""
        T = 0.50
        n = int(T / self.dt)
        t = np.arange(n) * self.dt
        f_in = 100.0
        vin = np.where((t * f_in) % 1.0 < 0.5, 5.0, 0.0)
        neuron = LIFNeuron(self.params)
        return _run_simulation(neuron, vin, self.dt, "EXP-5: Comparacion Experimental (100 Hz)")

    def run(self, verbose: bool = True) -> LIFValidationResults:
        if verbose:
            print("\n" + "="*60)
            print("  VALIDACION TSM-LIF — Neurona Fisiológica")
            print("="*60)
            print(self.params)
            print(f"  dt = {self.dt*1e3:.3f} ms\n")

        experiments = {
            "ramp":      self._exp1_ramp,
            "threshold": self._exp2_threshold,
            "single":    self._exp3_single_spike,
            "train":     self._exp4_spike_train,
            "comparison": self._exp5_experimental_comparison,
        }

        results_dict = {}
        for name, fn in experiments.items():
            res = fn()
            results_dict[name] = res
            if verbose:
                _print_experiment_summary(res)

        return LIFValidationResults(**results_dict)

def _print_experiment_summary(res: LIFExperimentResult):
    p = res.params
    print(f"  [{res.label}]")
    print(f"    Duración       : {res.time[-1]*1000:.1f} ms")
    print(f"    Vin rango      : [{res.current.min():.2f}, {res.current.max():.2f}] V")
    print(f"    Vc rango       : [{res.voltage.min():.2f}, {res.voltage.max():.2f}] V")
    print(f"    V_th / V_hold  : {p.V_th:.2f} / {p.V_hold:.2f} V")
    print(f"    Vout máx pico  : {res.vout.max():.3f} V")
    print(f"    Spikes         : {res.n_spikes}")
    if res.n_spikes > 0:
        print(f"    Primer spike   : {res.spike_times[0]*1000:.2f} ms")
        print(f"    Freq. media    : {res.mean_firing_rate_Hz:.2f} Hz")
    print()

