"""
simulations/lif_validation.py
==============================
Simulación de validación aislada de la neurona LIF.

Genera cuatro experimentos que demuestran cada capacidad fundamental:

  EXP-1: Rampa de corriente  → V aumenta con la corriente de entrada.
  EXP-2: Umbral (threshold)  → debajo no dispara, encima sí.
  EXP-3: Spike completo      → disparo + reset observados en el tiempo.
  EXP-4: Trenes de spikes    → múltiples disparos con corriente sostenida.

Uso::

    from memristor_simulator.simulations.lif_validation import LIFValidation
    results = LIFValidation().run(verbose=True)

Referencia:
    Gerstner & Kistler (2002) — Spiking Neuron Models, Cap. 4.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ..models.lif_neuron import LIFNeuron, LIFParameters, default_lif_params


# ─────────────────────────────────────────────────────────────────────────────
# Estructura de resultado por experimento
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LIFExperimentResult:
    """
    Resultado de un experimento de validación LIF.

    Atributos
    ---------
    label : str
        Nombre descriptivo del experimento.
    time : np.ndarray
        Vector de tiempo [s].
    voltage : np.ndarray
        Potencial de membrana V(t) [V].
    current : np.ndarray
        Corriente de entrada I(t) [A].
    spike_times : np.ndarray
        Instantes donde ocurrió un spike [s].
    params : LIFParameters
        Parámetros de la neurona usados.
    """
    label:       str
    time:        np.ndarray
    voltage:     np.ndarray
    current:     np.ndarray
    spike_times: np.ndarray
    params:      LIFParameters

    @property
    def voltage_mV(self) -> np.ndarray:
        """Potencial de membrana en mV."""
        return self.voltage * 1e3

    @property
    def current_nA(self) -> np.ndarray:
        """Corriente en nanoamperios."""
        return self.current * 1e9

    @property
    def n_spikes(self) -> int:
        """Número total de spikes registrados."""
        return len(self.spike_times)

    @property
    def mean_firing_rate_Hz(self) -> float:
        """Frecuencia media de disparo [Hz]."""
        T = self.time[-1] - self.time[0]
        return self.n_spikes / T if T > 0 else 0.0


@dataclass
class LIFValidationResults:
    """Colección de los cuatro experimentos de validación."""
    ramp:      LIFExperimentResult   # EXP-1: rampa de corriente
    threshold: LIFExperimentResult   # EXP-2: umbral
    single:    LIFExperimentResult   # EXP-3: spike único
    train:     LIFExperimentResult   # EXP-4: tren de spikes


# ─────────────────────────────────────────────────────────────────────────────
# Motor de validación
# ─────────────────────────────────────────────────────────────────────────────

def _run_simulation(neuron: LIFNeuron,
                    current_array: np.ndarray,
                    dt: float,
                    label: str) -> LIFExperimentResult:
    """
    Ejecuta la integración numérica paso a paso para un perfil de corriente dado.

    Parámetros
    ----------
    neuron : LIFNeuron
        Neurona a simular (se reinicia al inicio).
    current_array : np.ndarray
        Corriente de entrada I(t) [A], un valor por paso de tiempo.
    dt : float
        Paso de tiempo [s].
    label : str
        Etiqueta descriptiva del experimento.

    Retorna
    -------
    LIFExperimentResult con todos los arrays poblados.
    """
    n = len(current_array)
    neuron.reset()

    t_arr  = np.zeros(n)
    v_arr  = np.zeros(n)
    spikes = []

    for k in range(n):
        t = k * dt
        fired, V = neuron.step(current_array[k], dt)
        t_arr[k] = t
        v_arr[k] = V
        if fired:
            spikes.append(t)

    return LIFExperimentResult(
        label       = label,
        time        = t_arr,
        voltage     = v_arr,
        current     = current_array.copy(),
        spike_times = np.array(spikes),
        params      = neuron.params,
    )


class LIFValidation:
    """
    Batería de cuatro experimentos de validación de la neurona LIF.

    Los experimentos están diseñados para demostrar de forma independiente
    cada mecanismo fundamental antes de integrar la neurona con el memristor.

    Parámetros
    ----------
    params : LIFParameters, opcional
        Parámetros de la neurona. Por defecto = default_lif_params().
    dt : float
        Paso de tiempo [s]. Por defecto 0.1 ms.
    """

    def __init__(self,
                 params: Optional[LIFParameters] = None,
                 dt: float = 0.1e-3):
        self.params = params or default_lif_params()
        self.dt = dt

    # ── EXP-1: Rampa de corriente ─────────────────────────────────────────────

    def _exp1_ramp(self) -> LIFExperimentResult:
        """
        EXP-1 — Rampa de corriente.

        Aplica una corriente que aumenta linealmente de 0 a 6 nA en 300 ms.
        Demuestra que V crece a mayor velocidad cuando más corriente entra,
        y que la neurona dispara antes cuanto más fuerte es la entrada.
        """
        T = 0.30           # 300 ms
        I_max = 6e-9       # 6 nA
        n = int(T / self.dt)
        t_arr = np.arange(n) * self.dt
        # Rampa lineal: I aumenta de 0 a I_max
        current = np.linspace(0.0, I_max, n)

        neuron = LIFNeuron(self.params)
        return _run_simulation(neuron, current, self.dt,
                               "EXP-1: Rampa de corriente (0 -> 6 nA)")

    # ── EXP-2: Umbral (threshold) ─────────────────────────────────────────────

    def _exp2_threshold(self) -> LIFExperimentResult:
        """
        EXP-2 — Comparación por encima/debajo del umbral.

        Aplica dos niveles de corriente constante separados por una pausa:
          - Segmento A (0–200 ms):   I = 1 nA  → V sube, pero no llega a V_th.
          - Pausa     (200–250 ms):  I = 0     → V decae hacia E_L.
          - Segmento B (250–500 ms): I = 4 nA  → V supera V_th → spike.

        Demuestra la existencia del umbral: el mismo circuito, dos respuestas.
        """
        T_total    = 0.50   # 500 ms
        T_sub      = 0.20   # 200 ms subthreshold
        T_pause    = 0.05   # 50 ms de pausa
        T_supra    = 0.25   # 250 ms suprathreshold
        I_sub      = 1e-9   # 1 nA  — por debajo del umbral
        I_supra    = 4e-9   # 4 nA  — por encima del umbral

        n_total = int(T_total / self.dt)
        n_sub   = int(T_sub   / self.dt)
        n_pause = int(T_pause / self.dt)

        current = np.zeros(n_total)
        current[:n_sub]              = I_sub
        current[n_sub:n_sub+n_pause] = 0.0
        current[n_sub+n_pause:]      = I_supra

        neuron = LIFNeuron(self.params)
        return _run_simulation(neuron, current, self.dt,
                               "EXP-2: Por debajo / por encima del umbral")

    # ── EXP-3: Spike único ────────────────────────────────────────────────────

    def _exp3_single_spike(self) -> LIFExperimentResult:
        """
        EXP-3 — Spike único con zoom en el evento.

        Aplica un pulso de corriente corto (30 ms) que provoca exactamente
        un disparo, seguido de silencio para ver el reset completo.

        Muestra en detalle:
          - Subida gradual de V.
          - Disparo abrupto al alcanzar V_th.
          - Reset a V_reset.
          - Decaimiento post-spike hacia E_L.
        """
        T = 0.25           # 250 ms
        T_pulse = 0.030    # 30 ms de pulso
        I_pulse = 5e-9     # 5 nA — garantiza disparo en ≈20 ms

        n = int(T / self.dt)
        n_pulse = int(T_pulse / self.dt)

        current = np.zeros(n)
        current[:n_pulse] = I_pulse   # Pulso inicial

        neuron = LIFNeuron(self.params)
        return _run_simulation(neuron, current, self.dt,
                               "EXP-3: Spike unico - disparo y reset")

    # ── EXP-4: Tren de spikes ─────────────────────────────────────────────────

    def _exp4_spike_train(self) -> LIFExperimentResult:
        """
        EXP-4 — Tren de spikes con tres niveles de corriente.

        Divide la simulación en tres segmentos con corriente constante creciente:
          - Segmento 1 (0–300 ms):    I = 2 nA  → frecuencia baja.
          - Segmento 2 (300–600 ms):  I = 4 nA  → frecuencia media.
          - Segmento 3 (600–900 ms):  I = 7 nA  → frecuencia alta.

        Demuestra la curva f-I (firing rate vs input current) de la neurona.
        """
        T_seg = 0.30       # 300 ms por segmento
        levels = [2e-9, 4e-9, 7e-9]  # nA: baja, media, alta

        segments = []
        for I_level in levels:
            n_seg = int(T_seg / self.dt)
            segments.append(np.full(n_seg, I_level))
        current = np.concatenate(segments)

        neuron = LIFNeuron(self.params)
        return _run_simulation(neuron, current, self.dt,
                               "EXP-4: Tren de spikes (3 niveles de corriente)")

    # ── Ejecutar todos ────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> LIFValidationResults:
        """
        Ejecuta los cuatro experimentos de validación en secuencia.

        Parámetros
        ----------
        verbose : bool
            Si True, imprime resumen estadístico de cada experimento.

        Retorna
        -------
        LIFValidationResults con los cuatro experimentos.
        """
        if verbose:
            print("\n" + "="*60)
            print("  VALIDACION LIF — Neurona Aislada")
            print("  Paso 4 del proyecto de Computacion Neurormorfica")
            print("="*60)
            print(self.params)
            print(f"  dt = {self.dt*1e3:.2f} ms\n")

        experiments = {
            "ramp":      self._exp1_ramp,
            "threshold": self._exp2_threshold,
            "single":    self._exp3_single_spike,
            "train":     self._exp4_spike_train,
        }

        results_dict = {}
        for name, fn in experiments.items():
            res = fn()
            results_dict[name] = res
            if verbose:
                _print_experiment_summary(res)

        if verbose:
            print("\n" + "="*60)
            print("  [OK] Validacion completada.")
            print("="*60)

        return LIFValidationResults(**results_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _print_experiment_summary(res: LIFExperimentResult):
    """Imprime un resumen estadístico de un experimento."""
    p = res.params
    print(f"  [{res.label}]")
    print(f"    Duracion       : {res.time[-1]*1e3:.0f} ms")
    print(f"    I entrada      : [{res.current_nA.min():.2f}, "
          f"{res.current_nA.max():.2f}] nA")
    print(f"    V rango        : [{res.voltage_mV.min():.2f}, "
          f"{res.voltage_mV.max():.2f}] mV")
    print(f"    V_th           : {p.V_th*1e3:.2f} mV")
    print(f"    Spikes         : {res.n_spikes}")
    if res.n_spikes > 0:
        print(f"    Primer spike   : {res.spike_times[0]*1e3:.2f} ms")
        print(f"    Freq. media    : {res.mean_firing_rate_Hz:.2f} Hz")
    print()
