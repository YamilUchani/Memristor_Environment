"""
tests.test_lif_hybrid
======================
Regresión: verifica que los flujos de la GUI (Pestaña 2 y 3) disparan spikes con
los valores por defecto coherentes (C=100 nF, R=1 MΩ, V_th=0.95, V_reset=0.15,
t_ref=2 ms) usando los circuitos reales de neurolab.circuits.hybrid.
"""
import numpy as np

from neurolab.neurons import LIFConfig, LIFNeuron
from neurolab.circuits.hybrid import MemristorLIFCircuit, ResistorLIFCircuit
from neurolab.devices.presets import create_strukov_paper_device

# Parámetros que hoy definen las pestañas 2 y 3 de la GUI
DT = 1e-4
DURATION = 0.1      # 100 ms
F0 = 40.0           # 40 Hz -> 25 ms de período -> 5 ms de pulso (20% duty)


def _build_neuron() -> LIFNeuron:
    return LIFNeuron(LIFConfig(
        c_m=100e-9,   # 100 nF
        r_leak=1e6,   # 1 MΩ  -> tau = 100 ms
        v_rest=0.0,
        v_th=0.95,
        v_reset=0.15,
        t_ref=2e-3,
    ))


def _pulse_train_waveform(k: int) -> tuple:
    """Devuelve (i_uA, v_V) para el paso k del tren de pulsos unipolar al 20%."""
    phase = (k * DT * F0) % 1.0
    active = phase < 0.2
    return (20.0 if active else 0.0), (1.0 if active else 0.0)


def test_lif_tren_pulsos_dispara_spikes():
    """Pestaña 2: pulso de 20 µA @40 Hz debe producir ~4 spikes en 100 ms con intervalos de ~25 ms."""
    n = _build_neuron()
    steps = int(DURATION / DT)
    for k in range(steps):
        i, _ = _pulse_train_waveform(k)
        n.step(i * 1e-6, DT)

    assert len(n.spike_times) >= 3, f"Se esperaban ~4 spikes, se obtuvieron {len(n.spike_times)}"

    # Verificar periodicidad del tren de spikes (ISI ~ 25 ms para 40 Hz)
    intervals = np.diff(n.spike_times)
    assert all(0.020 <= iv <= 0.030 for iv in intervals), (
        f"Los spikes deben estar espaciados a ~25 ms (40 Hz). Obtenido: {intervals}"
    )


def test_lif_voltage_input_with_r_series():
    """Pestaña 2 (Fuente de Voltaje): V_IN = 5V @40 Hz con R_S = 100 kΩ debe disparar spikes periódicos."""
    n = LIFNeuron(LIFConfig(
        c_m=100e-9,      # 100 nF
        r_series=100e3,  # 100 kΩ
        r_leak=1e6,      # 1 MΩ
        v_rest=0.0,
        v_th=1.5,        # 1.5V umbral < V_inf * (1 - exp(-5ms/9ms)) = 1.95V
        v_reset=0.0,
        t_ref=2e-3
    ))
    steps = int(DURATION / DT)
    for k in range(steps):
        _, v_val = _pulse_train_waveform(k)
        v_in = v_val * 5.0  # Pulsos de 5V
        n.step(voltage_input=v_in, dt=DT)

    assert len(n.spike_times) >= 3, f"Se esperaban ~4 spikes con V_IN=5V y R_S=100kΩ, se obtuvieron {len(n.spike_times)}"

    # Verificar que los spikes coincidan con el periodo de los pulsos (~25 ms)
    intervals = np.diff(n.spike_times)
    assert all(0.020 <= iv <= 0.030 for iv in intervals), (
        f"Los intervalos entre spikes deben ser de ~25 ms. Obtenido: {intervals}"
    )


def test_hybrid_memristor_lif_dispara_spikes():
    """Pestaña 3 (modo Memristor): fuente 1 V @40 Hz debe excitar la neurona e incrementar conductancia."""
    mem = create_strukov_paper_device(initial_state=0.1)
    n = _build_neuron()
    circuit = MemristorLIFCircuit(memristor=mem, neuron=n)
    steps = int(DURATION / DT)
    for k in range(steps):
        _, v = _pulse_train_waveform(k)
        circuit.step(v, DT)

    assert len(n.spike_times) >= 3, f"Se esperaban ~4 spikes, se obtuvieron {len(n.spike_times)}"
    assert mem.x > 0.1, "El memristor debe conmutar progresivamente (aumentar x) bajo pulsos de excitación"


def test_hybrid_resistor_lif_dispara_spikes():
    """Pestaña 3 (modo Resistencia Fija): 10 kΩ con fuente 1 V debe excitar la neurona a 40 Hz."""
    n = _build_neuron()
    circuit = ResistorLIFCircuit(r_input=10_000.0, neuron=n)
    steps = int(DURATION / DT)
    for k in range(steps):
        _, v = _pulse_train_waveform(k)
        circuit.step(v, DT)

    assert len(n.spike_times) >= 3, f"Se esperaban ~4 spikes, se obtuvieron {len(n.spike_times)}"
    intervals = np.diff(n.spike_times)
    assert all(0.020 <= iv <= 0.030 for iv in intervals)


def test_clamp_corriente_equivale_pestana2():
    """
    Equivalencia EXACTA entre Pestaña 2 y Pestaña 3 (Resistencia Fija + clamp):
    si V_fuente(t) = V_m + I_objetivo(t)·R_syn, la corriente inyectada es idéntica
    a la fuente de corriente de la Pestaña 2 → mismos tiempos de spike y misma V(t).
    """
    r_syn = 10_000.0
    n_cur = _build_neuron()      # Pestaña 2: inyección directa de corriente
    n_cla = _build_neuron()      # Pestaña 3: clamp de corriente vía voltaje
    circuit = ResistorLIFCircuit(r_input=r_syn, neuron=n_cla)

    steps = int(DURATION / DT)
    for k in range(steps):
        i_uA, _ = _pulse_train_waveform(k)
        i_target = i_uA * 1e-6

        # Pestaña 2
        n_cur.step(i_target, DT)

        # Pestaña 3 (clamp): fuente de voltaje que cancela la caída por V_m
        v_source = n_cla.v_membrane + i_target * r_syn
        circuit.step(v_source, DT)

    # Corriente efectivamente inyectada por el clamp == i_target (diodo y R exactos)
    assert len(n_cla.spike_times) == len(n_cur.spike_times)
    np.testing.assert_allclose(n_cla.spike_times, n_cur.spike_times, atol=1e-12)
    assert len(n_cur.spike_times) >= 4  # sigue siendo la demo de 1 spike/pulso


def test_lif_refractory_period_suppression():
    """Verifica que durante el período refractario t_ref no se integre potencial ni se dispare spike secundario."""
    cfg = LIFConfig(c_m=100e-9, r_leak=1e6, v_rest=0.0, v_th=0.5, v_reset=0.0, t_ref=5e-3)
    n = LIFNeuron(cfg)

    # Paso 1: Inyectar corriente fuerte para forzar spike inmediato
    dt = 1e-4
    n.step(100e-6, dt)  # I = 100 µA -> dV = I/C * dt = 1e-4 / 1e-7 * 1e-4 = 0.1 V por paso

    # Avanzar hasta el spike
    for _ in range(10):
        n.step(100e-6, dt)
        if len(n.spike_times) > 0:
            break

    assert len(n.spike_times) == 1, "Debe haber ocurrido 1 spike"
    t_spike = n.spike_times[0]

    # Durante t_ref (5 ms = 50 pasos dt), inyectar corriente masiva (500 µA). V_m debe permanecer suprimido.
    for _ in range(40):
        n.step(500e-6, dt)
        assert len(n.spike_times) == 1, "No debe dispararse ningún spike secundario durante el período refractario"
        assert n.v_membrane < cfg.v_th, "V_m debe mantenerse suprimido bajo V_th durante t_ref"


def test_lif_analytical_validation_metrics():
    """Sección E: Verifica que la trayectoria numérica LIF reproduce la solución analítica diferencial exacta (R² > 0.99)."""
    from neurolab.core.lif_validation import compute_lif_validation_metrics
    cfg = LIFConfig(c_m=100e-9, r_series=100e3, r_leak=1e6, v_rest=0.0, v_th=2.5, v_reset=0.0)
    n = LIFNeuron(cfg)
    steps = 1000
    t = np.linspace(0, 0.1, steps)
    dt = t[1] - t[0]
    v_signal = np.full(steps, 2.0)  # Voltaje constante subumbral de 2.0V
    v_sim = np.zeros(steps)
    
    for k in range(steps):
        v_sim[k] = n.v_membrane
        n.step(voltage_input=v_signal[k], dt=dt)
        
    metrics = compute_lif_validation_metrics(t, v_sim, v_signal, cfg, is_voltage_input=True)
    assert metrics["r2"] > 0.99, f"Se esperaba R² > 0.99, se obtuvo {metrics['r2']}"
    assert metrics["mae"] < 0.05, f"Se esperaba MAE < 50 mV, se obtuvo {metrics['mae']}"