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
    """Pestaña 2: pulso de 20 µA @40 Hz debe producir al menos 1 spike por pulso."""
    n = _build_neuron()
    steps = int(DURATION / DT)
    for k in range(steps):
        i, _ = _pulse_train_waveform(k)
        n.step(i * 1e-6, DT)
    assert len(n.spike_times) >= 4, f"Se esperaban ~4 spikes, se obtuvieron {len(n.spike_times)}"


def test_hybrid_memristor_lif_dispara_spikes():
    """Pestaña 3 (modo Memristor): fuente 1 V @40 Hz debe excitar la neurona."""
    mem = create_strukov_paper_device(initial_state=0.1)
    n = _build_neuron()
    circuit = MemristorLIFCircuit(memristor=mem, neuron=n)
    steps = int(DURATION / DT)
    for k in range(steps):
        _, v = _pulse_train_waveform(k)
        circuit.step(v, DT)
    assert len(n.spike_times) >= 1, f"Se esperaban spikes, se obtuvieron {len(n.spike_times)}"


def test_hybrid_resistor_lif_dispara_spikes():
    """Pestaña 3 (modo Resistencia Fija): 10 kΩ con fuente 1 V debe excitar la neurona."""
    n = _build_neuron()
    circuit = ResistorLIFCircuit(r_input=10_000.0, neuron=n)
    steps = int(DURATION / DT)
    for k in range(steps):
        _, v = _pulse_train_waveform(k)
        circuit.step(v, DT)
    assert len(n.spike_times) >= 1, f"Se esperaban spikes, se obtuvieron {len(n.spike_times)}"


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