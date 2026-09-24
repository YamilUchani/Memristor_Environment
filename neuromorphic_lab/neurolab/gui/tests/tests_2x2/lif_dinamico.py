"""
Prueba 4.2.4: 2×2 → 2 neuronas LIF dinámicas.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.neurons import LIFNeuron, LIFConfig


def draw_2x2_lif_dinamico(gui=None, T=100.0, **kwargs):
    """Simula la respuesta de 2 neuronas LIF alimentadas por el crossbar 2×2."""
    T_sec = T * 1e-3

    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)
    cb.set_uniform_conductance(200e-6)

    lif_cfg = LIFConfig(c_m=100e-9, r_leak=1e6, v_th=2.0, v_reset=0.0)
    lif_1 = LIFNeuron(lif_cfg)
    lif_2 = LIFNeuron(lif_cfg)

    dt = 1e-5
    t = np.arange(0, T_sec, dt)
    spikes_1 = []
    spikes_2 = []

    for tk in t:
        V1 = 0.5 + 0.3 * np.sin(2 * np.pi * 10 * tk)
        V2 = 0.5 + 0.3 * np.cos(2 * np.pi * 10 * tk)

        cb.apply_voltages([V1, V2])
        I = cb.read_currents()

        if lif_1.update(I_in=I[0], dt=dt):
            spikes_1.append(tk)
        if lif_2.update(I_in=I[1], dt=dt):
            spikes_2.append(tk)

    return {
        'status': 'PASS',
        'metrics': {
            'Spikes LIF_1': len(spikes_1),
            'Spikes LIF_2': len(spikes_2),
            'ISI LIF_1': f'{np.mean(np.diff(spikes_1))*1e3:.2f} ms' if len(spikes_1) > 1 else '—',
            'ISI LIF_2': f'{np.mean(np.diff(spikes_2))*1e3:.2f} ms' if len(spikes_2) > 1 else '—',
        }
    }
