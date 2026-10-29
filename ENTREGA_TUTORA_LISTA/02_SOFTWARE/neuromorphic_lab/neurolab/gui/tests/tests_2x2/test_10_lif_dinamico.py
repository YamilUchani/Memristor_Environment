"""
Prueba 2x2_10: 2 neuronas LIF alimentadas por el crossbar 2×2.
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.neurons import LIFNeuron, LIFConfig


def draw_2x2_10(gui, T=100.0, freq=10.0, **kwargs):
    """
    Simula 2 neuronas LIF con entradas variables.
    """
    T = float(T) * 1e-3
    freq = float(freq)

    config = CrossbarConfig(n_rows=2, n_cols=2)
    cb = CrossbarIdeal(config)

    G = np.array([
        [200e-6, 150e-6],
        [100e-6, 250e-6],
    ])
    for i in range(2):
        for j in range(2):
            cb.set_conductance(i, j, G[i, j])

    lif_cfg = LIFConfig(c_m=100e-9, r_leak=1e6, v_th=2.0, v_reset=0.0)
    lif_1 = LIFNeuron(lif_cfg)
    lif_2 = LIFNeuron(lif_cfg)

    dt = 1e-5
    t = np.arange(0, T, dt)

    V1_t = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t)
    V2_t = 0.5 + 0.3 * np.cos(2 * np.pi * freq * t)

    Vm_1 = np.zeros_like(t)
    Vm_2 = np.zeros_like(t)
    I1_hist = np.zeros_like(t)
    I2_hist = np.zeros_like(t)
    spikes_1 = []
    spikes_2 = []

    for k in range(len(t)):
        cb.apply_voltages([V1_t[k], V2_t[k]])
        I = cb.read_currents()
        I1_hist[k] = I[0]
        I2_hist[k] = I[1]

        if lif_1.update(I_in=I[0], dt=dt):
            spikes_1.append(t[k])
        if lif_2.update(I_in=I[1], dt=dt):
            spikes_2.append(t[k])

        Vm_1[k] = lif_1.V_m
        Vm_2[k] = lif_2.V_m

    fig = gui.figure
    fig.clear()

    ax1 = fig.add_subplot(3, 1, 1)
    gui._style_axis(ax1)
    ax1.plot(t * 1e3, V1_t, 'b-', lw=1.5, label='V₁(t)')
    ax1.plot(t * 1e3, V2_t, 'r-', lw=1.5, label='V₂(t)')
    ax1.set_ylabel('V (V)', color='white')
    ax1.set_title(f'Prueba 2×2_10: 2 Neuronas LIF — {freq} Hz',
                  color='white', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', facecolor='#252526',
               edgecolor='#555', labelcolor='white', ncol=2)
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(3, 1, 2)
    gui._style_axis(ax2)
    ax2.plot(t * 1e3, I1_hist * 1e6, 'b-', lw=1.2, label='I₁(t)')
    ax2.plot(t * 1e3, I2_hist * 1e6, 'r-', lw=1.2, label='I₂(t)')
    ax2.set_ylabel('I (μA)', color='white')
    ax2.legend(loc='upper right', facecolor='#252526',
               edgecolor='#555', labelcolor='white', ncol=2)
    ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(3, 1, 3)
    gui._style_axis(ax3)
    ax3.plot(t * 1e3, Vm_1, 'b-', lw=1.2, label=f'LIF₁ ({len(spikes_1)} spikes)')
    ax3.plot(t * 1e3, Vm_2, 'r-', lw=1.2, label=f'LIF₂ ({len(spikes_2)} spikes)')
    ax3.axhline(2.0, color='white', ls='--', lw=1, alpha=0.4,
                label='V_th')
    ax3.set_xlabel('Tiempo (ms)', color='white')
    ax3.set_ylabel('V_m (V)', color='white')
    ax3.legend(loc='upper right', facecolor='#252526',
               edgecolor='#555', labelcolor='white', ncol=3)
    ax3.grid(True, alpha=0.2)

    fig.tight_layout()
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'Spikes LIF 1': len(spikes_1),
            'Spikes LIF 2': len(spikes_2),
            'ISI 1': f'{np.mean(np.diff(spikes_1))*1e3:.2f} ms' if len(spikes_1) > 1 else '—',
            'ISI 2': f'{np.mean(np.diff(spikes_2))*1e3:.2f} ms' if len(spikes_2) > 1 else '—',
        }
    }
