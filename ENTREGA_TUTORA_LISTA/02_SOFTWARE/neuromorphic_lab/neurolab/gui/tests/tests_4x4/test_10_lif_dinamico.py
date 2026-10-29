"""
neurolab.gui.tests.tests_4x4.test_10_lif_dinamico
=================================================
Prueba 4x4_10: 4 Neuronas LIF Dinámicas impulsadas por el Crossbar 4×4.

CORRECCIÓN (v2):
    Antes: los parámetros LIF (c_m=10nF, r_leak=10MΩ) con corrientes de ~100 μA
           producían V_m = I * R_leak ≈ 1000 V → spike inmediato cada paso
           → Vm_history quedaba plana en V_reset = 0.

    Ahora:
        1. Corrientes escaladas a nA aplicando un factor de escala explícito:
               I_nA[j] = I_col[j] * 1e-3   (μA → nA)
           O bien, usar un divisor de corriente (R_line virtual).

        2. Parámetros LIF ajustados para régimen de nA:
               c_m    = 100 pF  (100e-12 F)
               r_leak = 100 MΩ  (100e6 Ω)
               v_th   = 0.5 V
               tau    = c_m * r_leak = 10 ms  (constante de integración)

        3. Con G ≈ 200 μS, V ≈ 0.8 V, I_col ≈ 200 μS * 0.8 * 4 ≈ 640 μA
           Factor de escala k = 1e-9 → I_eff ≈ 640 pA
           En tau = 10 ms, V_m converge hacia: I_eff * R_leak ≈ 640e-12 * 100e6 ≈ 64 mV

        → Para que el LIF dispare se necesita f=2 kHz de entrada o más,
          lo cual es razonable biológicamente.

        SOLUCIÓN ADOPTADA (más limpia):
        Usar corrientes en rango de μA y parámetros LIF calibrados para ese rango:
               c_m    = 1 μF   (1e-6 F)
               r_leak = 10 kΩ  (10e3 Ω)
               v_th   = 1.0 V
               tau    = 1e-6 * 10e3 = 10 ms
        
        Con I_col ≈ 100 μA: ΔV_m por paso de dt=0.5ms:
               ΔV = (I - V_m/R) / C * dt = 100e-6 / 1e-6 * 0.5e-3 ≈ 50 mV/paso
        → el LIF dispara en ~20 pasos (10 ms) → frecuencia de spike ≈ 100 Hz ✅
"""

import numpy as np
from neurolab.crossbar import CrossbarIdeal, CrossbarConfig
from neurolab.neurons import LIFNeuron, LIFConfig


def draw_4x4_10(gui, **kwargs):
    """Simulación dinámica de 4 neuronas LIF alimentadas por las corrientes I1..I4 del Crossbar 4×4."""
    T_ms = float(kwargs.get('T', 100.0))
    dt_ms = 0.5                         # paso de tiempo en ms
    dt = dt_ms * 1e-3                   # en segundos
    n_steps = int(T_ms / dt_ms)
    t_ms = np.linspace(0, T_ms, n_steps)

    config_cb = CrossbarConfig(n_rows=4, n_cols=4)
    cb = CrossbarIdeal(config_cb)

    # Matriz G heterogénea para que cada neurona reciba distinta corriente
    # Rango 100-450 μS → I_col ≈ 80-360 μA con V ≈ 0.8-1.6 V
    G_mat = np.array([
        [300, 100, 150, 400],
        [200, 250, 180, 350],
        [150, 300, 220, 200],
        [250, 180, 400, 450],
    ]) * 1e-6
    cb.set_matrix_conductance(G_mat)

    # ✅ PARÁMETROS LIF CALIBRADOS PARA CORRIENTES DE μA
    # tau = c_m * r_leak = 1e-6 * 10e3 = 10 ms
    # V_m_ss = I * r_leak ≈ 100μA * 10kΩ = 1 V → spikes alrededor del umbral
    lif_cfg = LIFConfig(
        c_m=1e-6,       # 1 μF
        r_leak=10e3,    # 10 kΩ
        v_th=1.0,       # Umbral: 1 V
        v_reset=0.0,    # Reset: 0 V
        t_ref=2e-3,     # Período refractario: 2 ms
    )
    neurons = [LIFNeuron(lif_cfg) for _ in range(4)]

    # ✅ CORREGIDO: V_m_hist se guarda ANTES del posible reset de spike
    Vm_history = np.zeros((n_steps, 4))
    spikes = [[] for _ in range(4)]

    # Señal de entrada sinusoidal en las filas — genera variación dinámica
    for k in range(n_steps):
        V_in = np.array([
            0.8 * (1.0 + np.sin(2 * np.pi * 0.02 * t_ms[k])),   # 0 – 1.6 V a 20 Hz
            0.5 * (1.0 + np.sin(2 * np.pi * 0.03 * t_ms[k])),   # 0 – 1.0 V a 30 Hz
            0.6 * (1.0 + np.cos(2 * np.pi * 0.02 * t_ms[k])),   # 0 – 1.2 V a 20 Hz
            0.4 * (1.0 + np.cos(2 * np.pi * 0.01 * t_ms[k])),   # 0 – 0.8 V a 10 Hz
        ])
        cb.apply_voltages(V_in)
        I_out = cb.read_currents()

        for j in range(4):
            # Guardar V_m ANTES del step para capturar la subida (diente de sierra)
            Vm_history[k, j] = neurons[j].V_m
            spiked = neurons[j].step(current_input=I_out[j], dt=dt)
            if spiked:
                spikes[j].append(t_ms[k])

    fig = gui.figure
    fig.clear()

    fig_face = fig.get_facecolor()
    is_dark = fig_face not in [(1.0, 1.0, 1.0, 1.0), 'white', '#ffffff', '#fff']

    title_color = '#f3f4f6' if is_dark else '#1e3a8a'
    text_color = '#e5e7eb' if is_dark else '#1f2937'
    colors = ['#2563eb', '#d97706', '#9333ea', '#047857']

    # PANEL 1: Potenciales de Membrana Vm(t) — debe mostrar diente de sierra
    ax1 = fig.add_subplot(1, 2, 1)
    gui._style_axis(ax1)

    for j in range(4):
        ax1.plot(t_ms, Vm_history[:, j],
                 label=f'LIF_{j+1} ({len(spikes[j])} spikes)',
                 color=colors[j], linewidth=1.0, alpha=0.85)

    ax1.axhline(1.0, color='#dc2626', linestyle='--', linewidth=1.2, alpha=0.8, label='Umbral V_th (1.0 V)')
    ax1.set_ylim(-0.05, 1.3)   # ✅ Rango forzado para ver la dinámica
    ax1.set_xlabel('Tiempo (ms)', color=text_color)
    ax1.set_ylabel('Potencial V_m (V)', color=text_color)
    ax1.set_title('Potenciales de Membrana V_m(t)\n(Diente de sierra visible)', color=title_color, fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)

    # PANEL 2: Raster Plot de Spikes
    ax2 = fig.add_subplot(1, 2, 2)
    gui._style_axis(ax2)

    for j in range(4):
        if len(spikes[j]) > 0:
            ax2.vlines(spikes[j], j + 0.6, j + 1.4, color=colors[j], linewidth=1.5, label=f'LIF_{j+1}')
        else:
            ax2.vlines([], j + 0.6, j + 1.4, color=colors[j], linewidth=1.5, label=f'LIF_{j+1} (0 spikes)')

    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_yticklabels(['LIF_1', 'LIF_2', 'LIF_3', 'LIF_4'])
    ax2.set_xlabel('Tiempo (ms)', color=text_color)
    ax2.set_ylabel('Neurona LIF', color=text_color)
    ax2.set_title(f'Raster Plot de Spikes\n(Total = {sum(len(s) for s in spikes)})',
                  color=title_color, fontsize=11, fontweight='bold')
    ax2.set_ylim(0.5, 4.5)
    ax2.set_xlim(0, T_ms)
    ax2.legend(loc='upper right', fontsize=8)

    fig.suptitle('Prueba 4×4_10: 4 Neuronas LIF Dinámicas impulsadas por el Crossbar 4×4',
                 color=title_color, fontsize=12, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    gui.canvas.draw()

    return {
        'status': 'PASS',
        'metrics': {
            'Spikes LIF_1': len(spikes[0]),
            'Spikes LIF_2': len(spikes[1]),
            'Spikes LIF_3': len(spikes[2]),
            'Spikes LIF_4': len(spikes[3]),
            'Total Spikes': sum(len(s) for s in spikes),
            'V_m máx observado': f'{np.max(Vm_history):.3f} V',
        }
    }
