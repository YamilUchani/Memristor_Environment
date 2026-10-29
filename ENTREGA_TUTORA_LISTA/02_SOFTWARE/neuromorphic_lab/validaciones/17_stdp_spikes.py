"""
validacion_stdp_spikes.py
=========================
Muestra las formas de onda de los spikes presináptico (PRE) y postsináptico (POST)
y la diferencia temporal Δt = t_post - t_pre para casos LTP, nulo y LTD.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reconfigurar codificación de consola para Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def spike_waveform(t, t_spike, width=2.0):
    """Genera un pulso cuadrado de ancho `width` en t_spike."""
    return np.where((t >= t_spike) & (t < t_spike + width), 1.0, 0.0)


def main():
    base_dir = os.path.dirname(__file__)
    png_path = os.path.join(base_dir, "validacion_stdp_spikes.png")
    csv_path = os.path.join(base_dir, "validacion_stdp_spikes.csv")

    t = np.linspace(0, 100, 1000)  # 100 ms

    casos = [
        {"dt": +20.0, "titulo": "Δt = +20 ms → LTP (Potenciación: PRE antes que POST)"},
        {"dt": 0.0,   "titulo": "Δt = 0 ms → Coincidencia exacta (Sin cambio / Nulo)"},
        {"dt": -20.0, "titulo": "Δt = −20 ms → LTD (Depresión: POST antes que PRE)"},
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    df_list = []

    for ax, caso in zip(axes, casos):
        t_pre = 40.0
        t_post = t_pre + caso["dt"]

        V_pre = spike_waveform(t, t_pre, width=2.0)
        V_post = spike_waveform(t, t_post, width=2.0)

        ax.plot(t, V_pre + 1.5, 'b-', lw=2, label='Spike PRE (Electrodo Superior)')
        ax.plot(t, V_post, 'r-', lw=2, label='Spike POST (Electrodo Inferior)')
        ax.axvline(t_pre, color='b', ls='--', alpha=0.4)
        ax.axvline(t_post, color='r', ls='--', alpha=0.4)

        if caso["dt"] != 0:
            x_left = min(t_pre, t_post)
            x_right = max(t_pre, t_post)
            ax.annotate('', xy=(x_right, 1.25), xytext=(x_left, 1.25),
                        arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
            ax.text((t_pre + t_post) / 2, 1.35,
                    f'Δt = {caso["dt"]:+.0f} ms', ha='center', fontsize=10, fontweight='bold')

        ax.set_ylabel('Voltaje (V)', fontsize=10)
        ax.set_title(caso["titulo"], fontsize=11, fontweight='bold')
        ax.set_yticks([0, 1, 1.5, 2.5])
        ax.set_yticklabels(['0', '1 (POST)', '0 (PRE)', '1 (PRE)'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        df_list.append(pd.DataFrame({'t_ms': t, 'V_pre': V_pre, 'V_post': V_post, 'delta_t': caso['dt']}))

    axes[-1].set_xlabel('Tiempo (ms)', fontsize=11)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    pd.concat(df_list, ignore_index=True).to_csv(csv_path, index=False)

    print("=" * 65)
    print("  CARACTERIZACIÓN DE SPIKES STDP EN EL TIEMPO")
    print("=" * 65)
    print("Casos visualizados: Δt = +20 ms (LTP), Δt = 0 ms (Coincidencia), Δt = -20 ms (LTD)")
    print(f"Gráfica guardada en: {png_path}")
    print(f"Datos exportados a:  {csv_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
