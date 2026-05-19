"""
visualization/plot_state.py
=============================
Gráficas de la evolución temporal del estado interno x(t) y M(t).
"""

import numpy as np
import matplotlib.pyplot as plt
from ..simulations.iv_characterization import SimulationResult


def plot_state_evolution(result: SimulationResult,
                          ax: plt.Axes = None,
                          save_path: str = None,
                          dpi: int = 150) -> plt.Figure:
    """
    Grafica x(t) = w(t)/D y v(t) en el mismo eje temporal.

    La evolución de x muestra cómo los dopantes se desplazan bajo
    el campo eléctrico aplicado (Ec. 6 del paper).
    """
    standalone = ax is None
    if standalone:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    else:
        fig = ax.get_figure()
        axes = [ax]

    t = result.time
    v = result.voltage
    x = result.state_variable
    r = result.resistance

    if standalone:
        # Panel superior: voltaje
        axes[0].plot(t, v, 'b-', lw=2, label='Voltaje v(t)')
        axes[0].set_ylabel('Voltaje (V)', color='b', fontsize=11)
        axes[0].tick_params(axis='y', labelcolor='b')
        axes[0].grid(True, alpha=0.25, ls=':')
        axes[0].legend(loc='upper right')
        axes[0].set_title(f'Evolución temporal — {result.label}',
                          fontsize=12, fontweight='bold')

        # Panel inferior: estado x(t)
        ax2 = axes[1]
        ax2.plot(t, x, 'r-', lw=2, label='x(t) = w/D')
        ax2.axhline(0, color='k', ls='--', alpha=0.3, lw=1)
        ax2.axhline(1, color='k', ls='--', alpha=0.3, lw=1)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel('x = w/D', color='r', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_xlabel('Tiempo (s)', fontsize=11)
        ax2.grid(True, alpha=0.25, ls=':')
        ax2.legend(loc='upper right')

        # Memristancia en eje gemelo
        ax2r = ax2.twinx()
        ax2r.plot(t, r / 1000, 'purple', lw=1.5, alpha=0.6, ls='--',
                  label='M(t) (kΩ)')
        ax2r.set_ylabel('Memristancia M (kΩ)', color='purple', fontsize=11)
        ax2r.tick_params(axis='y', labelcolor='purple')
        ax2r.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  [FIG] Guardado: {save_path}")

    return fig


def plot_charge_flux(result: SimulationResult,
                      ax: plt.Axes = None,
                      save_path: str = None,
                      dpi: int = 150) -> plt.Figure:
    """
    Grafica q(Φ): carga vs flujo magnético.

    El paper requiere que q sea función univaluada de Φ para que
    el dispositivo se comporte como un memristor ideal (Figura 2b, inset).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    flux = result.flux
    charge = result.charge

    ax.plot(flux, charge, 'b-', lw=2)
    ax.set_xlabel('Flujo Φ = ∫V dt  (V·s)', fontsize=11)
    ax.set_ylabel('Carga q = ∫I dt  (C)', fontsize=11)
    ax.set_title('Relación Carga–Flujo\n(función univaluada = memristor ideal)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, ls=':')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  [FIG] Guardado: {save_path}")

    return fig
