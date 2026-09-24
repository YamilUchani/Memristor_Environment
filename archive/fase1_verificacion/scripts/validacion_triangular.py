"""
validaciones/validacion_triangular.py
======================================
Validación del modelo memristor Strukov con señal triangular V(t).
Complementa la validación senoidal (validacion_strukov.py) de la Etapa 1.1.

Salidas:
  - neuromorphic_lab/validaciones/validacion_triangular.png
  - neuromorphic_lab/validaciones/validacion_triangular.csv
"""

import sys
from pathlib import Path

# Reconfigurar encoding para consola Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Garantizar que el directorio raíz de neuromorphic_lab esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from neurolab.devices import MemristorStrukov
from neurolab.configs import StrukovConfig


def generate_triangular(t: np.ndarray, freq: float, amplitude: float, offset: float = 0.0) -> np.ndarray:
    """Genera onda triangular simétrica de frecuencia `freq` y amplitud `amplitude`."""
    period = 1.0 / freq
    phase = (t % period) / period
    tri = np.where(phase < 0.5, 4.0 * phase - 1.0, 3.0 - 4.0 * phase)
    return amplitude * tri + offset


def main():
    print("=" * 80)
    print(" VALIDACIÓN DEL MODELO MEMRISTOR STRUKOV — SEÑAL TRIANGULAR V(t)")
    print("=" * 80)

    output_dir = Path(__file__).resolve().parent
    fig_subfolder = output_dir.parent / "figuras"
    fig_subfolder.mkdir(parents=True, exist_ok=True)

    # --- Configuración del dispositivo ---
    cfg = StrukovConfig(
        RON=100.0,
        ROFF=16_000.0,
        x0=0.10,
        D=10e-9,
        mu_v=1e-14,
        clip_x=True,
    )
    mem = MemristorStrukov(cfg)

    # --- Señal triangular ---
    f = 0.5        # Hz
    A = 1.0        # V
    T = 6.0        # s
    dt = 1e-4      # s
    t = np.arange(0, T, dt)
    steps = len(t)
    V = generate_triangular(t, f, A)

    # --- Simulación ---
    I = np.zeros(steps)
    x = np.zeros(steps)
    R = np.zeros(steps)
    G = np.zeros(steps)

    for k in range(steps):
        I[k] = mem.current(V[k])
        R[k] = mem.resistance
        G[k] = mem.conductance
        x[k] = mem.x
        mem.update(V[k], dt)

    # --- Figura en tema blanco académico ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='#ffffff')
    fig.suptitle("Validación del Memristor de Strukov ante Excitación Triangular V(t)", color='#0f172a', fontsize=13, fontweight='bold')

    for ax in axes.flat:
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#000000', labelcolor='#000000')
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            spine.set_linewidth(1.1)

    # Panel 1: Entrada V(t)
    axes[0, 0].plot(t, V, color='#0f172a', lw=1.2)
    axes[0, 0].set_xlabel('Tiempo (s)', fontweight='bold')
    axes[0, 0].set_ylabel('Voltaje V (V)', fontweight='bold')
    axes[0, 0].set_title('① Señal Triangular de Entrada V(t)', fontweight='bold')

    # Panel 2: Respuesta I(t)
    axes[0, 1].plot(t, I * 1e3, color='#2563eb', lw=1.2)
    axes[0, 1].set_xlabel('Tiempo (s)', fontweight='bold')
    axes[0, 1].set_ylabel('Corriente I (mA)', fontweight='bold')
    axes[0, 1].set_title('② Corriente de Salida I(t)', fontweight='bold')

    # Panel 3: Estado interno x(t)
    axes[1, 0].plot(t, x, color='#059669', lw=1.2)
    axes[1, 0].set_xlabel('Tiempo (s)', fontweight='bold')
    axes[1, 0].set_ylabel('Estado Normalizado x(t)', fontweight='bold')
    axes[1, 0].set_title('③ Evolución del Estado Interno x(t) ∈ [0,1]', fontweight='bold')

    # Panel 4: Lazo de Histéresis Pinzado I-V
    axes[1, 1].plot(V, I * 1e3, color='#dc2626', lw=1.5)
    axes[1, 1].axhline(0, color='#64748b', linestyle=':', lw=0.8)
    axes[1, 1].axvline(0, color='#64748b', linestyle=':', lw=0.8)
    axes[1, 1].set_xlabel('Voltaje V (V)', fontweight='bold')
    axes[1, 1].set_ylabel('Corriente I (mA)', fontweight='bold')
    axes[1, 1].set_title('④ Lazo de Histéresis Pinzado I-V (Triangular)', fontweight='bold')

    plt.tight_layout()
    png_path = output_dir / "validacion_triangular.png"
    plt.savefig(png_path, dpi=300)
    plt.savefig(fig_subfolder / "validacion_triangular.png", dpi=300)
    plt.close()

    # --- Guardar CSV ---
    csv_path = output_dir / "validacion_triangular.csv"
    with open(csv_path, "w", encoding="utf-8") as f_out:
        f_out.write("t_s,V_V,I_A,x_adim,R_ohm,G_S\n")
        step_ds = max(1, steps // 2000)
        for k in range(0, steps, step_ds):
            f_out.write(f"{t[k]:.6f},{V[k]:.6f},{I[k]:.6e},{x[k]:.6f},{R[k]:.2f},{G[k]:.6e}\n")

    # --- Métricas ---
    print(f" Rango de estado x(t):   [{x.min():.4f}, {x.max():.4f}]")
    print(f" Rango de Resistencia R: [{R.min():.2f}, {R.max():.2f}] Ohm")
    print(f" Rango de Corriente I:   [{I.min()*1e3:.4f}, {I.max()*1e3:.4f}] mA")
    print(f" Puntos simulados:       {len(V):,}")
    print(f"\n[OK] Archivos generados exitosamente:")
    print(f"   - {png_path}")
    print(f"   - {csv_path}")


if __name__ == '__main__':
    main()
