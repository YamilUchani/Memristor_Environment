"""
neurolab.gui.widgets.hybrid_plot_canvas
========================================
Lienzo Matplotlib para el circuito Híbrido Memristor-LIF (Pestaña 3).

Muestra 4 señales sincronizadas:
  ① V_source(t)  — Voltaje de entrada de la fuente (V)
  ② R_M(t)       — Resistencia dinámica del memristor (kΩ) [la novedad]
  ③ I_mem(t)     — Corriente sináptica inyectada a la neurona (µA)
  ④ V_c(t)       — Potencial de membrana + líneas V_th/V_reset + Raster
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HybridMplCanvas(QWidget):
    """Lienzo de dibujo Matplotlib para visualización del circuito Híbrido (Memristor+LIF)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 8), facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # 4 subplots con eje X compartido
        self.ax_vs = self.figure.add_subplot(411)
        self.ax_r  = self.figure.add_subplot(412, sharex=self.ax_vs)
        self.ax_i  = self.figure.add_subplot(413, sharex=self.ax_vs)
        self.ax_v  = self.figure.add_subplot(414, sharex=self.ax_vs)

        self.figure.subplots_adjust(hspace=0.65, left=0.10, right=0.95, top=0.93, bottom=0.08)
        self._setup_axes()

    # ── Configuración de ejes ────────────────────────────────────────────────

    def _setup_axes(self):
        for ax in [self.ax_vs, self.ax_r, self.ax_i, self.ax_v]:
            ax.set_facecolor("#181825")
            ax.tick_params(colors="#a6adc8", labelcolor="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_color("#45475a")

        self.ax_vs.set_title("① Fuente de Voltaje ($V_{source}$)", color="#cdd6f4", fontsize=10)
        self.ax_vs.set_ylabel("V (V)", color="#cdd6f4")
        self.ax_vs.grid(True, color="#313244", linestyle="--", linewidth=0.6)

        self.ax_r.set_title("② Resistencia Sináptica Dinámica ($R_M$)", color="#cdd6f4", fontsize=10)
        self.ax_r.set_ylabel(r"$R_M$ (k$\Omega$)", color="#cdd6f4")
        self.ax_r.grid(True, color="#313244", linestyle="--", linewidth=0.6)
        self.ax_r.ticklabel_format(useOffset=False, style="plain", axis="y")

        self.ax_i.set_title("③ Corriente Sináptica ($I_{mem} = V_{source} / R_M$)", color="#cdd6f4", fontsize=10)
        self.ax_i.set_ylabel(r"I ($\mu$A)", color="#cdd6f4")
        self.ax_i.grid(True, color="#313244", linestyle="--", linewidth=0.6)

        self.ax_v.set_title("④ Integración Neuronal + Disparos ($V_c$)", color="#cdd6f4", fontsize=10)
        self.ax_v.set_ylabel("$V_c$ (V)", color="#cdd6f4")
        self.ax_v.set_xlabel("Tiempo (s)", color="#cdd6f4")
        self.ax_v.grid(True, color="#313244", linestyle="--", linewidth=0.6)

    # ── Actualización de datos ───────────────────────────────────────────────

    def plot_results(
        self,
        t,
        v_source,
        r_m_hist,
        i_mem_hist,
        v_m_hist,
        spike_times,
        v_th,
        v_reset,
        v_rest,
        x_state_hist=None,
    ):
        """
        Actualiza todos los subplots con los datos del último paso de simulación.
        """
        self.ax_vs.clear()
        self.ax_r.clear()
        if hasattr(self, 'ax_r_twin') and self.ax_r_twin is not None:
            self.ax_r_twin.clear()
            self.ax_r_twin.remove()
            self.ax_r_twin = None
            
        self.ax_i.clear()
        self.ax_v.clear()
        self._setup_axes()

        # ── ① V_source ──────────────────────────────────────────────────────
        self.ax_vs.plot(t, v_source, color="#cba6f7", linewidth=1.5)
        self.ax_vs.set_ylim(bottom=min(0, np.min(v_source)) - 0.05)

        # ── ② R_M(t) dinámica y x(t) en eje derecho ─────────────────────────
        r_kohm = r_m_hist / 1e3
        l1 = self.ax_r.plot(t, r_kohm, color="#a6e3a1", linewidth=1.5, label=r"$R_M$ (k$\Omega$)")
        r_min, r_max = np.min(r_kohm), np.max(r_kohm)
        margin = max((r_max - r_min) * 0.1, 0.5)
        self.ax_r.set_ylim(max(0, r_min - margin), r_max + margin)

        if x_state_hist is not None:
            self.ax_r_twin = self.ax_r.twinx()
            self.ax_r_twin.tick_params(colors="#89dceb", labelcolor="#89dceb")
            self.ax_r_twin.set_ylabel("$x(t)$ Estado", color="#89dceb")
            self.ax_r_twin.set_ylim(-0.05, 1.05)
            for spine in self.ax_r_twin.spines.values():
                spine.set_color("#45475a")
            l2 = self.ax_r_twin.plot(t, x_state_hist, color="#89dceb", linewidth=1.2, linestyle="--", label="$x(t)$ Estado Memristivo")
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            self.ax_r.legend(lines, labels, loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

        # ── ③ I_mem(t) en µA ────────────────────────────────────────────────
        i_ua = i_mem_hist * 1e6
        self.ax_i.plot(t, i_ua, color="#f9e2af", linewidth=1.5)
        self.ax_i.set_ylim(bottom=min(0, np.min(i_ua)) - 0.5)

        # ── ④ V_c(t) = V_m - V_rest, + líneas umbral/reset ─────────────────
        v_c      = v_m_hist - v_rest
        v_th_c   = v_th   - v_rest
        v_reset_c = v_reset - v_rest

        self.ax_v.plot(t, v_c, color="#89b4fa", linewidth=1.5)
        self.ax_v.axhline(v_th_c,    color="#f38ba8", linestyle="--", alpha=0.85, linewidth=1.2, label="$V_{th}$ relativo")
        self.ax_v.axhline(v_reset_c, color="#a6e3a1", linestyle=":",  alpha=0.85, linewidth=1.2, label="$V_{reset}$ relativo")
        self.ax_v.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=9)

        # Marcas de spike superpuestas en V_c
        if len(spike_times) > 0:
            for st in spike_times:
                self.ax_v.axvline(st, color="#f38ba8", linewidth=0.8, alpha=0.5, linestyle=":")

        self.canvas.draw()
