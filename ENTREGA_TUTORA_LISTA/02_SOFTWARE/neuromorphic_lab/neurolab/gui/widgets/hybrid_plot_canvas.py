"""
neurolab.gui.widgets.hybrid_plot_canvas
========================================
Canvas Matplotlib para el circuito Híbrido Memristor-LIF.
"""

import numpy as np
from .canvas_base import BaseMplCanvas


class HybridMplCanvas(BaseMplCanvas):
    """Canvas para el circuito Híbrido (Memristor + LIF)."""

    def __init__(self, parent=None):
        super().__init__(parent, rows=4, cols=1, figsize=(8, 8), sharex=True)
        self.ax_vs, self.ax_r, self.ax_i, self.ax_v = self.axes
        self.ax_r_twin = None
        self._setup_titles()

    def _setup_titles(self):
        self.ax_vs.set_title("① Fuente de Voltaje ($V_{source}$)", fontsize=10)
        self.ax_vs.set_ylabel("V (V)")
        self.ax_r.set_title("② Resistencia Sináptica Dinámica ($R_M$)", fontsize=10)
        self.ax_r.set_ylabel(r"$R_M$ (k$\Omega$)")
        self.ax_r.ticklabel_format(useOffset=False, style="plain", axis="y")
        self.ax_i.set_title("③ Corriente Sináptica ($I_{mem} = V_{source} / R_M$)", fontsize=10)
        self.ax_i.set_ylabel(r"I ($\mu$A)")
        self.ax_v.set_title("④ Integración Neuronal + Disparos ($V_c$)", fontsize=10)
        self.ax_v.set_ylabel("$V_c$ (V)")
        self.ax_v.set_xlabel("Tiempo (s)")

    def plot_results(self, t, v_source, r_m_hist, i_mem_hist, v_m_hist,
                     spike_times, v_th, v_reset, v_rest, x_state_hist=None):
        """Actualiza todos los subplots."""
        self.clear_axes()
        if self.ax_r_twin is not None:
            self.ax_r_twin.remove()
            self.ax_r_twin = None
        self._setup_titles()

        idx_ds = self._decimate(t)
        t_ds = t[idx_ds]
        v_source_ds = v_source[idx_ds]
        r_m_hist_ds = r_m_hist[idx_ds]
        i_mem_hist_ds = i_mem_hist[idx_ds]
        v_m_hist_ds = v_m_hist[idx_ds]
        x_state_hist_ds = x_state_hist[idx_ds] if x_state_hist is not None else None

        # ① V_source
        self.ax_vs.plot(t_ds, v_source_ds, color="#cba6f7", linewidth=1.5)
        self.ax_vs.set_ylim(bottom=min(0, np.min(v_source_ds)) - 0.05)

        # ② R_M(t)
        r_kohm = r_m_hist_ds / 1e3
        l1 = self.ax_r.plot(t_ds, r_kohm, color="#a6e3a1", linewidth=1.5,
                            label=r"$R_M$ (k$\Omega$)")
        r_min, r_max = np.min(r_kohm), np.max(r_kohm)
        margin = max((r_max - r_min) * 0.1, 0.5)
        self.ax_r.set_ylim(max(0, r_min - margin), r_max + margin)

        if x_state_hist_ds is not None:
            self.ax_r_twin = self.ax_r.twinx()
            self.ax_r_twin.tick_params(colors="#89dceb", labelcolor="#89dceb")
            self.ax_r_twin.set_ylabel("$x(t)$ Estado", color="#89dceb")
            self.ax_r_twin.set_ylim(-0.05, 1.05)
            for spine in self.ax_r_twin.spines.values():
                spine.set_color("#45475a")
            l2 = self.ax_r_twin.plot(t_ds, x_state_hist_ds, color="#89dceb",
                                     linewidth=1.2, linestyle="--",
                                     label="$x(t)$ Estado Memristivo")
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            self.ax_r.legend(lines, labels, loc="upper right", facecolor="#181825",
                             edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

        # ③ I_mem(t)
        i_ua = i_mem_hist_ds * 1e6
        self.ax_i.plot(t_ds, i_ua, color="#f9e2af", linewidth=1.5)
        self.ax_i.set_ylim(bottom=min(0, np.min(i_ua)) - 0.5)

        # ④ V_c(t)
        v_c = v_m_hist_ds - v_rest
        v_th_c = v_th - v_rest
        v_reset_c = v_reset - v_rest
        self.ax_v.plot(t_ds, v_c, color="#89b4fa", linewidth=1.5)
        self.ax_v.axhline(v_th_c, color="#f38ba8", linestyle="--",
                          alpha=0.85, linewidth=1.2, label="$V_{th}$ relativo")
        self.ax_v.axhline(v_reset_c, color="#a6e3a1", linestyle=":",
                          alpha=0.85, linewidth=1.2, label="$V_{reset}$ relativo")
        self.ax_v.legend(loc="upper right", facecolor="#181825",
                         edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=9)
        if len(spike_times) > 0:
            for st in spike_times:
                self.ax_v.axvline(st, color="#f38ba8", linewidth=0.8,
                                  alpha=0.5, linestyle=":")
        self.refresh()
