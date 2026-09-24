"""
neurolab.gui.widgets.neuron_plot_canvas
========================================
Canvas Matplotlib para la neurona LIF (5 subplots).
"""

import numpy as np
from .canvas_base import BaseMplCanvas


class NeuronMplCanvas(BaseMplCanvas):
    """Canvas para visualización de la neurona LIF (5 subplots sincronizados)."""

    def __init__(self, parent=None):
        super().__init__(parent, rows=5, cols=1, figsize=(8, 8), sharex=True)
        self.ax_rs, self.ax_rl, self.ax_i, self.ax_v, self.ax_s = self.axes
        self._setup_titles()

    def _setup_titles(self):
        self.ax_rs.set_title("① Resistencia Serie ($R_S$)", fontsize=9)
        self.ax_rs.set_ylabel(r"$R_S$ (k$\Omega$)", color="#89dceb")
        self.ax_rs.ticklabel_format(useOffset=False, style='plain', axis='y')

        self.ax_rl.set_title("② Resistencia de Fuga ($R_{leak}$)", fontsize=9)
        self.ax_rl.set_ylabel(r"$R_{leak}$ (k$\Omega$)", color="#a6e3a1")
        self.ax_rl.ticklabel_format(useOffset=False, style='plain', axis='y')

        self.ax_i.set_title("③ Estímulo de Entrada: Voltaje ($V_{IN}$) y Corriente ($I_{in}$)", fontsize=9)
        self.ax_i.set_ylabel("$V_{IN}$ (V)", color="#89b4fa")

        self.ax_v.set_title("④ Potencial de Membrana / Capacitor ($V_c$)", fontsize=9)
        self.ax_v.set_ylabel("$V_c$ (V)")

        self.ax_s.set_title("⑤ Salida de Voltaje de Disparo ($V_{OUT}$)", fontsize=9)
        self.ax_s.set_ylabel("$V_{OUT}$ (V)")
        self.ax_s.set_xlabel("Tiempo (s)")

    def plot_results(self, t, signal_in, v_m, r_m_hist, spike_times,
                     v_th, v_reset, v_rest, i_in=None, r_series_hist=None,
                     val_metrics=None):
        """Actualiza todos los subplots."""
        self.clear_axes()
        self._setup_titles()

        idx_ds = self._decimate(t)
        t_ds = t[idx_ds]
        signal_in_ds = signal_in[idx_ds]
        v_m_ds = v_m[idx_ds]
        r_m_hist_ds = r_m_hist[idx_ds]
        r_series_hist_ds = r_series_hist[idx_ds] if r_series_hist is not None else None
        i_in_ds = i_in[idx_ds] if i_in is not None else None

        # --- ① R_S(t) ---
        if r_series_hist_ds is not None:
            r_series_kohm = r_series_hist_ds / 1e3
        else:
            r_series_kohm = np.full_like(t_ds, 100.0)
        self.ax_rs.plot(t_ds, r_series_kohm, color="#89dceb", linewidth=1.5, label=r"$R_S$ (Serie)")
        r_min, r_max = np.min(r_series_kohm), np.max(r_series_kohm)
        margin = max((r_max - r_min) * 0.15, 1.0)
        self.ax_rs.set_ylim(max(0, r_min - margin), r_max + margin)
        self.ax_rs.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a",
                          labelcolor="#cdd6f4", fontsize=8)

        # --- ② R_leak(t) ---
        r_leak_kohm = r_m_hist_ds / 1e3
        self.ax_rl.plot(t_ds, r_leak_kohm, color="#a6e3a1", linewidth=1.5, label=r"$R_{leak}$ (Fuga)")
        r_min_l, r_max_l = np.min(r_leak_kohm), np.max(r_leak_kohm)
        margin_l = max((r_max_l - r_min_l) * 0.15, 10.0)
        self.ax_rl.set_ylim(max(0, r_min_l - margin_l), r_max_l + margin_l)
        self.ax_rl.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a",
                          labelcolor="#cdd6f4", fontsize=8)

        # --- ③ V_IN(t) + I_in(t) ---
        max_abs = np.max(np.abs(signal_in_ds))
        if max_abs > 0.01:
            l1 = self.ax_i.plot(t_ds, signal_in_ds, color="#89b4fa", linewidth=1.5, label="$V_{IN}$ (V)")
            self.ax_i.set_ylabel("$V_{IN}$ (V)", color="#89b4fa")
            ax_i_twin = self.ax_i.twinx()
            ax_i_twin.tick_params(colors="#f9e2af", labelcolor="#f9e2af")
            ax_i_twin.set_ylabel(r"$I_{in}$ ($\mu$A)", color="#f9e2af")
            for spine in ax_i_twin.spines.values():
                spine.set_color("#45475a")
            i_uA = i_in_ds * 1e6 if i_in_ds is not None else signal_in_ds * 10.0
            l2 = ax_i_twin.plot(t_ds, i_uA, color="#f9e2af", linewidth=1.2,
                                linestyle="--", label=r"$I_{in}$ ($\mu$A)")
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            self.ax_i.legend(lines, labels, loc="upper right", facecolor="#181825",
                             edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)
        else:
            self.ax_i.plot(t_ds, signal_in_ds * 1e6, color="#f9e2af", linewidth=1.5)

        # --- ④ V_c(t) ---
        r2_str = ""
        if val_metrics and "v_analytical" in val_metrics:
            v_ana_ds = val_metrics["v_analytical"][idx_ds]
            r2_val = val_metrics.get("r2", 1.0)
            r2_str = f" [Validación R² = {r2_val:.4f}]"
            self.ax_v.plot(t_ds, v_ana_ds, color="#f9e2af", linewidth=1.4,
                           linestyle="--", alpha=0.95,
                           label=f"[Analítico] Solución Exacta (R²={r2_val:.4f})")
        self.ax_v.plot(t_ds, v_m_ds, color="#cba6f7", linewidth=1.5, label="$V_m$ (Simulación)")
        self.ax_v.set_title(f"④ Potencial de Membrana / Capacitor ($V_c$){r2_str}",
                            fontsize=9)
        self.ax_v.axhline(v_th, color="#f38ba8", linestyle="--", alpha=0.8,
                          linewidth=1.2, label=f"$V_{{th}}$ = {v_th:+.2f} V")
        self.ax_v.axhline(v_reset, color="#a6e3a1", linestyle=":", alpha=0.8,
                          linewidth=1.2, label=f"$V_{{reset}}$ = {v_reset:+.2f} V")
        self.ax_v.axhline(v_rest, color="#585b70", linestyle="-.", alpha=0.6,
                          linewidth=0.8, label=f"$V_{{rest}}$ = {v_rest:+.2f} V")

        if len(spike_times) > 0:
            spike_v = [v_th] * len(spike_times)
            self.ax_v.scatter(spike_times, spike_v, color="#f38ba8", s=25,
                              zorder=5, marker="^",
                              label=f"Spikes ({len(spike_times)})")

        y_lo = min(v_reset, v_rest, np.min(v_m_ds)) - abs(v_th) * 0.15
        y_hi = max(v_th, np.max(v_m_ds)) + abs(v_th) * 0.15
        self.ax_v.set_ylim(y_lo, y_hi)
        self.ax_v.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a",
                         labelcolor="#cdd6f4", fontsize=8)

        # --- ⑤ V_OUT Spikes ---
        dt_ds = t_ds[1] - t_ds[0] if len(t_ds) > 1 else 1e-4
        v_out = 0.02 + 0.005 * np.sin(2.0 * np.pi * 50.0 * t_ds)
        v_peak = max(0.53, 0.55 * v_th)
        for st in spike_times:
            idx = int(np.searchsorted(t_ds, st))
            if 0 <= idx < len(t_ds):
                tail_len = int(0.002 / dt_ds)
                for m in range(min(tail_len, len(t_ds) - idx)):
                    decay = np.exp(-m * dt_ds / 0.0004)
                    v_out[idx + m] = max(v_out[idx + m], 0.02 + (v_peak - 0.02) * decay)
        self.ax_s.plot(t_ds, v_out, color="#f38ba8", linewidth=1.5, label="$V_{OUT}$ Spikes")
        self.ax_s.set_ylim(-0.02, max(0.6, v_peak + 0.08))
        self.ax_s.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a",
                         labelcolor="#cdd6f4", fontsize=8)
        self.refresh()
