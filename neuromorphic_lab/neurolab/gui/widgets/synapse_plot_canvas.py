"""
neurolab.gui.widgets.synapse_plot_canvas
=========================================
Canvas Matplotlib para plasticidad sináptica (LTP, LTD, STDP, spikes).
"""

import numpy as np
from .canvas_base import BaseMplCanvas


class SynapseMplCanvas(BaseMplCanvas):
    """Canvas para experimentos de plasticidad sináptica."""

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        super().__init__(parent, rows=1, cols=1, figsize=(width, height),
                         dpi=dpi, sharex=False)

    def plot_experiment(self, exp_type: str, sim_data: dict):
        """Dibuja el experimento seleccionado."""
        self.figure.clear()

        if exp_type in ["ltp_ltd_sep", "ltp", "ltd"]:
            self._plot_ltp_ltd_sep(sim_data)
        elif exp_type == "ltp_ltd_ciclo":
            self._plot_ciclo(sim_data)
        elif exp_type in ["stdp_curves", "stdp_hebbian", "stdp_anti"]:
            self._plot_stdp(sim_data)
        elif exp_type == "stdp_spikes":
            self._plot_stdp_spikes(sim_data)
        elif exp_type == "stdp_temporal":
            self._plot_stdp_temporal(sim_data)
        elif exp_type == "virg_evidence":
            self._plot_evidence(sim_data)
        elif exp_type == "iv_hysteresis":
            self._plot_iv_hysteresis(sim_data)
        elif exp_type == "jo2010_stdp":
            self._plot_jo2010_stdp(sim_data)
        elif exp_type == "jo2010_ltp_ltd":
            self._plot_jo2010_ltp_ltd(sim_data)

        self._apply_style(list(self.figure.axes))
        self.refresh()

    # ================================================================
    # EXPERIMENTOS
    # ================================================================

    def _plot_ltp_ltd_sep(self, sim_data):
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)
        G_ltp = sim_data.get("G_ltp", np.array([]))
        G_ltd = sim_data.get("G_ltd", np.array([]))
        if len(G_ltp) > 0:
            ax1.plot(range(len(G_ltp)), G_ltp * 1e6, 'b-o', ms=4, lw=1.5, label='LTP (+1V)')
            ax1.set_title("LTP — Potenciación", fontsize=10, fontweight='bold')
            ax1.set_xlabel("Número de pulsos"); ax1.set_ylabel("Conductancia (μS)")
            ax1.legend(loc='lower right')
        if len(G_ltd) > 0:
            ax2.plot(range(len(G_ltd)), G_ltd * 1e6, 'r-s', ms=4, lw=1.5, label='LTD (-1V)')
            ax2.set_title("LTD — Depresión", fontsize=10, fontweight='bold')
            ax2.set_xlabel("Número de pulsos"); ax2.set_ylabel("Conductancia (μS)")
            ax2.legend(loc='upper right')

    def _plot_ciclo(self, sim_data):
        ax = self.figure.add_subplot(1, 1, 1)
        G = sim_data.get("G_history", np.array([]))
        if len(G) > 0:
            n_half = (len(G) - 1) // 2
            pulses = np.arange(len(G))
            ax.plot(pulses[:n_half+1], G[:n_half+1] * 1e6, 'b-o', ms=4, lw=1.5, label='LTP')
            ax.plot(pulses[n_half:], G[n_half:] * 1e6, 'r-s', ms=4, lw=1.5, label='LTD')
            ax.axvline(n_half, color='#a6adc8', ls='--', alpha=0.7)
            ax.set_title("Ciclo Reversible LTP → LTD", fontsize=11, fontweight='bold')
            ax.set_xlabel("Pulsos"); ax.set_ylabel("G (μS)")
            ax.legend(loc='upper right')

    def _plot_stdp(self, sim_data):
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)
        dt_ms = sim_data.get("dt_ms", np.array([]))
        dW_h = sim_data.get("dW_hebbian", np.array([]))
        dW_a = sim_data.get("dW_anti", np.array([]))
        if len(dt_ms) > 0:
            ax1.plot(dt_ms, dW_h, 'b-o', ms=4, lw=1.5)
            ax1.axhline(0, color='#cdd6f4', ls='--', lw=0.6)
            ax1.axvline(0, color='#cdd6f4', ls='--', lw=0.6)
            ax1.set_title("STDP Hebbiano", fontsize=10, fontweight='bold')
            ax1.set_xlabel("Δt (ms)"); ax1.set_ylabel("ΔW")

            ax2.plot(dt_ms, dW_a, 'r-s', ms=4, lw=1.5)
            ax2.axhline(0, color='#cdd6f4', ls='--', lw=0.6)
            ax2.axvline(0, color='#cdd6f4', ls='--', lw=0.6)
            ax2.set_title("STDP Anti-Hebbiano", fontsize=10, fontweight='bold')
            ax2.set_xlabel("Δt (ms)"); ax2.set_ylabel("ΔW")

    def _plot_stdp_spikes(self, sim_data):
        casos = sim_data.get("casos_spikes", [])
        for idx, item in enumerate(casos):
            ax = self.figure.add_subplot(len(casos), 1, idx + 1)
            t = item["t"]; v_pre = item["v_pre"]; v_post = item["v_post"]
            ax.plot(t, v_pre + 1.5, color='#89b4fa', lw=1.8, label='Spike PRE')
            ax.plot(t, v_post, color='#f38ba8', lw=1.8, label='Spike POST')
            ax.set_title(item["titulo"], fontsize=9, fontweight='bold')
            ax.set_ylabel("V (V)")
            if idx == len(casos) - 1:
                ax.set_xlabel("Tiempo (ms)")
            ax.legend(loc='upper right', fontsize=7)

    def _plot_stdp_temporal(self, sim_data):
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)
        dt_seq = sim_data.get("dt_sequence", np.array([]))
        W_hist = sim_data.get("W_history", np.array([]))
        if len(dt_seq) > 0:
            ax1.stem(range(1, len(dt_seq) + 1), dt_seq, basefmt='k-')
            ax1.axhline(0, color='#cdd6f4', ls='--', lw=0.6)
            ax1.set_title("Secuencia de Δt", fontsize=9, fontweight='bold')
            ax1.set_ylabel("Δt (ms)")
        if len(W_hist) > 0:
            ax2.plot(range(len(W_hist)), W_hist, 'g-o', ms=4, lw=1.5)
            ax2.axhline(W_hist[0], color='#cdd6f4', ls='--', lw=0.6)
            ax2.set_title("Evolución W(t)", fontsize=9, fontweight='bold')
            ax2.set_xlabel("Nº par de spikes"); ax2.set_ylabel("W")
            ax2.legend(loc='upper left', fontsize=7)

    def _plot_evidence(self, sim_data):
        keys = ["V", "I", "R", "G"]
        y_labels = ["V (V)", "I (μA)", "R (kΩ)", "G (μS)"]
        y_scales = [1, 1e6, 1/1e3, 1e6]
        for i, (key, ylab, scale) in enumerate(zip(keys, y_labels, y_scales)):
            ax = self.figure.add_subplot(4, 1, i+1)
            t_ms = sim_data.get("t_ms", np.array([]))
            ltp = sim_data.get(f"{key}_ltp", np.array([]))
            ltd = sim_data.get(f"{key}_ltd", np.array([]))
            if len(t_ms) > 0:
                ax.plot(t_ms, ltp * scale, 'b-', lw=1.2, label='LTP')
                ax.plot(t_ms, ltd * scale, 'r-', lw=1.2, label='LTD')
                ax.set_ylabel(ylab); ax.legend(loc='upper right', fontsize=7)
        self.figure.axes[-1].set_xlabel("Tiempo (ms)")

    def _plot_iv_hysteresis(self, sim_data):
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)
        V = sim_data.get("V_sweep", np.array([]))
        I = sim_data.get("I_sweep", np.array([]))
        R = sim_data.get("R_sweep", np.array([]))
        if len(V) > 0:
            ax1.plot(V, I * 1e3, 'b-', lw=1.5)
            ax1.axhline(0, color='#cdd6f4', ls='--', lw=0.6)
            ax1.axvline(0, color='#cdd6f4', ls='--', lw=0.6)
            ax1.set_title("Curva I-V (Histéresis Pinzada)", fontsize=10, fontweight='bold')
            ax1.set_xlabel("V (V)"); ax1.set_ylabel("I (mA)")

            ax2.plot(V, R / 1e3, 'r-', lw=1.5)
            ax2.axvline(0, color='#cdd6f4', ls='--', lw=0.6)
            ax2.set_title("R vs V", fontsize=10, fontweight='bold')
            ax2.set_xlabel("V (V)"); ax2.set_ylabel("R (kΩ)")

    def _plot_jo2010_stdp(self, sim_data):
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)
        dt_exp = sim_data.get("dt_exp_ms", np.array([]))
        dw_exp = sim_data.get("dw_exp_pct", np.array([]))
        dt_sim = sim_data.get("dt_sim_ms", np.array([]))
        dw_sim = sim_data.get("dw_sim_pct", np.array([]))
        r2 = sim_data.get("r2", 0.0)
        if len(dt_exp) > 0:
            ax1.plot(dt_exp, dw_exp, 'o', ms=8, color='#f38ba8', label='Experimental (Jo 2010)')
            idx = np.argsort(dt_sim)
            ax1.plot(dt_sim[idx], dw_sim[idx], '-', lw=2.2, color='#89b4fa', label='Simulación')
            ax1.axhline(0, color='#cdd6f4', ls='--', lw=0.8, alpha=0.4)
            ax1.axvline(0, color='#cdd6f4', ls='--', lw=0.8, alpha=0.4)
            ax1.set_title(f"Jo 2010 — STDP (R² = {r2:.4f})", fontsize=10, fontweight='bold')
            ax1.set_xlabel("Δt (ms)"); ax1.set_ylabel("ΔW (%)")
            ax1.legend(loc='upper left', fontsize=8)

            error = sim_data.get("dw_sim_at_exp", np.array([])) - dw_exp
            ax2.bar(dt_exp, error, width=3.0, color='#cba6f7', edgecolor='white')
            ax2.axhline(0, color='#cdd6f4', ls='--', lw=0.8, alpha=0.5)
            ax2.set_title("Error punto a punto", fontsize=10, fontweight='bold')
            ax2.set_xlabel("Δt (ms)"); ax2.set_ylabel("Error (%)")

    def _plot_jo2010_ltp_ltd(self, sim_data):
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)
        pulse = sim_data.get("pulse_num", np.array([]))
        current = sim_data.get("current_exp", np.array([]))
        idx_peak = sim_data.get("idx_peak", 10)
        G_sim = sim_data.get("G_sim", np.array([]))
        if len(pulse) > 0:
            ax1.plot(pulse[:idx_peak+1], current[:idx_peak+1], 's-', ms=5, color='#89b4fa', label='LTP')
            ax1.plot(pulse[idx_peak:], current[idx_peak:], 'o-', ms=5, color='#f38ba8', label='LTD')
            if len(G_sim) > 0:
                G_norm = (G_sim - G_sim.min()) / (G_sim.max() - G_sim.min() + 1e-12) * (current.max() - current.min()) + current.min()
                pulse_sim = np.linspace(pulse[0], pulse[-1], len(G_sim))
                ax1.plot(pulse_sim, G_norm, '--', lw=1.8, color='#a6e3a1', label='Simulación')
            ax1.set_title("Jo 2010 — LTP/LTD", fontsize=10, fontweight='bold')
            ax1.set_xlabel("Pulso"); ax1.set_ylabel("Corriente")
            ax1.legend(loc='lower left', fontsize=8)

        labels = ['LTP', 'LTD']
        slopes = [sim_data.get('P_slope', 0.0), sim_data.get('D_slope', 0.0)]
        ax2.bar(labels, slopes, color=['#89b4fa', '#f38ba8'], edgecolor='white', width=0.5)
        ax2.axhline(0, color='#cdd6f4', ls='--', lw=0.8, alpha=0.5)
        ax2.set_ylabel("Pendiente (nA/pulso)")
        ax2.set_title("Tasas de cambio", fontsize=10, fontweight='bold')
