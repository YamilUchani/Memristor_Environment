"""
neurolab.gui.widgets.plot_canvas
=================================
Canvas Matplotlib para el panel de memristor (MplCanvas).
"""

import numpy as np
from .canvas_base import BaseMplCanvas


class MplCanvas(BaseMplCanvas):
    """
    Canvas para la simulación del memristor.
    Subplots dinámicos según `active_plots`.
    """

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        # Grilla 2x3 por defecto (6 subplots posibles)
        super().__init__(parent, rows=2, cols=3,
                         figsize=(width, height), dpi=dpi, sharex=False)

    def plot_results(
        self,
        t: np.ndarray,
        v_in: np.ndarray,
        v_drop: np.ndarray,
        i_out: np.ndarray,
        x_state: np.ndarray,
        r_hist: np.ndarray,
        g_hist: np.ndarray,
        val_data: dict | None = None,
        active_plots: list[str] | None = None,
        ensemble_data: list[dict] | None = None
    ):
        """Actualiza y dibuja los resultados de la simulación."""
        if not active_plots:
            active_plots = ["iv", "vt_it", "x", "r", "g", "p"]

        self.figure.clear()
        n = len(active_plots)

        if n == 0:
            self.canvas.draw()
            return

        if n == 1:
            rows, cols = 1, 1
        elif n == 2:
            rows, cols = 1, 2
        elif n == 3:
            rows, cols = 1, 3
        elif n == 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3

        self._rows = rows
        self._cols = cols
        self.axes = self._create_axes(rows, cols, sharex=False)

        t_ms = t * 1e3
        idx_ds = self._decimate(t)
        idx_ens = self._decimate_ensemble(t)

        t_ms_ds = t_ms[idx_ds]
        v_in_ds = v_in[idx_ds]
        v_drop_ds = v_drop[idx_ds]
        i_out_ds = i_out[idx_ds]
        x_state_ds = x_state[idx_ds]
        r_hist_ds = r_hist[idx_ds]
        g_hist_ds = g_hist[idx_ds]

        es_csv_estatico = (
            val_data is not None and
            (val_data.get("wd_val") is None or not np.isfinite(val_data["wd_val"]).any())
        )
        val_data_time = None if es_csv_estatico else val_data

        for idx, plot_id in enumerate(active_plots):
            ax = self.axes[idx]

            if plot_id == "iv":
                if ensemble_data:
                    for ens_idx, k_item in enumerate(ensemble_data):
                        label_ens = 'Población D2D' if ens_idx == 0 else None
                        ax.plot(k_item["v_drop"][idx_ens], k_item["i_out"][idx_ens] * 1e3,
                                color='#f5c2e7', alpha=0.3, linewidth=0.8, label=label_ens)
                ax.plot(v_drop_ds, i_out_ds * 1e3, color='#f38ba8', linewidth=1.8, label='Simulado (Base)')
                if val_data is not None:
                    v_ref = val_data.get("v_interp")
                    i_ref_mA = val_data.get("i_val_mA")
                    wd_ref = val_data.get("wd_val")
                    if v_ref is not None and i_ref_mA is not None and (wd_ref is None or not np.isfinite(wd_ref).any()):
                        mask_set = v_ref >= 0.0
                        mask_reset = v_ref < 0.0
                        ax.scatter(v_ref[mask_set], i_ref_mA[mask_set], color='#f9e2af',
                                   s=16, alpha=0.85, label='CSV SET (Prezioso)', zorder=3)
                        ax.scatter(v_ref[mask_reset], i_ref_mA[mask_reset], color='#89dceb',
                                   s=16, alpha=0.85, label='CSV RESET (Prezioso)', zorder=3)
                    elif v_ref is not None and i_ref_mA is not None:
                        ax.plot(v_ref, i_ref_mA, color='#f9e2af', linewidth=1.2,
                                linestyle='--', label='Validación CSV')
                ax.legend(loc='upper left', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Curva I-V (Histéresis)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Voltaje $V_{drop}$ (V)")
                ax.set_ylabel("Corriente (mA)")

            elif plot_id == "vt_it":
                ax.plot(t_ms_ds, v_in_ds, color='#cdd6f4', linewidth=1.0, linestyle='--', label='V_in (Gen)')
                ax.plot(t_ms_ds, v_drop_ds, color='#89b4fa', linewidth=1.5, label='V_drop (Mem)')
                if val_data_time is not None:
                    ax.plot(val_data_time["t_v"] * 1e3, val_data_time["v_val"],
                            color='#f9e2af', linewidth=1.2, linestyle=':', label='V_val (CSV)')
                ax.set_title("Voltaje y Corriente vs Tiempo", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Voltaje (V)", color='#89b4fa')
                ax.tick_params(axis='y', labelcolor='#89b4fa')

                ax_i_twin = ax.twinx()
                if ensemble_data:
                    for k_item in ensemble_data:
                        ax_i_twin.plot(t_ms[idx_ens], k_item["i_out"][idx_ens] * 1e3,
                                       color='#f5c2e7', alpha=0.25, linewidth=0.8)
                ax_i_twin.plot(t_ms_ds, i_out_ds * 1e3, color='#f38ba8', linewidth=1.2,
                               linestyle='--', label='Corriente I(t)')
                if val_data_time is not None:
                    ax_i_twin.plot(val_data_time["t_i"] * 1e3, val_data_time["i_val_mA"],
                                   color='#fab387', linewidth=1.2, linestyle=':', label='I_val (CSV)')
                ax_i_twin.set_ylabel("Corriente (mA)", color='#f38ba8')
                ax_i_twin.tick_params(colors='#f38ba8', labelsize=8)
                for spine in ax_i_twin.spines.values():
                    spine.set_color('#45475a')

                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax_i_twin.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper right',
                          fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')

            elif plot_id == "x":
                if ensemble_data:
                    for ens_idx, k_item in enumerate(ensemble_data):
                        label_ens = 'Población D2D' if ens_idx == 0 else None
                        ax.plot(t_ms[idx_ens], k_item["x_state"][idx_ens],
                                color='#a6e3a1', alpha=0.3, linewidth=0.8, label=label_ens)
                ax.plot(t_ms_ds, x_state_ds, color='#a6e3a1', linewidth=1.8, label='x(t) Simulado')
                if val_data_time is not None:
                    ax.plot(val_data_time["t_w"] * 1e3, val_data_time["wd_val"],
                            color='#f9e2af', linewidth=1.5, linestyle='--', label='w/d (CSV)')
                ax.legend(loc='lower right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.axhline(0.0, color='#f38ba8', linestyle=':', alpha=0.5)
                ax.axhline(1.0, color='#f38ba8', linestyle=':', alpha=0.5)
                ax.set_title("Estado Interno Normalizado x(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("x (0.0 = OFF, 1.0 = ON)")
                max_x_val = np.max(x_state_ds)
                min_x_val = np.min(x_state_ds)
                if max_x_val <= 1.1 and min_x_val >= -0.1:
                    ax.set_ylim(-0.05, 1.05)

            elif plot_id == "r":
                if ensemble_data:
                    for ens_idx, k_item in enumerate(ensemble_data):
                        label_ens = 'Población D2D' if ens_idx == 0 else None
                        ax.plot(t_ms[idx_ens], k_item["r_hist"][idx_ens] / 1e3,
                                color='#f9e2af', alpha=0.3, linewidth=0.8, label=label_ens)
                ax.plot(t_ms_ds, r_hist_ds / 1e3, color='#fab387', linewidth=1.8, label='R(t) Simulado')
                if val_data_time is not None and "r_val_kohm" in val_data_time:
                    ax.plot(val_data_time["t_i"] * 1e3, val_data_time["r_val_kohm"],
                            color='#f9e2af', linewidth=1.2, linestyle='--', label='R_val (CSV)')
                ax.legend(loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Resistencia Instantánea R(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Resistencia (kΩ)")

            elif plot_id == "g":
                if ensemble_data:
                    for k_item in ensemble_data:
                        ax.plot(t_ms[idx_ens], k_item["g_hist"][idx_ens] * 1e6,
                                color='#94e2d5', alpha=0.3, linewidth=0.8)
                ax.plot(t_ms_ds, g_hist_ds * 1e6, color='#89dceb', linewidth=1.8, label='G(t) Simulado')
                if val_data_time is not None and "g_val_us" in val_data_time:
                    ax.plot(val_data_time["t_i"] * 1e3, val_data_time["g_val_us"],
                            color='#f9e2af', linewidth=1.2, linestyle='--', label='G_val (CSV)')
                ax.legend(loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Conductancia Instantánea G(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Conductancia (µS)")

            elif plot_id == "p":
                if ensemble_data:
                    for k_item in ensemble_data:
                        ax.plot(t_ms[idx_ens],
                                k_item["v_drop"][idx_ens] * (k_item["i_out"][idx_ens] * 1e3),
                                color='#cba6f7', alpha=0.25, linewidth=0.8)
                p_mw_ds = v_drop_ds * (i_out_ds * 1e3)
                ax.plot(t_ms_ds, p_mw_ds, color='#cba6f7', linewidth=1.8, label='P(t) Simulado')
                if val_data_time is not None:
                    p_csv_mw = val_data_time["v_interp"] * val_data_time["i_val_mA"]
                    ax.plot(val_data_time["t_i"] * 1e3, p_csv_mw,
                            color='#f9e2af', linewidth=1.2, linestyle='--', label='P_val (CSV)')
                ax.legend(loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Potencia Instantánea P(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Potencia (mW)")

        self._apply_style(self.axes[:n])
        self.refresh()
