import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MplCanvas(QWidget):
    """
    Widget de Matplotlib embebido en PySide6 para la visualización en tiempo real
    de los resultados de la simulación del Memristor.
    """

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        super().__init__(parent)

        # Configurar figura con tema oscuro elegante
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e2e')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Disposición visual
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _style_axes_list(self, axes: list):
        """Aplica estilos visuales oscuros y armoniosos a las gráficas dinámicas."""
        for ax in axes:
            ax.set_facecolor('#181825')
            ax.tick_params(colors='#cdd6f4', labelsize=8)
            ax.xaxis.label.set_color('#cdd6f4')
            ax.yaxis.label.set_color('#cdd6f4')
            ax.title.set_color('#89b4fa')
            for spine in ax.spines.values():
                spine.set_color('#45475a')
            ax.grid(True, linestyle='--', alpha=0.3, color='#585b70')

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
        """Actualiza y dibuja los resultados de la simulación seleccionados dinámicamente."""
        if not active_plots:
            active_plots = ["iv", "vt_it", "x", "r", "g", "p"]

        self.figure.clear()
        n = len(active_plots)
        
        if n == 0:
            self.canvas.draw()
            return
            
        if n == 1: rows, cols = 1, 1
        elif n == 2: rows, cols = 1, 2
        elif n == 3: rows, cols = 1, 3
        elif n == 4: rows, cols = 2, 2
        else: rows, cols = 2, 3

        axes_to_style = []
        t_ms = t * 1e3

        for idx, plot_id in enumerate(active_plots, start=1):
            ax = self.figure.add_subplot(rows, cols, idx)
            axes_to_style.append(ax)

            if plot_id == "iv":
                if ensemble_data:
                    for ens_idx, k_item in enumerate(ensemble_data):
                        label_ens = 'Población D2D' if ens_idx == 0 else None
                        ax.plot(k_item["v_drop"], k_item["i_out"] * 1e3, color='#f5c2e7', alpha=0.3, linewidth=0.8, label=label_ens)
                ax.plot(v_drop, i_out * 1e3, color='#f38ba8', linewidth=1.8, label='Simulado (Base)')
                if val_data is not None:
                    ax.plot(val_data["v_interp"], val_data["i_val_mA"], color='#f9e2af', linewidth=1.2, linestyle='--', label='Validación CSV')
                ax.legend(loc='upper left', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Curva I-V (Histéresis)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Voltaje $V_{drop}$ (V)")
                ax.set_ylabel("Corriente (mA)")

            elif plot_id == "vt_it":
                ax.plot(t_ms, v_in, color='#cdd6f4', linewidth=1.0, linestyle='--', label='V_in (Gen)')
                ax.plot(t_ms, v_drop, color='#89b4fa', linewidth=1.5, label='V_drop (Mem)')
                if val_data is not None:
                    ax.plot(val_data["t_v"] * 1e3, val_data["v_val"], color='#f9e2af', linewidth=1.2, linestyle=':', label='V_val (CSV)')
                ax.set_title("Voltaje y Corriente vs Tiempo", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Voltaje (V)", color='#89b4fa')
                ax.tick_params(axis='y', labelcolor='#89b4fa')

                ax_i_twin = ax.twinx()
                if ensemble_data:
                    for k_item in ensemble_data:
                        ax_i_twin.plot(t_ms, k_item["i_out"] * 1e3, color='#f5c2e7', alpha=0.25, linewidth=0.8)
                ax_i_twin.plot(t_ms, i_out * 1e3, color='#f38ba8', linewidth=1.2, linestyle='--', label='Corriente I(t)')
                if val_data is not None:
                    ax_i_twin.plot(val_data["t_i"] * 1e3, val_data["i_val_mA"], color='#fab387', linewidth=1.2, linestyle=':', label='I_val (CSV)')
                ax_i_twin.set_ylabel("Corriente (mA)", color='#f38ba8')
                
                ax_i_twin.tick_params(colors='#f38ba8', labelsize=8)
                for spine in ax_i_twin.spines.values():
                    spine.set_color('#45475a')

                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax_i_twin.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')

            elif plot_id == "x":
                if ensemble_data:
                    for ens_idx, k_item in enumerate(ensemble_data):
                        label_ens = 'Población D2D' if ens_idx == 0 else None
                        ax.plot(t_ms, k_item["x_state"], color='#a6e3a1', alpha=0.3, linewidth=0.8, label=label_ens)
                ax.plot(t_ms, x_state, color='#a6e3a1', linewidth=1.8, label='x(t) Simulado')
                if val_data is not None:
                    ax.plot(val_data["t_w"] * 1e3, val_data["wd_val"], color='#f9e2af', linewidth=1.5, linestyle='--', label='w/d (CSV)')
                ax.legend(loc='lower right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.axhline(0.0, color='#f38ba8', linestyle=':', alpha=0.5)
                ax.axhline(1.0, color='#f38ba8', linestyle=':', alpha=0.5)
                ax.set_title("Estado Interno Normalizado x(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("x (0.0 = OFF, 1.0 = ON)")
                
                # Auto-escalar límites en Y si x(t) se desborda (>1.1 o <-0.1 en Modo Ideal)
                max_x_val = np.max(x_state)
                min_x_val = np.min(x_state)
                if max_x_val <= 1.1 and min_x_val >= -0.1:
                    ax.set_ylim(-0.05, 1.05)

            elif plot_id == "r":
                if ensemble_data:
                    for ens_idx, k_item in enumerate(ensemble_data):
                        label_ens = 'Población D2D' if ens_idx == 0 else None
                        ax.plot(t_ms, k_item["r_hist"] / 1e3, color='#f9e2af', alpha=0.3, linewidth=0.8, label=label_ens)
                ax.plot(t_ms, r_hist / 1e3, color='#fab387', linewidth=1.8, label='R(t) Simulado')
                if val_data is not None and "r_val_kohm" in val_data:
                    ax.plot(val_data["t_i"] * 1e3, val_data["r_val_kohm"], color='#f9e2af', linewidth=1.2, linestyle='--', label='R_val (CSV)')
                ax.legend(loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Resistencia Instantánea R(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Resistencia (kΩ)")

            elif plot_id == "g":
                if ensemble_data:
                    for k_item in ensemble_data:
                        ax.plot(t_ms, k_item["g_hist"] * 1e6, color='#94e2d5', alpha=0.3, linewidth=0.8)
                ax.plot(t_ms, g_hist * 1e6, color='#89dceb', linewidth=1.8, label='G(t) Simulado')
                if val_data is not None and "g_val_us" in val_data:
                    ax.plot(val_data["t_i"] * 1e3, val_data["g_val_us"], color='#f9e2af', linewidth=1.2, linestyle='--', label='G_val (CSV)')
                ax.legend(loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Conductancia Instantánea G(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Conductancia (µS)")

            elif plot_id == "p":
                if ensemble_data:
                    for k_item in ensemble_data:
                        ax.plot(t_ms, k_item["v_drop"] * (k_item["i_out"] * 1e3), color='#cba6f7', alpha=0.25, linewidth=0.8)
                p_mw = v_drop * (i_out * 1e3)
                ax.plot(t_ms, p_mw, color='#cba6f7', linewidth=1.8, label='P(t) Simulado')
                if val_data is not None:
                    p_csv_mw = val_data["v_interp"] * val_data["i_val_mA"]
                    ax.plot(val_data["t_i"] * 1e3, p_csv_mw, color='#f9e2af', linewidth=1.2, linestyle='--', label='P_val (CSV)')
                ax.legend(loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Potencia Instantánea P(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Potencia (mW)")

        self._style_axes_list(axes_to_style)
        try:
            self.figure.tight_layout(pad=1.8)
        except Exception:
            pass
        self.canvas.draw()
