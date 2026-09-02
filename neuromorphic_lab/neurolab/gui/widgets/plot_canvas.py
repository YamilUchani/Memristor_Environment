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

        # Crear 6 subgráficos: (2 filas, 3 columnas)
        self.ax_iv = self.figure.add_subplot(231)
        self.ax_vi_t = self.figure.add_subplot(232)
        self.ax_i_twin = self.ax_vi_t.twinx()
        self.ax_x = self.figure.add_subplot(233)
        self.ax_r = self.figure.add_subplot(234)
        self.ax_g = self.figure.add_subplot(235)
        self.ax_p = self.figure.add_subplot(236)

        self._style_axes()

    def _style_axes(self):
        """Aplica estilos visuales oscuros y armoniosos a todas las gráficas."""
        axes = [self.ax_iv, self.ax_vi_t, self.ax_x, self.ax_r, self.ax_g, self.ax_p]
        for ax in axes:
            ax.set_facecolor('#181825')
            ax.tick_params(colors='#cdd6f4', labelsize=8)
            ax.xaxis.label.set_color('#cdd6f4')
            ax.yaxis.label.set_color('#cdd6f4')
            ax.title.set_color('#89b4fa')
            for spine in ax.spines.values():
                spine.set_color('#45475a')
            ax.grid(True, linestyle='--', alpha=0.3, color='#585b70')

        self.ax_i_twin.tick_params(colors='#f38ba8', labelsize=8)
        self.ax_i_twin.yaxis.label.set_color('#f38ba8')
        for spine in self.ax_i_twin.spines.values():
            spine.set_color('#45475a')

        self.figure.tight_layout(pad=2.0)

    def plot_results(self, t: np.ndarray, v_in: np.ndarray, v_drop: np.ndarray, i_out: np.ndarray, x_state: np.ndarray, r_hist: np.ndarray, g_hist: np.ndarray):
        """Actualiza y dibuja los resultados de la simulación."""
        # Limpiar ejes
        self.ax_iv.clear()
        self.ax_vi_t.clear()
        self.ax_i_twin.clear()
        self.ax_x.clear()
        self.ax_r.clear()
        self.ax_g.clear()
        self.ax_p.clear()

        # Restaurar estilo
        self._style_axes()

        t_ms = t * 1e3  # Convertir a ms para mejor visualización

        # 1. Curva I-V (Histéresis)
        self.ax_iv.plot(v_drop, i_out * 1e6, color='#f38ba8', linewidth=1.5)
        self.ax_iv.set_title("Curva I-V (Histéresis)", fontsize=9, fontweight='bold')
        self.ax_iv.set_xlabel("Voltaje $V_{drop}$ (V)")
        self.ax_iv.set_ylabel("Corriente (µA)")
        self.ax_iv.grid(True, linestyle='--', alpha=0.3)

        # 2. Señales Temporales V(t) e I(t) en el mismo gráfico (Doble Eje Y)
        self.ax_vi_t.plot(t_ms, v_in, color='#cdd6f4', linewidth=1.0, linestyle='--', label='V_in (Gen)')
        self.ax_vi_t.plot(t_ms, v_drop, color='#89b4fa', linewidth=1.5, label='V_drop (Mem)')
        self.ax_vi_t.set_title("Voltaje y Corriente vs Tiempo", fontsize=9, fontweight='bold')
        self.ax_vi_t.set_xlabel("Tiempo (ms)")
        self.ax_vi_t.set_ylabel("Voltaje (V)", color='#89b4fa')
        self.ax_vi_t.tick_params(axis='y', labelcolor='#89b4fa')
        self.ax_vi_t.grid(True, linestyle='--', alpha=0.3)
        self.ax_i_twin.plot(t_ms, i_out * 1e3, color='#f38ba8', linewidth=1.2, linestyle='--', label='Corriente I(t)')
        self.ax_i_twin.set_ylabel("Corriente (mA)", color='#f38ba8')
        self.ax_i_twin.yaxis.set_label_position("right")
        self.ax_i_twin.yaxis.tick_right()
        self.ax_i_twin.tick_params(axis='y', labelcolor='#f38ba8')

        lines, labels = self.ax_vi_t.get_legend_handles_labels()
        lines2, labels2 = self.ax_i_twin.get_legend_handles_labels()
        self.ax_vi_t.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')

        # 3. Estado Interno x(t)
        self.ax_x.plot(t_ms, x_state, color='#a6e3a1', linewidth=1.5, label='x(t)')
        self.ax_x.axhline(0.0, color='#f38ba8', linestyle=':', alpha=0.5)
        self.ax_x.axhline(1.0, color='#f38ba8', linestyle=':', alpha=0.5)
        self.ax_x.set_title("Estado Interno Normalizado x(t)", fontsize=9, fontweight='bold')
        self.ax_x.set_xlabel("Tiempo (ms)")
        self.ax_x.set_ylabel("x (0.0 = OFF, 1.0 = ON)")
        self.ax_x.set_ylim(-0.05, 1.05)

        # 4. Resistencia R(t)
        self.ax_r.plot(t_ms, r_hist / 1e3, color='#fab387', linewidth=1.5)
        self.ax_r.set_title("Resistencia Instantánea R(t)", fontsize=9, fontweight='bold')
        self.ax_r.set_xlabel("Tiempo (ms)")
        self.ax_r.set_ylabel("Resistencia (kΩ)")

        # 5. Conductancia G(t)
        self.ax_g.plot(t_ms, g_hist * 1e6, color='#89dceb', linewidth=1.5)
        self.ax_g.set_title("Conductancia Instantánea G(t)", fontsize=9, fontweight='bold')
        self.ax_g.set_xlabel("Tiempo (ms)")
        self.ax_g.set_ylabel("Conductancia (µS)")

        # 6. Potencia Instantánea P(t) = V_drop(t) * I(t)
        p_mw = v_drop * (i_out * 1e3)
        self.ax_p.plot(t_ms, p_mw, color='#cba6f7', linewidth=1.5)
        self.ax_p.set_title("Potencia Instantánea P(t)", fontsize=9, fontweight='bold')
        self.ax_p.set_xlabel("Tiempo (ms)")
        self.ax_p.set_ylabel("Potencia (mW)")

        try:
            self.figure.tight_layout(pad=1.8)
        except Exception:
            pass
        self.canvas.draw()
