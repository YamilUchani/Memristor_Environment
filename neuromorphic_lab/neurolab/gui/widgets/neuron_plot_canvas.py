import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class NeuronMplCanvas(QWidget):
    """Lienzo de dibujo Matplotlib para visualización de la neurona LIF."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # 4 subplots: R_M(t), I(t), V_m(t), Raster
        self.ax_r = self.figure.add_subplot(411)
        self.ax_i = self.figure.add_subplot(412, sharex=self.ax_r)
        self.ax_v = self.figure.add_subplot(413, sharex=self.ax_r)
        self.ax_s = self.figure.add_subplot(414, sharex=self.ax_r)
        
        self.figure.subplots_adjust(hspace=0.6, left=0.1, right=0.95, top=0.92, bottom=0.1)
        self._setup_axes()

    def _setup_axes(self):
        for ax in [self.ax_r, self.ax_i, self.ax_v, self.ax_s]:
            ax.set_facecolor("#181825")
            ax.tick_params(colors="#a6adc8", labelcolor="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_color("#45475a")
                
        self.ax_r.set_title("① Función de Sinapsis Variable ($R_M$)", color="#cdd6f4", fontsize=10)
        self.ax_r.set_ylabel(r"$R_M$ (k$\Omega$)", color="#cdd6f4")
        self.ax_r.grid(True, color="#313244", linestyle="--")
        self.ax_r.ticklabel_format(useOffset=False, style='plain', axis='y')
        
        self.ax_i.set_title("② Control de Corriente Inyectada ($I_{mem}$)", color="#cdd6f4", fontsize=10)
        self.ax_i.set_ylabel(r"I ($\mu$A)", color="#cdd6f4")
        self.ax_i.grid(True, color="#313244", linestyle="--")
        
        self.ax_v.set_title("③ Integración Neuronal ($V_c$ puro sin offset)", color="#cdd6f4", fontsize=10)
        self.ax_v.set_ylabel("$V_c$ (V)", color="#cdd6f4")
        self.ax_v.grid(True, color="#313244", linestyle="--")
        
        self.ax_s.set_title("④ Disparo Neuronal (Raster Plot)", color="#cdd6f4", fontsize=10)
        self.ax_s.set_ylabel("Eventos", color="#cdd6f4")
        self.ax_s.set_xlabel("Tiempo (s)", color="#cdd6f4")
        self.ax_s.grid(True, color="#313244", linestyle="--")
        self.ax_s.set_yticks([])

    def plot_results(self, t, i_in, v_m, r_m_hist, spike_times, v_th, v_reset, v_rest):
        self.ax_r.clear()
        self.ax_i.clear()
        self.ax_v.clear()
        self.ax_s.clear()
        self._setup_axes()
        
        # Resistencia
        r_m_kohm = r_m_hist / 1e3
        self.ax_r.plot(t, r_m_kohm, color="#a6e3a1", linewidth=1.5)
        
        # Ajuste dinámico de los límites del eje y para la resistencia
        r_min, r_max = np.min(r_m_kohm), np.max(r_m_kohm)
        margin = max((r_max - r_min) * 0.1, 1.0)
        self.ax_r.set_ylim(max(0, r_min - margin), r_max + margin)
        
        # Corriente (interpretada como A, graficada en uA)
        self.ax_i.plot(t, i_in * 1e6, color="#f9e2af", linewidth=1.5)
        
        # Potencial Puro del Capacitor (V_c = V_m - V_rest)
        v_c = v_m - v_rest
        v_th_c = v_th - v_rest
        v_reset_c = v_reset - v_rest
        
        self.ax_v.plot(t, v_c, color="#89b4fa", linewidth=1.5)
        self.ax_v.axhline(v_th_c, color="#f38ba8", linestyle="--", alpha=0.8, label="$V_{th}$ relativo")
        self.ax_v.axhline(v_reset_c, color="#a6e3a1", linestyle=":", alpha=0.8, label="$V_{reset}$ relativo")
        self.ax_v.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")
        
        # Raster
        if len(spike_times) > 0:
            self.ax_s.vlines(spike_times, 0, 1, color="#f38ba8", linewidth=2.0)
            
        self.ax_s.set_ylim(0, 1)
        self.canvas.draw()
