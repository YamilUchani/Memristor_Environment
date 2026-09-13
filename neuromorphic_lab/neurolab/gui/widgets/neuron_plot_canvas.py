import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class NeuronMplCanvas(QWidget):
    """Lienzo de dibujo Matplotlib para visualización de la neurona LIF (5 subplots sincronizados)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 8), facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # 5 subplots: R_S(t), R_leak(t), I(t), V_m(t), V_out(t)
        self.ax_rs = self.figure.add_subplot(511)
        self.ax_rl = self.figure.add_subplot(512, sharex=self.ax_rs)
        self.ax_i  = self.figure.add_subplot(513, sharex=self.ax_rs)
        self.ax_v  = self.figure.add_subplot(514, sharex=self.ax_rs)
        self.ax_s  = self.figure.add_subplot(515, sharex=self.ax_rs)
        
        self.figure.subplots_adjust(hspace=0.65, left=0.10, right=0.95, top=0.94, bottom=0.08)
        self._setup_axes()

    def _setup_axes(self):
        for ax in [self.ax_rs, self.ax_rl, self.ax_i, self.ax_v, self.ax_s]:
            ax.set_facecolor("#181825")
            ax.tick_params(colors="#a6adc8", labelcolor="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_color("#45475a")
                
        self.ax_rs.set_title("① Resistencia Serie ($R_S$)", color="#cdd6f4", fontsize=9)
        self.ax_rs.set_ylabel(r"$R_S$ (k$\Omega$)", color="#89dceb")
        self.ax_rs.grid(True, color="#313244", linestyle="--", linewidth=0.6)
        self.ax_rs.ticklabel_format(useOffset=False, style='plain', axis='y')

        self.ax_rl.set_title("② Resistencia de Fuga ($R_{leak}$)", color="#cdd6f4", fontsize=9)
        self.ax_rl.set_ylabel(r"$R_{leak}$ (k$\Omega$)", color="#a6e3a1")
        self.ax_rl.grid(True, color="#313244", linestyle="--", linewidth=0.6)
        self.ax_rl.ticklabel_format(useOffset=False, style='plain', axis='y')
        
        self.ax_i.set_title("③ Estímulo de Entrada: Voltaje ($V_{IN}$) y Corriente ($I_{in}$)", color="#cdd6f4", fontsize=9)
        self.ax_i.set_ylabel("$V_{IN}$ (V)", color="#89b4fa")
        self.ax_i.grid(True, color="#313244", linestyle="--", linewidth=0.6)
        
        self.ax_v.set_title("④ Potencial de Membrana / Capacitor ($V_c$)", color="#cdd6f4", fontsize=9)
        self.ax_v.set_ylabel("$V_c$ (V)", color="#cdd6f4")
        self.ax_v.grid(True, color="#313244", linestyle="--", linewidth=0.6)
        
        self.ax_s.set_title("⑤ Salida de Voltaje de Disparo ($V_{OUT}$)", color="#cdd6f4", fontsize=9)
        self.ax_s.set_ylabel("$V_{OUT}$ (V)", color="#cdd6f4")
        self.ax_s.set_xlabel("Tiempo (s)", color="#cdd6f4")
        self.ax_s.grid(True, color="#313244", linestyle="--", linewidth=0.6)

    def plot_results(self, t, signal_in, v_m, r_m_hist, spike_times, v_th, v_reset, v_rest, i_in=None, r_series_hist=None):
        self.ax_rs.clear()
        self.ax_rl.clear()
        self.ax_i.clear()
        if hasattr(self, 'ax_i_twin') and self.ax_i_twin is not None:
            self.ax_i_twin.clear()
            self.ax_i_twin.remove()
            self.ax_i_twin = None
            
        self.ax_v.clear()
        self.ax_s.clear()
        self._setup_axes()
        
        # 1. Resistencia Serie R_S (Subplot ①)
        if r_series_hist is not None:
            r_series_kohm = r_series_hist / 1e3
        else:
            r_series_kohm = np.full_like(t, 100.0)
            
        self.ax_rs.plot(t, r_series_kohm, color="#89dceb", linewidth=1.5, label=r"$R_S$ (Serie)")
        r_min, r_max = np.min(r_series_kohm), np.max(r_series_kohm)
        margin = max((r_max - r_min) * 0.15, 1.0)
        self.ax_rs.set_ylim(max(0, r_min - margin), r_max + margin)
        self.ax_rs.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

        # 2. Resistencia de Fuga R_leak (Subplot ②)
        r_leak_kohm = r_m_hist / 1e3
        self.ax_rl.plot(t, r_leak_kohm, color="#a6e3a1", linewidth=1.5, label=r"$R_{leak}$ (Fuga)")
        r_min_l, r_max_l = np.min(r_leak_kohm), np.max(r_leak_kohm)
        margin_l = max((r_max_l - r_min_l) * 0.15, 10.0)
        self.ax_rl.set_ylim(max(0, r_min_l - margin_l), r_max_l + margin_l)
        self.ax_rl.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

        # 3. Señal de entrada V_IN e I_in (Subplot ③)
        max_abs = np.max(np.abs(signal_in))
        if max_abs > 0.01:
            l1 = self.ax_i.plot(t, signal_in, color="#89b4fa", linewidth=1.5, label="$V_{IN}$ (V)")
            self.ax_i.set_ylabel("$V_{IN}$ (V)", color="#89b4fa")
            
            self.ax_i_twin = self.ax_i.twinx()
            self.ax_i_twin.tick_params(colors="#f9e2af", labelcolor="#f9e2af")
            self.ax_i_twin.set_ylabel(r"$I_{in}$ ($\mu$A)", color="#f9e2af")
            for spine in self.ax_i_twin.spines.values():
                spine.set_color("#45475a")
                
            if i_in is not None:
                i_uA = i_in * 1e6
            else:
                i_uA = signal_in * 10.0
                
            l2 = self.ax_i_twin.plot(t, i_uA, color="#f9e2af", linewidth=1.2, linestyle="--", label=r"$I_{in}$ ($\mu$A)")
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            self.ax_i.legend(lines, labels, loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)
        else:
            self.ax_i.set_title("③ Control de Corriente Inyectada ($I_{in}$)", color="#cdd6f4", fontsize=9)
            self.ax_i.set_ylabel(r"$I_{in}$ ($\mu$A)", color="#f9e2af")
            self.ax_i.plot(t, signal_in * 1e6, color="#f9e2af", linewidth=1.5)

        # 4. Potencial del Capacitor V_c (Subplot ④)
        # Mostramos V_m directamente (no V_m - V_rest) para que V_rest, V_th y V_reset
        # sean líneas absolutas interpretables con sus valores físicos reales.
        self.ax_v.plot(t, v_m, color="#cba6f7", linewidth=1.5, label="$V_m$ (membrana)")

        # Líneas de referencia
        self.ax_v.axhline(v_th,    color="#f38ba8", linestyle="--", alpha=0.8, linewidth=1.2, label=f"$V_{{th}}$ = {v_th:+.2f} V")
        self.ax_v.axhline(v_reset, color="#a6e3a1", linestyle=":",  alpha=0.8, linewidth=1.2, label=f"$V_{{reset}}$ = {v_reset:+.2f} V")
        self.ax_v.axhline(v_rest,  color="#585b70", linestyle="-.", alpha=0.6, linewidth=0.8, label=f"$V_{{rest}}$ = {v_rest:+.2f} V")

        # Picos visuales en los instantes de spike (Punto 3 auditoría v2026.09.11)
        # Se dibujan marcadores al nivel de V_th para que los spikes sean visibles.
        # No modifica los datos de V_m — solo es un artefacto gráfico educativo.
        if len(spike_times) > 0:
            spike_v_vals = [v_th] * len(spike_times)
            self.ax_v.scatter(spike_times, spike_v_vals,
                              color="#f38ba8", s=25, zorder=5, marker="^",
                              label=f"Spikes ({len(spike_times)})")

        # Rango Y automático: incluye V_reset negativo (AHP) + V_th + margen
        y_lo = min(v_reset, v_rest, np.min(v_m)) - abs(v_th) * 0.15
        y_hi = max(v_th, np.max(v_m)) + abs(v_th) * 0.15
        self.ax_v.set_ylim(y_lo, y_hi)
        self.ax_v.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)


        # 5. Salida V_OUT Spikes (Subplot ⑤)
        steps = len(t)
        dt = t[1] - t[0] if steps > 1 else 1e-4
        v_out = 0.02 + 0.005 * np.sin(2.0 * np.pi * 50.0 * t)
        v_peak = max(0.53, 0.55 * v_th)
        
        for st in spike_times:
            idx = int(np.searchsorted(t, st))
            if 0 <= idx < steps:
                tail_len = int(0.002 / dt)
                for m in range(min(tail_len, steps - idx)):
                    decay = np.exp(-m * dt / 0.0004)
                    v_out[idx + m] = max(v_out[idx + m], 0.02 + (v_peak - 0.02) * decay)
                    
        self.ax_s.plot(t, v_out, color="#f38ba8", linewidth=1.5, label="$V_{OUT}$ Spikes")
        self.ax_s.set_ylim(-0.02, max(0.6, v_peak + 0.08))
        self.ax_s.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)
        self.canvas.draw()
