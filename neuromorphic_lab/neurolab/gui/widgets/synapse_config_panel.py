"""
neurolab.gui.widgets.synapse_config_panel
=========================================
Panel de control de PySide6 para la configuración e interacción con el módulo de
plasticidad sináptica (Fase 3: LTP, LTD, STDP, Spikes y Evidencia V-I-R-G).
"""

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QComboBox,
    QLabel, QPushButton, QTabWidget
)
from PySide6.QtCore import Signal, Qt
from neurolab.gui.widgets.arrow_spinbox import ArrowDoubleSpinBox
from neurolab.gui.widgets.config_panel import ConfigPanel
from neurolab.core.memristor import Memristor
from neurolab.synapses import (
    MemristiveSynapse, LTPRule, LTDRule, STDPRule, AntiSTDPRule
)


class SynapseConfigPanel(QWidget):
    """Panel de configuración e interacción con plasticidad sináptica."""

    param_changed = Signal()
    simulation_requested = Signal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # ── GroupBox: Selección de Experimento ──────────────────────────────
        group_exp = QGroupBox("⚡ Experimento de Plasticidad Sináptica")
        lay_exp = QVBoxLayout(group_exp)

        self.combo_exp = QComboBox()
        self.combo_exp.addItems([
            "⚡ 1 · LTP y LTD (Pulsos Directos +1V / -1V)",
            "🔄 2 · Ciclo Reversible LTP (50) → LTD (50)",
            "🎯 3 · STDP Hebbiano vs. Anti-Hebbiano (Ventana)",
            "⚡ 4 · Spikes PRE y POST en el Tiempo (Waveforms)",
            "📈 5 · Evolución Temporal de W(t) (Secuencia)",
            "🔬 6 · Evidencia V-I-R-G (4 Paneles Multivariable)",
            "🔄 7 · Curva I-V e Histéresis Pinzada",
            "📊 8 · Validación Jo 2010 — STDP (jo2010_stdp.csv)",
            "📊 9 · Validación Jo 2010 — LTP/LTD 200 Pulsos (jo2010_ltp_ltd.csv)"
        ])
        self.combo_exp.setStyleSheet("color: #cdd6f4; background-color: #181825; border: 1px solid #45475a; padding: 6px; border-radius: 4px; font-weight: bold;")
        self.combo_exp.currentIndexChanged.connect(self._on_exp_changed)
        lay_exp.addWidget(self.combo_exp)
        layout.addWidget(group_exp)

        # ── GroupBox: Subpestañas Físicas de Memristor ────────────────────────
        group_dev = QGroupBox("🔬 Dispositivo Memristivo Base (Física Completa)")
        lay_dev = QVBoxLayout(group_dev)
        lay_dev.setContentsMargins(4, 8, 4, 8)

        self.memristor_subtabs = QTabWidget()
        self.config_panel_1 = ConfigPanel()   # Strukov Ideal
        self.config_panel_2 = ConfigPanel()   # HfO₂ Serie Neurona
        self.config_panel_3 = ConfigPanel()   # Prezioso 2014

        # Preconfigurar perfiles por defecto
        self.config_panel_1.txt_device_name.setText("Strukov TiO₂ (Ideal)")
        self.config_panel_1.combo_material.setCurrentText("TiO₂ (Dióxido de Titanio - Strukov 2008)")
        self.config_panel_1.spin_r_on.setValue(100.0)
        self.config_panel_1.combo_ron_unit.setCurrentText("Ω")
        self.config_panel_1.spin_r_off.setValue(16.0)
        self.config_panel_1.combo_roff_unit.setCurrentText("kΩ")
        self.config_panel_1.spin_x0.setValue(0.10)

        self.config_panel_2.txt_device_name.setText("HfO₂ Serie Neurona")
        self.config_panel_2.combo_material.setCurrentText("HfO₂ (Óxido de Hafnio - CMOS LIF 2025)")
        self.config_panel_2.combo_model_name.setCurrentText("strukov")
        self.config_panel_2.spin_r_on.setValue(1.0)
        self.config_panel_2.combo_ron_unit.setCurrentText("kΩ")
        self.config_panel_2.spin_r_off.setValue(1.0)
        self.config_panel_2.combo_roff_unit.setCurrentText("MΩ")
        self.config_panel_2.spin_x0.setValue(0.05)
        self.config_panel_2.combo_realism_mode.setCurrentIndex(3)
        self.config_panel_2.chk_volatile.setChecked(True)

        self.config_panel_3.txt_device_name.setText("Prezioso 2014 (Al₂O₃/TiO₂-x)")
        self.config_panel_3.combo_material.setCurrentText("TiO₂ (Dióxido de Titanio - Strukov 2008)")
        self.config_panel_3.combo_model_name.setCurrentText("prezioso")
        self.config_panel_3.spin_r_on.setValue(4.75)
        self.config_panel_3.combo_ron_unit.setCurrentText("kΩ")
        self.config_panel_3.spin_r_off.setValue(1.0)
        self.config_panel_3.combo_roff_unit.setCurrentText("MΩ")
        self.config_panel_3.spin_x0.setValue(0.05)
        self.config_panel_3.combo_realism_mode.setCurrentIndex(1)

        self.memristor_subtabs.addTab(self.config_panel_1, "🔬 1 · Strukov Ideal")
        self.memristor_subtabs.addTab(self.config_panel_2, "🧠 2 · HfO₂ Neurona")
        self.memristor_subtabs.addTab(self.config_panel_3, "📊 3 · Prezioso 2014")

        lay_dev.addWidget(self.memristor_subtabs)
        layout.addWidget(group_dev)

        # ── GroupBox: Parámetros STDP ───────────────────────────────────────
        group_stdp = QGroupBox("🎯 Parámetros STDP (Bi & Poo / Gerstner)")
        form_stdp = QFormLayout(group_stdp)

        self.spin_aplus = ArrowDoubleSpinBox(value=0.20, min_val=0.001, max_val=0.5, step=0.01)
        self.spin_aminus = ArrowDoubleSpinBox(value=-0.20, min_val=-0.5, max_val=-0.001, step=0.005)
        self.spin_tauplus = ArrowDoubleSpinBox(value=30.0, min_val=1.0, max_val=100.0, step=1.0, suffix=" ms")
        self.spin_tauminus = ArrowDoubleSpinBox(value=30.0, min_val=1.0, max_val=100.0, step=1.0, suffix=" ms")

        form_stdp.addRow("Amplitud A+ (LTP):", self.spin_aplus)
        form_stdp.addRow("Amplitud A− (LTD):", self.spin_aminus)
        form_stdp.addRow("Constante τ+:", self.spin_tauplus)
        form_stdp.addRow("Constante τ−:", self.spin_tauminus)
        layout.addWidget(group_stdp)

        # Botón de Simulación
        btn_run = QPushButton("🚀 Ejecutar Simulación de Plasticidad")
        btn_run.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
                font-size: 13px;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
        """)
        btn_run.clicked.connect(self.run_simulation)
        layout.addWidget(btn_run)

        # Panel de métricas cuantitativas
        self.metrics_label = QLabel()
        self.metrics_label.setTextFormat(Qt.RichText)
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("""
            QLabel {
                background-color: #181825;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
                color: #cdd6f4;
            }
        """)
        layout.addWidget(self.metrics_label)
        layout.addStretch()

        # Conectar cambios de spinbox y subpestañas
        for spin in (self.spin_aplus, self.spin_aminus, self.spin_tauplus, self.spin_tauminus):
            spin.spin.valueChanged.connect(self._on_param_changed)

        self.config_panel_1.param_changed.connect(lambda *_: self._on_param_changed())
        self.config_panel_2.param_changed.connect(lambda *_: self._on_param_changed())
        self.config_panel_3.param_changed.connect(lambda *_: self._on_param_changed())
        self.memristor_subtabs.currentChanged.connect(lambda *_: self.run_simulation())

    def _on_param_changed(self):
        self.param_changed.emit()

    def _on_exp_changed(self, idx: int):
        if idx == 7:  # Jo 2010 STDP
            for s in (self.spin_aplus.spin, self.spin_aminus.spin, self.spin_tauplus.spin, self.spin_tauminus.spin):
                s.blockSignals(True)
            self.spin_aplus.setValue(0.20)
            self.spin_aminus.setValue(-0.20)
            self.spin_tauplus.setValue(30.0)
            self.spin_tauminus.setValue(30.0)
            for s in (self.spin_aplus.spin, self.spin_aminus.spin, self.spin_tauplus.spin, self.spin_tauminus.spin):
                s.blockSignals(False)
        self.run_simulation()

    def get_active_config_panel(self) -> ConfigPanel:
        curr_tab = self.memristor_subtabs.currentIndex()
        if curr_tab == 0: return self.config_panel_1
        elif curr_tab == 1: return self.config_panel_2
        else: return self.config_panel_3

    def get_active_device_name(self) -> str:
        return self.get_active_config_panel().txt_device_name.text()

    def _create_device(self) -> Memristor:
        return self.get_active_config_panel().build_memristor()

    def run_simulation(self):
        idx = self.combo_exp.currentIndex()
        dev_name = self.get_active_device_name()

        if idx == 0:  # LTP / LTD sep
            exp_type = "ltp_ltd_sep"
            mem = self._create_device()
            syn = MemristiveSynapse(mem)

            ltp = LTPRule(n_pulses=40, V_pulse=+1.0)
            G_ltp = ltp.apply(syn, dt=1e-3)
            syn.reset()

            ltd = LTDRule(n_pulses=40, V_pulse=-1.0)
            G_ltd = ltd.apply(syn, dt=1e-3)

            sim_data = {"G_ltp": G_ltp, "G_ltd": G_ltd}
            dG_ltp = (G_ltp[-1] - G_ltp[0]) * 1e6
            dG_ltd = (G_ltd[-1] - G_ltd[0]) * 1e6
            self.metrics_label.setText(
                f"<b>Dispositivo:</b> {dev_name}<br>"
                f"<b>LTP (+1V):</b> G_ini = {G_ltp[0]*1e6:.2f} μS → G_fin = {G_ltp[-1]*1e6:.2f} μS | ΔG = +{dG_ltp:.2f} μS<br>"
                f"<b>LTD (-1V):</b> G_ini = {G_ltd[0]*1e6:.2f} μS → G_fin = {G_ltd[-1]*1e6:.2f} μS | ΔG = {dG_ltd:.2f} μS"
            )

        elif idx == 1:  # LTP -> LTD ciclo
            exp_type = "ltp_ltd_ciclo"
            mem = self._create_device()
            syn = MemristiveSynapse(mem)

            G_hist = [syn.conductance]
            for _ in range(50):
                syn.update(pre_spike=True, post_spike=False, dt=1e-3, V_applied=+1.0)
                G_hist.append(syn.conductance)
            for _ in range(50):
                syn.update(pre_spike=False, post_spike=True, dt=1e-3, V_applied=-1.0)
                G_hist.append(syn.conductance)

            G_hist = np.array(G_hist)
            sim_data = {"G_history": G_hist}
            self.metrics_label.setText(
                f"<b>Dispositivo:</b> {dev_name}<br>"
                f"<b>Ciclo Reversible (100 pulsos):</b><br>"
                f"• G_inicial (0): {G_hist[0]*1e6:.2f} μS<br>"
                f"• G_max (50 LTP): {G_hist[50]*1e6:.2f} μS<br>"
                f"• G_final (100 LTD): {G_hist[-1]*1e6:.2f} μS (Reversible: ✓)"
            )

        elif idx == 2:  # STDP curves
            exp_type = "stdp_curves"
            dt_values = np.linspace(-0.08, 0.08, 33)
            h = STDPRule(A_plus=self.spin_aplus.value(), A_minus=self.spin_aminus.value(),
                         tau_plus=self.spin_tauplus.value()*1e-3, tau_minus=self.spin_tauminus.value()*1e-3)
            a = AntiSTDPRule(A_plus=self.spin_aplus.value(), A_minus=self.spin_aminus.value(),
                             tau_plus=self.spin_tauplus.value()*1e-3, tau_minus=self.spin_tauminus.value()*1e-3)

            dW_h = np.array([h.delta_w(dt_v) for dt_v in dt_values])
            dW_a = np.array([a.delta_w(dt_v) for dt_v in dt_values])

            sim_data = {"dt_ms": dt_values * 1e3, "dW_hebbian": dW_h, "dW_anti": dW_a}
            self.metrics_label.setText(
                f"<b>Ventana STDP Biológica ({dev_name}):</b><br>"
                f"• ΔW_max Hebb: +{dW_h.max():.4f} (Δt = +5 ms)<br>"
                f"• ΔW_min Hebb: {dW_h.min():.4f} (Δt = -5 ms)<br>"
                f"• τ+ = {self.spin_tauplus.value():.1f} ms, τ− = {self.spin_tauminus.value():.1f} ms"
            )

        elif idx == 3:  # STDP spikes
            exp_type = "stdp_spikes"
            t = np.linspace(0, 100, 1000)
            def spike_w(t_arr, t_sp):
                return np.where((t_arr >= t_sp) & (t_arr < t_sp + 2.0), 1.0, 0.0)

            casos = [
                {"dt": +20.0, "titulo": "Δt = +20 ms → LTP (Potenciación)", "t": t, "v_pre": spike_w(t, 40.0), "v_post": spike_w(t, 60.0)},
                {"dt": 0.0,   "titulo": "Δt = 0 ms → Coincidencia Nula", "t": t, "v_pre": spike_w(t, 40.0), "v_post": spike_w(t, 40.0)},
                {"dt": -20.0, "titulo": "Δt = −20 ms → LTD (Depresión)", "t": t, "v_pre": spike_w(t, 60.0), "v_post": spike_w(t, 40.0)},
            ]
            sim_data = {"casos_spikes": casos}
            self.metrics_label.setText(
                f"<b>Formas de Onda STDP ({dev_name}):</b><br>"
                f"• PRE antes que POST (Δt > 0) → LTP<br>"
                f"• POST antes que PRE (Δt < 0) → LTD"
            )

        elif idx == 4:  # STDP temporal
            exp_type = "stdp_temporal"
            dt_seq = np.concatenate([np.linspace(0.005, 0.040, 20), np.linspace(-0.005, -0.040, 20)])
            mem = self._create_device()
            syn = MemristiveSynapse(mem)
            h = STDPRule(A_plus=self.spin_aplus.value(), A_minus=self.spin_aminus.value(),
                         tau_plus=self.spin_tauplus.value()*1e-3, tau_minus=self.spin_tauminus.value()*1e-3)

            W_hist = [syn.weight]
            for dt_v in dt_seq:
                h.apply(syn, dt_v)
                W_hist.append(syn.weight)

            W_hist = np.array(W_hist)
            sim_data = {"dt_sequence": dt_seq * 1e3, "W_history": W_hist}
            self.metrics_label.setText(
                f"<b>Dispositivo:</b> {dev_name}<br>"
                f"<b>Evolución Temporal de W(t):</b><br>"
                f"• W inicial: {W_hist[0]:.4f}<br>"
                f"• W max (fase LTP): {W_hist[20]:.4f}<br>"
                f"• W final (fase LTD): {W_hist[-1]:.4f}"
            )

        elif idx == 5:  # V-I-R-G evidence
            exp_type = "virg_evidence"
            def sim_p(v_val):
                m = self._create_device()
                t_arr = np.arange(0, 40e-3, 1e-4)
                V = np.array([v_val if (tk % 1e-3) < 0.5e-3 else 0.0 for tk in t_arr])
                I, R, G = np.zeros_like(V), np.zeros_like(V), np.zeros_like(V)
                for k, vk in enumerate(V):
                    I[k] = m.current(vk)
                    R[k] = m.resistance
                    G[k] = 1.0 / R[k]
                    m.update(vk, 1e-4)
                return t_arr * 1e3, V, I, R, G

            t_ms, V_ltp, I_ltp, R_ltp, G_ltp = sim_p(+1.0)
            _, V_ltd, I_ltd, R_ltd, G_ltd = sim_p(-1.0)

            sim_data = {
                "t_ms": t_ms, "V_ltp": V_ltp, "V_ltd": V_ltd,
                "I_ltp": I_ltp, "I_ltd": I_ltd, "R_ltp": R_ltp, "R_ltd": R_ltd,
                "G_ltp": G_ltp, "G_ltd": G_ltd
            }
            self.metrics_label.setText(
                f"<b>Dispositivo:</b> {dev_name}<br>"
                f"• LTP (+1V): I_mean = {I_ltp[I_ltp>0].mean()*1e6:.1f} μA, ΔR = -{ (R_ltp[0]-R_ltp[-1])/1e3:.2f} kΩ<br>"
                f"• LTD (-1V): I_mean = {I_ltd[I_ltd<0].mean()*1e6:.1f} μA, ΔR = +{ (R_ltd[-1]-R_ltd[0])/1e3:.2f} kΩ"
            )

        elif idx == 6:  # IV hysteresis
            exp_type = "iv_hysteresis"
            m = self._create_device()
            t_arr = np.linspace(0, 2, 800)
            V = 1.0 * np.sin(2 * np.pi * t_arr)
            I, R = np.zeros_like(V), np.zeros_like(V)
            for k, vk in enumerate(V):
                I[k] = m.current(vk)
                R[k] = m.resistance
                m.update(vk, 1e-4)

            sim_data = {"V_sweep": V, "I_sweep": I, "R_sweep": R}
            self.metrics_label.setText(
                f"<b>Dispositivo:</b> {dev_name}<br>"
                f"• I_max = {I.max()*1e3:.3f} mA, I_min = {I.min()*1e3:.3f} mA<br>"
                f"• R_min = {R.min()/1e3:.2f} kΩ, R_max = {R.max()/1e3:.2f} kΩ"
            )

        elif idx == 7:  # Jo 2010 STDP Validation (jo2010_stdp.csv)
            exp_type = "jo2010_stdp"
            from neurolab.validation import validate_jo2010_stdp

            A_plus = self.spin_aplus.value()
            A_minus = self.spin_aminus.value()
            tau_plus = self.spin_tauplus.value() * 1e-3
            tau_minus = self.spin_tauminus.value() * 1e-3

            stdp = STDPRule(A_plus=A_plus, A_minus=A_minus, tau_plus=tau_plus, tau_minus=tau_minus)
            dt_values = np.linspace(-80.0, 80.0, 161)
            dw_sim = np.array([stdp.delta_w(dt * 1e-3) * 100.0 for dt in dt_values])

            results = validate_jo2010_stdp(dt_values, dw_sim)
            sim_data = results

            status_icon = "PASS [OK]" if results['r2'] > 0.85 else "REVISAR [!]"
            color_hex = "#a6e3a1" if results['r2'] > 0.85 else "#fab387"

            self.metrics_label.setText(
                f"<b>Validación Jo 2010 — STDP (Fig. 3a) [{dev_name}]:</b><br>"
                f"• <b>Fuente CSV:</b> {results['source']} (jo2010_stdp.csv)<br>"
                f"• <b>R² (Ajuste):</b> <font color='{color_hex}'><b>{results['r2']:.4f}</b></font> ({status_icon})<br>"
                f"• <b>MAE:</b> {results['mae']:.4f} | <b>RMSE:</b> {results['rmse']:.4f}<br>"
                f"• <b>Error Máx:</b> {results['err_max']:.2f} %"
            )

        elif idx == 8:  # Jo 2010 LTP/LTD Validation (jo2010_ltp_ltd.csv)
            exp_type = "jo2010_ltp_ltd"
            from neurolab.validation import validate_jo2010_ltp_ltd

            results = validate_jo2010_ltp_ltd()

            # Simulación con dispositivo memristivo activo y regla de saturación no lineal
            mem = self._create_device()
            syn = MemristiveSynapse(mem)
            ltp = LTPRule(n_pulses=100, V_pulse=+1.0, saturation=True, tau_sat=35.0)
            G_ltp = ltp.apply(syn, dt=1e-3)
            ltd = LTDRule(n_pulses=100, V_pulse=-1.0, saturation=True, tau_sat=35.0)
            G_ltd = ltd.apply(syn, dt=1e-3)
            results['G_sim'] = np.concatenate([G_ltp, G_ltd[1:]])

            sim_data = results

            p_mono = "✓" if results['P_monotonic'] else "✗"
            d_mono = "✓" if results['D_monotonic'] else "✗"

            self.metrics_label.setText(
                f"<b>Validación Jo 2010 — LTP/LTD (Fig. 2a) [{dev_name}]:</b><br>"
                f"• <b>Fuente CSV:</b> {results['source']} (jo2010_ltp_ltd.csv)<br>"
                f"• <b>Pendiente LTP (P):</b> {results['P_slope']:+.4f} (Monótona: {p_mono})<br>"
                f"• <b>Pendiente LTD (D):</b> {results['D_slope']:+.4f} (Monótona: {d_mono})<br>"
                f"• <b>Peak Current:</b> {results['peak_current']:.3f} (100nA) | <b>Retención:</b> {results['retention_pct']:.1f} %"
            )

        self.simulation_requested.emit(exp_type, sim_data)
