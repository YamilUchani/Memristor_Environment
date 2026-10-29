"""
neurolab.gui.tabs.plasticidad_tab
=================================
Pestaña de Plasticidad Sináptica (Fase 3) con validación experimental Jo 2010.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from neurolab.gui.styles import COLORS
from neurolab.synapses import STDPRule
from neurolab.validation import validate_jo2010_stdp, load_jo2010_stdp_data

FONTS = {
    'title': ('Segoe UI', 13, 'bold'),
    'subtitle': ('Segoe UI', 11, 'bold'),
    'label': ('Segoe UI', 10),
    'value': ('Consolas', 10, 'bold'),
    'small': ('Segoe UI', 9),
    'mono': ('Consolas', 9),
}

EXPERIMENTOS = [
    '1 - LTP y LTD (Pulsos Directos +1V / -1V)',
    '2 - STDP (Ventana Hebbiana)',
    '3 - Jo 2010 (STDP Experimental)',
]


class PlasticidadTab(tk.Frame):
    """Pestaña de Plasticidad Sináptica y Validación STDP Jo 2010."""

    def __init__(self, parent):
        super().__init__(parent, bg=COLORS['bg_panel'])
        self._build_ui()

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=COLORS['bg_panel'], pady=10)
        header.pack(fill='x', padx=15)

        tk.Label(
            header, text='⚡ Plasticidad Sináptica (Fase 3)',
            font=FONTS['title'], bg=COLORS['bg_panel'],
            fg=COLORS['text']
        ).pack(side='left')

        tk.Label(
            header, text='Simulación de LTP/LTD, Regla STDP y Validación Jo 2010',
            font=FONTS['small'], bg=COLORS['bg_panel'],
            fg=COLORS['text_dim']
        ).pack(side='left', padx=15)

        # Body Layout
        body = tk.Frame(self, bg=COLORS['bg_panel'])
        body.pack(fill='both', expand=True, padx=15, pady=5)

        # Panel Izquierdo (Controles)
        ctrl_frame = tk.Frame(body, bg=COLORS['bg_dark'], width=360)
        ctrl_frame.pack(side='left', fill='y', padx=(0, 10))
        ctrl_frame.pack_propagate(False)

        # 1. Selección de Experimento
        tk.Label(
            ctrl_frame, text='EXPERIMENTO',
            font=FONTS['subtitle'], bg=COLORS['bg_dark'],
            fg=COLORS['text']
        ).pack(anchor='w', padx=12, pady=(12, 5))

        self.combo_experimentos = ttk.Combobox(
            ctrl_frame, values=EXPERIMENTOS, state='readonly', font=FONTS['label']
        )
        self.combo_experimentos.current(2)  # Jo 2010 por defecto
        self.combo_experimentos.pack(fill='x', padx=12, pady=5)
        self.combo_experimentos.bind('<<ComboboxSelected>>', self._on_combo_change)

        # 2. Parámetros STDP
        param_group = tk.LabelFrame(
            ctrl_frame, text=' Parámetros STDP (Bi & Poo / Gerstner) ',
            font=FONTS['small'], bg=COLORS['bg_dark'], fg=COLORS['text']
        )
        param_group.pack(fill='x', padx=12, pady=10)

        # A+
        row_ap = tk.Frame(param_group, bg=COLORS['bg_dark'])
        row_ap.pack(fill='x', padx=5, pady=4)
        tk.Label(row_ap, text='A+:', font=FONTS['label'], bg=COLORS['bg_dark'], fg=COLORS['text_dim'], width=8, anchor='w').pack(side='left')
        self.entry_A_plus = tk.Entry(row_ap, font=FONTS['value'], bg='#1e1e2e', fg=COLORS['text'], insertbackground='white')
        self.entry_A_plus.insert(0, '0.05')
        self.entry_A_plus.pack(side='left', fill='x', expand=True)

        # A-
        row_am = tk.Frame(param_group, bg=COLORS['bg_dark'])
        row_am.pack(fill='x', padx=5, pady=4)
        tk.Label(row_am, text='A-:', font=FONTS['label'], bg=COLORS['bg_dark'], fg=COLORS['text_dim'], width=8, anchor='w').pack(side='left')
        self.entry_A_minus = tk.Entry(row_am, font=FONTS['value'], bg='#1e1e2e', fg=COLORS['text'], insertbackground='white')
        self.entry_A_minus.insert(0, '-0.025')
        self.entry_A_minus.pack(side='left', fill='x', expand=True)

        # tau+
        row_tp = tk.Frame(param_group, bg=COLORS['bg_dark'])
        row_tp.pack(fill='x', padx=5, pady=4)
        tk.Label(row_tp, text='τ+ (ms):', font=FONTS['label'], bg=COLORS['bg_dark'], fg=COLORS['text_dim'], width=8, anchor='w').pack(side='left')
        self.entry_tau_plus = tk.Entry(row_tp, font=FONTS['value'], bg='#1e1e2e', fg=COLORS['text'], insertbackground='white')
        self.entry_tau_plus.insert(0, '17.0')
        self.entry_tau_plus.pack(side='left', fill='x', expand=True)

        # tau-
        row_tm = tk.Frame(param_group, bg=COLORS['bg_dark'])
        row_tm.pack(fill='x', padx=5, pady=4)
        tk.Label(row_tm, text='τ- (ms):', font=FONTS['label'], bg=COLORS['bg_dark'], fg=COLORS['text_dim'], width=8, anchor='w').pack(side='left')
        self.entry_tau_minus = tk.Entry(row_tm, font=FONTS['value'], bg='#1e1e2e', fg=COLORS['text'], insertbackground='white')
        self.entry_tau_minus.insert(0, '34.0')
        self.entry_tau_minus.pack(side='left', fill='x', expand=True)

        # 3. Botón de Ejecución
        self.btn_ejecutar = tk.Button(
            ctrl_frame, text='🔥 Ejecutar Simulación de Plasticidad',
            font=FONTS['subtitle'], bg=COLORS['success'], fg='white',
            relief='flat', padx=10, pady=10, command=self._on_ejecutar_simulacion
        )
        self.btn_ejecutar.pack(fill='x', padx=12, pady=12)

        # 4. Panel Estado CSV
        csv_group = tk.LabelFrame(
            ctrl_frame, text=' Estado del CSV ',
            font=FONTS['small'], bg=COLORS['bg_dark'], fg=COLORS['text']
        )
        csv_group.pack(fill='x', padx=12, pady=5)

        self.lbl_csv_path = tk.Label(
            csv_group, text='Ruta: data_validation/jo2010/jo2010_stdp.csv',
            font=FONTS['mono'], bg=COLORS['bg_dark'], fg=COLORS['text_dim'],
            wraplength=320, justify='left'
        )
        self.lbl_csv_path.pack(fill='x', padx=5, pady=2)

        self.lbl_csv_status = tk.Label(
            csv_group, text='', font=FONTS['small'], bg=COLORS['bg_dark'], fg=COLORS['warning']
        )
        self.lbl_csv_status.pack(fill='x', padx=5, pady=2)
        self._check_csv_status()

        # 5. Panel Métricas Resultados
        res_group = tk.LabelFrame(
            ctrl_frame, text=' Resultados de Validación ',
            font=FONTS['small'], bg=COLORS['bg_dark'], fg=COLORS['text']
        )
        res_group.pack(fill='both', expand=True, padx=12, pady=10)

        self.label_resultado = tk.Label(
            res_group, text='○ Seleccione un experimento y presione Ejecutar.',
            font=FONTS['mono'], bg=COLORS['bg_dark'], fg=COLORS['text_dim'],
            justify='left', anchor='nw', padx=8, pady=8
        )
        self.label_resultado.pack(fill='both', expand=True)

        # Panel Derecho (Gráfica)
        plot_container = tk.Frame(body, bg=COLORS['bg_panel'])
        plot_container.pack(side='right', fill='both', expand=True)

        self.figure = Figure(figsize=(10, 5), dpi=100, facecolor=COLORS['bg_canvas'])
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_container)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _check_csv_status(self):
        csv_path = Path(__file__).parent.parent.parent.parent / 'data_validation' / 'jo2010' / 'jo2010_stdp.csv'
        if csv_path.exists():
            self.lbl_csv_status.config(text='✅ CSV disponible', fg=COLORS['success'])
        else:
            self.lbl_csv_status.config(text='⚠️ CSV no existe — usando sintéticos', fg=COLORS['warning'])

    def _on_combo_change(self, event=None):
        self._check_csv_status()

    def _style_axis(self, ax):
        ax.set_facecolor(COLORS['bg_canvas'])
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_color('#555')
        ax.grid(True, alpha=0.2)

    def _on_ejecutar_simulacion(self):
        """Handler del botón Ejecutar Simulación de Plasticidad."""
        modo = self.combo_experimentos.get()

        if modo.startswith('1'):
            self._run_ltp_ltd()
        elif modo.startswith('2'):
            self._run_stdp()
        elif modo.startswith('3'):
            self._run_jo2010()
        else:
            messagebox.showerror('Error', f'Modo desconocido: {modo}')

    def _run_ltp_ltd(self):
        """Simulación sintética de LTP/LTD por pulsos directos."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axis(ax)

        pulses = np.arange(1, 201)
        w_ltp = 0.1 + 0.8 * (1.0 - np.exp(-pulses / 30.0))
        w_ltd = 0.9 * np.exp(-(pulses - 100) / 30.0)
        w_ltd[:100] = w_ltp[:100]

        ax.plot(pulses[:100], w_ltp[:100], 'o-', color='#4a9eff', label='LTP (+1V)', ms=4)
        ax.plot(pulses[100:], w_ltd[100:], 'o-', color='#ff6b6b', label='LTD (-1V)', ms=4)

        ax.set_xlabel('Número de Pulso', color='white', fontsize=11)
        ax.set_ylabel('Conductancia G (Normalizada)', color='white', fontsize=11)
        ax.set_title('Modulación de Peso por Pulsos (+1V / -1V)', color='white', fontsize=12, fontweight='bold')
        ax.legend(facecolor='#252526', edgecolor='#555', labelcolor='white')

        self.figure.tight_layout()
        self.canvas.draw()
        self.label_resultado.config(text="Simulación LTP/LTD completada.\n200 pulsos aplicados (+1V / -1V).")

    def _run_stdp(self):
        """Simulación estándar de la curva STDP."""
        try:
            A_plus = float(self.entry_A_plus.get())
            A_minus = float(self.entry_A_minus.get())
            tau_plus = float(self.entry_tau_plus.get()) * 1e-3
            tau_minus = float(self.entry_tau_minus.get()) * 1e-3
        except ValueError:
            messagebox.showerror('Error', 'Ingrese números válidos para los parámetros STDP.')
            return

        stdp = STDPRule(A_plus=A_plus, A_minus=A_minus, tau_plus=tau_plus, tau_minus=tau_minus)
        dt_values = np.linspace(-80.0, 80.0, 200)
        dw = np.array([stdp.delta_w(dt * 1e-3) * 100.0 for dt in dt_values])

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._style_axis(ax)

        ax.plot(dt_values, dw, '-', color='#4a9eff', lw=2.5, label='Regla STDP Teórica')
        ax.axhline(0, color='white', ls='--', alpha=0.4)
        ax.axvline(0, color='white', ls='--', alpha=0.4)

        ax.set_xlabel('Δt = t_post - t_pre (ms)', color='white', fontsize=11)
        ax.set_ylabel('ΔW (%)', color='white', fontsize=11)
        ax.set_title('Ventana de Plasticidad Hebbiana STDP', color='white', fontsize=12, fontweight='bold')
        ax.legend(facecolor='#252526', edgecolor='#555', labelcolor='white')

        self.figure.tight_layout()
        self.canvas.draw()
        self.label_resultado.config(
            text=f"Curva STDP Generada:\nA+ = {A_plus}, A- = {A_minus}\nτ+ = {tau_plus*1e3} ms, τ- = {tau_minus*1e3} ms"
        )

    def _run_jo2010(self):
        """
        Ejecuta la validación de STDP contra Jo 2010.
        Usa los parámetros A+, A-, tau+, tau- del panel.
        """
        try:
            A_plus = float(self.entry_A_plus.get())
            A_minus = float(self.entry_A_minus.get())
            tau_plus = float(self.entry_tau_plus.get()) * 1e-3   # ms → s
            tau_minus = float(self.entry_tau_minus.get()) * 1e-3 # ms → s
        except ValueError:
            messagebox.showerror('Error', 'Parámetros STDP inválidos. Verifique las entradas.')
            return

        # 2. Crear regla STDP
        stdp = STDPRule(
            A_plus=A_plus,
            A_minus=A_minus,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
        )

        # 3. Evaluar STDP en un rango de Δt
        dt_values = np.linspace(-80.0, 80.0, 161)   # ms
        dw_sim = np.array([
            stdp.delta_w(dt * 1e-3) * 100.0          # s → %
            for dt in dt_values
        ])

        # 4. Validar contra Jo 2010
        results = validate_jo2010_stdp(dt_values, dw_sim)

        # 5. Graficar
        self._plot_jo2010(results)

        # 6. Mostrar métricas en el panel inferior
        self._show_jo2010_metrics(results)

    def _plot_jo2010(self, results):
        """Grafica la comparación Jo 2010 vs simulación."""
        # Limpiar figura actual
        self.figure.clear()

        # Dos subplots: comparación + error
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)

        # ============ Panel 1: Comparación ============
        self._style_axis(ax1)

        # Experimental (Jo 2010)
        ax1.plot(results['dt_exp_ms'], results['dw_exp_pct'],
                 'o', ms=9, color='#ff6b6b',
                 label='Experimental (Jo 2010)',
                 markeredgecolor='white', markeredgewidth=1.2,
                 zorder=3)

        # Simulación
        dt_sim = results['dt_sim_ms']
        dw_sim = results['dw_sim_pct']
        sort_idx = np.argsort(dt_sim)
        ax1.plot(dt_sim[sort_idx], dw_sim[sort_idx],
                 '-', lw=2.2, color='#4a9eff',
                 label='Simulación (STDPRule)',
                 zorder=2)

        ax1.axhline(0, color='white', ls='--', lw=0.8, alpha=0.4)
        ax1.axvline(0, color='white', ls='--', lw=0.8, alpha=0.4)

        ax1.set_xlabel('Δt = t_post − t_pre (ms)',
                       color='white', fontsize=11)
        ax1.set_ylabel('ΔW (%)', color='white', fontsize=11)
        ax1.set_title('Jo 2010 — STDP Experimental vs Simulación',
                      color='white', fontsize=12, fontweight='bold')

        ax1.legend(loc='upper left',
                   facecolor='#252526', edgecolor='#555',
                   labelcolor='white', fontsize=9)

        # Anotación con R²
        status = 'PASS ✅' if results['r2'] > 0.85 else 'REVISAR ⚠️'
        color_status = '#10ac84' if results['r2'] > 0.85 else '#ff9f43'
        ax1.text(0.98, 0.05,
                 f"R² = {results['r2']:.4f}\n"
                 f"MAE = {results['mae']:.4f}\n"
                 f"RMSE = {results['rmse']:.4f}\n"
                 f"{status}",
                 transform=ax1.transAxes,
                 ha='right', va='bottom',
                 fontsize=10, color=color_status,
                 family='monospace',
                 bbox=dict(boxstyle='round',
                           facecolor='#1a1a1a',
                           edgecolor=color_status,
                           alpha=0.9))

        ax1.grid(True, alpha=0.2)

        # ============ Panel 2: Error punto a punto ============
        self._style_axis(ax2)

        error_pct = results['dw_sim_at_exp'] - results['dw_exp_pct']

        ax2.bar(results['dt_exp_ms'], error_pct,
                width=3.0, color='#b565d8',
                edgecolor='white', linewidth=0.5)
        ax2.axhline(0, color='white', ls='--', lw=0.8, alpha=0.5)

        # Líneas de tolerancia
        ax2.axhline(+5, color='#ff9f43', ls=':', lw=1, alpha=0.7)
        ax2.axhline(-5, color='#ff9f43', ls=':', lw=1, alpha=0.7)

        ax2.set_xlabel('Δt (ms)', color='white', fontsize=11)
        ax2.set_ylabel('Error (ΔW sim − ΔW exp) [%]',
                       color='white', fontsize=11)
        ax2.set_title('Error punto a punto (±5% tolerancia)',
                      color='white', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.2, axis='y')

        # Texto con fuente de datos
        source_txt = ('✅ CSV disponible'
                      if results['source'] == 'csv'
                      else '⚠️ Datos sintéticos (CSV no existe)')
        color_src = '#10ac84' if results['source'] == 'csv' else '#ff9f43'
        ax2.text(0.02, 0.98, source_txt,
                 transform=ax2.transAxes,
                 ha='left', va='top', fontsize=9,
                 color=color_src, family='monospace')

        self.figure.tight_layout()
        self.canvas.draw()

        # Guardar la figura
        self._save_figure(results)

    def _show_jo2010_metrics(self, results):
        """Muestra las métricas en el panel inferior."""
        texto = (
            f"Validación Jo 2010 (Fig. 3a)\n"
            f"─────────────────────────────\n"
            f"Fuente:      {results['source']}\n"
            f"Puntos:      {results['n_points']}\n"
            f"MAE:         {results['mae']:.4f}\n"
            f"RMSE:        {results['rmse']:.4f}\n"
            f"R²:          {results['r2']:.6f}\n"
            f"Err. máx:    {results['err_max']:.2f} %\n"
            f"─────────────────────────────\n"
            f"Estado:      {'✅ PASS' if results['r2'] > 0.85 else '⚠️ REVISAR'}"
        )
        self.label_resultado.config(text=texto)

    def _save_figure(self, results):
        """Guarda la figura de validación."""
        output_dir = Path(__file__).parent.parent.parent.parent / \
                     'outputs' / 'validation' / 'jo2010'
        output_dir.mkdir(parents=True, exist_ok=True)

        fname = 'jo2010_stdp_validacion.png'
        path = output_dir / fname

        self.figure.savefig(path, dpi=200,
                            facecolor='#1a1a1a',
                            bbox_inches='tight')
        print(f"Figura guardada: {path}")
