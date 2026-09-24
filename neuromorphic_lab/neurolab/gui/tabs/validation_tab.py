"""
neurolab.gui.tabs.validation_tab
================================
Pestaña de Validación contra papers experimentales.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from neurolab.gui.styles import COLORS
from neurolab.synapses import STDPRule
from neurolab.validation import (
    validate_jo2010_stdp,
    validate_jo2010_ltp_ltd,
    validate_jo2010_endurance,
)

# Adaptación de fuentes para compatibilidad con Tkinter
FONTS = {
    'title': ('Segoe UI', 13, 'bold'),
    'subtitle': ('Segoe UI', 11, 'bold'),
    'label': ('Segoe UI', 10),
    'value': ('Consolas', 10, 'bold'),
    'small': ('Segoe UI', 9),
    'mono': ('Consolas', 9),
}

# =====================================================================
# CATÁLOGO DE VALIDACIONES
# =====================================================================

VALIDATIONS_CATALOG = [
    {
        'id': 'jo2010_stdp',
        'name': 'Jo 2010 — STDP',
        'category': 'Plasticidad',
        'paper': 'Jo et al., Nano Letters 10(4), 2010',
        'description': 'Curva STDP con 22 puntos experimentales (Fig. 3a)',
        'function': validate_jo2010_stdp,
        'params': [
            {'name': 'A_plus', 'label': 'A+', 'type': 'float', 'default': 0.13},
            {'name': 'A_minus', 'label': 'A-', 'type': 'float', 'default': -0.18},
            {'name': 'tau_plus', 'label': 'τ+ (ms)', 'type': 'float', 'default': 20.0},
            {'name': 'tau_minus', 'label': 'τ- (ms)', 'type': 'float', 'default': 20.0},
        ],
    },
    {
        'id': 'jo2010_ltp_ltd',
        'name': 'Jo 2010 — LTP/LTD',
        'category': 'Plasticidad',
        'paper': 'Jo et al., Nano Letters 10(4), 2010',
        'description': 'Curva LTP/LTD con 200 pulsos (Fig. 2a)',
        'function': validate_jo2010_ltp_ltd,
        'params': [],
    },
]


class ValidationTab(tk.Frame):
    """Pestaña de Validación con visualización en tiempo real."""

    def __init__(self, parent):
        super().__init__(parent, bg=COLORS['bg_panel'])

        self.current_validation_id = None
        self.is_running = False

        self._build_ui()
        self._populate_catalog()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=COLORS['bg_panel'], pady=10)
        header.pack(fill='x', padx=15)

        tk.Label(header, text='📊 Validación contra Papers',
                 font=FONTS['title'], bg=COLORS['bg_panel'],
                 fg=COLORS['text']).pack(side='left')

        tk.Label(header, text='Comparación cuantitativa con datos experimentales',
                 font=FONTS['small'], bg=COLORS['bg_panel'],
                 fg=COLORS['text_dim']).pack(side='left', padx=15)

        # Contenedor principal
        main = tk.Frame(self, bg=COLORS['bg_panel'])
        main.pack(fill='both', expand=True, padx=15, pady=10)

        # Columna izquierda: catálogo
        self._build_catalog(main)

        # Columna centro: detalle
        self._build_detail_panel(main)

        # Columna derecha: resultados
        self._build_results_panel(main)

        # Panel inferior: gráfica
        self._build_plot_panel()

    def _build_catalog(self, parent):
        frame = tk.Frame(parent, bg=COLORS['bg_dark'], width=240)
        frame.pack(side='left', fill='y', padx=(0, 10))
        frame.pack_propagate(False)

        tk.Label(frame, text='VALIDACIONES',
                 font=FONTS['subtitle'], bg=COLORS['bg_dark'],
                 fg=COLORS['text']).pack(anchor='w', padx=10, pady=(10, 5))

        self.tree = ttk.Treeview(frame, show='tree', height=15)
        self.tree.pack(fill='both', expand=True, padx=5, pady=5)
        self.tree.bind('<<TreeviewSelect>>', self._on_selected)

    def _build_detail_panel(self, parent):
        frame = tk.Frame(parent, bg=COLORS['bg_dark'], width=350)
        frame.pack(side='left', fill='y', padx=(0, 10))
        frame.pack_propagate(False)

        tk.Label(frame, text='DETALLE',
                 font=FONTS['subtitle'], bg=COLORS['bg_dark'],
                 fg=COLORS['text']).pack(anchor='w', padx=10, pady=(10, 5))

        # Nombre
        self.detail_name = tk.Label(frame, text='—',
                                     font=FONTS['value'],
                                     bg=COLORS['bg_dark'],
                                     fg=COLORS['success'],
                                     anchor='w', padx=10, pady=5)
        self.detail_name.pack(fill='x')

        # Paper
        tk.Label(frame, text='Paper de referencia:',
                 font=FONTS['small'], bg=COLORS['bg_dark'],
                 fg=COLORS['text_dim'],
                 anchor='w', padx=10).pack(fill='x', pady=(10, 2))

        self.detail_paper = tk.Label(frame, text='—',
                                      font=FONTS['small'],
                                      bg=COLORS['bg_dark'],
                                      fg=COLORS['text'],
                                      anchor='w', padx=10, pady=5,
                                      wraplength=320, justify='left')
        self.detail_paper.pack(fill='x')

        # Descripción
        tk.Label(frame, text='Descripción:',
                 font=FONTS['small'], bg=COLORS['bg_dark'],
                 fg=COLORS['text_dim'],
                 anchor='w', padx=10).pack(fill='x', pady=(10, 2))

        self.detail_desc = tk.Label(frame, text='—',
                                     font=FONTS['small'],
                                     bg=COLORS['bg_dark'],
                                     fg=COLORS['text'],
                                     anchor='w', padx=10, pady=5,
                                     wraplength=320, justify='left')
        self.detail_desc.pack(fill='x')

        # Info CSV
        tk.Label(frame, text='Archivo CSV esperado:',
                 font=FONTS['small'], bg=COLORS['bg_dark'],
                 fg=COLORS['text_dim'],
                 anchor='w', padx=10).pack(fill='x', pady=(10, 2))

        self.detail_csv = tk.Label(frame, text='—',
                                    font=FONTS['mono'],
                                    bg=COLORS['bg_dark'],
                                    fg=COLORS['warning'],
                                    anchor='w', padx=10, pady=5,
                                    wraplength=320, justify='left')
        self.detail_csv.pack(fill='x')

        # Estado del CSV
        self.csv_status = tk.Label(frame, text='',
                                    font=FONTS['small'],
                                    bg=COLORS['bg_dark'],
                                    fg=COLORS['text_dim'],
                                    anchor='w', padx=10)
        self.csv_status.pack(fill='x')

        # Botón ejecutar
        btn_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        btn_frame.pack(side='bottom', fill='x', padx=10, pady=10)

        self.btn_run = tk.Button(
            btn_frame, text='▶ VALIDAR',
            font=FONTS['subtitle'],
            bg=COLORS['success'], fg='white',
            relief='flat', padx=15, pady=10,
            command=self._run_current)
        self.btn_run.pack(fill='x')

    def _build_results_panel(self, parent):
        frame = tk.Frame(parent, bg=COLORS['bg_dark'], width=280)
        frame.pack(side='left', fill='y')
        frame.pack_propagate(False)

        tk.Label(frame, text='RESULTADOS',
                 font=FONTS['subtitle'], bg=COLORS['bg_dark'],
                 fg=COLORS['text']).pack(anchor='w', padx=10, pady=(10, 5))

        self.status_display = tk.Label(
            frame, text='○ Sin ejecutar',
            font=FONTS['label'], bg=COLORS['bg_dark'],
            fg=COLORS['text_dim'], anchor='w', padx=10)
        self.status_display.pack(fill='x', pady=5)

        self.metrics_frame = tk.Frame(frame, bg=COLORS['bg_dark'])
        self.metrics_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.time_label = tk.Label(frame, text='',
                                    font=FONTS['small'],
                                    bg=COLORS['bg_dark'],
                                    fg=COLORS['text_dim'],
                                    anchor='w', padx=10)
        self.time_label.pack(side='bottom', fill='x', pady=5)

    def _build_plot_panel(self):
        plot_frame = tk.Frame(self, bg=COLORS['bg_panel'])
        plot_frame.pack(fill='both', expand=True, padx=15, pady=(0, 10))

        self.figure = Figure(figsize=(14, 5), dpi=100,
                              facecolor=COLORS['bg_canvas'])
        self.ax = self.figure.add_subplot(111)
        self._style_axis(self.ax)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _style_axis(self, ax):
        ax.set_facecolor(COLORS['bg_canvas'])
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_color('#555')
        ax.grid(True, alpha=0.2)

    # ------------------------------------------------------------------
    # CATÁLOGO
    # ------------------------------------------------------------------

    def _populate_catalog(self):
        current_cat = None
        for v in VALIDATIONS_CATALOG:
            if v['category'] != current_cat:
                current_cat = v['category']
                parent = self.tree.insert('', 'end',
                                           iid=f'cat_{current_cat}',
                                           text=f'▼ {current_cat}',
                                           open=True)
            self.tree.insert(parent, 'end',
                             iid=v['id'],
                             text=f'  {v["name"]}')

    def _on_selected(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        v_id = sel[0]
        if v_id.startswith('cat_'):
            return

        v = next((x for x in VALIDATIONS_CATALOG if x['id'] == v_id), None)
        if v is None:
            return

        self.current_validation_id = v_id
        self.detail_name.config(text=v['name'])
        self.detail_paper.config(text=v['paper'])
        self.detail_desc.config(text=v['description'])

        # Info CSV
        csv_name = f'{v_id}.csv'
        csv_path = Path(__file__).parent.parent.parent.parent / \
                    'data_validation' / 'jo2010' / csv_name

        try:
            rel_path = csv_path.relative_to(Path(__file__).parent.parent.parent.parent)
            self.detail_csv.config(text=str(rel_path))
        except ValueError:
            self.detail_csv.config(text=csv_name)

        # Estado del CSV
        if csv_path.exists():
            self.csv_status.config(text='✅ CSV disponible',
                                    fg=COLORS['success'])
        else:
            self.csv_status.config(
                text='⚠️ CSV no existe — usando datos sintéticos',
                fg=COLORS['warning'])

        self._clear_results()

    # ------------------------------------------------------------------
    # EJECUCIÓN
    # ------------------------------------------------------------------

    def _run_current(self):
        if self.current_validation_id is None or self.is_running:
            return

        v = next(x for x in VALIDATIONS_CATALOG
                 if x['id'] == self.current_validation_id)

        self.is_running = True
        self.btn_run.config(state='disabled', text='⏳ VALIDANDO...')
        self.status_display.config(text='⏳ Ejecutando...',
                                    fg=COLORS['warning'])

        def worker():
            t0 = time.time()
            try:
                # Ejecutar según el tipo
                if v['id'] == 'jo2010_stdp':
                    stdp = STDPRule(A_plus=0.05, A_minus=-0.025,
                                     tau_plus=17e-3, tau_minus=34e-3)
                    result = validate_jo2010_stdp(stdp)
                else:
                    result = v['function']()

                elapsed = time.time() - t0
                self.after(0, lambda: self._on_result(v, result, elapsed))

            except Exception as e:
                self.after(0, lambda: self._on_error(e))
            finally:
                self.after(0, self._on_finish)

        threading.Thread(target=worker, daemon=True).start()

    def _on_result(self, v, result, elapsed):
        self.status_display.config(text='✅ COMPLETADO',
                                    fg=COLORS['success'])

        # Métricas
        for w in self.metrics_frame.winfo_children():
            w.destroy()

        metrics_map = {
            'jo2010_stdp': [
                ('Fuente', result.get('source', '—')),
                ('Puntos', str(result.get('n_points', '—'))),
                ('MAE', f'{result.get("mae", 0):.4f}'),
                ('RMSE', f'{result.get("rmse", 0):.4f}'),
                ('R²', f'{result.get("r2", 0):.6f}'),
                ('Err. máx', f'{result.get("err_rel_max", 0):.2f} %'),
            ],
            'jo2010_ltp_ltd': [
                ('Fuente', result.get('source', '—')),
                ('Pendiente LTP', f'{result.get("P_slope", 0):+.4f}'),
                ('Pendiente LTD', f'{result.get("D_slope", 0):+.4f}'),
                ('Monotonía P', '✅' if result.get('P_monotonic') else '❌'),
                ('Monotonía D', '✅' if result.get('D_monotonic') else '❌'),
                ('Retención', f'{result.get("retention_pct", 0):.1f} %'),
            ],
            'jo2010_endurance': [
                ('Fuente', result.get('source', '—')),
                ('Corriente máx', f'{result.get("current_max", 0):.3f} nA'),
                ('Corriente mín', f'{result.get("current_min", 0):.3f} nA'),
                ('Ventana', f'{result.get("window", 0):.3f} nA'),
                ('Retención', f'{result.get("retention_window", 0):.1f} %'),
                ('Degradación', f'{result.get("degradation", 0):.1f} %'),
            ],
        }

        for key, value in metrics_map.get(v['id'], []):
            row = tk.Frame(self.metrics_frame, bg=COLORS['bg_dark'])
            row.pack(fill='x', pady=2)

            tk.Label(row, text=f'{key}:',
                     font=FONTS['small'],
                     bg=COLORS['bg_dark'],
                     fg=COLORS['text_dim'],
                     width=15, anchor='w').pack(side='left')

            tk.Label(row, text=value,
                     font=FONTS['value'],
                     bg=COLORS['bg_dark'],
                     fg=COLORS['success'],
                     anchor='w').pack(side='left')

        self.time_label.config(text=f'⏱ {elapsed*1e3:.0f} ms')

        # Graficar
        self._plot(v['id'], result)

    def _on_error(self, error):
        self.status_display.config(text='❌ ERROR', fg=COLORS['error'])
        messagebox.showerror('Error de validación', str(error))

    def _on_finish(self):
        self.is_running = False
        self.btn_run.config(state='normal', text='▶ VALIDAR')

    def _clear_results(self):
        for w in self.metrics_frame.winfo_children():
            w.destroy()
        self.status_display.config(text='○ Sin ejecutar',
                                    fg=COLORS['text_dim'])
        self.time_label.config(text='')
        self.ax.clear()
        self._style_axis(self.ax)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # GRÁFICAS
    # ------------------------------------------------------------------

    def _plot(self, v_id, result):
        self.figure.clear()

        if v_id == 'jo2010_stdp':
            self._plot_stdp(result)
        elif v_id == 'jo2010_ltp_ltd':
            self._plot_ltp_ltd(result)

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_stdp(self, result):
        ax = self.figure.add_subplot(111)
        self._style_axis(ax)

        dt = result['dt_ms']
        dw_exp = result['dw_exp_pct']
        dw_sim = result['dw_sim_pct']

        # Experimental
        ax.plot(dt, dw_exp, 'o', ms=8, color='#ff6b6b',
                label='Experimental (Jo 2010)', zorder=3)

        # Simulación
        dt_sorted_idx = np.argsort(dt)
        dt_s = dt[dt_sorted_idx]
        dw_s = dw_sim[dt_sorted_idx]
        ax.plot(dt_s, dw_s, '-', lw=2, color='#4a9eff',
                label='Simulación (STDPRule)', zorder=2)

        ax.axhline(0, color='white', ls='--', lw=0.8, alpha=0.5)
        ax.axvline(0, color='white', ls='--', lw=0.8, alpha=0.5)

        ax.set_xlabel('Δt = t_post − t_pre (ms)',
                       color='white', fontsize=11)
        ax.set_ylabel('ΔW (%)', color='white', fontsize=11)
        ax.set_title(f'Validación STDP — Jo et al. 2010 (R² = {result["r2"]:.4f})',
                     color='white', fontsize=13, fontweight='bold')

        ax.legend(loc='upper left', facecolor='#252526',
                  edgecolor='#555', labelcolor='white')

        # Anotación
        status = '✅ PASS' if result['r2'] > 0.85 else '⚠️ REVISAR'
        text = (
            f'MAE = {result["mae"]:.4f}\n'
            f'RMSE = {result["rmse"]:.4f}\n'
            f'R² = {result["r2"]:.6f}\n'
            f'Err. máx = {result["err_rel_max"]:.2f} %\n'
            f'{status}'
        )
        ax.text(0.98, 0.05, text,
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, color='#10ac84', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                          edgecolor='#10ac84', alpha=0.9))

        ax.grid(True, alpha=0.2)

    def _plot_ltp_ltd(self, result):
        ax1 = self.figure.add_subplot(1, 2, 1)
        self._style_axis(ax1)

        pulse = result['pulse_num']
        current = result['current_exp']
        idx_peak = result['idx_peak']

        ax1.plot(pulse[:idx_peak + 1], current[:idx_peak + 1],
                 's-', ms=6, color='#4a9eff', label='LTP (P)')
        ax1.plot(pulse[idx_peak:], current[idx_peak:],
                 'o-', ms=6, color='#ff6b6b', label='LTD (D)')
        ax1.axvline(pulse[idx_peak], color='gray',
                    ls='--', lw=1, alpha=0.7)

        ax1.set_xlabel('Número de pulso', color='white', fontsize=11)
        ax1.set_ylabel('Corriente (100 nA)', color='white', fontsize=11)
        ax1.set_title('LTP/LTD — Jo 2010 Fig. 2a',
                      color='white', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower left', facecolor='#252526',
                   edgecolor='#555', labelcolor='white')
        ax1.grid(True, alpha=0.2)

        # Panel derecho: pendientes
        ax2 = self.figure.add_subplot(1, 2, 2)
        self._style_axis(ax2)

        labels = ['LTP (P)', 'LTD (D)']
        slopes = [result['P_slope'], result['D_slope']]
        colors = ['#4a9eff', '#ff6b6b']

        bars = ax2.bar(labels, slopes, color=colors, edgecolor='white')

        for bar, s in zip(bars, slopes):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     h + (0.001 if h > 0 else -0.001),
                     f'{s:+.4f}',
                     ha='center',
                     va='bottom' if h > 0 else 'top',
                     color='white', fontsize=10, fontweight='bold')

        ax2.axhline(0, color='white', ls='--', lw=0.8, alpha=0.5)
        ax2.set_ylabel('Pendiente (nA/pulso)', color='white', fontsize=11)
        ax2.set_title('Tasas de cambio',
                      color='white', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.2, axis='y')
