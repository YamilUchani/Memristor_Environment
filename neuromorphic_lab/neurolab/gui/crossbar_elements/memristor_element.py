"""
neurolab.gui.crossbar_elements.memristor_element
=================================================
Elemento Memristor Strukov — NO Volátil.
"""

import random
from typing import Dict, Any
from neurolab.gui.crossbar_elements.base_element import VisualElement


class MemristorElement(VisualElement):
    """
    Memristor Strukov / Prezioso — NO volátil.
    Contiene la configuración completa de la Pestaña 1.
    """

    def __init__(self, element_id: str, row: int, col: int, x: float, y: float):
        super().__init__(element_id, x, y)
        self.row = row
        self.col = col
        # --- Modificadores Estocásticos ---
        self.params = {
            # --- Identificación & Modelo ---
            'model': 'strukov',
            'name': 'Memristor Strukov',
            'is_volatile': False,
            'reference': 'Strukov et al., Nature 453, 2008 / Prezioso et al., Nature 518, 2014',

            # --- Parámetros físicos ---
            'RON': 100.0,            # Ω
            'ROFF': 16_000.0,        # Ω
            'x0': 0.10,              # adim.
            'D': 10e-9,              # m (10 nm)
            'mu_v': 1e-14,           # m²/(V·s)
            'clip_x': True,

            # --- Ventana ---
            'window_type': 'Biolek',
            'window_p': 2,

            # --- Modificadores Estocásticos por Celda (D2D / C2C / Ruido) ---
            'seed': random.randint(10000, 999999),
            'chk_d2d': True,
            'd2d_sigma': 0.08,
            'd2d_factor': 1.0,
            'chk_c2c': True,
            'c2c_sigma': 0.05,
            'c2c_theta': 1.0,
            'chk_noise': False,
            'noise_std': 1e-6,

            # --- Estado actual ---
            'x': 0.10,
            'R': 14410.0,
            'G': 69.4e-6,
            'G_11': 69.4e-6,
            'R_11': 14410.0,

            # --- Integración ---
            'dt': 1e-4,              # s

            # --- Métricas de validación ---
            'R2_strukov': 0.9900,
            'MAE_validacion': 0.0100,
        }
        self.on_params_changed()

    def on_params_changed(self):
        """Recalcula R, G y el factor de dispersión D2D cuando cambian los parámetros."""
        p = self.params
        model_str = str(p.get('model', 'strukov')).lower()
        if 'prezioso' in model_str:
            p['name'] = 'Prezioso 2014 (Al₂O₃/TiO₂-x)'
            p['reference'] = 'Prezioso et al., Nature 518, 2014'
        elif 'hfo' in model_str or 'volatil' in model_str:
            p['name'] = 'HfO₂ Volátil Neurona'
            p['reference'] = 'Wang et al., CMOS LIF 2025'
            p['is_volatile'] = True
        else:
            p['name'] = 'Memristor Strukov'
            p['reference'] = 'Strukov et al., Nature 453, 2008'

        RON = float(p.get('RON', p.get('R_on', 100.0)))
        ROFF = float(p.get('ROFF', p.get('R_off', 16000.0)))
        x_val = float(p.get('x', p.get('x0', 0.10)))



        # Factor D2D por semilla de celda
        seed_val = int(p.get('seed', random.randint(10000, 999999)))
        chk_d2d = bool(p.get('chk_d2d', True))
        d2d_sigma = float(p.get('d2d_sigma', 0.08))

        if chk_d2d and d2d_sigma > 0:
            import numpy as np
            rng = np.random.default_rng(seed_val)
            sample = float(rng.normal(0.0, d2d_sigma))
            sample = float(np.clip(sample, -2.0 * d2d_sigma, 2.0 * d2d_sigma))
            d2d_factor = 1.0 + sample
        else:
            d2d_factor = 1.0

        p['d2d_factor'] = d2d_factor

        R_nominal = RON * x_val + ROFF * (1.0 - x_val)
        G_nominal = 1.0 / R_nominal if R_nominal > 0 else 0.0
        G = G_nominal * d2d_factor
        R = 1.0 / G if G > 0 else R_nominal

        p['x'] = x_val
        p['R'] = R
        p['G'] = G
        p['R_11'] = R
        p['G_11'] = G

    def hit_test(self, mx: float, my: float) -> bool:
        return abs(mx - self.x) <= 50 and abs(my - self.y) <= 40

    def get_config(self) -> Dict[str, Any]:
        model_str = str(self.params.get('model', 'strukov')).lower()
        is_vol = bool(self.params.get('is_volatile', False))
        return {
            'title': f'Memristor M{self.row+1}{self.col+1} ({self.params.get("name", "Strukov")})',
            'sections': [
                {
                    'name': '1. Identificación y Modelo Físico',
                    'fields': [
                        {'key': 'model', 'label': 'Modelo Matemático Base',
                         'type': 'combo',
                         'values': ['strukov', 'prezioso', 'hfo2_volatile'],
                         'value': model_str,
                         'help': 'Selecciona el modelo memristivo (Strukov 2008, Prezioso 2014, o HfO₂ Volátil)'},
                        {'key': 'reference', 'label': 'Referencia Paper',
                         'type': 'readonly',
                         'value': self.params.get('reference', 'Strukov et al. 2008 / Prezioso et al. 2014')},
                        {'key': 'is_volatile', 'label': 'Modo Volátil (Relajación Difusiva)',
                         'type': 'check', 'value': is_vol},
                        {'key': 'volatile_tau_relax', 'label': 'Tiempo Relajación τ_relax (s)',
                         'type': 'float', 'value': float(self.params.get('volatile_tau_relax', 0.5)),
                         'min': 0.001, 'max': 10.0, 'step': 0.05},
                    ]
                },
                {
                    'name': '2. Parámetros Físicos del Memristor',
                    'fields': [
                        {'key': 'RON', 'label': 'Resistencia R_ON (Ω)',
                         'type': 'float', 'value': float(self.params.get('RON', 100.0)),
                         'min': 0.001, 'max': 1e6, 'step': 1.0},
                        {'key': 'ROFF', 'label': 'Resistencia R_OFF (Ω)',
                         'type': 'float', 'value': float(self.params.get('ROFF', 16000.0)),
                         'min': 0.001, 'max': 1e9, 'step': 100.0},
                        {'key': 'x0', 'label': 'Estado Inicial x_0 (0 a 1)',
                         'type': 'float', 'value': float(self.params.get('x0', 0.10)),
                         'min': 0.0, 'max': 1.0, 'step': 0.01},
                        {'key': 'mu_v', 'label': 'Movilidad Iónica μ_v (m²/V·s)',
                         'type': 'sci', 'value': float(self.params.get('mu_v', 1e-14)),
                         'min': 1e-20, 'max': 1e-8},
                        {'key': 'clip_x', 'label': 'Limitar x a [0, 1]',
                         'type': 'check', 'value': self.params.get('clip_x', True)},
                    ]
                },
                {
                    'name': '3. Función de Ventana',
                    'fields': [
                        {'key': 'window_type', 'label': 'Tipo de Ventana',
                         'type': 'combo',
                         'values': ['Biolek', 'Joglekar', 'Sin Ventana'],
                         'value': self.params.get('window_type', 'Biolek')},
                        {'key': 'window_p', 'label': 'Exponente Ventana (p)',
                         'type': 'float', 'value': float(self.params.get('window_p', 2)),
                         'min': 1.0, 'max': 5.0, 'step': 1.0},
                    ]
                },
                {
                    'name': '4. Modificadores Estocásticos (D2D / C2C / Semilla)',
                    'fields': [
                        {'key': 'seed', 'label': 'Semilla Aleatoria de Celda (Seed)',
                         'type': 'int', 'value': int(self.params.get('seed', random.randint(10000, 999999))),
                         'min': 1, 'max': 999999, 'step': 1},
                        {'key': 'chk_d2d', 'label': 'Activar Variabilidad D2D (Dispositivo a Dispositivo)',
                         'type': 'check', 'value': bool(self.params.get('chk_d2d', True))},
                        {'key': 'd2d_sigma', 'label': 'σ_D2D (Desviación Fabricación D2D)',
                         'type': 'float', 'value': float(self.params.get('d2d_sigma', 0.08)),
                         'min': 0.0, 'max': 0.50, 'step': 0.01},
                        {'key': 'chk_c2c', 'label': 'Activar Variabilidad C2C (Ciclo a Ciclo)',
                         'type': 'check', 'value': bool(self.params.get('chk_c2c', True))},
                        {'key': 'c2c_sigma', 'label': 'σ_C2C (Fluctuación por Pulso C2C)',
                         'type': 'float', 'value': float(self.params.get('c2c_sigma', 0.05)),
                         'min': 0.0, 'max': 0.50, 'step': 0.01},
                        {'key': 'chk_noise', 'label': 'Activar Ruido Térmico en Corriente',
                         'type': 'check', 'value': bool(self.params.get('chk_noise', False))},
                        {'key': 'noise_std', 'label': 'σ_ruido Intensidad Corriente',
                         'type': 'sci', 'value': float(self.params.get('noise_std', 1e-6)),
                         'min': 1e-12, 'max': 1e-3},
                    ]
                },
                {
                    'name': '5. Estado Actual',
                    'fields': [
                        {'key': 'x', 'label': 'x (adim.)',
                         'type': 'readonly',
                         'value': f'{self.params.get("x", 0.10):.6f}'},
                        {'key': 'R', 'label': 'R (kΩ)',
                         'type': 'readonly',
                         'value': f'{self.params.get("R", 14410.0)/1e3:.4f}'},
                        {'key': 'G', 'label': 'G (μS)',
                         'type': 'readonly',
                         'value': f'{self.params.get("G", 69.4e-6)*1e6:.4f}'},
                    ]
                },
            ]
        }
