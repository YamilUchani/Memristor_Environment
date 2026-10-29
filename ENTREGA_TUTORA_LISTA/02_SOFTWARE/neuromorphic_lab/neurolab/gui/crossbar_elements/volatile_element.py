"""
neurolab.gui.crossbar_elements.volatile_element
================================================
Elemento Memristor Volátil HfO₂.
"""

import random
from typing import Dict, Any
from neurolab.gui.crossbar_elements.base_element import VisualElement


class VolatileMemristorElement(VisualElement):
    def __init__(self, element_id: str, x: float, y: float):
        super().__init__(element_id, x, y)
        self.params = {
            'model': 'HfO2_volatile',
            'is_volatile': True,
            'reference': 'Wang et al., Nanomaterials 15, 2025',
            'RON': 10e3,
            'ROFF': 500e3,
            'x0': 0.05,
            'tau_relax': 0.30,
            'volatile_tau_relax': 0.30,
            'x_eq': 0.05,
            'V_th': 0.5,
            'k_stim': 1.0,
            'window_type': 'Sin Ventana',
            'window_p': 2,
            'seed': random.randint(10000, 999999),
            'chk_c2c': False,
            'c2c_sigma': 0.05,
            'c2c_theta': 1.0,
            'chk_d2d': False,
            'd2d_sigma': 0.05,
            'chk_noise': False,
            'noise_std': 1e-6,
            'x': 0.05,
            'R': 475_500.0,
            'G': 2.103e-6,
            'dt': 1e-4,
            'first_ISI_ms': 42.42,
            'last_ISI_ms': 1.56,
            'ISI_reduction_pct': 96.32,
            'spikes_cycle1': 1373,
            'spikes_cycle2': 1374,
        }

    def on_params_changed(self):
        p = self.params
        RON = float(p.get('RON', p.get('volatile_R_on', 10e3)))
        ROFF = float(p.get('ROFF', p.get('volatile_R_off', 500e3)))
        x_val = float(p.get('x0', p.get('x', 0.05)))
        tau = float(p.get('tau_relax', p.get('volatile_tau_relax', 0.30)))

        R = RON * x_val + ROFF * (1.0 - x_val)
        G = 1.0 / R if R > 0 else 0.0

        p['x'] = x_val
        p['R'] = R
        p['G'] = G
        p['volatile_tau_relax'] = tau
        p['tau_relax'] = tau

    def hit_test(self, mx: float, my: float) -> bool:
        return abs(mx - self.x) <= 45 and abs(my - self.y) <= 35

    def get_config(self) -> Dict[str, Any]:
        return {
            'title': 'Memristor Volátil M_v — HfO₂',
            'sections': [
                {
                    'name': '1. Identificación y Modelo Volátil',
                    'fields': [
                        {'key': 'model', 'label': 'Modelo Volátil',
                         'type': 'combo',
                         'values': ['HfO2_volatile', 'diffusive_Ag_SiO2', 'NbOx_threshold'],
                         'value': self.params.get('model', 'HfO2_volatile')},
                        {'key': 'is_volatile', 'label': 'Volatilidad',
                         'type': 'readonly', 'value': 'SÍ (Decaimiento difusivo espontáneo)'},
                    ]
                },
                {
                    'name': '2. Parámetros Físicos y Relajación',
                    'fields': [
                        {'key': 'RON_kOm', 'label': 'Resistencia R_ON (kΩ)',
                         'type': 'float', 'value': float(self.params.get('RON', 10e3)) / 1e3,
                         'min': 0.1, 'max': 100.0, 'step': 0.5},
                        {'key': 'ROFF_kOm', 'label': 'Resistencia R_OFF (kΩ)',
                         'type': 'float', 'value': float(self.params.get('ROFF', 500e3)) / 1e3,
                         'min': 10.0, 'max': 5000.0, 'step': 10.0},
                        {'key': 'x0', 'label': 'Estado Inicial x_0',
                         'type': 'float', 'value': self.params.get('x0', 0.05),
                         'min': 0.0, 'max': 1.0, 'step': 0.01},
                        {'key': 'tau_relax', 'label': 'Tiempo Relajación τ_relax (s)',
                         'type': 'float', 'value': self.params.get('tau_relax', 0.30),
                         'min': 0.001, 'max': 10.0, 'step': 0.05},
                    ]
                },
                {
                    'name': '3. Estado Actual',
                    'fields': [
                        {'key': 'x', 'label': 'x (adim.)',
                         'type': 'readonly',
                         'value': f'{self.params.get("x", 0.05):.6f}'},
                        {'key': 'R', 'label': 'R (kΩ)',
                         'type': 'readonly',
                         'value': f'{self.params.get("R", 475500.0)/1e3:.4f}'},
                        {'key': 'G', 'label': 'G (μS)',
                         'type': 'readonly',
                         'value': f'{self.params.get("G", 2.103e-6)*1e6:.4f}'},
                    ]
                },
            ]
        }
