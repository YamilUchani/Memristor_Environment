"""
neurolab.gui.crossbar_elements.neuron_element
==============================================
Elemento Neurona LIF.
"""

from typing import Dict, Any
from neurolab.gui.crossbar_elements.base_element import VisualElement


class NeuronElement(VisualElement):
    def __init__(self, element_id: str, x: float, y: float):
        super().__init__(element_id, x, y)
        self.params = {
            'name': 'Neurona LIF',
            'model': 'LIF_Integrate_and_Fire',
            'C_m': 100e-9,           # F (100 nF)
            'R_series': 100e3,       # Ω (100 kΩ)
            'R_leak': 1e6,           # Ω (1 MΩ)
            'V_th': 2.5,             # V
            'V_reset': 0.0,          # V
            'V_rest': 0.0,           # V
            'tau_ref': 2e-3,         # s (2 ms)
            'tau_m': 100e-3,         # s (τ_m = R_leak·C_m)
            'tau_eq': 9.09e-3,       # s (τ_eq = (R_S || R_leak)·C_m)
            'V_m': 0.0,
            'dt': 1e-4,              # s
            'spike_count': 0,
            'last_isi': 0.0,
            'R2_analitica': 1.0000,
        }

    def on_params_changed(self):
        p = self.params
        R_leak = float(p.get('R_leak', 1e6))
        R_series = float(p.get('R_series', 100e3))
        C_m = float(p.get('C_m', 100e-9))

        p['tau_m'] = R_leak * C_m
        if (R_series + R_leak) > 0:
            r_eq = (R_series * R_leak) / (R_series + R_leak)
            p['tau_eq'] = r_eq * C_m
        else:
            p['tau_eq'] = 0.0

    def hit_test(self, mx: float, my: float) -> bool:
        return abs(mx - self.x) <= 50 and abs(my - self.y) <= 50

    def get_config(self) -> Dict[str, Any]:
        C_m_nF = float(self.params.get('C_m', 100e-9)) * 1e9
        R_series_kOm = float(self.params.get('R_series', 100e3)) / 1e3
        R_leak_kOm = float(self.params.get('R_leak', 1e6)) / 1e3
        tau_ref_ms = float(self.params.get('tau_ref', 2e-3)) * 1e3
        tau_m_ms = float(self.params.get('tau_m', 100e-3)) * 1e3
        tau_eq_ms = float(self.params.get('tau_eq', 9.09e-3)) * 1e3

        return {
            'title': 'Neurona LIF',
            'sections': [
                {
                    'name': '1. Identificación',
                    'fields': [
                        {'key': 'name', 'label': 'Nombre',
                         'type': 'text', 'value': self.params.get('name', 'Neurona LIF')},
                        {'key': 'model', 'label': 'Modelo Neuronal',
                         'type': 'combo',
                         'values': ['LIF_Integrate_and_Fire', 'LIF_Leaky_Hardware'],
                         'value': self.params.get('model', 'LIF_Integrate_and_Fire')},
                    ]
                },
                {
                    'name': '2. Parámetros Eléctricos LIF',
                    'fields': [
                        {'key': 'C_m_nF', 'label': 'Capacitancia C_m (nF)',
                         'type': 'float', 'value': C_m_nF,
                         'min': 0.001, 'max': 10000.0, 'step': 1.0},
                        {'key': 'R_leak_kOm', 'label': 'Resistencia Fuga R_leak (kΩ)',
                         'type': 'float', 'value': R_leak_kOm,
                         'min': 0.001, 'max': 1000000.0, 'step': 10.0},
                        {'key': 'V_th', 'label': 'Potencial Umbral V_th (V)',
                         'type': 'float', 'value': float(self.params.get('V_th', 2.5)),
                         'min': -100.0, 'max': 100.0, 'step': 0.05},
                        {'key': 'V_reset', 'label': 'Potencial Reset V_reset (V)',
                         'type': 'float', 'value': float(self.params.get('V_reset', 0.0)),
                         'min': -100.0, 'max': 100.0, 'step': 0.05},
                    ]
                },
                {
                    'name': '3. Estado Actual',
                    'fields': [
                        {'key': 'V_m', 'label': 'Potencial Membrana Vm (V)',
                         'type': 'readonly',
                         'value': f'{self.params.get("V_m", 0.0):.4f}'},
                        {'key': 'spike_count', 'label': 'Conteo de Spikes',
                         'type': 'readonly',
                         'value': f'{self.params.get("spike_count", 0)}'},
                    ]
                },
            ]
        }
