"""
neurolab.gui.crossbar_elements.sensor_element
==============================================
Elemento Sensor.
"""

from typing import Dict, Any
from neurolab.gui.crossbar_elements.base_element import VisualElement


class SensorElement(VisualElement):
    def __init__(self, element_id: str, x: float, y: float):
        super().__init__(element_id, x, y)
        self.params = {
            'name': 'Sensor Frontal',
            'type': 'infrarrojo',
            'V_out': 1.0,           # V
            'f_max': 200.0,         # Hz
            'd_min': 0.05,          # m
            'd_max': 2.0,           # m
            'd_actual': 0.30,       # m
            'noise_enabled': False,
            'noise_sigma': 0.02,
            # Codificación Poisson
            'f_min_poisson': 10.0,   # Hz
            'f_max_poisson': 300.0,  # Hz
        }

    def hit_test(self, mx: float, my: float) -> bool:
        return abs(mx - self.x) <= 45 and abs(my - self.y) <= 40

    def get_config(self) -> Dict[str, Any]:
        return {
            'title': f'Sensor: {self.params["name"]}',
            'sections': [
                {
                    'name': 'Identificación',
                    'fields': [
                        {'key': 'name', 'label': 'Nombre',
                         'type': 'text', 'value': self.params['name']},
                        {'key': 'type', 'label': 'Tipo',
                         'type': 'combo',
                         'values': ['infrarrojo', 'ultrasonido', 'lidar'],
                         'value': self.params['type']},
                    ]
                },
                {
                    'name': 'Señal Eléctrica',
                    'fields': [
                        {'key': 'V_out', 'label': 'V_out (V)',
                         'type': 'float', 'value': self.params['V_out'],
                         'min': 0.1, 'max': 5.0, 'step': 0.1},
                        {'key': 'f_max', 'label': 'Frecuencia máx (Hz)',
                         'type': 'float', 'value': self.params['f_max'],
                         'min': 10.0, 'max': 1000.0, 'step': 10.0},
                    ]
                },
                {
                    'name': 'Rango de Detección',
                    'fields': [
                        {'key': 'd_min', 'label': 'Distancia mín (m)',
                         'type': 'float', 'value': self.params['d_min'],
                         'min': 0.0, 'max': 5.0, 'step': 0.01},
                        {'key': 'd_max', 'label': 'Distancia máx (m)',
                         'type': 'float', 'value': self.params['d_max'],
                         'min': 0.1, 'max': 10.0, 'step': 0.1},
                        {'key': 'd_actual', 'label': 'Distancia actual (m)',
                         'type': 'float', 'value': self.params['d_actual'],
                         'min': 0.0, 'max': 10.0, 'step': 0.01},
                    ]
                },
                {
                    'name': 'Codificación Poisson',
                    'fields': [
                        {'key': 'f_min_poisson', 'label': 'f_mín (Hz)',
                         'type': 'float', 'value': self.params['f_min_poisson'],
                         'min': 1.0, 'max': 100.0, 'step': 1.0},
                        {'key': 'f_max_poisson', 'label': 'f_máx (Hz)',
                         'type': 'float', 'value': self.params['f_max_poisson'],
                         'min': 50.0, 'max': 1000.0, 'step': 10.0},
                    ]
                },
                {
                    'name': 'Ruido',
                    'fields': [
                        {'key': 'noise_enabled', 'label': 'Activar ruido',
                         'type': 'check', 'value': self.params['noise_enabled']},
                        {'key': 'noise_sigma', 'label': 'σ relativo',
                         'type': 'float', 'value': self.params['noise_sigma'],
                         'min': 0.0, 'max': 0.5, 'step': 0.01},
                    ]
                },
            ]
        }
