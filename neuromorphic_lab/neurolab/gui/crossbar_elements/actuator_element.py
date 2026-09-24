"""
neurolab.gui.crossbar_elements.actuator_element
================================================
Elemento Actuador.
"""

from typing import Dict, Any
from neurolab.gui.crossbar_elements.base_element import VisualElement


class ActuatorElement(VisualElement):
    def __init__(self, element_id: str, x: float, y: float):
        super().__init__(element_id, x, y)
        self.params = {
            'name': 'Motor',
            'action': 'retroceder',
            'V_threshold': 0.5,
            'speed': 1.0,
        }

    def hit_test(self, mx: float, my: float) -> bool:
        return abs(mx - self.x) <= 45 and abs(my - self.y) <= 30

    def get_config(self) -> Dict[str, Any]:
        return {
            'title': f'Actuador: {self.params["name"]}',
            'sections': [
                {
                    'name': 'Identificación',
                    'fields': [
                        {'key': 'name', 'label': 'Nombre',
                         'type': 'text', 'value': self.params['name']},
                        {'key': 'action', 'label': 'Acción',
                         'type': 'combo',
                         'values': ['avanzar', 'retroceder',
                                    'girar_izquierda', 'girar_derecha'],
                         'value': self.params['action']},
                    ]
                },
                {
                    'name': 'Control',
                    'fields': [
                        {'key': 'V_threshold', 'label': 'V_umbral (V)',
                         'type': 'float', 'value': self.params['V_threshold'],
                         'min': 0.0, 'max': 5.0, 'step': 0.1},
                        {'key': 'speed', 'label': 'Velocidad',
                         'type': 'float', 'value': self.params['speed'],
                         'min': 0.0, 'max': 2.0, 'step': 0.1},
                    ]
                },
            ]
        }
