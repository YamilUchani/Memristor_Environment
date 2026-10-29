"""
neurolab.gui.crossbar_elements.base_element
============================================
Elemento visual base.
"""

from typing import Dict, Any


class VisualElement:
    def __init__(self, element_id: str, x: float, y: float):
        self.element_id = element_id
        self.x = float(x)
        self.y = float(y)
        self.selected = False
        self.hover = False
        self.params: Dict[str, Any] = {}

    def hit_test(self, mx: float, my: float) -> bool:
        raise NotImplementedError

    def update_params(self, new_params: Dict[str, Any]):
        self.params.update(new_params)
        if hasattr(self, 'on_params_changed'):
            self.on_params_changed()
