"""
neurolab.gui.crossbar.crossbar_nxn_view
=======================================
Vista interactiva del crossbar N×N (Fase 4.4).
"""

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from neurolab.gui.styles import COLORS, FONTS

class CrossbarNxNView(QWidget):
    """Vista interactiva del crossbar N×N."""
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lbl = QLabel("🔶 Crossbar N×N (Fase 4.4)")
        lbl.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text']};")
        lay.addWidget(lbl)
