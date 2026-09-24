"""
neurolab.gui.widgets.canvas_base
=================================
Clase base común para todos los Canvas Matplotlib de la GUI.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


COLORS = {
    'bg_figure': '#1e1e2e',
    'bg_axes':   '#181825',
    'spine':     '#45475a',
    'grid':      '#585b70',
    'tick':      '#cdd6f4',
    'title':     '#89b4fa',
    'text':      '#cdd6f4',
}

MAX_DISPLAY_POINTS = 4000
ENSEMBLE_DISPLAY_POINTS = 1200


class BaseMplCanvas(QWidget):
    """Clase base para todos los Canvas Matplotlib de la GUI."""

    def __init__(self, parent=None, rows: int = 1, cols: int = 1,
                 figsize: tuple = (10, 8), dpi: int = 100,
                 sharex: bool = True, add_toolbar: bool = True):
        super().__init__(parent)

        self.figure = Figure(figsize=figsize, dpi=dpi, facecolor=COLORS['bg_figure'])
        self.canvas = FigureCanvas(self.figure)

        self.toolbar = None
        if add_toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        if self.toolbar is not None:
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self._rows = rows
        self._cols = cols
        self._sharex = sharex
        self.axes = self._create_axes(rows, cols, sharex)
        self._apply_style(self.axes)

    def _create_axes(self, rows, cols, sharex):
        axes = []
        for i in range(rows * cols):
            if i == 0:
                ax = self.figure.add_subplot(rows, cols, i + 1)
            else:
                if sharex:
                    ax = self.figure.add_subplot(rows, cols, i + 1, sharex=axes[0])
                else:
                    ax = self.figure.add_subplot(rows, cols, i + 1)
            axes.append(ax)
        return axes

    def _apply_style(self, axes=None):
        if axes is None:
            axes = self.axes
        for ax in axes:
            ax.set_facecolor(COLORS['bg_axes'])
            ax.tick_params(colors=COLORS['tick'], labelsize=8)
            ax.xaxis.label.set_color(COLORS['text'])
            ax.yaxis.label.set_color(COLORS['text'])
            ax.title.set_color(COLORS['title'])
            for spine in ax.spines.values():
                spine.set_color(COLORS['spine'])
            ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])

    def clear_axes(self, axes=None):
        if axes is None:
            axes = self.axes
        for ax in axes:
            ax.clear()
        self._apply_style(axes)

    def clear_figure(self):
        self.figure.clear()
        self.axes = self._create_axes(self._rows, self._cols, self._sharex)
        self._apply_style(self.axes)

    def refresh(self):
        try:
            self.figure.tight_layout(pad=1.8)
        except Exception:
            pass
        self.canvas.draw()

    @staticmethod
    def _decimate(x, max_points=MAX_DISPLAY_POINTS):
        n = len(x)
        if n <= max_points:
            return slice(None)
        step = max(1, n // max_points)
        return slice(None, None, step)

    @staticmethod
    def _decimate_ensemble(x):
        return BaseMplCanvas._decimate(x, ENSEMBLE_DISPLAY_POINTS)
