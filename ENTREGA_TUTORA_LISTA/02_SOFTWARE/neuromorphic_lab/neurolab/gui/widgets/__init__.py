"""
Widgets componentes de la interfaz gráfica neurolab GUI.
"""

from neurolab.gui.widgets.canvas_base import BaseMplCanvas
from neurolab.gui.widgets.plot_canvas import MplCanvas
from neurolab.gui.widgets.neuron_plot_canvas import NeuronMplCanvas
from neurolab.gui.widgets.hybrid_plot_canvas import HybridMplCanvas
from neurolab.gui.widgets.synapse_plot_canvas import SynapseMplCanvas
from neurolab.gui.widgets.config_panel import ConfigPanel
from neurolab.gui.widgets.signal_panel import SignalPanel

__all__ = [
    "BaseMplCanvas",
    "MplCanvas",
    "NeuronMplCanvas",
    "HybridMplCanvas",
    "SynapseMplCanvas",
    "ConfigPanel",
    "SignalPanel",
]
