"""
models/ — Núcleo físico: Strukov (2008), estocasticidad C2C, efectos térmicos,
          neurona LIF (Leaky Integrate-and-Fire).
"""
from .strukov_model import StrukovMemristor, calculate_resistance, dxdt_strukov
from .stochastic_model import StochasticMemristor, C2CConfig
from .thermal_model import ThermalModel
from .lif_neuron import LIFNeuron, LIFParameters, default_lif_params, fast_lif_params
