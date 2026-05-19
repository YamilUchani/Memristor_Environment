"""
models/ — Núcleo físico: Strukov (2008), estocasticidad C2C, efectos térmicos.
"""
from .strukov_model import StrukovMemristor, calculate_resistance, dxdt_strukov
from .stochastic_model import StochasticMemristor, C2CConfig
from .thermal_model import ThermalModel
