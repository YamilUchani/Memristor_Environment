"""
neurolab.crossbar
=================
Módulo de crossbar memristivo (UNIFICADO).

Componentes:
- Crossbar: clase principal (unificada)
- CrossbarConfig: configuración
- ProgrammingMode: enum de modos
- AddressDecoder, RowDriver, ColumnDriver: hardware de dirección
- ReadConfig, WriteConfig: configuración explícita
- LineResistanceModel: modelo de resistencias de línea
- SneakPathModel: modelo de sneak paths
- CrossbarGUIHelper: integración con GUI
- Trace, STDPRule, RSTDPRule: plasticidad
"""

from .configs import CrossbarConfig, ProgrammingMode
from .core import Crossbar
from .line_resistance import LineResistanceModel
from .gui_integration import CrossbarGUIHelper
from .sneak_paths import SneakPathModel
from .addressing import (
    AddressDecoder,
    RowDriver,
    ColumnDriver,
    ReadConfig,
    WriteConfig,
)
from .validation import (
    validate_read_operation,
    validate_programming_selectivity,
    validate_line_resistance,
    validate_scalability,
    validate_sneak_paths,
    validate_d2d_c2c_variability,
    validate_all,
)

from .plasticity import (
    Trace,
    STDPConfig,
    STDPRule,
    RSTDPConfig,
    RSTDPRule,
)

# Alias para retrocompatibilidad
CrossbarIdeal = Crossbar
CrossbarSneak = Crossbar
CrossbarLine = Crossbar

__all__ = [
    'CrossbarConfig',
    'ProgrammingMode',
    'Crossbar',
    'LineResistanceModel',
    'CrossbarGUIHelper',
    'SneakPathModel',
    'CrossbarIdeal',
    'CrossbarSneak',
    'CrossbarLine',
    'AddressDecoder',
    'RowDriver',
    'ColumnDriver',
    'ReadConfig',
    'WriteConfig',
    'Trace',
    'STDPConfig',
    'STDPRule',
    'RSTDPConfig',
    'RSTDPRule',
    'validate_read_operation',
    'validate_programming_selectivity',
    'validate_line_resistance',
    'validate_scalability',
    'validate_sneak_paths',
    'validate_d2d_c2c_variability',
    'validate_all',
]

__version__ = '2.0.0'
