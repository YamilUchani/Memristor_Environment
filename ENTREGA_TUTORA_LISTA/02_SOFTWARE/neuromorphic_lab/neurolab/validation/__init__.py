"""
neurolab.validation
===================
Módulo de validación contra papers experimentales.
"""

from neurolab.validation.jo2010_stdp import (
    load_jo2010_stdp_data,
    validate_jo2010_stdp,
)
from neurolab.validation.jo2010_ltp_ltd import (
    load_jo2010_ltp_ltd_data,
    validate_jo2010_ltp_ltd,
)
from neurolab.validation.prezioso2015_s3 import (
    simulate_iv_virgin,
    simulate_conductance_map,
    compute_histogram_stats,
    validate_prezioso2015_s3,
    PREZIOSO_VIRGIN_DEFAULTS,
)

__all__ = [
    'load_jo2010_stdp_data',
    'validate_jo2010_stdp',
    'load_jo2010_ltp_ltd_data',
    'validate_jo2010_ltp_ltd',
    'simulate_iv_virgin',
    'simulate_conductance_map',
    'compute_histogram_stats',
    'validate_prezioso2015_s3',
    'PREZIOSO_VIRGIN_DEFAULTS',
]
