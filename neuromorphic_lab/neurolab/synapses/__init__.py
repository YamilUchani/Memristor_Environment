"""
neurolab.synapses
=================
Módulo de sinapsis y plasticidad sináptica (LTP, LTD, STDP).
"""

from neurolab.synapses.base import Synapse
from neurolab.synapses.memristive_synapse import MemristiveSynapse
from neurolab.synapses.plasticity import LTPRule, LTDRule
from neurolab.synapses.stdp import STDPRule, AntiSTDPRule
from neurolab.synapses.configs import PlasticityConfig

__all__ = [
    "Synapse",
    "MemristiveSynapse",
    "LTPRule",
    "LTDRule",
    "STDPRule",
    "AntiSTDPRule",
    "PlasticityConfig",
]
