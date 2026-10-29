"""
neurolab.neurons
================
Módulo neuronal independiente.
Contiene la arquitectura base y los modelos neuronales, completamente
desacoplados del sistema de memristores.
"""

from neurolab.neurons.config import LIFConfig
from neurolab.neurons.base import BaseNeuron
from neurolab.neurons.lif import LIFNeuron

__all__ = ["LIFConfig", "BaseNeuron", "LIFNeuron"]
