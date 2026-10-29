"""
neurolab.circuits
==================
Circuitos que acoplan una fuente de voltaje a una neurona LIF a través de un
elemento sináptico intermedio (memristor dinámico o resistencia fija).
"""
from neurolab.circuits.hybrid import MemristorLIFCircuit, ResistorLIFCircuit

__all__ = ["MemristorLIFCircuit", "ResistorLIFCircuit"]