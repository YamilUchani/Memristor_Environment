"""
neurolab.circuits.hybrid
========================
Módulo que define los circuitos híbridos que acoplan una fuente de voltaje a una neurona LIF
a través de un elemento intermedio (resistencia fija o memristor).
"""

from typing import Dict
from neurolab.neurons.base import BaseNeuron
from neurolab.core.memristor import Memristor

class ResistorLIFCircuit:
    """
    Etapa 2.12: Circuito con Resistencia Fija.
    Acopla una fuente de voltaje a una neurona LIF mediante una resistencia.
    Calcula la corriente real basada en la diferencia de potencial.
    """

    def __init__(self, r_input: float, neuron: BaseNeuron):
        """
        Args:
            r_input: Valor de la resistencia de entrada en Ohmios.
            neuron: Instancia de la neurona LIF.
        """
        self.r_input = r_input
        self.neuron = neuron

    def step(self, v_source: float, dt: float) -> Dict[str, float]:
        """
        Avanza el circuito un paso temporal.

        Args:
            v_source: Voltaje de entrada de la fuente en este instante (V).
            dt: Incremento de tiempo (s).

        Returns:
            Dict con el estado actual del circuito.
        """
        # Calcular el voltaje efectivo sobre la resistencia
        v_m = self.neuron.v_membrane
        
        # Añadimos comportamiento rectificador (Diodo Ideal) al igual que en el híbrido
        if v_source >= v_m:
            i_in = (v_source - v_m) / self.r_input if self.r_input > 0 else 0.0
        else:
            i_in = 0.0
        
        # Avanzar el estado de la neurona con esta corriente
        has_spiked = self.neuron.step(i_in, dt)
        
        return {
            "v_source": v_source,
            "v_m": self.neuron.v_membrane,  # Estado actualizado tras el paso
            "i_in": i_in,
            "has_spiked": bool(has_spiked)
        }


class MemristorLIFCircuit:
    """
    Etapas 2.14 y 2.15: Módulo Híbrido Memristor-LIF.
    Acopla una fuente de voltaje a una neurona LIF a través de un Memristor.
    """

    def __init__(self, memristor: Memristor, neuron: BaseNeuron):
        """
        Args:
            memristor: Instancia del Memristor (ya configurado).
            neuron: Instancia de la neurona LIF.
        """
        self.memristor = memristor
        self.neuron = neuron

    def step(self, v_source: float, dt: float) -> Dict[str, float]:
        """
        Avanza el circuito híbrido un paso temporal.
        Calcula el acoplamiento físico en el orden correcto.

        Args:
            v_source: Voltaje de entrada de la fuente en este instante (V).
            dt: Incremento de tiempo (s).

        Returns:
            Dict con el estado completo del circuito en este instante.
        """
        # 1. El voltaje aplicado sobre el memristor depende del V_m previo
        v_m_prev = self.neuron.v_membrane
        
        # Añadimos comportamiento rectificador (Diodo Ideal / Sinapsis Biológica)
        # La corriente solo fluye desde la fuente hacia la neurona.
        if v_source >= v_m_prev:
            v_M = v_source - v_m_prev
        else:
            v_M = 0.0  # El diodo bloquea la corriente inversa
            
        # 2. El memristor avanza su estado y nos dice qué corriente lo atraviesa
        i_M = self.memristor.step(v_M, dt)
        
        # 3. Esa corriente ingresa a la neurona, la cual avanza su estado
        has_spiked = self.neuron.step(i_M, dt)
        
        return {
            "v_source": v_source,
            "v_m": self.neuron.v_membrane,  # Estado actualizado de V_m
            "v_M": v_M,
            "i_M": i_M,
            "x": self.memristor.x,
            "r_M": self.memristor.resistance,
            "has_spiked": bool(has_spiked)
        }
