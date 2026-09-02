"""
neurolab.neurons.base
=====================
Define la interfaz general para futuras neuronas en el simulador.
"""
from abc import ABC, abstractmethod

class BaseNeuron(ABC):
    """
    Interfaz abstracta para todas las neuronas del simulador.
    Cualquier modelo neuronal debe implementar estos métodos básicos.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reinicia el estado interno de la neurona a sus valores por defecto (reposo)."""
        pass

    @abstractmethod
    def step(self, current_input: float, dt: float) -> bool:
        """
        Avanza la simulación de la neurona un paso temporal.

        Args:
            current_input: Corriente o señal de entrada.
            dt: Paso de tiempo de integración.

        Returns:
            bool: True si la neurona disparó un spike en este paso, False en caso contrario.
        """
        pass
