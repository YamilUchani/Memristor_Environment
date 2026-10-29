from abc import ABC, abstractmethod
from typing import Any
from neurolab.core.config import ElectricalConfig

class BaseMathModel(ABC):
    """Interfaz abstracta para modelos matemáticos físicos puros de memristor."""

    @abstractmethod
    def compute_dxdt(self, state: float, voltage: float, current: float, 
                     electrical: ElectricalConfig, model_config: Any) -> float:
        """
        Calcula la velocidad instantánea de cambio del estado normalizado dx/dt.

        Args:
            state: Estado normalizado actual x in [0.0, 1.0].
            voltage: Voltaje aplicado instantáneo (V).
            current: Corriente instantánea que circula por el dispositivo (A).
            electrical: Configuración eléctrica universal (R_on, R_off, etc.).
            model_config: Configuración física específica del modelo.

        Returns:
            float: Derivada temporal dx/dt.
        """
        pass

class BaseRealismModifier(ABC):
    """Interfaz abstracta para modificadores de realismo (Ventanas, Ruido, D2D, C2C, etc.)."""

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        """
        Aplica una modificación o atenuación a la tasa de cambio dx/dt.

        Args:
            dxdt: Derivada de estado calculada previamente por el modelo base u otros modificadores.
            state: Estado normalizado x in [0.0, 1.0].
            voltage: Voltaje instantáneo (V).
            current: Corriente instantánea (A).
            electrical: Configuración eléctrica universal.
            model_config: Configuración física específica.

        Returns:
            float: Derivada dx/dt modificada.
        """
        return dxdt

    def modify_current(self, current: float, voltage: float, state: float) -> float:
        """Hook opcional para inyectar ruido o perturbación en la corriente calculada."""
        return current
