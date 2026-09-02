"""
neurolab.neurons.lif
====================
Implementación de la neurona Leaky Integrate-and-Fire (LIF).
"""
from typing import List
from neurolab.neurons.base import BaseNeuron
from neurolab.neurons.config import LIFConfig


class LIFNeuron(BaseNeuron):
    """
    Implementación del modelo Leaky Integrate-and-Fire (LIF).
    
    Dinámica subumbral:
    C * dV_m/dt = I_in - (V_m - V_rest) / R
    """

    def __init__(self, config: LIFConfig = None):
        self.config = config or LIFConfig()
        self.v_membrane = self.config.v_rest
        self.refractory_time_left = 0.0
        
        # Variables de seguimiento temporal y eventos (Etapa 2.4)
        self.t = 0.0
        self.spike_times: List[float] = []
        self.has_spiked = False

    def reset(self) -> None:
        """Reinicia la neurona a su estado de reposo y limpia el historial."""
        self.v_membrane = self.config.v_rest
        self.refractory_time_left = 0.0
        self.t = 0.0
        self.spike_times.clear()
        self.has_spiked = False

    def step(self, current_input: float, dt: float) -> bool:
        """
        Ejecuta un paso de integración usando el método de Euler explícito.
        
        Args:
            current_input: Corriente de entrada (I_in) en Amperios.
            dt: Incremento temporal en Segundos.
            
        Returns:
            bool: True si se generó un spike en este paso, False en caso contrario.
        """
        # Actualizamos el reloj interno de la neurona
        self.t += dt
        
        # Reiniciamos el estado discreto de spike para este paso específico
        self.has_spiked = False

        # Etapa 2.5 - Control del Período Refractario (Realista)
        is_refractory = False
        if self.refractory_time_left > 0.0:
            self.refractory_time_left -= dt
            is_refractory = True

        # --- Dinámica Subumbral (Integración) ---
        # Corriente de fuga: (V_m - V_rest) / R
        leak_current = (self.v_membrane - self.config.v_rest) / self.config.r_leak
        
        # Si está en periodo refractario, el switch S1 está abierto, la corriente de entrada es 0
        effective_input_current = 0.0 if is_refractory else current_input
        
        # dV_m = (1/C) * (I_in - I_leak) * dt
        dv = ((effective_input_current - leak_current) / self.config.c_m) * dt
        
        self.v_membrane += dv

        if is_refractory:
            return False

        # --- Etapa 2.4 - Mecanismo de Disparo (Spike) y Reset ---
        if self.v_membrane >= self.config.v_th:
            # 1. Registrar el evento de spike (separado del estado físico de membrana)
            self.has_spiked = True
            
            # 2. Registrar el instante temporal exacto
            self.spike_times.append(self.t)
            
            # 3. Reiniciar el potencial de membrana
            self.v_membrane = self.config.v_reset
            
            # 4. Iniciar periodo refractario (sólo si está configurado > 0)
            if self.config.t_ref > 0.0:
                self.refractory_time_left = self.config.t_ref

        return self.has_spiked
