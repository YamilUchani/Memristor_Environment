"""
neurolab.neurons.lif
====================
Implementación de la neurona Leaky Integrate-and-Fire (LIF).

Cambios (v2026.09.11):
  - Bug #1 fix: modo voltage_input usa diodo ideal max(V_in-V_m, 0) para consistencia
    con la Pestaña 3 (sinapsis excitatoria unidireccional). Sin reflujo de corriente
    hacia la fuente cuando V_IN baja a 0.
"""
from typing import List
from neurolab.neurons.base import BaseNeuron
from neurolab.neurons.config import LIFConfig


class LIFNeuron(BaseNeuron):
    """
    Implementación del modelo Leaky Integrate-and-Fire (LIF) impulsado por Voltaje o Corriente.
    
    Circuito físico (Fuente de voltajes + Diodo Ideal + R_series en serie con R_leak || C_m):
    C_m * dV_m/dt = max(V_in - V_m, 0) / R_series - (V_m - V_rest) / R_leak
    
    El diodo ideal en la entrada modela una sinapsis excitatoria unidireccional:
    la corriente solo fluye de la fuente hacia la neurona, nunca al revés.
    Esto hace la Pestaña 2 consistente con la Pestaña 3 (Híbrida).
    """

    def __init__(self, config: LIFConfig = None):
        self.config = config or LIFConfig()
        self.v_membrane = self.config.v_rest
        self.v_th = float(self.config.v_th_base)
        self.refractory_time_left = 0.0
        
        # Variables de seguimiento temporal y eventos (Etapa 2.4)
        self.t = 0.0
        self.spike_times: List[float] = []
        self.has_spiked = False

    @property
    def V_m(self) -> float:
        """Alias para v_membrane."""
        return self.v_membrane

    @V_m.setter
    def V_m(self, value: float) -> None:
        self.v_membrane = float(value)

    def update(self, I_in: float = 0.0, dt: float = 1e-4, V_in: float = None) -> bool:
        """Alias para step."""
        return self.step(current_input=I_in, dt=dt, voltage_input=V_in)

    def reset(self) -> None:
        """Reinicia la neurona a su estado de reposo y limpia el historial."""

        self.v_membrane = self.config.v_rest
        self.v_th = float(self.config.v_th_base)
        self.refractory_time_left = 0.0
        self.t = 0.0
        self.spike_times.clear()
        self.has_spiked = False

    def step(self, current_input: float = 0.0, dt: float = 1e-4, voltage_input: float = None) -> bool:
        """
        Ejecuta un paso de integración usando el método de Euler explícito.
        
        Args:
            current_input: Corriente de entrada (I_in) en Amperios (modo corriente).
            dt: Incremento temporal en Segundos.
            voltage_input: Voltaje de entrada (V_in) en Voltios (modo circuito serie-paralelo).
            
        Returns:
            bool: True si se generó un spike en este paso, False en caso contrario.
        """
        # Actualizamos el reloj interno de la neurona
        self.t += dt
        
        # Reiniciamos el estado discreto de spike para este paso específico
        self.has_spiked = False

        # --- Adaptación de Umbral (Spike-Frequency Adaptation) ---
        decay_factor = dt / max(1e-4, self.config.tau_adapt)
        self.v_th += (self.config.v_th_base - self.v_th) * decay_factor

        # Control del Período Refractario
        is_refractory = False
        if self.refractory_time_left > 0.0:
            self.refractory_time_left -= dt
            is_refractory = True

        # --- Dinámica Subumbral (Integración Física) ---
        # Corriente de fuga a través de R_leak: (V_m - V_rest) / R_leak
        leak_current = (self.v_membrane - self.config.v_rest) / self.config.r_leak
        
        if is_refractory:
            effective_input_current = 0.0
        elif voltage_input is not None:
            # Bug #1 fix: Diodo Ideal — sinapsis excitatoria unidireccional.
            # La corriente solo fluye de la fuente hacia la neurona: max(V_in - V_m, 0).
            # Esto evita el reflujo de carga hacia la fuente cuando V_IN baja a 0,
            # siendo consistente con el modelo híbrido de la Pestaña 3.
            v_drop = max(voltage_input - self.v_membrane, 0.0)
            effective_input_current = v_drop / self.config.r_series
        else:
            # Entrada directa por corriente I_in
            effective_input_current = current_input
        
        # dV_m = (1/C_m) * (I_in_eff - I_leak) * dt
        dv = ((effective_input_current - leak_current) / self.config.c_m) * dt
        
        self.v_membrane += dv

        if is_refractory:
            return False

        # --- Mecanismo de Disparo (Spike) y Reset con Umbral Adaptativo ---
        if self.v_membrane >= self.v_th:
            # 1. Registrar el evento de spike
            self.has_spiked = True
            
            # 2. Registrar el instante temporal exacto
            self.spike_times.append(self.t)
            
            # 3. Reiniciar el potencial de membrana
            self.v_membrane = self.config.v_reset

            # 4. Incrementar umbral adaptativo (Auto-frenado homeostático)
            self.v_th += self.config.v_adapt_inc
            
            # 5. Iniciar periodo refractario (sólo si está configurado > 0)
            if self.config.t_ref > 0.0:
                self.refractory_time_left = self.config.t_ref

        return self.has_spiked
