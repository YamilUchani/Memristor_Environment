import numpy as np
import math
from typing import Any
from neurolab.core.base_device import BaseMathModel
from neurolab.core.config import ElectricalConfig, CustomConfig

class CustomMathModel(BaseMathModel):
    """
    Modelo matemático genérico definido por el usuario mediante ecuaciones de texto.
    Evalúa 'equation_dxdt' y opcionalmente 'equation_i' y 'equation_window'.
    """

    def _eval_eq(self, eq_str: str, x: float, v: float, i: float, R_on: float, R_off: float, D: float, mu_v: float, tau_relax: float, x_eq: float) -> float:
        local_env = {
            "x": x,
            "v": v,
            "i": i,
            "R_on": R_on,
            "R_off": R_off,
            "D": D,
            "mu_v": mu_v,
            "tau_relax": tau_relax,
            "x_eq": x_eq,
            "np": np,
            "math": math,
            "sin": math.sin,
            "cos": math.cos,
            "exp": math.exp,
            "log": math.log,
            "abs": abs,
            "sign": lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        }
        try:
            return float(eval(eq_str, {"__builtins__": None}, local_env))
        except Exception as e:
            # En caso de error de sintaxis o evaluación, retornamos 0 para no romper la simulación
            print(f"Error evaluando ecuación custom '{eq_str}': {e}")
            return 0.0

    def compute_dxdt(self, state: float, voltage: float, current: float, 
                     electrical: ElectricalConfig, model_config: Any) -> float:
        if not isinstance(model_config, CustomConfig):
            return 0.0
            
        dxdt = self._eval_eq(model_config.equation_dxdt, state, voltage, current, electrical.r_on, electrical.r_off, model_config.D, model_config.mu_v, model_config.tau_relax, model_config.x_eq)
        window = self._eval_eq(model_config.equation_window, state, voltage, current, electrical.r_on, electrical.r_off, model_config.D, model_config.mu_v, model_config.tau_relax, model_config.x_eq)
        
        return dxdt * window
        
    def compute_resistance(self, state: float, electrical: ElectricalConfig, model_config: Any, voltage: float) -> float:
        if not isinstance(model_config, CustomConfig) or not model_config.equation_i:
            # Fallback
            return electrical.r_on * state + electrical.r_off * (1.0 - state)
            
        # Para compute resistance, si equation_i es definida, tratamos de sacar i y luego r = v/i
        # En la ecuacion original de I, x, v, R_on, R_off están definidos, pero 'i' aún no se conoce.
        # Por lo tanto enviamos i = 0.
        i = self._eval_eq(model_config.equation_i, state, voltage, 0.0, electrical.r_on, electrical.r_off, model_config.D, model_config.mu_v, model_config.tau_relax, model_config.x_eq)
        
        if abs(i) < 1e-12:
             return electrical.r_off # Para evitar division por cero
        
        # R = V / I
        # Pero si V es 0, R es indefinido. 
        # Asi que evaluamos la corriente con un V de prueba pequeño si V es muy chico.
        if abs(voltage) < 1e-12:
            v_test = 1e-6
            i_test = self._eval_eq(model_config.equation_i, state, v_test, 0.0, electrical.r_on, electrical.r_off, model_config.D, model_config.mu_v, model_config.tau_relax, model_config.x_eq)
            if abs(i_test) > 1e-18:
                return abs(v_test / i_test)
            return electrical.r_off
            
        r = abs(voltage / i)
        return max(r, 1.0)
