import numpy as np
from neurolab.core.base_device import BaseDevice

class StrukovMemristor(BaseDevice):
    """
    Modelo físico determinista de Strukov et al. (2008) con regularización
    de ventana de Biolek y parámetros libremente configurables.
    """
    def __init__(self, 
                 R_on: float = 100.0, 
                 R_off: float = 500_000.0, 
                 D: float = 10e-9, 
                 mu_v: float = 1e-14, 
                 w_init: float = 0.1, 
                 p_window: int = 5,
                 enable_nonlinear_drift: bool = True):
        """
        Inicializa un memristor de Strukov parametrizable.

        Args:
            R_on: Resistencia en estado completamente dopado (Ohm).
            R_off: Resistencia en estado completamente aislante (Ohm).
            D: Espesor físico total de la película activa (m).
            mu_v: Movilidad de vacancias de oxígeno (m^2 / (V * s)).
            w_init: Fracción de estado inicial w/D (rango [0.0, 1.0]).
            p_window: Exponente para la ventana de Biolek.
            enable_nonlinear_drift: Si es True, aplica la ventana de Biolek en los bordes.
        """
        self.R_on = R_on
        self.R_off = R_off
        self.D = D
        self.mu_v = mu_v
        self.w = np.clip(w_init, 0.0, 1.0)
        self.p_window = p_window
        self.enable_nonlinear_drift = enable_nonlinear_drift
        
        # Resistencia inicial
        self._resistance = self._calculate_resistance(self.w)

    def _calculate_resistance(self, w: float) -> float:
        """Calcula la resistencia instantánea para un dopaje w."""
        return self.R_on * w + self.R_off * (1.0 - w)

    @property
    def resistance(self) -> float:
        return self._resistance

    def step(self, voltage: float, dt: float) -> float:
        """
        Calcula la corriente instantánea e integra el estado del memristor
        dada una tensión aplicada.

        Args:
            voltage: Voltaje aplicado instantáneo (V).
            dt: Intervalo temporal de integración (s).

        Returns:
            float: Corriente instantánea (A).
        """
        # 1. Calcular corriente instantánea I = V / R
        current = voltage / self._resistance

        # 2. Calcular la derivada de estado dw/dt
        # dw/dt = (mu_v * R_on / D^2) * I(t)
        # Nota: Usamos D^2 porque w es el ratio normalizado w/D.
        dwdt = (self.mu_v * self.R_on / (self.D ** 2)) * current

        # 3. Aplicar ventana de Biolek si está activa
        if self.enable_nonlinear_drift:
            # Ventana de Biolek
            # f(w) = 1 - (2w - 1)^(2p)
            # Para evitar discontinuidades, evaluamos según la polaridad de la corriente
            sign = 1.0 if current >= 0 else 0.0
            f_w = 1.0 - (2.0 * self.w - 1.0) ** (2 * self.p_window)
            
            # En la conmutación extrema de bordes, Biolek se desactiva si el voltaje empuja
            # al dispositivo de regreso del borde.
            # f(w, i) = 1 - (w - step(-i))^(2p)
            f_w_biolek = 1.0 - (self.w - (1.0 - sign)) ** (2 * self.p_window)
            dwdt *= f_w_biolek

        # 4. Integrar usando Euler Explícito
        self.w = float(np.clip(self.w + dwdt * dt, 0.0, 1.0))

        # 5. Recalcular la resistencia para el próximo paso
        self._resistance = self._calculate_resistance(self.w)

        return current
