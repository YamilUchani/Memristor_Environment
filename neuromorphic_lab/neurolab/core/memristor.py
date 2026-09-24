import numpy as np
from typing import List, Optional, Any
from neurolab.core.config import DeviceIdentity, ElectricalConfig
from neurolab.core.base_device import BaseMathModel, BaseRealismModifier

class Memristor:
    """
    Abstracción universal de un Dispositivo Memristivo.

    Organización en 5 niveles:
    - NIVEL A: Identidad y tipo de dispositivo (DeviceIdentity).
    - NIVEL B: Parámetros eléctricos universales (ElectricalConfig: R_on, R_off, G_on, G_off).
    - NIVEL C: Estado interno normalizado x in [0.0, 1.0].
    - NIVEL D: Modificadores de realismo (Decoradores/Pipeline de efectos: Biolek, D2D, C2C, ruido).
    - NIVEL E: Modelo matemático físico base (BaseMathModel).
    """

    def __init__(
        self,
        math_model: BaseMathModel,
        electrical: Optional[ElectricalConfig] = None,
        identity: Optional[DeviceIdentity] = None,
        model_config: Optional[Any] = None,
        modifiers: Optional[List[BaseRealismModifier]] = None,
        clip_x: bool = True
    ):
        self.math_model = math_model
        self.electrical = electrical or ElectricalConfig()
        self.identity = identity or DeviceIdentity()
        self.model_config = model_config
        self.modifiers: List[BaseRealismModifier] = modifiers or []
        self.clip_x: bool = clip_x

        # NIVEL C: Estado interno normalizado x in [0.0, 1.0] (o libre si clip_x=False)
        val = float(self.electrical.initial_state)
        self._x: float = val if not self.clip_x else (1.0 if val > 1.0 else (0.0 if val < 0.0 else val))

    @property
    def x(self) -> float:
        """Estado interno normalizado x (0.0 = OFF / R_off, 1.0 = ON / R_on)."""
        return self._x

    @property
    def RON(self) -> float:
        return self.electrical.r_on

    @property
    def ROFF(self) -> float:
        return self.electrical.r_off

    @property
    def r_on(self) -> float:
        return self.electrical.r_on

    @property
    def r_off(self) -> float:
        return self.electrical.r_off


    @x.setter
    def x(self, value: float) -> None:
        val = float(value)
        self._x = val if not self.clip_x else (1.0 if val > 1.0 else (0.0 if val < 0.0 else val))

    @property
    def state(self) -> float:
        """Alias para el estado interno normalizado x."""
        return self.x

    @property
    def resistance(self) -> float:
        """
        Resistencia instantánea R(x) en Ohmios.
        R(x) = R_on * x + R_off * (1.0 - x)

        Soft-clamp (v2026.09.11): La resistencia tiene un piso de 1 Ω.
        Esto evita explosiones numéricas cuando R_ON es muy pequeño y el
        paso temporal es grande (corrientes artificiales de miles de amperios).
        Físicamente justificado: ningún óxido memristivo real tiene R < 1 Ω.
        """
        import numpy as np
        x_eff = float(np.clip(self._x, 0.0, 1.0))
        r_min_phys = max(1.0, float(self.electrical.r_on))
        v_curr = getattr(self, "_last_v", 0.0)
        if hasattr(self.math_model, "compute_resistance"):
            r_raw = self.math_model.compute_resistance(x_eff, self.electrical, self.model_config, v_curr)
        else:
            r_raw = self.electrical.r_on * x_eff + self.electrical.r_off * (1.0 - x_eff)
        return max(float(r_raw), r_min_phys)


    @property
    def conductance(self) -> float:
        """
        Conductancia instantánea G(x) en Siemens.
        G(x) = 1.0 / R(x)
        """
        return 1.0 / self.resistance

    def current(self, voltage: float) -> float:
        """
        Calcula la corriente instantánea que circula por el memristor (A).
        Aplica los modificadores de realismo en la corriente si están presentes.
        """
        self._last_v = voltage
        r_current = self.resistance
        curr = voltage / r_current if r_current > 0 else 0.0
        for modifier in self.modifiers:
            curr = modifier.modify_current(curr, voltage, self._x)
        return curr

    def update(self, voltage: float, dt: float) -> None:
        """
        Actualiza el estado interno integrando dx/dt en el paso temporal dt.
        """
        self._last_v = voltage
        r_current = self.resistance
        curr = voltage / r_current if r_current > 0 else 0.0

        dxdt = self.math_model.compute_dxdt(
            state=self._x,
            voltage=voltage,
            current=curr,
            electrical=self.electrical,
            model_config=self.model_config
        )

        for modifier in self.modifiers:
            dxdt = modifier.modify_dxdt(
                dxdt=dxdt,
                state=self._x,
                voltage=voltage,
                current=curr,
                electrical=self.electrical,
                model_config=self.model_config
            )

        new_x = self._x + dxdt * dt
        if self.clip_x:
            self._x = 1.0 if new_x > 1.0 else (0.0 if new_x < 0.0 else new_x)
        else:
            self._x = new_x

    def step(self, voltage: float, dt: float) -> float:
        """
        Ejecuta un paso de integración temporal en el dispositivo.

        Args:
            voltage: Voltaje instantáneo aplicado (V).
            dt: Intervalo de integración temporal en segundos (s).

        Returns:
            float: Corriente instantánea que circula por el memristor (A).
        """
        i = self.current(voltage)
        self.update(voltage, dt)
        return i

    def reset(self, initial_state: Optional[float] = None) -> None:
        """Reinicia el estado interno al estado inicial especificado o configurado."""
        val = float(initial_state if initial_state is not None else self.electrical.initial_state)
        if self.clip_x:
            self._x = 1.0 if val > 1.0 else (0.0 if val < 0.0 else val)
        else:
            self._x = val
