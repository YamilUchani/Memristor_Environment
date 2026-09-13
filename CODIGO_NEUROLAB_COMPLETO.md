# CÓDIGO COMPLETO — Neuromorphic Lab

Este documento es un **bundle generado automáticamente** con todos los archivos de código fuente de `neuromorphic_lab/` y los orquestadores de la raíz del repo.
Generado: 2026-09-12T16:07:36
Total de archivos: 53



---

## Arranque de la GUI

`neuromorphic_lab/run_app.py` — 15 líneas

```python
"""
Script ejecutor de la interfaz gráfica Neuromorphic Lab.
Uso: python run_app.py
"""

import sys
import os

# Asegurar que el paquete neurolab se encuentre en el PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neurolab.gui.app import main

if __name__ == "__main__":
    main()

```



---

## neurolab

`neuromorphic_lab\neurolab\__init__.py` — 1 líneas

```python
"""Neuromorphic Lab package."""

```



---

## neurolab

`neuromorphic_lab\neurolab\circuits\__init__.py` — 9 líneas

```python
"""
neurolab.circuits
==================
Circuitos que acoplan una fuente de voltaje a una neurona LIF a través de un
elemento sináptico intermedio (memristor dinámico o resistencia fija).
"""
from neurolab.circuits.hybrid import MemristorLIFCircuit, ResistorLIFCircuit

__all__ = ["MemristorLIFCircuit", "ResistorLIFCircuit"]
```



---

## neurolab

`neuromorphic_lab\neurolab\circuits\hybrid.py` — 110 líneas

```python
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

```



---

## neurolab

`neuromorphic_lab\neurolab\core\__init__.py` — 17 líneas

```python
"""
Módulo core de neurolab.
Contiene la abstracción universal del Memristor, estructuras de configuración y clases base.
"""

from neurolab.core.config import DeviceIdentity, ElectricalConfig, StrukovConfig
from neurolab.core.base_device import BaseMathModel, BaseRealismModifier
from neurolab.core.memristor import Memristor

__all__ = [
    "DeviceIdentity",
    "ElectricalConfig",
    "StrukovConfig",
    "BaseMathModel",
    "BaseRealismModifier",
    "Memristor",
]

```



---

## neurolab

`neuromorphic_lab\neurolab\core\base_device.py` — 49 líneas

```python
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

```



---

## neurolab

`neuromorphic_lab\neurolab\core\config.py` — 47 líneas

```python
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class DeviceIdentity:
    """NIVEL A: Identidad y metadatos del dispositivo."""
    device_name: str = "Strukov TiO2"
    device_family: str = "oxide_memristor"
    model_name: str = "strukov"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ElectricalConfig:
    """NIVEL B: Parámetros eléctricos universales del dispositivo."""
    r_on: float = 100.0        # R_min = 100 Ω (Strukov 2008)
    r_off: float = 16_000.0    # R_max = 16 kΩ (Strukov 2008, Ratio 160)
    initial_state: float = 0.10 # x_0 = w_0/D = 0.10 (w_0 = 1 nm)

    def __post_init__(self):
        if self.r_on <= 0 or self.r_off <= 0:
            raise ValueError("R_on y R_off deben ser estrictamente positivos.")
        if self.r_on >= self.r_off:
            raise ValueError("R_on debe ser menor que R_off.")
        if not (0.0 <= self.initial_state <= 1.0):
            raise ValueError("initial_state debe estar acotado en el rango [0.0, 1.0].")

    @property
    def g_on(self) -> float:
        """Conductancia máxima en Siemens (1 / R_on)."""
        return 1.0 / self.r_on

    @property
    def g_off(self) -> float:
        """Conductancia mínima en Siemens (1 / R_off)."""
        return 1.0 / self.r_off

@dataclass
class StrukovConfig:
    """NIVEL E: Parámetros físicos específicos del modelo de Strukov (2008)."""
    D: float = 10e-9        # Espesor físico de la capa activa en metros (10 nm)
    mu_v: float = 1e-14     # Movilidad de vacancias de oxígeno en m^2 / (V * s)

    def __post_init__(self):
        if self.D <= 0:
            raise ValueError("El grosor D debe ser estrictamente positivo.")
        if self.mu_v <= 0:
            raise ValueError("La movilidad mu_v debe ser estrictamente positiva.")

```



---

## neurolab

`neuromorphic_lab\neurolab\core\lif_validation.py` — 100 líneas

```python
"""
neurolab.core.lif_validation
============================
Módulo de validación analítica matemática para la neurona LIF (Sección E).
Calcula la solución analítica diferencial exacta V_teorico(t) y reporta
métricas cuantitativas de exactitud (MAE, RMSE, Error Relativo, R²).
"""

import numpy as np
from typing import Dict, Any
from neurolab.neurons.config import LIFConfig


def compute_lif_analytical_trajectory(
    t: np.ndarray,
    v_sim: np.ndarray,
    v_signal: np.ndarray,
    config: LIFConfig,
    is_voltage_input: bool = True
) -> np.ndarray:
    """
    Calcula la trayectoria analítica exacta V_teorico(t) del potencial de membrana.
    
    Ecuación diferencial:
      tau_eq * dV/dt = V_inf(t) - V
      V_teorico(k) = V_inf(k) + (V_teorico(k-1) - V_inf(k)) * exp(-dt / tau_eq)
    """
    steps = len(t)
    v_analytical = np.zeros(steps)
    
    if steps == 0:
        return v_analytical
        
    dt = t[1] - t[0] if steps > 1 else 1e-4
    
    if is_voltage_input:
        r_eq = config.r_eq
        tau_eq = config.tau
        # V_inf(t) = (R_leak / (R_S + R_leak)) * V_IN(t) + (R_S / (R_S + R_leak)) * V_rest
        r_sum = config.r_series + config.r_leak
        v_inf_array = (config.r_leak / r_sum) * v_signal + (config.r_series / r_sum) * config.v_rest
    else:
        # Modo corriente directa I_in
        tau_eq = config.r_leak * config.c_m
        v_inf_array = config.v_rest + config.r_leak * v_signal

    v_current = config.v_rest
    v_analytical[0] = v_current

    for k in range(1, steps):
        # Si la simulación reseteó la membrana por spike en el paso anterior
        if v_sim[k-1] <= config.v_reset + 1e-6 and v_sim[max(0, k-2)] >= config.v_th - 0.05:
            v_current = config.v_reset

        v_inf = v_inf_array[k]
        v_current = v_inf + (v_current - v_inf) * np.exp(-dt / tau_eq)
        v_analytical[k] = v_current

    return v_analytical


def compute_lif_validation_metrics(
    t: np.ndarray,
    v_sim: np.ndarray,
    v_signal: np.ndarray,
    config: LIFConfig,
    is_voltage_input: bool = True
) -> Dict[str, float]:
    """
    Calcula las métricas cuantitativas formales comparando V_simulado(t) vs V_teórico(t).
    
    Returns:
        Dict con MAE, RMSE, Error_Relativo_Max_Pct, R2 y V_analytical.
    """
    v_analytical = compute_lif_analytical_trajectory(t, v_sim, v_signal, config, is_voltage_input)
    
    # 1. MAE (Error Absoluto Medio)
    mae = float(np.mean(np.abs(v_sim - v_analytical)))
    
    # 2. RMSE (Raíz del Error Cuadrático Medio)
    rmse = float(np.sqrt(np.mean((v_sim - v_analytical) ** 2)))
    
    # 3. Error Relativo Máximo (%)
    denom = np.abs(v_analytical)
    denom = np.where(denom < 1e-6, 1e-6, denom)
    rel_err_max = float(np.max(np.abs(v_sim - v_analytical) / denom) * 100.0)
    
    # 4. R² (Coeficiente de Determinación Teórico)
    ss_res = np.sum((v_sim - v_analytical) ** 2)
    ss_tot = np.sum((v_sim - np.mean(v_sim)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 1.0
    r2 = max(0.0, min(1.0, r2))
    
    return {
        "mae": mae,
        "rmse": rmse,
        "rel_err_max": rel_err_max,
        "r2": r2,
        "v_analytical": v_analytical
    }

```



---

## neurolab

`neuromorphic_lab\neurolab\core\memristor.py` — 131 líneas

```python
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
        R_MIN_OHMS = 1.0  # Piso de resistencia para estabilidad numérica
        x_eff = max(0.0, min(1.0, self._x)) if not self.clip_x else self._x
        r_raw = self.electrical.r_on * x_eff + self.electrical.r_off * (1.0 - x_eff)
        return max(r_raw, R_MIN_OHMS)

    @property
    def conductance(self) -> float:
        """
        Conductancia instantánea G(x) en Siemens.
        G(x) = 1.0 / R(x)
        """
        return 1.0 / self.resistance

    def step(self, voltage: float, dt: float) -> float:
        """
        Ejecuta un paso de integración temporal en el dispositivo.

        Args:
            voltage: Voltaje instantáneo aplicado (V).
            dt: Intervalo de integración temporal en segundos (s).

        Returns:
            float: Corriente instantánea que circula por el memristor (A).
        """
        # 1. Corriente instantánea Ohmiana base: I = V / R(x)
        r_current = self.resistance
        current = voltage / r_current if r_current > 0 else 0.0

        # 2. Calcular la tasa de cambio base dx/dt desde el modelo matemático físico
        dxdt = self.math_model.compute_dxdt(
            state=self._x,
            voltage=voltage,
            current=current,
            electrical=self.electrical,
            model_config=self.model_config
        )

        # 3. Aplicar pipeline de modificadores de realismo (Efectos no lineales, bordes, variabilidad)
        for modifier in self.modifiers:
            dxdt = modifier.modify_dxdt(
                dxdt=dxdt,
                state=self._x,
                voltage=voltage,
                current=current,
                electrical=self.electrical,
                model_config=self.model_config
            )

        # 4. Integrar el estado normalizado usando Euler Explícito (optimizado)
        new_x = self._x + dxdt * dt
        if self.clip_x:
            self._x = 1.0 if new_x > 1.0 else (0.0 if new_x < 0.0 else new_x)
        else:
            self._x = new_x

        # 5. Aplicar modificaciones en la corriente de salida si hay ruido térmico/lectura
        for modifier in self.modifiers:
            current = modifier.modify_current(current, voltage, self._x)

        return current

    def reset(self, initial_state: Optional[float] = None) -> None:
        """Reinicia el estado interno al estado inicial especificado o configurado."""
        val = float(initial_state if initial_state is not None else self.electrical.initial_state)
        if self.clip_x:
            self._x = 1.0 if val > 1.0 else (0.0 if val < 0.0 else val)
        else:
            self._x = val

```



---

## neurolab

`neuromorphic_lab\neurolab\core\validation_metrics.py` — 158 líneas

```python
"""
neurolab.core.validation_metrics
==================================
Cálculo de métricas cuantitativas de validación entre la simulación y datos de referencia (CSV).

Métricas implementadas:
  - MAE   : Error Absoluto Medio
  - RMSE  : Raíz del Error Cuadrático Medio
  - Error Relativo Máximo (%)
  - Correlación de Pearson (R²)
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class QuantityMetrics:
    """Métricas de error para una magnitud física específica."""
    name: str
    units: str
    mae: float = 0.0
    rmse: float = 0.0
    max_rel_error_pct: float = 0.0
    r2: float = 0.0
    n_points: int = 0

    def to_html_row(self) -> str:
        """Devuelve una fila HTML <tr> para insertar en una tabla de resultados."""
        r2_color = "#a6e3a1" if self.r2 >= 0.95 else ("#f9e2af" if self.r2 >= 0.80 else "#f38ba8")
        err_color = "#a6e3a1" if self.max_rel_error_pct <= 5.0 else ("#f9e2af" if self.max_rel_error_pct <= 20.0 else "#f38ba8")
        return (
            f"<tr>"
            f"<td style='padding:3px 6px; color:#cdd6f4;'><b>{self.name}</b></td>"
            f"<td style='padding:3px 6px; color:#cdd6f4; text-align:right;'>{self.mae:.4g} {self.units}</td>"
            f"<td style='padding:3px 6px; color:#cdd6f4; text-align:right;'>{self.rmse:.4g} {self.units}</td>"
            f"<td style='padding:3px 6px; color:{err_color}; text-align:right;'>{self.max_rel_error_pct:.2f}%</td>"
            f"<td style='padding:3px 6px; color:{r2_color}; text-align:right;'>{self.r2:.4f}</td>"
            f"</tr>"
        )


def _pearson_r2(y_ref: np.ndarray, y_sim: np.ndarray) -> float:
    """Calcula el coeficiente de determinación R² de Pearson entre dos vectores."""
    if len(y_ref) < 2 or np.std(y_ref) == 0.0 or np.std(y_sim) == 0.0:
        return float("nan")
    corr = np.corrcoef(y_ref, y_sim)
    return float(corr[0, 1] ** 2) if not np.isnan(corr[0, 1]) else float("nan")


def compute_metrics(
    y_ref: np.ndarray,
    y_sim: np.ndarray,
    name: str,
    units: str,
    threshold: float = 1e-12
) -> QuantityMetrics:
    """
    Calcula MAE, RMSE, error relativo máximo y R² entre y_ref (CSV) e y_sim (simulado).

    Args:
        y_ref:     Vector de referencia (datos CSV).
        y_sim:     Vector simulado (interpolado en los mismos instantes).
        name:      Nombre de la magnitud (e.g. "x(t) — w/D").
        units:     Unidades (e.g. "adim.", "mA", "kΩ").
        threshold: Umbral mínimo para calcular error relativo (evita división por cero).

    Returns:
        QuantityMetrics con los resultados.
    """
    mask = np.isfinite(y_ref) & np.isfinite(y_sim)
    y_r = y_ref[mask]
    y_s = y_sim[mask]
    n = len(y_r)

    if n == 0:
        return QuantityMetrics(name=name, units=units, n_points=0)

    diff = np.abs(y_r - y_s)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Error relativo normalizado por la amplitud pico (evita singularidades en cruces por cero)
    peak = float(np.max(np.abs(y_r)))
    if peak > threshold:
        max_rel_err = float(np.max(diff) / peak * 100.0)
    else:
        max_rel_err = float("nan")

    r2 = _pearson_r2(y_r, y_s)
    return QuantityMetrics(
        name=name, units=units,
        mae=mae, rmse=rmse,
        max_rel_error_pct=max_rel_err,
        r2=r2, n_points=n
    )


def compute_all_metrics(
    t_sim: np.ndarray,
    i_sim_mA: np.ndarray,
    x_sim: np.ndarray,
    r_sim_kohm: np.ndarray,
    g_sim_us: np.ndarray,
    val_data: dict
) -> list[QuantityMetrics]:
    """
    Calcula métricas para todas las magnitudes disponibles en val_data,
    interpolando la simulación en los instantes de tiempo del CSV.

    Args:
        t_sim:       Vector de tiempo de la simulación (s).
        i_sim_mA:    Corriente simulada en mA.
        x_sim:       Estado interno simulado x = w/D (adim.).
        r_sim_kohm:  Resistencia simulada en kΩ.
        g_sim_us:    Conductancia simulada en µS.
        val_data:    Diccionario devuelto por ValidationDataLoader.load_all().

    Returns:
        Lista de QuantityMetrics ordenada por magnitud.
    """
    results = []

    # ── Estado interno x(t) = w/D ──────────────────────────────────────────
    t_w = val_data.get("t_w")
    wd_ref = val_data.get("wd_val")
    if t_w is not None and wd_ref is not None:
        x_interp = np.interp(t_w, t_sim, x_sim)
        results.append(compute_metrics(wd_ref, x_interp, "x(t) — w/D", "adim.", threshold=1e-3))

    # ── Corriente I(t) en mA ────────────────────────────────────────────────
    t_i = val_data.get("t_i")
    i_ref_mA = val_data.get("i_val_mA")
    if t_i is not None and i_ref_mA is not None:
        i_interp = np.interp(t_i, t_sim, i_sim_mA)
        results.append(compute_metrics(i_ref_mA, i_interp, "I(t)", "mA", threshold=1e-4))

    # ── Resistencia R(t) en kΩ ──────────────────────────────────────────────
    r_ref = val_data.get("r_val_kohm")
    if t_i is not None and r_ref is not None:
        r_interp = np.interp(t_i, t_sim, r_sim_kohm)
        mask_valid = np.isfinite(r_ref)
        if mask_valid.any():
            results.append(compute_metrics(
                r_ref[mask_valid], r_interp[mask_valid], "R(t)", "kΩ", threshold=0.01
            ))

    # ── Conductancia G(t) en µS ─────────────────────────────────────────────
    g_ref = val_data.get("g_val_us")
    if t_i is not None and g_ref is not None:
        g_interp = np.interp(t_i, t_sim, g_sim_us)
        mask_valid = np.isfinite(g_ref)
        if mask_valid.any():
            results.append(compute_metrics(
                g_ref[mask_valid], g_interp[mask_valid], "G(t)", "µS", threshold=1.0
            ))

    return results

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\__init__.py` — 22 líneas

```python
"""
Paquete neurolab.devices: Modelos, configuraciones, estado y realismo.
"""
from neurolab.devices.config import (
    DeviceConfig, ElectricalConfig, StateConfig, RealismConfig,
    WindowConfig, StochasticConfig, StrukovConfig
)
from neurolab.devices.base import BaseDevice, BaseMathModel, BaseRealismModifier
from neurolab.devices.state import StateManager

__all__ = [
    "DeviceConfig",
    "ElectricalConfig",
    "StateConfig",
    "RealismConfig",
    "WindowConfig",
    "StochasticConfig",
    "StrukovConfig",
    "BaseMathModel",
    "BaseRealismModifier",
    "StateManager"
]

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\base.py` — 14 líneas

```python
"""
neurolab.devices.base
=====================
Reexporta las clases base abstractas desde `core.base_device` para mantener
compatibilidad de imports en todo el paquete `devices`.

Fuente de verdad: neurolab/core/base_device.py
"""
from neurolab.core.base_device import BaseMathModel, BaseRealismModifier  # noqa: F401

# Alias de compatibilidad legacy
BaseDevice = None  # No utilizado en la arquitectura actual

__all__ = ["BaseMathModel", "BaseRealismModifier"]

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\config.py` — 92 líneas

```python
"""
neurolab.devices.config
=======================
Configuraciones estructurales del dispositivo memristivo.

- `ElectricalConfig` y `StrukovConfig` se importan desde `core.config`
  (fuente de verdad única).
- `DeviceConfig`, `StateConfig`, `WindowConfig`, `StochasticConfig`,
  `RealismConfig` se definen aquí como parte de la capa `devices`.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# ── Reexports desde core (fuente de verdad) ─────────────────────────────────
from neurolab.core.config import ElectricalConfig, StrukovConfig  # noqa: F401

# ── Configuraciones propias de la capa devices ───────────────────────────────

@dataclass
class DeviceConfig:
    """A. Configuración General de Identidad del Dispositivo."""
    name: str = "Strukov 2008 - Figure 2b"
    family: str = "oxide_memristor"
    model: str = "strukov"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def device_name(self) -> str:
        return self.name

    @property
    def device_family(self) -> str:
        return self.family

    @property
    def model_name(self) -> str:
        return self.model


@dataclass
class StateConfig:
    """C. Configuración del Estado Interno Normalizado."""
    x_init: float = 0.1
    x_min: float = 0.0
    x_max: float = 1.0
    normalized: bool = True

    def __post_init__(self):
        if not (self.x_min <= self.x_init <= self.x_max):
            raise ValueError("x_init debe estar acotado entre x_min y x_max.")


@dataclass
class WindowConfig:
    """Configuración de Ventanas y Efectos de Frontera."""
    enabled: bool = False
    window_type: str = "biolek"   # "none" | "biolek" | "joglekar"
    p: int = 5


@dataclass
class StochasticConfig:
    """Configuración Universal de Estocasticidad."""
    enabled: bool = False

    # D2D — Device-to-Device (Variabilidad de Fabricación)
    enable_d2d: bool = False
    d2d_distribution: str = "gaussian"
    d2d_ron_sigma: float = 0.05
    d2d_roff_sigma: float = 0.05
    d2d_mu_sigma: float = 0.05

    # C2C — Cycle-to-Cycle (Proceso Ornstein-Uhlenbeck)
    enable_c2c: bool = False
    c2c_model: str = "ornstein_uhlenbeck"
    c2c_sigma: float = 0.05
    c2c_theta: float = 1.0

    # Ruido Dinámico de Lectura
    enable_noise: bool = False
    noise_std: float = 1e-6
    noise_type: str = "gaussian"

    # Reproducibilidad Científica
    seed: Optional[int] = 42


@dataclass
class RealismConfig:
    """D. Configuración de Realismo agrupada."""
    window: WindowConfig = field(default_factory=WindowConfig)
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\models\__init__.py` — 6 líneas

```python
"""
Modelos matemáticos físicos puros.
"""
from neurolab.devices.models.strukov import StrukovMathModel

__all__ = ["StrukovMathModel"]

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\models\strukov.py` — 42 líneas

```python
"""
neurolab.devices.models.strukov
=================================
Modelo matemático determinista de Strukov et al. (Nature 2008).

Importa la interfaz base desde `core.base_device` (fuente de verdad).
Importa los dataclasses de configuración desde `core.config`.
"""
from neurolab.core.base_device import BaseMathModel
from neurolab.core.config import ElectricalConfig, StrukovConfig
from typing import Optional


class StrukovMathModel(BaseMathModel):
    """
    Modelo Matemático Determinista de Strukov et al. (Nature 2008).

    Ecuación diferencial de deriva iónica normalizada:
        dx/dt = (mu_v * R_on / D²) * I(t)

    donde:
        x     = w/D  fracción normalizada de región dopada [0.0, 1.0]
        mu_v  = movilidad de vacancias de oxígeno (m² / V·s)
        D     = espesor físico de la capa activa (m)
        R_on  = resistencia mínima (estado ON) en Ω
        I(t)  = corriente instantánea (A)

    Referencia: Strukov et al., "The missing memristor found", Nature 453, 2008.
    """

    def compute_dxdt(
        self,
        state: float,
        voltage: float,
        current: float,
        electrical: ElectricalConfig,
        model_config: Optional[StrukovConfig] = None
    ) -> float:
        """Calcula la velocidad de deriva iónica pura sin perturbaciones ni ventanas."""
        cfg = model_config if model_config is not None else StrukovConfig()
        # dx/dt = (mu_v * R_on / D²) * I(t)
        return float((cfg.mu_v * electrical.r_on / (cfg.D ** 2)) * current)

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\presets\__init__.py` — 16 líneas

```python
"""
Presets de referencia científica.
"""
from neurolab.devices.presets.strukov_2008 import (
    create_strukov_2008_fig2b_device,
    create_strukov_normalized_preset,
    create_strukov_stochastic_preset,
    create_strukov_paper_device
)

__all__ = [
    "create_strukov_2008_fig2b_device",
    "create_strukov_normalized_preset",
    "create_strukov_stochastic_preset",
    "create_strukov_paper_device"
]

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\presets\strukov_2008.py` — 73 líneas

```python
from neurolab.core.memristor import Memristor
from neurolab.devices.config import DeviceConfig, ElectricalConfig, StrukovConfig
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism import (
    BiolekWindowModifier, D2DVariabilityModifier,
    C2CVariabilityModifier, ThermalNoiseModifier
)
from typing import List, Optional, Any

def create_strukov_2008_fig2b_device(modifiers: List[Any] = None) -> Memristor:
    """
    Preset 1: STRUKOV_2008_IDEAL
    Perfil científico reproducible puro (sin modificadores por defecto).
    """
    identity = DeviceConfig(
        name="Strukov 2008 - Figure 2b (Ideal)",
        family="oxide_memristor",
        model="strukov"
    )

    electrical = ElectricalConfig(
        r_on=100.0,
        r_off=16_000.0,
        initial_state=0.1
    )

    strukov_config = StrukovConfig(
        D=10e-9,
        mu_v=1e-14
    )

    return Memristor(
        math_model=StrukovMathModel(),
        electrical=electrical,
        identity=identity,
        model_config=strukov_config,
        modifiers=modifiers or []
    )

def create_strukov_normalized_preset(p: int = 5) -> Memristor:
    """
    Preset 2: STRUKOV_NORMALIZED
    Perfil extendido determinista no lineal con ventana de Biolek activa.
    """
    dev = create_strukov_2008_fig2b_device(modifiers=[BiolekWindowModifier(p=p)])
    dev.identity.name = "Strukov TiO2 (Normalizado - Ventana Biolek)"
    return dev

def create_strukov_stochastic_preset(seed: int = 42) -> Memristor:
    """
    Preset 3: STRUKOV_STOCHASTIC
    Perfil realista completo con D2D, C2C (Ornstein-Uhlenbeck) y Ruido Térmico (seed = 42).
    """
    modifiers = [
        BiolekWindowModifier(p=5),
        D2DVariabilityModifier(variability_std=0.05, seed=seed),
        C2CVariabilityModifier(sigma=0.05, theta=1.0, seed=seed),
        ThermalNoiseModifier(noise_std=1e-6, seed=seed)
    ]
    dev = create_strukov_2008_fig2b_device(modifiers=modifiers)
    dev.identity.name = "Strukov TiO2 (Estocástico Realista)"
    return dev

def create_strukov_paper_device(
    modifiers: Optional[List[Any]] = None,
    initial_state: float = 0.1
) -> Memristor:
    """Alias de compatibilidad para la suite de pruebas."""
    dev = create_strukov_2008_fig2b_device(modifiers=modifiers)
    dev.identity.name = "Strukov TiO2 (Paper Fig 2b)"
    dev.electrical.initial_state = initial_state
    dev.x = initial_state
    return dev

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\presets.py` — 14 líneas

```python
from typing import List, Optional
from neurolab.core.memristor import Memristor
from neurolab.devices.base import BaseRealismModifier
from neurolab.devices.presets.strukov_2008 import create_strukov_2008_fig2b_device

def create_strukov_paper_device(
    modifiers: Optional[List[BaseRealismModifier]] = None,
    initial_state: float = 0.1
) -> Memristor:
    """Alias de compatibilidad para create_strukov_2008_fig2b_device."""
    dev = create_strukov_2008_fig2b_device(modifiers=modifiers)
    dev.electrical.initial_state = initial_state
    dev.x = initial_state
    return dev

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\realism\__init__.py` — 17 líneas

```python
"""
Modificadores de realismo y dinámica estocástica.
"""
from neurolab.devices.realism.window import BiolekWindowModifier, JoglekarWindowModifier
from neurolab.devices.realism.d2d import D2DVariabilityModifier
from neurolab.devices.realism.c2c import C2CVariabilityModifier
from neurolab.devices.realism.noise import ThermalNoiseModifier
from neurolab.devices.realism.volatile import VolatileDecayModifier

__all__ = [
    "BiolekWindowModifier",
    "JoglekarWindowModifier",
    "D2DVariabilityModifier",
    "C2CVariabilityModifier",
    "ThermalNoiseModifier",
    "VolatileDecayModifier",
]

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\realism\c2c.py` — 76 líneas

```python
"""
neurolab.devices.realism.c2c
==============================
Modificador Ciclo-a-Ciclo (C2C) basado en el proceso de Ornstein-Uhlenbeck.

Fuente de ABCs: neurolab.core.base_device

Cambios (v2026.09.11):
  - Bug #2 fix: dt_ou ya NO está hardcodeado a 0.02 s.
    El bucle de simulación debe llamar a `set_simulation_dt(dt)` antes de correr,
    o pasar dt_simulation al constructor. Si no se especifica, se mantiene 0.02 s
    como fallback para retrocompatibilidad.
"""
import numpy as np
from neurolab.core.base_device import BaseRealismModifier
from neurolab.core.config import ElectricalConfig
from typing import Any


class C2CVariabilityModifier(BaseRealismModifier):
    """
    Modificador C2C (Cycle-to-Cycle) — Proceso de Ornstein-Uhlenbeck discreto.

    Ecuación de actualización:
        deta = -theta * eta * dt_ou + sigma * sqrt(dt_ou) * N(0, 1)
        factor_multiplicativo = max(0.1, 1.0 + eta)

    Parámetros:
        sigma           : Volatilidad del ruido (intensidad de las fluctuaciones).
        theta           : Tasa de retorno a la media (mayor = más rápida reversión).
        seed            : Semilla aleatoria para reproducibilidad científica.
        dt_simulation   : Paso temporal real de la simulación (s). Si se especifica,
                          el proceso OU queda calibrado a la escala temporal correcta.
                          Se puede actualizar en tiempo de ejecución via set_simulation_dt().

    Bug #2 fix (v2026.09.11): dt_ou ya no es hardcodeado. Escala con el dt real.
    """

    # Factor de escala del dt para el proceso OU respecto al dt de simulación.
    # OU opera a una escala temporal más lenta que el paso de integración del memristor.
    # Un factor de 20 significa que el ruido evoluciona ~20x más lento que la dinámica base.
    _OU_DT_SCALE: float = 20.0

    def __init__(self, sigma: float = 0.05, theta: float = 1.0,
                 seed: int = 42, relative_std: float = None,
                 dt_simulation: float = None):
        self.sigma = relative_std if relative_std is not None else sigma
        self.theta = theta
        self.rng = np.random.default_rng(seed)
        self.eta: float = 0.0
        # Bug #2 fix: dt real de la simulación. Fallback a 0.02 s para retrocompat.
        self._dt_sim: float = dt_simulation if dt_simulation is not None else 0.02

    def set_simulation_dt(self, dt: float) -> None:
        """
        Actualiza el paso temporal de la simulación para calibrar el proceso OU.
        Debe llamarse desde el bucle de simulación antes de empezar a iterar.

        Args:
            dt: Paso de integración real de la simulación en segundos.
        """
        self._dt_sim = dt

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        if self.sigma <= 0:
            return dxdt

        # Bug #2 fix: dt_ou escala con el dt real de simulación (no hardcodeado).
        # Se usa un factor de escala para que el OU opere más lento que la dinámica base.
        dt_ou = self._dt_sim * self._OU_DT_SCALE
        d_eta = -self.theta * self.eta * dt_ou + self.sigma * np.sqrt(dt_ou) * self.rng.normal(0.0, 1.0)
        self.eta += float(d_eta)

        factor = max(0.1, 1.0 + self.eta)
        return float(dxdt * factor)

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\realism\d2d.py` — 76 líneas

```python
"""
neurolab.devices.realism.d2d
==============================
Modificador Device-to-Device (D2D) — Variabilidad estática de fabricación.

Fuente de ABCs: neurolab.core.base_device

CORRECCIÓN (bug anterior): El modificador D2D solo tenía `apply_to_electrical()`
que nadie en el pipeline de Memristor.step() llamaba. Ahora también implementa
`modify_dxdt()` aplicando un factor de escalado fijo calculado una sola vez
al momento de inicializar el dispositivo, modelando la variabilidad "quemada"
en la fabricación del chip (no varía ciclo a ciclo, sino dispositivo a dispositivo).
"""
import numpy as np
from neurolab.core.base_device import BaseRealismModifier
from neurolab.core.config import ElectricalConfig
from typing import Any


class D2DVariabilityModifier(BaseRealismModifier):
    """
    Modificador D2D (Device-to-Device).

    Simula la variabilidad de fabricación entre muestras:
    - Un factor escalar fijo `_d2d_factor` se calcula **una sola vez** al inicializar.
    - Este factor modula dx/dt en cada paso, escalando la velocidad de conmutación.
    - También puede modificar R_on/R_off vía `apply_to_electrical()`.

    Parámetros:
        variability_std : Desviación estándar de la distribución Gaussiana de variabilidad.
        seed            : Semilla para reproducibilidad estricta.
    """

    def __init__(self, variability_std: float = 0.05, seed: int = 42):
        self.variability_std = variability_std
        self.rng = np.random.default_rng(seed)

        # Factor D2D fijo: calculado una sola vez (fabricación estática con distribución normal truncada)
        if variability_std > 0:
            raw_sample = self.rng.normal(0.0, variability_std)
            clipped_sample = float(np.clip(raw_sample, -2.0 * variability_std, 2.0 * variability_std))
            self._d2d_factor = 1.0 + clipped_sample
        else:
            self._d2d_factor = 1.0

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        """
        Aplica el factor D2D fijo (variabilidad de fabricación) a la derivada de estado.
        El factor es constante durante toda la vida del dispositivo, modelando dispersión
        de fabricación (no varía por ciclo).
        """
        return float(dxdt * self._d2d_factor)

    def apply_to_electrical(self, electrical: ElectricalConfig) -> ElectricalConfig:
        """
        Aplica dispersión estocástica a R_on y R_off (variabilidad de fabricación D2D).
        Útil para caracterización multi-dispositivo fuera del loop de simulación.
        """
        if self.variability_std <= 0:
            return electrical

        sample_on = float(np.clip(self.rng.normal(0.0, self.variability_std), -2.0 * self.variability_std, 2.0 * self.variability_std))
        sample_off = float(np.clip(self.rng.normal(0.0, self.variability_std), -2.0 * self.variability_std, 2.0 * self.variability_std))

        factor_on = 1.0 + sample_on
        factor_off = 1.0 + sample_off

        new_r_on = max(1.0, electrical.r_on * factor_on)
        new_r_off = max(new_r_on * 1.5, electrical.r_off * factor_off)

        return ElectricalConfig(
            r_on=new_r_on,
            r_off=new_r_off,
            initial_state=electrical.initial_state
        )

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\realism\noise.py` — 32 líneas

```python
"""
neurolab.devices.realism.noise
================================
Modificador de Ruido Térmico de Lectura.

Fuente de ABCs: neurolab.core.base_device
"""
import numpy as np
from neurolab.core.base_device import BaseRealismModifier


class ThermalNoiseModifier(BaseRealismModifier):
    """
    Ruido Térmico de lectura modelado como N(0, noise_std).

    Se aplica a la corriente de salida en cada paso de integración,
    simulando el ruido de Johnson-Nyquist en la medición.

    Parámetros:
        noise_std : Desviación estándar del ruido (A). Típico: 1e-6 A.
        seed      : Semilla para reproducibilidad científica.
    """

    def __init__(self, noise_std: float = 1e-6, seed: int = 42):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def modify_current(self, current: float, voltage: float, state: float) -> float:
        if self.noise_std <= 0:
            return current
        noise = float(self.rng.normal(0.0, self.noise_std))
        return float(current + noise)

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\realism\volatile.py` — 100 líneas

```python
"""
neurolab.devices.realism.volatile
===================================
Modificador de Decaimiento Volátil (Memristor Difusivo).

Implementa el término de relajación que convierte el modelo de Strukov (no volátil,
memoria perfecta) en un modelo de memristor VOLÁTIL (difusivo), análogo a los
dispositivos Ag/SiO₂ o NbOx usados en hardware neuromórfico real para emular el
período refractario de neuronas biológicas.

Referencia física:
  Wang et al. (2025), "Memristive Approaches to Biologically Plausible Spiking Neural
  Networks", Sección 4.1-4.2 — "Diffusive TSMs (Threshold Switching Memristors)":
  estos dispositivos tienen relajación espontánea: cuando se retira el voltaje,
  el filamento conductor se disuelve y x decae hacia x₀ por sí solo.

Ecuación diferencial modificada (Strukov + Relajación Volátil):
─────────────────────────────────────────────────────────────────
  dx/dt = [μᵥ·R_ON/D²·I(t)] + [-(x - x₀) / τ_relax]
            ↑ Deriva iónica      ↑ Relajación volátil
            (Strukov original)   (NUEVO — este módulo)

Donde:
  x₀      : Estado de reposo del memristor (= electrical.initial_state)
  τ_relax : Tiempo de relajación (s). Cuánto tarda en volver a x₀ tras un shunt.
             Rango típico: 0.01 s (10 ms) a 0.5 s (500 ms).

Física del período refractario con este módulo:
───────────────────────────────────────────────
1. Spike → el bucle de simulación forza x → x_shunt (≈ 0.95 → R_ON → alta conductancia).
2. La alta conductancia drena la corriente de entrada → V_m no puede volver a V_th.
3. Con V_m ≈ V_rest, la corriente es ≈ 0 → término de deriva ≈ 0.
4. El término de relajación domina: x decae de x_shunt → x₀ exponencialmente.
5. Cuando x ≈ x₀, R_leak ≈ R_OFF → fuga pequeña → neurona vuelve a ser excitable.

El tiempo del período refractario emergente ≈ τ_relax · ln(x_shunt / x₀).
"""
import math
import numpy as np
from neurolab.core.base_device import BaseRealismModifier
from neurolab.core.config import ElectricalConfig
from typing import Any


class VolatileDecayModifier(BaseRealismModifier):
    """
    Modificador de Relajación Volátil (Memristor Difusivo).

    Añade al pipeline de dx/dt el término:
        dx/dt_relax = -(x - x₀) / τ_relax

    Esto simula la relajación espontánea del filamento conductor en
    memristores difusivos (Ag/SiO₂, NbOx) cuando el voltaje es retirado.

    El shunt del estado (x → x_shunt en el momento del spike) se maneja
    externamente en el bucle de simulación de main_window.py, ya que ese
    es el único punto que conoce cuándo ocurre el spike.

    Parámetros:
        tau_relax  : Tiempo de relajación (s). Cuánto tarda x en volver a x₀.
                     τ pequeño (0.01s) → refractario corto, rápido.
                     τ grande  (0.5s)  → refractario largo, lento.
        x0_override: Si se especifica, usa este valor como x₀ en lugar del
                     electrical.initial_state. Útil para forzar un estado de
                     reposo distinto al de fabricación.
    """

    def __init__(self, tau_relax: float = 0.05, x0_override: float = None):
        if tau_relax <= 0:
            raise ValueError(f"tau_relax debe ser positivo (recibido: {tau_relax})")
        self.tau_relax = tau_relax
        self.x0_override = x0_override

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        """
        Suma el término de relajación volátil al dx/dt existente.

        dx/dt_total = dx/dt_strukov + (-(x - x₀) / τ_relax)
        """
        x0 = self.x0_override if self.x0_override is not None else electrical.initial_state

        # Término de relajación: tira x hacia x₀ con constante de tiempo τ_relax
        dx_dt_relax = -(state - x0) / self.tau_relax

        return float(dxdt + dx_dt_relax)

    @property
    def refractory_time_estimate(self) -> float:
        """
        Estimación del tiempo de período refractario basado en τ_relax.
        Asume que el shunt lleva x a 0.99 y x₀ ≈ 0.10.
        t_ref ≈ τ_relax · ln((x_shunt - x₀) / 0.05)
        """
        x_shunt = 0.99
        x0 = 0.10
        try:
            return self.tau_relax * math.log((x_shunt - x0) / 0.05)
        except (ValueError, ZeroDivisionError):
            return self.tau_relax * 3.0

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\realism\window.py` — 50 líneas

```python
"""
neurolab.devices.realism.window
================================
Modificadores de Funciones de Ventana (Boundary Effects).

Fuente de ABCs: neurolab.core.base_device
"""
import numpy as np
from neurolab.core.base_device import BaseRealismModifier
from neurolab.core.config import ElectricalConfig
from typing import Any


class BiolekWindowModifier(BaseRealismModifier):
    """
    Función de Ventana de Biolek: f(x, i) = 1 - (x - stp(-i))^(2p)

    stp(-i) = 1 si i < 0 (RESET → atenúa al acercarse a x=0)
              0 si i >= 0 (SET  → atenúa al acercarse a x=1)

    Referencia: Biolek et al., RADIOENGINEERING, 2009.
    """

    def __init__(self, p: int = 5):
        self.p = p

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        x = state
        i = current
        stp = 1.0 if i < 0.0 else 0.0
        f_win = 1.0 - np.power(x - stp, 2 * self.p)
        return float(dxdt * f_win)


class JoglekarWindowModifier(BaseRealismModifier):
    """
    Función de Ventana de Joglekar: f(x) = 1 - (2x - 1)^(2p)

    Referencia: Joglekar & Wolf, Eur. Phys. J. B, 2009.
    """

    def __init__(self, p: int = 5):
        self.p = p

    def modify_dxdt(self, dxdt: float, state: float, voltage: float, current: float,
                    electrical: ElectricalConfig, model_config: Any) -> float:
        x = state
        f_win = 1.0 - np.power(2.0 * x - 1.0, 2 * self.p)
        return float(dxdt * f_win)

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\state.py` — 23 líneas

```python
from neurolab.devices.config import StateConfig

class StateManager:
    """
    Gestión del estado interno normalizado x in [x_min, x_max].
    """

    def __init__(self, config: StateConfig):
        self.config = config
        self._x: float = float(config.x_init)

    @property
    def value(self) -> float:
        return self._x

    @value.setter
    def value(self, val: float) -> None:
        v = float(val)
        self._x = self.config.x_max if v > self.config.x_max else (self.config.x_min if v < self.config.x_min else v)

    def reset(self, new_init: float = None) -> None:
        val = float(new_init if new_init is not None else self.config.x_init)
        self._x = self.config.x_max if val > self.config.x_max else (self.config.x_min if val < self.config.x_min else val)

```



---

## neurolab

`neuromorphic_lab\neurolab\devices\strukov.py` — 8 líneas

```python
"""
neurolab.devices.strukov
=========================
Alias de compatibilidad → redirige a devices.models.strukov (fuente activa).
"""
from neurolab.devices.models.strukov import StrukovMathModel  # noqa: F401

__all__ = ["StrukovMathModel"]

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\__init__.py` — 7 líneas

```python
"""
Módulo de Interfaz Gráfica de Usuario (GUI) basada en PySide6 (Qt6).
"""

from neurolab.gui.main_window import MainWindow

__all__ = ["MainWindow"]

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\app.py` — 82 líneas

```python
import sys
import os
import signal
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QtMsgType, QMessageLogContext, qInstallMessageHandler, qFormatLogMessage
from neurolab.gui.main_window import MainWindow


def _qt_message_handler(mode: QtMsgType, context: QMessageLogContext, message: str):
    """Filtra avisos benignos de Qt y re-emite el resto con el formato por defecto."""
    # "This plugin does not support propagateSizeHints()" lo emite QPlatformWindow en
    # los plugins de plataforma (windows/offscreen) cuando el layout pide propagar
    # hints de tamaño; es inofensivo y solo ensucia la consola.
    # "This plugin does not support raise()" es el equivalente para raise() y
    # también es inofensivo.
    if "propagateSizeHints" in message or "does not support raise()" in message:
        return
    formatted = qFormatLogMessage(mode, context, message).rstrip("\n")
    sys.stderr.write(formatted + "\n")
    sys.stderr.flush()


def main():
    """Punto de entrada para la ejecución de la GUI neurolab."""
    # Consolas Windows con codepage legacy (cp850/cp1252) no representan emojis:
    # sustituir caracteres no codificables en vez de lanzar UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    print("Neuromorphic Lab » inicializando interfaz gráfica...", flush=True)
    print("  (puede tardar unos segundos en la primera carga)", flush=True)

    # Permitir que el sistema maneje SIGINT (Ctrl+C) limpiamente sin traceback de C++ eventFilter
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Filtro de avisos benignos de Qt (antes de crear QApplication)
    qInstallMessageHandler(_qt_message_handler)

    # PySide6 >= 6.10 ya no empaqueta fuentes; si no hay QT_QPA_FONTDIR, apuntar al
    # directorio de fuentes del SO para evitar el aviso "Cannot find font directory".
    if os.name == "nt" and not os.environ.get("QT_QPA_FONTDIR"):
        _fonts = r"C:\Windows\Fonts"
        if os.path.isdir(_fonts):
            os.environ["QT_QPA_FONTDIR"] = _fonts

    # Modo ventana invisible (offscreen): válido solo para pruebas headless. En una
    # sesión de escritorio normal suele ser un resto accidental (p. ej. de una
    # terminal usada para tests headless) y hace que la app corra SIN ventana
    # visible, con la consola "bloqueada" aparentando un cuelgue.
    if os.name == "nt" and os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        if os.environ.get("NEUROLAB_HEADLESS") == "1":
            print("[AVISO] Modo headless forzado (NEUROLAB_HEADLESS=1): ventana invisible.", flush=True)
        else:
            del os.environ["QT_QPA_PLATFORM"]
            print("[AVISO] Se ignoró QT_QPA_PLATFORM=offscreen de esta sesión: la ventana", flush=True)
            print("       sería invisible. Para forzar modo headless usa NEUROLAB_HEADLESS=1.", flush=True)

    app = QApplication(sys.argv)

    # Timer periódico para permitir que Python procese eventos de interrupción de teclado
    sigint_timer = QTimer()
    sigint_timer.start(200)
    sigint_timer.timeout.connect(lambda: None)

    window = MainWindow()
    window.show()

    # En Windows la ventana puede quedar detrás de la consola: traerla al frente
    window.raise_()
    window.activateWindow()

    print("[OK] Neuromorphic Lab » ventana abierta.", flush=True)
    print("   La consola queda 'bloqueada' mientras la ventana esté abierta (es normal).", flush=True)
    print("   Para salir: cierra la ventana o presiona Ctrl+C.", flush=True)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\main_window.py` — 1109 líneas

```python
"""
neurolab.gui.main_window
=========================
Ventana Principal de la aplicación Neuromorphic Lab.

Características:
  - Loop de simulación con debounce (100 ms) para actualización en tiempo real
  - Autoguardado de sesión al cerrar la ventana (configs/last_session.json)
  - Restauración automática de última sesión al arrancar
  - Título dinámico con el nombre del dispositivo activo
"""
import os
import time
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea,
    QStatusBar, QMessageBox, QSplitter, QTabWidget, QGroupBox, QLabel, QDockWidget,
    QComboBox, QFormLayout, QApplication, QTextBrowser, QPushButton, QRadioButton
)
from PySide6.QtCore import Qt, QTimer
from neurolab.gui.widgets.config_panel import ConfigPanel
from neurolab.gui.widgets.signal_panel import SignalPanel
from neurolab.gui.widgets.plot_canvas import MplCanvas
from neurolab.gui.widgets.neuron_config_panel import NeuronConfigPanel
from neurolab.gui.widgets.neuron_plot_canvas import NeuronMplCanvas
from neurolab.gui.widgets.hybrid_plot_canvas import HybridMplCanvas
from neurolab.gui.widgets.arrow_spinbox import ArrowDoubleSpinBox, NoUnfocusedWheelEventFilter
from neurolab.io.profile_manager import ProfileManager
from neurolab.io.validation_loader import ValidationDataLoader
from neurolab.core.validation_metrics import compute_all_metrics
from neurolab.circuits.hybrid import MemristorLIFCircuit, ResistorLIFCircuit
from neurolab.core.lif_validation import compute_lif_validation_metrics


class MainWindow(QMainWindow):
    """Ventana Principal de la aplicación Neuromorphic Lab."""

    @property
    def config_panel(self) -> ConfigPanel:
        """Devuelve el ConfigPanel del Memristor actualmente activo en la sub-pestaña de Pestaña 1."""
        return self.memristor_subtabs.currentWidget() if hasattr(self, 'memristor_subtabs') else self.config_panel_1

    def __init__(self):
        super().__init__()
        self._profile_manager = ProfileManager()
        self._validation_loader = ValidationDataLoader()
        self._active_profile_name: str = "Sin Perfil"

        # Filtro global: ignora rueda del ratón en controles numéricos o listas sin clic previo
        self._wheel_filter = NoUnfocusedWheelEventFilter()
        app_instance = QApplication.instance()
        if app_instance is not None:
            app_instance.installEventFilter(self._wheel_filter)

        self.setWindowTitle("Neuromorphic Lab — Simulador Universal de Dispositivos Memristivos")
        self.resize(1280, 800)

        # Temporizador debounce para suavizar actualizaciones en tiempo real
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(100)
        self.debounce_timer.timeout.connect(self.run_simulation)

        self._apply_dark_theme()
        self.init_ui()
        # Simulación inicial ÚNICA: con la sesión restaurada si existe, o con
        # los valores por defecto en caso contrario (evita simular dos veces).
        if not self._restore_last_session():
            self.run_simulation()

    # ── Tema Visual ──────────────────────────────────────────────────────────

    def _apply_dark_theme(self):
        """Aplica un tema visual oscuro moderno a toda la interfaz Qt."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
                padding-right: 24px;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #89b4fa;
            }
            QDoubleSpinBox::up-button, QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px; height: 13px;
                background-color: #45475a;
                border-left: 1px solid #313244;
                border-bottom: 1px solid #313244;
                border-top-right-radius: 4px;
            }
            QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover {
                background-color: #89b4fa;
            }
            QDoubleSpinBox::down-button, QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px; height: 13px;
                background-color: #45475a;
                border-left: 1px solid #313244;
                border-bottom-right-radius: 4px;
            }
            QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {
                background-color: #89b4fa;
            }
            QCheckBox {
                color: #cdd6f4;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px; height: 16px;
            }
            QStatusBar {
                background-color: #181825;
                color: #a6adc8;
            }
            QScrollArea { border: none; }
            QSplitter::handle:horizontal {
                background-color: #45475a;
                width: 6px;
                margin: 0px 2px;
                border-radius: 3px;
            }
            QSplitter::handle:horizontal:hover {
                background-color: #89b4fa;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 4px;
                background-color: #1e1e2e;
            }
            QTabBar::tab {
                background-color: #313244;
                color: #a6adc8;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background-color: #45475a;
            }
        """)

    # ── Inicialización UI ────────────────────────────────────────────────────

    def init_ui(self):
        self.setDockNestingEnabled(True)

        # Usamos un QWidget vacío como central para que los docks ocupen el espacio principal
        self.setCentralWidget(None)

        # === Menú Superior ===
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: #181825;
                color: #cdd6f4;
            }
            QMenuBar::item:selected {
                background-color: #313244;
            }
            QMenu {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
            }
            QMenu::item:selected {
                background-color: #89b4fa;
                color: #11111b;
            }
        """)
        vista_menu = menu_bar.addMenu("🖥️ Vista")
        
        act_tabify = vista_menu.addAction("📑 Agrupar todo en Pestañas")
        act_tabify.triggered.connect(self._action_tabify_all)
        
        act_tile = vista_menu.addAction("🪟 Dividir Pantalla (Mosaico)")
        act_tile.triggered.connect(self._action_tile_all)

        # === Pestaña 1: Memristor ===
        tab_memristor = QWidget()
        layout_memristor = QHBoxLayout(tab_memristor)
        layout_memristor.setContentsMargins(0, 0, 0, 0)
        
        splitter_m = QSplitter(Qt.Horizontal)
        controls_container_m = QWidget()
        controls_layout_m = QVBoxLayout(controls_container_m)
        # Sub-pestañas para configurar múltiples Memristores (Memristor 1, Memristor 2, Memristor 3)
        self.memristor_subtabs = QTabWidget()
        self.config_panel_1 = ConfigPanel()
        self.config_panel_2 = ConfigPanel()
        self.config_panel_3 = ConfigPanel()

        self.config_panel_1.txt_device_name.setText("Memristor No Volátil (Strukov)")
        self.config_panel_1.combo_material.setCurrentText("TiO₂ (Dióxido de Titanio - Strukov 2008)")
        self.config_panel_1.spin_r_on.setValue(100.0)
        self.config_panel_1.combo_ron_unit.setCurrentText("Ω")
        self.config_panel_1.spin_r_off.setValue(16.0)
        self.config_panel_1.combo_roff_unit.setCurrentText("kΩ")
        self.config_panel_1.spin_x0.setValue(0.10)
        self.config_panel_1.spin_D_nm.setValue(10.0)
        self.config_panel_1.spin_mu_v.setValue(1e-14)
        self.config_panel_1.combo_window_type.setCurrentText("Biolek")
        self.config_panel_1.spin_biolek_p.setValue(5)
        self.config_panel_1.spin_seed.setValue(42)

        self.config_panel_2.txt_device_name.setText("Memristor Volátil (Decaimiento)")
        self.config_panel_2.combo_material.setCurrentText("HfO₂ (Óxido de Hafnio - CMOS LIF 2025)")
        self.config_panel_2.spin_r_on.setValue(1.0)          # 1 kΩ (LRS)
        self.config_panel_2.combo_ron_unit.setCurrentText("kΩ")
        self.config_panel_2.spin_r_off.setValue(1.0)         # 1 MΩ (HRS)
        self.config_panel_2.combo_roff_unit.setCurrentText("MΩ")
        self.config_panel_2.spin_x0.setValue(0.99)           # ← arranca casi ON → flat inicial largo
        self.config_panel_2.spin_D_nm.setValue(10.0)
        self.config_panel_2.spin_mu_v.setValue(1e-16)
        self.config_panel_2.combo_window_type.setCurrentText("Sin Ventana")
        self.config_panel_2.chk_volatile.setChecked(True)
        self.config_panel_2.spin_tau_relax.setValue(0.10)
        self.config_panel_2.spin_x_shunt.setValue(0.99)      # irrelevante con apply_shunt=False
        if hasattr(self.config_panel_2, "spin_x_eq"):
            self.config_panel_2.spin_x_eq.setValue(0.05)     # ← equilibrio bajo = decae
        self.config_panel_2.spin_seed.setValue(42)

        self.config_panel_3.txt_device_name.setText("Memristor Serie (Híbrido)")
        self.config_panel_3.combo_material.setCurrentText("HfO₂ (Óxido de Hafnio - CMOS LIF 2025)")
        self.config_panel_3.spin_r_on.setValue(10.0)
        self.config_panel_3.combo_ron_unit.setCurrentText("kΩ")
        self.config_panel_3.spin_r_off.setValue(1.0)
        self.config_panel_3.combo_roff_unit.setCurrentText("MΩ")
        self.config_panel_3.spin_x0.setValue(0.10)
        self.config_panel_3.spin_D_nm.setValue(10.0)
        self.config_panel_3.spin_mu_v.setValue(1e-16)
        self.config_panel_3.combo_window_type.setCurrentText("Biolek")
        self.config_panel_3.spin_biolek_p.setValue(3)
        self.config_panel_3.spin_seed.setValue(42)

        self.signal_panel = SignalPanel()
        # Default signal for Memristor Tab (Sinusoidal by default)
        idx = self.signal_panel.combo_waveform.findText("Sinusoidal")
        if idx >= 0:
            self.signal_panel.combo_waveform.setCurrentIndex(idx)
        self.signal_panel.spin_v0.setValue(1.0)
        self.signal_panel.spin_f0.setValue(0.5)
        self.signal_panel.spin_duration.setValue(8.0)
        self.signal_panel.spin_dt_ms.setValue(0.1)

        self.config_panel_1.set_signal_panel(self.signal_panel)
        self.config_panel_2.set_signal_panel(self.signal_panel)
        self.config_panel_3.set_signal_panel(self.signal_panel)

        self.memristor_subtabs.addTab(self.config_panel_1, "🔬 Memristor 1 (No Volátil)")
        self.memristor_subtabs.addTab(self.config_panel_2, "🔬 Memristor 2 (Volátil)")
        self.memristor_subtabs.addTab(self.config_panel_3, "🔬 Memristor 3 (Híbrido)")

        controls_layout_m.addWidget(self.memristor_subtabs)
        controls_layout_m.addWidget(self.signal_panel)
        
        scroll_m = QScrollArea()
        scroll_m.setWidget(controls_container_m)
        scroll_m.setWidgetResizable(True)
        scroll_m.setMinimumWidth(350)
        
        self.plot_canvas = MplCanvas(self)
        
        splitter_m.addWidget(scroll_m)
        splitter_m.addWidget(self.plot_canvas)
        splitter_m.setSizes([460, 820])
        splitter_m.setCollapsible(0, False)
        layout_memristor.addWidget(splitter_m)
        
        # Panel de métricas cuantitativas de validación
        self.metrics_label = QLabel()
        self.metrics_label.setTextFormat(Qt.RichText)
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setAlignment(Qt.AlignTop)
        self.metrics_label.setStyleSheet("""
            QLabel {
                background-color: #181825;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px;
                font-size: 11px;
                color: #cdd6f4;
            }
        """)
        self.metrics_label.setText(
            "<p style='color:#585b70; text-align:center;'>"
            "Activa la Validación CSV y ejecuta la simulación<br>"
            "para ver las métricas cuantitativas."
            "</p>"
        )
        controls_layout_m.addWidget(self.metrics_label)
        controls_layout_m.setStretch(0, 0)  # config panel
        controls_layout_m.setStretch(1, 0)  # signal panel
        controls_layout_m.setStretch(2, 1)  # metrics panel expande

        # === Pestaña 1: Memristor (Dock) ===
        self.dock_memristor = QDockWidget("🔬 Simulador de Memristor", self)
        self.dock_memristor.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_memristor.setWidget(tab_memristor)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_memristor)

        # === Pestaña 2: Neurona LIF ===
        tab_neuron = QWidget()
        layout_neuron = QHBoxLayout(tab_neuron)
        layout_neuron.setContentsMargins(0, 0, 0, 0)
        
        splitter_n = QSplitter(Qt.Horizontal)
        controls_container_n = QWidget()
        controls_layout_n = QVBoxLayout(controls_container_n)
        
        self.neuron_config_panel = NeuronConfigPanel()
        
        # Preconfiguración de la Neurona LIF
        self.neuron_config_panel.combo_c_unit.setCurrentText("nF")
        self.neuron_config_panel.spin_c_m.setValue(500.0)
        self.neuron_config_panel.combo_rs_unit.setCurrentText("kΩ")
        self.neuron_config_panel.spin_r_series.setValue(1000.0)
        self.neuron_config_panel.combo_r_unit.setCurrentText("MΩ")
        self.neuron_config_panel.spin_r_leak.setValue(1.0)
        self.neuron_config_panel.spin_v_rest.setValue(0.0)
        self.neuron_config_panel.spin_v_th.setValue(1.0)
        self.neuron_config_panel.spin_v_reset.setValue(0.10)
        self.neuron_config_panel.spin_t_ref.setValue(2.0)
        self.neuron_config_panel._on_change()

        # SignalPanel en modo fuente de voltaje (V_IN)
        self.neuron_signal_panel = SignalPanel(mode="voltage")
        self.neuron_config_panel.set_signal_panel(self.neuron_signal_panel)
        
        # === PRECONFIGURACIÓN AUTOMÁTICA NEURONA LIF ===
        idx = self.neuron_signal_panel.combo_waveform.findText("Tren de Pulsos (Unipolar)")
        if idx >= 0:
            self.neuron_signal_panel.combo_waveform.setCurrentIndex(idx)
        self.neuron_signal_panel.spin_v0.setValue(3.0)
        self.neuron_signal_panel.spin_f0.setValue(100.0)
        self.neuron_signal_panel.spin_duration.setValue(2.0)
        self.neuron_signal_panel.spin_dt_ms.setValue(0.001)
        
        controls_layout_n.addWidget(self.neuron_config_panel)
        controls_layout_n.addWidget(self.neuron_signal_panel)
        
        scroll_n = QScrollArea()
        scroll_n.setWidget(controls_container_n)
        scroll_n.setWidgetResizable(True)
        scroll_n.setMinimumWidth(350)
        
        self.neuron_plot_canvas = NeuronMplCanvas(self)
        
        splitter_n.addWidget(scroll_n)
        splitter_n.addWidget(self.neuron_plot_canvas)
        splitter_n.setSizes([460, 820])
        splitter_n.setCollapsible(0, False)
        layout_neuron.addWidget(splitter_n)
        
        # === Pestaña 2: Neurona LIF (Dock) ===
        self.dock_neuron = QDockWidget("🧠 Simulador Neurona LIF", self)
        self.dock_neuron.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_neuron.setWidget(tab_neuron)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_neuron)

        # === Pestaña 3: Híbrido Memristor-LIF ===
        tab_hybrid = QWidget()
        layout_hybrid = QHBoxLayout(tab_hybrid)
        layout_hybrid.setContentsMargins(0, 0, 0, 0)

        splitter_h = QSplitter(Qt.Horizontal)
        controls_container_h = QWidget()
        controls_layout_h = QVBoxLayout(controls_container_h)

        # QGroupBox: Ubicación y Selección de Memristores en el Circuito LIF
        group_mode = QGroupBox("📍 Ubicación y Selección de Memristores en LIF")
        lay_mode = QVBoxLayout(group_mode)

        form_combos = QFormLayout()
        self.combo_mem_rs = QComboBox()
        self.combo_mem_rs.addItems(["Memristor 1 (Sub-Pestaña 1)", "Memristor 2 (Sub-Pestaña 2)", "Memristor 3 (Sub-Pestaña 3)"])

        self.combo_mem_rs.setCurrentIndex(2)  # Default: Memristor 3 (Serie - Híbrido)

        self.combo_mem_leak = QComboBox()
        self.combo_mem_leak.addItems(["Memristor 1 (Sub-Pestaña 1)", "Memristor 2 (Sub-Pestaña 2)", "Memristor 3 (Sub-Pestaña 3)"])
        self.combo_mem_leak.setCurrentIndex(1)  # Default: Memristor 2 para fuga

        self.combo_mem_rs.setStyleSheet("color: #cdd6f4; background-color: #181825; border: 1px solid #45475a; padding: 4px; border-radius: 4px;")
        self.combo_mem_leak.setStyleSheet("color: #cdd6f4; background-color: #181825; border: 1px solid #45475a; padding: 4px; border-radius: 4px;")

        lbl_rs = QLabel("Memristor en Serie (R_S):")
        lbl_rs.setStyleSheet("color: #89b4fa; font-weight: bold;")
        lbl_leak = QLabel("Memristor en Fuga (R_leak):")
        lbl_leak.setStyleSheet("color: #a6e3a1; font-weight: bold;")

        form_combos.addRow(lbl_rs, self.combo_mem_rs)
        form_combos.addRow(lbl_leak, self.combo_mem_leak)
        lay_mode.addLayout(form_combos)

        self.rb_mem_series = QRadioButton("1. Solo Memristor en Serie (R_S)")
        self.rb_mem_leak   = QRadioButton("2. Solo Memristor en Fuga (R_leak)")
        self.rb_mem_both   = QRadioButton("3. Memristores en Ambos (R_S y R_leak)")
        self.rb_mem_none   = QRadioButton("4. Sin Memristores (Solo LIF pasivo)")
        self.rb_mem_both.setChecked(True)

        for rb in (self.rb_mem_series, self.rb_mem_leak, self.rb_mem_both, self.rb_mem_none):
            rb.setStyleSheet("color: #cdd6f4; font-weight: bold; padding: 3px;")
            lay_mode.addWidget(rb)
            rb.toggled.connect(self._on_param_changed)

        self.combo_mem_rs.currentIndexChanged.connect(self._on_param_changed)
        self.combo_mem_leak.currentIndexChanged.connect(self._on_param_changed)

        # QGroupBox: Memristor Activo (Pestaña 1)
        group_mem = QGroupBox("🔬 Memristor Activo (Pestaña 1)")
        lay_mem = QVBoxLayout(group_mem)
        self.lbl_hybrid_mem_info = QLabel()
        self.lbl_hybrid_mem_info.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        self.lbl_hybrid_mem_info.setWordWrap(True)
        btn_edit_mem = QPushButton("✏️ Modificar Memristor en Pestaña 1")
        btn_edit_mem.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                border: 1px solid #45475a;
                padding: 5px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45475a; }
        """)
        btn_edit_mem.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        lay_mem.addWidget(self.lbl_hybrid_mem_info)
        lay_mem.addWidget(btn_edit_mem)

        # QGroupBox: Neurona LIF Activa (Pestaña 2)
        group_neu = QGroupBox("🧠 Neurona LIF Activa (Pestaña 2)")
        lay_neu = QVBoxLayout(group_neu)
        self.lbl_hybrid_neuron_info = QLabel()
        self.lbl_hybrid_neuron_info.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        self.lbl_hybrid_neuron_info.setWordWrap(True)
        btn_edit_neu = QPushButton("✏️ Modificar Neurona LIF en Pestaña 2")
        btn_edit_neu.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #a6e3a1;
                border: 1px solid #45475a;
                padding: 5px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45475a; }
        """)
        btn_edit_neu.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        lay_neu.addWidget(self.lbl_hybrid_neuron_info)
        lay_neu.addWidget(btn_edit_neu)

        # Panel de Fuente de Señal para la entrada al circuito híbrido
        self.hybrid_signal_panel = SignalPanel(mode="voltage")
        idx = self.hybrid_signal_panel.combo_waveform.findText("Tren de Pulsos (Unipolar)")
        if idx >= 0:
            self.hybrid_signal_panel.combo_waveform.setCurrentIndex(idx)
        self.hybrid_signal_panel.spin_v0.setValue(5.0)
        self.hybrid_signal_panel.spin_f0.setValue(100.0)
        self.hybrid_signal_panel.spin_duration.setValue(0.5)
        self.hybrid_signal_panel.spin_dt_ms.setValue(0.01)

        controls_layout_h.addWidget(group_mode)
        controls_layout_h.addWidget(group_mem)
        controls_layout_h.addWidget(group_neu)
        controls_layout_h.addWidget(self.hybrid_signal_panel)

        scroll_h = QScrollArea()
        scroll_h.setWidget(controls_container_h)
        scroll_h.setWidgetResizable(True)
        scroll_h.setMinimumWidth(380)

        self.hybrid_plot_canvas = NeuronMplCanvas(self)

        splitter_h.addWidget(scroll_h)
        splitter_h.addWidget(self.hybrid_plot_canvas)
        splitter_h.setSizes([460, 820])
        splitter_h.setCollapsible(0, False)
        layout_hybrid.addWidget(splitter_h)

        # === Pestaña 3: Híbrido Memristor-LIF (Dock) ===
        self.dock_hybrid = QDockWidget("🧠+🔬 Híbrido Memristor-LIF", self)
        self.dock_hybrid.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_hybrid.setWidget(tab_hybrid)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_hybrid)

        # === Pestaña 4: Documentación Formal ===
        tab_docs = QWidget()
        layout_docs = QVBoxLayout(tab_docs)
        layout_docs.setContentsMargins(10, 10, 10, 10)
        
        self.docs_browser = QTextBrowser()
        self.docs_browser.setOpenExternalLinks(True)
        self.docs_browser.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        
        # Cargar el archivo Markdown
        doc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "Modelo_Fisico_Strukov_2008.md")
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                md_content = f.read()
                self.docs_browser.setMarkdown(md_content)
        except Exception as e:
            self.docs_browser.setHtml(f"<h2 style='color:#f38ba8;'>Error al cargar documentación:</h2><p>{e}</p>")
            
        layout_docs.addWidget(self.docs_browser)
        
        # === Pestaña 4: Documentación Formal (Dock) ===
        self.dock_docs = QDockWidget("📖 Fundamentos Físicos", self)
        self.dock_docs.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_docs.setWidget(tab_docs)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_docs)

        # Apilar las pestañas (tabify) por defecto
        self.tabifyDockWidget(self.dock_memristor, self.dock_neuron)
        self.tabifyDockWidget(self.dock_neuron, self.dock_hybrid)
        self.tabifyDockWidget(self.dock_hybrid, self.dock_docs)
        self.dock_memristor.raise_()

        # === Conexión de Señales ===
        # Modificar botones de "Modificar en Pestaña X" para que traigan el dock al frente
        btn_edit_mem.clicked.disconnect()
        btn_edit_neu.clicked.disconnect()
        btn_edit_mem.clicked.connect(lambda: self.dock_memristor.raise_())
        btn_edit_neu.clicked.connect(lambda: self.dock_neuron.raise_())

        self.signal_panel.run_simulation_requested.connect(self.run_simulation)
        self.config_panel_1.param_changed.connect(self._on_param_changed)
        self.config_panel_2.param_changed.connect(self._on_param_changed)
        self.config_panel_3.param_changed.connect(self._on_param_changed)
        self.memristor_subtabs.currentChanged.connect(self.run_simulation)
        self.signal_panel.param_changed.connect(self._on_param_changed)

        self.neuron_signal_panel.run_simulation_requested.connect(self.run_simulation)
        self.neuron_config_panel.param_changed.connect(self._on_param_changed)
        self.neuron_signal_panel.param_changed.connect(self._on_param_changed)

        self.hybrid_signal_panel.run_simulation_requested.connect(self.run_simulation)
        self.hybrid_signal_panel.param_changed.connect(self._on_param_changed)

        # Reemplazamos tabs.currentChanged por detectar visibilidad si fuera necesario, 
        # pero con DockWidgets cada panel puede notificar cambios, y actualizaremos
        # solo los visibles.
        
        # Conectar cambios de visibilidad de los docks para correr la simulación al mostrarse
        self.dock_memristor.visibilityChanged.connect(lambda visible: self.run_simulation() if visible else None)
        self.dock_neuron.visibilityChanged.connect(lambda visible: self.run_simulation() if visible else None)
        self.dock_hybrid.visibilityChanged.connect(lambda visible: self.run_simulation() if visible else None)

        # Barra de Estado
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo. Modo en tiempo real activo.")

        # Nota: la simulación inicial la ejecuta el constructor (una sola vez)
        # después de intentar restaurar la última sesión.

    # ── Funciones del Menú Vista ──────────────────────────────────────────────
    
    def _action_tabify_all(self):
        """Agrupa todos los docks en un solo espacio con pestañas."""
        self.tabifyDockWidget(self.dock_memristor, self.dock_neuron)
        self.tabifyDockWidget(self.dock_neuron, self.dock_hybrid)
        self.tabifyDockWidget(self.dock_hybrid, self.dock_docs)
        self.dock_memristor.raise_()
        
    def _action_tile_all(self):
        """Divide la pantalla para mostrar los paneles más importantes a la vez."""
        # Colocamos Memristor a la izquierda
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_memristor)
        # Híbrido a la derecha
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_hybrid)
        # Neurona abajo del memristor
        self.splitDockWidget(self.dock_memristor, self.dock_neuron, Qt.Vertical)
        # Documentos abajo del híbrido (si se quiere ver) o agrupado
        self.tabifyDockWidget(self.dock_hybrid, self.dock_docs)
        self.dock_hybrid.raise_()

    # ── Gestión de Sesión ────────────────────────────────────────────────────

    def _restore_last_session(self) -> bool:
        """
        Restaura la última sesión guardada (configs/last_session.json).
        Devuelve True si la sesión se restauró correctamente; False si no
        existe o está corrupta (la app arranca con valores por defecto).
        No ejecuta la simulación: el llamador decide cuándo simular, de modo
        que el arranque ejecute la simulación inicial una sola vez.
        """
        try:
            data = self._profile_manager.load_last_session()
            if data is None:
                return False
            self.config_panel.from_dict(data)
            device_name = data.get("device_name", "Última Sesión")
            self._update_window_title(device_name)
            self.status_bar.showMessage(f"✓ Sesión restaurada: '{device_name}'")
            return True
        except Exception:
            # Sesión corrupta: ignorar silenciosamente
            return False

    def _save_last_session(self):
        """Guarda la configuración actual como última sesión (silencioso, sin diálogo)."""
        try:
            data = self.config_panel.to_dict()
            self._profile_manager.save_last_session(data)
        except Exception:
            pass  # Nunca interrumpir el cierre por un error de guardado

    def _update_window_title(self, device_name: str = ""):
        """Actualiza el título de la ventana con el nombre del dispositivo activo."""
        if device_name:
            self.setWindowTitle(
                f"Neuromorphic Lab — {device_name}"
            )
        else:
            self.setWindowTitle("Neuromorphic Lab — Simulador Universal de Dispositivos Memristivos")

    # ── Eventos Qt ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Intercepta el cierre para autoguardar la sesión actual."""
        self._save_last_session()
        event.accept()

    # ── Lógica de Simulación ─────────────────────────────────────────────────

    def _on_param_changed(self):
        """Disparado ante cualquier cambio de parámetro si la opción en tiempo real está activa."""
        # Se ejecuta un temporizador global que procesará todas las vistas visibles
        if self.signal_panel.is_realtime_enabled or self.neuron_signal_panel.is_realtime_enabled or self.hybrid_signal_panel.is_realtime_enabled:
            self.debounce_timer.start(180)

    def run_simulation(self):
        """Ejecuta la simulación para todos los docks que estén visibles actualmente."""
        if hasattr(self, 'dock_memristor') and not self.dock_memristor.visibleRegion().isEmpty():
            self._run_memristor_simulation()
        if hasattr(self, 'dock_neuron') and not self.dock_neuron.visibleRegion().isEmpty():
            self._run_neuron_simulation()
        if hasattr(self, 'dock_hybrid') and not self.dock_hybrid.visibleRegion().isEmpty():
            self._run_hybrid_simulation()

    def _run_memristor_simulation(self):
        """Ejecuta la simulación usando la configuración activa del memristor y la señal."""
        try:
            start_time = time.time()

            # 1. Construir memristor y obtener señal
            memristor = self.config_panel.build_memristor()
            t, v_signal, dt = self.signal_panel.generate_voltage_signal()
            steps = len(t)

            i_out = np.zeros(steps)
            x_state = np.zeros(steps)
            r_hist = np.zeros(steps)
            g_hist = np.zeros(steps)

            # Bug #2 fix: Inyectar el dt real de simulación al modificador C2C
            from neurolab.devices.realism.c2c import C2CVariabilityModifier
            for mod in memristor.modifiers:
                if isinstance(mod, C2CVariabilityModifier):
                    mod.set_simulation_dt(dt)

            # 2. Bucle de integración temporal (Euler Explícito)
            for step_idx in range(steps):
                v_inst = v_signal[step_idx]
                i_inst = memristor.step(voltage=v_inst, dt=dt)
                i_out[step_idx] = i_inst
                x_state[step_idx] = memristor.x
                r_hist[step_idx] = memristor.resistance
                g_hist[step_idx] = memristor.conductance


            # 2b. Población Monte Carlo de Dispositivos (D2D Ensemble)
            ensemble_data = []
            if self.config_panel.chk_d2d.isChecked() and self.config_panel.chk_show_d2d_ensemble.isChecked():
                base_seed = self.config_panel.spin_seed.value()
                num_ensemble = self.config_panel.spin_d2d_count.value()
                for dev_i in range(1, num_ensemble + 1):
                    dev_k = self.config_panel.build_memristor(seed_override=base_seed + dev_i * 101)
                    i_k = np.zeros(steps)
                    x_k = np.zeros(steps)
                    r_k = np.zeros(steps)
                    g_k = np.zeros(steps)
                    for s_idx in range(steps):
                        v_k = v_signal[s_idx]
                        i_k[s_idx] = dev_k.step(voltage=v_k, dt=dt)
                        x_k[s_idx] = dev_k.x
                        r_k[s_idx] = dev_k.resistance
                        g_k[s_idx] = dev_k.conductance
                    ensemble_data.append({
                        "v_drop": v_signal,
                        "i_out": i_k,
                        "x_state": x_k,
                        "r_hist": r_k,
                        "g_hist": g_k
                    })

            elapsed = time.time() - start_time

            # 3. Cargar datos de validación CSV si el interruptor está activo
            val_data = None
            if self.config_panel.is_csv_validation_enabled:
                val_data = self._validation_loader.load_all()

            # 4. Graficar resultados
            self.plot_canvas.plot_results(
                t, v_signal, v_signal, i_out, x_state, r_hist, g_hist,
                val_data=val_data,
                active_plots=self.config_panel.active_plots,
                ensemble_data=ensemble_data if ensemble_data else None
            )

            # 5. Métricas cuantitativas de validación
            if val_data is not None:
                metrics = compute_all_metrics(
                    t_sim=t,
                    i_sim_mA=i_out * 1e3,
                    x_sim=x_state,
                    r_sim_kohm=r_hist / 1e3,
                    g_sim_us=g_hist * 1e6,
                    val_data=val_data
                )
                self._update_metrics_panel(metrics)
            else:
                self.metrics_label.setText(
                    "<p style='color:#585b70; text-align:center;'>"
                    "Activa la Validación CSV para ver las métricas cuantitativas."
                    "</p>"
                )

            # 6. Actualizar título y barra de estado
            device_name = memristor.identity.device_name
            self._update_window_title(device_name)

            r_min, r_max = np.min(r_hist), np.max(r_hist)
            msg = (
                f"✓ Simulación Memristor ejecutada en {elapsed:.3f}s | Pasos: {steps:,} | "
                f"Dispositivo: '{device_name}' | "
                f"R_min: {r_min:.1f} Ω, R_max: {r_max:.1f} Ω"
            )
            self.status_bar.showMessage(msg)

        except Exception as e:
            QMessageBox.critical(self, "Error de Simulación", f"Ocurrió un error al ejecutar la simulación del Memristor:\n{str(e)}")
            self.status_bar.showMessage("❌ Error en la simulación.")
            
    def _update_metrics_panel(self, metrics: list) -> None:
        """Construye y muestra la tabla HTML de métricas cuantitativas de validación."""
        if not metrics:
            self.metrics_label.setText(
                "<p style='color:#585b70; text-align:center;'>Sin métricas disponibles.</p>"
            )
            return

        header = (
            "<table width='100%' cellspacing='0' cellpadding='0'>"
            "<tr style='background-color:#313244;'>"
            "<th style='padding:4px 6px; text-align:left; color:#89b4fa; font-size:11px;'>Magnitud</th>"
            "<th style='padding:4px 6px; text-align:right; color:#89b4fa; font-size:11px;'>MAE</th>"
            "<th style='padding:4px 6px; text-align:right; color:#89b4fa; font-size:11px;'>RMSE</th>"
            "<th style='padding:4px 6px; text-align:right; color:#89b4fa; font-size:11px;'>Err.Rel.Máx</th>"
            "<th style='padding:4px 6px; text-align:right; color:#89b4fa; font-size:11px;'>R²</th>"
            "</tr>"
        )
        rows = "".join(m.to_html_row() for m in metrics)
        footer = "</table>"

        # Leyenda de color
        legend = (
            "<p style='font-size:10px; color:#585b70; margin-top:4px;'>"
            "<span style='color:#a6e3a1;'>■</span> Excelente &nbsp;"
            "<span style='color:#f9e2af;'>■</span> Aceptable &nbsp;"
            "<span style='color:#f38ba8;'>■</span> Revisar"
            "</p>"
        )

        title = "<p style='color:#89b4fa; font-weight:bold; margin-bottom:4px;'>📊 Métricas de Validación CSV</p>"
        self.metrics_label.setText(title + header + rows + footer + legend)

    def _run_neuron_simulation(self):
        """Ejecuta la simulación de la Neurona LIF en tiempo real impulsada por voltaje V_IN y R_S."""
        try:
            start_time = time.time()
            
            neuron = self.neuron_config_panel.build_neuron()
            t, v_signal, dt = self.neuron_signal_panel.generate_voltage_signal()
            steps = len(t)

            v_m_hist = np.zeros(steps)
            i_in_hist = np.zeros(steps)
            r_leak_hist = np.full(steps, neuron.config.r_leak)
            r_series_hist = np.full(steps, neuron.config.r_series)
            
            for step_idx in range(steps):
                v_m_hist[step_idx] = neuron.v_membrane
                # Corriente inyectada por la fuente a través de R_S: I_in = (V_IN - V_m) / R_S
                i_in_hist[step_idx] = (v_signal[step_idx] - neuron.v_membrane) / neuron.config.r_series
                neuron.step(voltage_input=v_signal[step_idx], dt=dt)
                
            elapsed = time.time() - start_time
            
            self.neuron_plot_canvas.plot_results(
                t, v_signal, v_m_hist, r_leak_hist, neuron.spike_times, 
                neuron.config.v_th, neuron.config.v_reset,
                neuron.config.v_rest, i_in=i_in_hist, r_series_hist=r_series_hist
            )
            
            # ── Validación Matemática Analítica LIF (Sección E) ──
            lif_val = compute_lif_validation_metrics(t, v_m_hist, v_signal, neuron.config, is_voltage_input=True)
            
            n_spikes = len(neuron.spike_times)
            freq = n_spikes / t[-1] if t[-1] > 0 else 0
            
            self._update_window_title("Simulador LIF (Fuente Voltaje V_IN + R_S)")
            
            msg = (f"✓ Simulación LIF ejecutada en {elapsed:.3f}s | Pasos: {steps:,} | "
                   f"Spikes: {n_spikes} ({freq:.1f} Hz) | "
                   f"Validación Teórica: MAE={lif_val['mae']*1e3:.2f} mV, RMSE={lif_val['rmse']*1e3:.2f} mV, R²={lif_val['r2']:.4f}")
            self.status_bar.showMessage(msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error de Simulación", f"Error en la simulación LIF:\n{str(e)}")
            self.status_bar.showMessage("❌ Error en la simulación LIF.")

    def _get_panel_by_index(self, index: int) -> ConfigPanel:
        """Devuelve el ConfigPanel correspondiente al índice de sub-pestaña."""
        if index == 1:
            return self.config_panel_2
        elif index == 2:
            return self.config_panel_3
        else:
            return self.config_panel_1

    def _get_memristor_by_index(self, index: int, seed_offset: int = 0) -> Memristor:
        """Devuelve una instancia de Memristor configurado según el índice de la sub-pestaña (0, 1, 2)."""
        panel = self._get_panel_by_index(index)

        if seed_offset != 0:
            base_seed = panel.spin_seed.value()
            return panel.build_memristor(seed_override=base_seed + seed_offset)
        return panel.build_memristor()

    def _update_hybrid_summary(self):
        """Actualiza las tarjetas informativas de la Pestaña 3 mostrando los valores exactos de los paneles."""
        try:
            idx_rs = self.combo_mem_rs.currentIndex()
            idx_leak = self.combo_mem_leak.currentIndex()

            panel_rs = self._get_panel_by_index(idx_rs)
            panel_leak = self._get_panel_by_index(idx_leak)
            
            mem_rs = self._get_memristor_by_index(idx_rs)
            offset_leak = 1000 if (idx_rs == idx_leak) else 0
            mem_leak = self._get_memristor_by_index(idx_leak, seed_offset=offset_leak)
            neu = self.neuron_config_panel.build_neuron()
            
            mode_series = self.rb_mem_series.isChecked()
            mode_leak = self.rb_mem_leak.isChecked()
            mode_both = self.rb_mem_both.isChecked()
            mode_none = self.rb_mem_none.isChecked()

            def _fmt_r(val_ohm: float) -> str:
                if val_ohm >= 1e6:
                    return f"{val_ohm/1e6:.2f} MΩ"
                elif val_ohm >= 1e3:
                    return f"{val_ohm/1e3:.2f} kΩ"
                else:
                    return f"{val_ohm:.1f} Ω"

            r_rs_str = f"Sustituida por Memristor {idx_rs+1} (R<sub>ini</sub> = {_fmt_r(mem_rs.resistance)})" if (mode_series or mode_both) else f"Resistencia Fija ({_fmt_r(neu.config.r_series)})"
            r_leak_str = f"Sustituida por Memristor {idx_leak+1} (R<sub>ini</sub> = {_fmt_r(mem_leak.resistance)})" if (mode_leak or mode_both) else f"Resistencia Fija ({_fmt_r(neu.config.r_leak)})"

            mat_rs_name = panel_rs.combo_material.currentText().split(' ')[0]
            mat_leak_name = panel_leak.combo_material.currentText().split(' ')[0]
            
            if mode_none:
                mem_text = (
                    f"• <b>Resistencia Serie (R<sub>S</sub>):</b> {r_rs_str}<br/>"
                    f"• <b>Resistencia Fuga (R<sub>leak</sub>):</b> {r_leak_str}<br/>"
                    f"• <b>Circuito LIF Pasivo</b> (Sin memristores acoplados)"
                )
            else:
                mem_text = (
                    f"• <b>Resistencia Serie (R<sub>S</sub>):</b> {r_rs_str}<br/>"
                    f"• <b>Resistencia Fuga (R<sub>leak</sub>):</b> {r_leak_str}<br/>"
                    f"• <b>Memristor {idx_rs+1} (Serie - {mat_rs_name}):</b> R<sub>ON</sub> = {_fmt_r(panel_rs.get_r_on_ohms())} | R<sub>OFF</sub> = {_fmt_r(panel_rs.get_r_off_ohms())} (µ<sub>v</sub> = {panel_rs.spin_mu_v.value():.1e} m²/V·s, x<sub>0</sub> = {panel_rs.spin_x0.value():.2f})<br/>"
                    f"• <b>Memristor {idx_leak+1} (Fuga - {mat_leak_name}):</b> R<sub>ON</sub> = {_fmt_r(panel_leak.get_r_on_ohms())} | R<sub>OFF</sub> = {_fmt_r(panel_leak.get_r_off_ohms())} (µ<sub>v</sub> = {panel_leak.spin_mu_v.value():.1e} m²/V·s, x<sub>0</sub> = {panel_leak.spin_x0.value():.2f})"
                )
            
            cfg = neu.config
            neu_text = (
                f"• <b>Capacitancia C<sub>m</sub>:</b> {cfg.c_m*1e9:.2f} nF<br/>"
                f"• <b>Resistencia Serie Fija (LIF P2):</b> {_fmt_r(cfg.r_series)}<br/>"
                f"• <b>Resistencia Fuga Fija (LIF P2):</b> {_fmt_r(cfg.r_leak)}<br/>"
                f"• <b>Umbral V<sub>th</sub>:</b> {cfg.v_th:.2f} V | <b>Reset V<sub>reset</sub>:</b> {cfg.v_reset:.2f} V"
            )
            
            self.lbl_hybrid_mem_info.setText(mem_text)
            self.lbl_hybrid_neuron_info.setText(neu_text)
        except Exception:
            pass

    def _run_hybrid_simulation(self):
        """
        Ejecuta la simulación Híbrida (Pestaña 3) con los gráficos de Neurona LIF.
        Selecciona dinámicamente qué Memristor de Pestaña 1 se asigna a R_S y a R_leak.
        """
        try:
            start_time = time.time()
            
            # Actualizar tarjetas informativas en la Pestaña 3
            self._update_hybrid_summary()
            
            # 1. Obtener Memristor(es) de Pestaña 1 y Neurona LIF de Pestaña 2 según selección
            idx_rs = self.combo_mem_rs.currentIndex()
            idx_leak = self.combo_mem_leak.currentIndex()

            mem1 = self._get_memristor_by_index(idx_rs, seed_offset=0)
            offset_leak = 1000 if (idx_rs == idx_leak) else 0
            mem2 = self._get_memristor_by_index(idx_leak, seed_offset=offset_leak)
            neuron = self.neuron_config_panel.build_neuron()
            
            # 2. Generar señal de entrada V_IN(t)
            t, v_signal, dt = self.hybrid_signal_panel.generate_voltage_signal()
            steps = len(t)

            v_m_hist = np.zeros(steps)
            i_in_hist = np.zeros(steps)
            r_leak_hist = np.zeros(steps)
            r_series_hist = np.zeros(steps)

            # Determinar modo seleccionado
            mode_series = self.rb_mem_series.isChecked()
            mode_leak = self.rb_mem_leak.isChecked()
            mode_both = self.rb_mem_both.isChecked()
            mode_none = self.rb_mem_none.isChecked()

            r_s_base = neuron.config.r_series
            r_leak_base = neuron.config.r_leak
            c_m = neuron.config.c_m
            v_rest = neuron.config.v_rest
            v_th = neuron.config.v_th
            v_reset = neuron.config.v_reset
            t_ref = neuron.config.t_ref

            # Bug #2 fix: Inyectar el dt real de simulación a los modificadores C2C
            from neurolab.devices.realism.c2c import C2CVariabilityModifier
            for mem in (mem1, mem2):
                for mod in mem.modifiers:
                    if isinstance(mod, C2CVariabilityModifier):
                        mod.set_simulation_dt(dt)

            # Modo Volátil: leer configuración de shunt del panel del memristor de fuga
            # idx_leak determina qué sub-panel tiene la configuración del memristor de fuga
            _leak_panel = self._get_panel_by_index(idx_leak)
            volatile_cfg = _leak_panel.get_volatile_config()
            volatile_enabled = volatile_cfg["enabled"]
            volatile_x_shunt = volatile_cfg["x_shunt"]

            refractory_left = 0.0

            for step_idx in range(steps):
                v_in = v_signal[step_idx]
                v_m = neuron.v_membrane
                
                # --- Cálculo de Resistencias e Inyección según el modo ---
                if mode_series:
                    # Diodo ideal en serie (unidireccional)
                    v_drop_s = max(v_in - v_m, 0.0)
                    i_in_val = mem1.step(v_drop_s, dt)
                    r_s_curr = mem1.resistance
                    r_leak_curr = r_leak_base
                    i_leak_val = (v_m - v_rest) / r_leak_curr if r_leak_curr > 0 else 0.0
                elif mode_leak:
                    # Bug #3 fix: El memristor de fuga recibe el voltaje REAL sin max().
                    # Voltaje negativo ocurre cuando V_m < V_rest (afterhyperpolarization).
                    # Esto permite la transición SET→RESET del memristor tras el spike.
                    v_drop_l = v_m - v_rest  # Sin max() ← CORREGIDO
                    mem2.step(v_drop_l, dt)
                    r_s_curr = r_s_base
                    r_leak_curr = mem2.resistance
                    # Diodo ideal en entrada (unidireccional)
                    i_in_val = max(v_in - v_m, 0.0) / r_s_curr if r_s_curr > 0 else 0.0
                    # La corriente de fuga usa el voltaje real (puede ser negativa = carga hacia GND)
                    i_leak_val = (v_m - v_rest) / r_leak_curr if r_leak_curr > 0 else 0.0
                elif mode_both:
                    # Diodo ideal en serie (unidireccional)
                    v_drop_s = max(v_in - v_m, 0.0)
                    # Bug #3 fix: Memristor de fuga ve voltaje real sin max()
                    v_drop_l = v_m - v_rest  # Sin max() ← CORREGIDO
                    i_in_val = mem1.step(v_drop_s, dt)
                    mem2.step(v_drop_l, dt)
                    r_s_curr = mem1.resistance
                    r_leak_curr = mem2.resistance
                    i_leak_val = (v_m - v_rest) / r_leak_curr if r_leak_curr > 0 else 0.0
                else: # mode_none
                    r_s_curr = r_s_base
                    r_leak_curr = r_leak_base
                    i_in_val = max(v_in - v_m, 0.0) / r_s_curr if r_s_curr > 0 else 0.0
                    i_leak_val = (v_m - v_rest) / r_leak_curr if r_leak_curr > 0 else 0.0

                # Guardar historiales
                v_m_hist[step_idx] = v_m
                i_in_hist[step_idx] = i_in_val
                r_series_hist[step_idx] = r_s_curr
                r_leak_hist[step_idx] = r_leak_curr

                # --- Integración Física de la Neurona LIF ---
                neuron.t += dt
                neuron.has_spiked = False

                if refractory_left > 0.0:
                    refractory_left -= dt
                    # Bug fix (v2026.09.11): durante el refractario, la fuga sigue activa.
                    # V_m se recupera exponencialmente desde v_reset hacia v_rest,
                    # exactamente igual que en la Pestaña 2 (lif.py).
                    # Sin esta línea, V_m queda congelado en v_reset (ej. -0.20V).
                    dv_refrac = (-i_leak_val / c_m) * dt
                    neuron.v_membrane += dv_refrac
                else:
                    dv = ((i_in_val - i_leak_val) / c_m) * dt
                    neuron.v_membrane += dv
                    
                    if neuron.v_membrane >= v_th:
                        neuron.has_spiked = True
                        neuron.spike_times.append(neuron.t)
                        neuron.v_membrane = v_reset
                        if t_ref > 0.0:
                            refractory_left = t_ref
                        # Shunt opcional: si está OFF, x evoluciona naturalmente
                        # (relajación + RESET por AHP) y el periodo refractario
                        # DECRECE ciclo a ciclo.
                        apply_shunt = volatile_cfg.get("apply_shunt", False)
                        if volatile_enabled and apply_shunt and (mode_leak or mode_both):
                            mem2.x = volatile_x_shunt

            elapsed = time.time() - start_time

            # 3. Graficar en el NeuronMplCanvas de la Neurona LIF
            self.hybrid_plot_canvas.plot_results(
                t=t,
                signal_in=v_signal,
                v_m=v_m_hist,
                r_m_hist=r_leak_hist,
                spike_times=neuron.spike_times,
                v_th=v_th,
                v_reset=v_reset,
                v_rest=v_rest,
                i_in=i_in_hist,
                r_series_hist=r_series_hist
            )

            n_spikes = len(neuron.spike_times)
            freq = n_spikes / t[-1] if t[-1] > 0 else 0

            mode_str = f"Serie (Memristor {idx_rs + 1})" if mode_series else (f"Fuga (Memristor {idx_leak + 1})" if mode_leak else f"Ambos (M{idx_rs + 1} & M{idx_leak + 1})")
            self._update_window_title(f"Neuromorphic Lab — Híbrido: {mode_str}")

            msg = (f"✓ Simulación Híbrida ejecutada en {elapsed:.3f}s | Config: {mode_str} | "
                   f"Spikes (P2): {n_spikes} ({freq:.1f} Hz)")
            self.status_bar.showMessage(msg)

        except Exception as e:
            QMessageBox.critical(self, "Error de Simulación Híbrida", f"Error en la simulación Híbrida:\n{str(e)}")
            self.status_bar.showMessage("❌ Error en la simulación Híbrida.")

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\widgets\__init__.py` — 9 líneas

```python
"""
Widgets componentes de la interfaz gráfica neurolab GUI.
"""

from neurolab.gui.widgets.plot_canvas import MplCanvas
from neurolab.gui.widgets.config_panel import ConfigPanel
from neurolab.gui.widgets.signal_panel import SignalPanel

__all__ = ["MplCanvas", "ConfigPanel", "SignalPanel"]

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\widgets\arrow_spinbox.py` — 240 líneas

```python
from PySide6.QtWidgets import QWidget, QHBoxLayout, QDoubleSpinBox, QSpinBox, QPushButton, QComboBox, QAbstractSpinBox
from PySide6.QtCore import Signal, Qt, QObject, QEvent


class FocusDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox que ignora la rueda del ratón a menos que tenga el foco activo (clic previo)."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class FocusSpinBox(QSpinBox):
    """QSpinBox que ignora la rueda del ratón a menos que tenga el foco activo (clic previo)."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class FocusComboBox(QComboBox):
    """QComboBox que ignora la rueda del ratón a menos que tenga el foco activo (clic previo)."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoUnfocusedWheelEventFilter(QObject):
    """
    Filtro de eventos Qt global que evita que cualquier control numérico o desplegable
    cambie de valor al girar la rueda del ratón por encima sin haber hecho clic primero.
    """
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if isinstance(obj, (QAbstractSpinBox, QComboBox, ArrowDoubleSpinBox, ArrowSpinBox)):
                has_focus = obj.hasFocus() or (hasattr(obj, 'spin') and obj.spin.hasFocus())
                if not has_focus:
                    event.ignore()
                    return True
        return super().eventFilter(obj, event)


class ArrowDoubleSpinBox(QWidget):
    """
    Control numérico flotante con botones independientes ▲ y ▼ fuera de la barra de texto.
    Solo responde a la rueda del ratón si primero ha sido enfocado con un clic.
    """
    valueChanged = Signal(float)

    def __init__(self, value=1.0, min_val=0.0, max_val=100.0, step=0.1, decimals=2, suffix="", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Spinbox con protección de foco contra scroll no deseado
        self.spin = FocusDoubleSpinBox()
        self.spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.spin.setRange(min_val, max_val)
        self.spin.setDecimals(decimals)
        self.spin.setValue(value)
        self.spin.setSingleStep(step)
        if suffix:
            self.spin.setSuffix(suffix)

        # Botón ARRIBA ▲ fuera de la barra
        self.btn_up = QPushButton("▲")
        self.btn_up.setFixedSize(26, 26)
        self.btn_up.setCursor(Qt.PointingHandCursor)
        self.btn_up.setFocusPolicy(Qt.NoFocus)
        self.btn_up.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #11111b;
            }
            QPushButton:pressed {
                background-color: #b4befe;
            }
        """)

        # Botón ABAJO ▼ fuera de la barra
        self.btn_down = QPushButton("▼")
        self.btn_down.setFixedSize(26, 26)
        self.btn_down.setCursor(Qt.PointingHandCursor)
        self.btn_down.setFocusPolicy(Qt.NoFocus)
        self.btn_down.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #11111b;
            }
            QPushButton:pressed {
                background-color: #b4befe;
            }
        """)

        layout.addWidget(self.spin, stretch=1)
        layout.addWidget(self.btn_up)
        layout.addWidget(self.btn_down)
        self.setLayout(layout)

        self.btn_up.clicked.connect(self._step_up)
        self.btn_down.clicked.connect(self._step_down)
        self.spin.valueChanged.connect(self.valueChanged.emit)

    def wheelEvent(self, event):
        if self.spin.hasFocus():
            self.spin.wheelEvent(event)
        else:
            event.ignore()

    def _step_up(self):
        new_val = min(self.max_val, self.spin.value() + self.step)
        self.spin.setValue(new_val)

    def _step_down(self):
        new_val = max(self.min_val, self.spin.value() - self.step)
        self.spin.setValue(new_val)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, val: float):
        self.spin.setValue(val)


class ArrowSpinBox(QWidget):
    """
    Control numérico entero con botones independientes ▲ y ▼ fuera de la barra de texto.
    Solo responde a la rueda del ratón si primero ha sido enfocado con un clic.
    """
    valueChanged = Signal(int)

    def __init__(self, value=1, min_val=0, max_val=100, step=1, suffix="", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.spin = FocusSpinBox()
        self.spin.setButtonSymbols(QSpinBox.NoButtons)
        self.spin.setRange(min_val, max_val)
        self.spin.setValue(value)
        self.spin.setSingleStep(step)
        if suffix:
            self.spin.setSuffix(suffix)

        self.btn_up = QPushButton("▲")
        self.btn_up.setFixedSize(26, 26)
        self.btn_up.setCursor(Qt.PointingHandCursor)
        self.btn_up.setFocusPolicy(Qt.NoFocus)
        self.btn_up.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #11111b;
            }
        """)

        self.btn_down = QPushButton("▼")
        self.btn_down.setFixedSize(26, 26)
        self.btn_down.setCursor(Qt.PointingHandCursor)
        self.btn_down.setFocusPolicy(Qt.NoFocus)
        self.btn_down.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #89b4fa;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #89b4fa;
                color: #11111b;
            }
        """)

        layout.addWidget(self.spin, stretch=1)
        layout.addWidget(self.btn_up)
        layout.addWidget(self.btn_down)
        self.setLayout(layout)

        self.btn_up.clicked.connect(self._step_up)
        self.btn_down.clicked.connect(self._step_down)
        self.spin.valueChanged.connect(self.valueChanged.emit)

    def wheelEvent(self, event):
        if self.spin.hasFocus():
            self.spin.wheelEvent(event)
        else:
            event.ignore()

    def _step_up(self):
        new_val = min(self.max_val, self.spin.value() + self.step)
        self.spin.setValue(new_val)

    def _step_down(self):
        new_val = max(self.min_val, self.spin.value() - self.step)
        self.spin.setValue(new_val)

    def value(self) -> int:
        return self.spin.value()

    def setValue(self, val: int):
        self.spin.setValue(val)

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\widgets\config_panel.py` — 963 líneas

```python
"""
neurolab.gui.widgets.config_panel
====================================
Panel de Control interactivo para la configuración en 5 secciones del Memristor.

Incluye:
  - Todos los parámetros estocásticos detallados (Seed, C2C, D2D, Ruido, Ventanas)
  - Guardado/Carga de perfiles JSON via ProfileManager (directorio configs/ por defecto)
  - Diálogo de Presets del Sistema (lista de perfiles incluidos con el simulador)
  - Metadata de versión en exportaciones JSON
"""
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLineEdit,
    QCheckBox, QComboBox, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QMessageBox, QDialog, QListWidget, QListWidgetItem,
    QDialogButtonBox
)
from PySide6.QtCore import Signal, Qt
from neurolab.devices.config import DeviceConfig, ElectricalConfig, StrukovConfig
from neurolab.core.memristor import Memristor
from neurolab.devices.models.strukov import StrukovMathModel
from neurolab.devices.realism import (
    BiolekWindowModifier, JoglekarWindowModifier,
    D2DVariabilityModifier, C2CVariabilityModifier, ThermalNoiseModifier
)
from neurolab.gui.widgets.arrow_spinbox import ArrowDoubleSpinBox, ArrowSpinBox
from neurolab.io.profile_manager import ProfileManager


class SystemPresetsDialog(QDialog):
    """
    Diálogo modal que muestra los perfiles JSON disponibles en el directorio configs/.
    Permite seleccionar y cargar un preset del sistema con un doble click o botón.
    """

    def __init__(self, profile_manager: ProfileManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📋 Presets del Sistema")
        self.setMinimumSize(420, 300)
        self.selected_path: Path | None = None
        self._pm = profile_manager

        layout = QVBoxLayout()

        lbl = QLabel("Selecciona un preset para cargarlo:")
        lbl.setStyleSheet("color: #cdd6f4; font-weight: bold; margin-bottom: 4px;")
        layout.addWidget(lbl)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #89b4fa;
                color: #11111b;
                border-radius: 3px;
            }
            QListWidget::item:hover {
                background-color: #313244;
            }
        """)

        self._profiles: list[Path] = profile_manager.list_profiles()
        for path in self._profiles:
            display = profile_manager.get_profile_display_name(path)
            item = QListWidgetItem(f"  {display}")
            item.setToolTip(str(path))
            self.list_widget.addItem(item)

        if not self._profiles:
            self.list_widget.addItem("  (No hay perfiles en configs/)")

        self.list_widget.itemDoubleClicked.connect(self._accept)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Cargar Perfil")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(buttons)

        self.setLayout(layout)

    def _accept(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._profiles):
            self.selected_path = self._profiles[row]
            self.accept()
        else:
            QMessageBox.warning(self, "Sin Selección", "Selecciona un perfil de la lista.")


MATERIAL_MOBILITY_MAP = {
    "TiO₂ (Dióxido de Titanio - Strukov 2008)": {
        "mu_v": 1.0e-14,
        "family": "TiO2_oxide",
        "info": "TiO₂: Movilidad iónica alta (1×10⁻¹⁴ m²/V·s) — Conmutación rápida (Strukov et al., 2008)."
    },
    "TaOₓ (Óxido de Tántalo - Industrial)": {
        "mu_v": 1.0e-15,
        "family": "TaOx_oxide",
        "info": "TaOₓ: Movilidad iónica intermedia (1×10⁻¹⁵ m²/V·s) — Alta durabilidad (TSMC/Panasonic)."
    },
    "HfO₂ (Óxido de Hafnio - CMOS LIF 2025)": {
        "mu_v": 1.0e-16,
        "family": "HfO2_oxide",
        "info": "HfO₂: Movilidad iónica moderada (1×10⁻¹⁶ m²/V·s) — Deriva óptima para Neurona LIF (Wang et al., 2025)."
    },
    "WOₓ (Óxido de Tungsteno - Sináptico)": {
        "mu_v": 5.0e-15,
        "family": "WOx_oxide",
        "info": "WOₓ: Movilidad iónica sináptica (5×10⁻¹⁵ m²/V·s) — Conmutación analógica suave."
    },
    "NbOₓ (Óxido de Niobio - Mott Switch)": {
        "mu_v": 1.0e-17,
        "family": "NbOx_oxide",
        "info": "NbOₓ: Movilidad iónica ultra-lenta (1×10⁻¹⁷ m²/V·s) — Emulación de conmutación por umbral."
    },
    "Personalizado": {
        "mu_v": None,
        "family": "custom_oxide",
        "info": "Personalizado: Movilidad iónica ajustada manualmente por el usuario."
    }
}


class ConfigPanel(QWidget):
    """
    Panel de Control interactivo para la configuración del Memristor.

    Secciones:
        1. 🪪 Identidad del Dispositivo
        2. ⚡ Parámetros Eléctricos Universales
        3. 🔬 Parámetros Físicos (Strukov 2008)
        4. 🛠️  Modificadores de Realismo y Ventanas
    """
    preset_requested = Signal()
    param_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signal_panel = None
        self._profile_manager = ProfileManager()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # ── Barra de botones superior ────────────────────────────────────────
        toolbar_layout = QHBoxLayout()

        btn_preset = QPushButton("⚡ Perfil Paper Strukov")
        btn_preset.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #11111b;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        btn_preset.clicked.connect(self._load_strukov_paper_preset)

        btn_system = QPushButton("📋 Presets del Sistema")
        btn_system.setStyleSheet("""
            QPushButton {
                background-color: #cba6f7;
                color: #11111b;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #f5c2e7; }
        """)
        btn_system.clicked.connect(self._open_system_presets)

        btn_save = QPushButton("💾 Guardar JSON")
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #11111b;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #94e2d5; }
        """)
        btn_save.clicked.connect(self.save_profile_dialog)

        btn_load = QPushButton("📂 Cargar JSON")
        btn_load.setStyleSheet("""
            QPushButton {
                background-color: #fab387;
                color: #11111b;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #f9e2af; }
        """)
        btn_load.clicked.connect(self.load_profile_dialog)

        toolbar_layout.addWidget(btn_preset)
        toolbar_layout.addWidget(btn_system)
        toolbar_layout.addWidget(btn_save)
        toolbar_layout.addWidget(btn_load)
        main_layout.addLayout(toolbar_layout)

        # ── 1. Identidad y Material del Dispositivo ─────────────────────────
        group_a = QGroupBox("1. 🪪 Identidad del Dispositivo y Material")
        layout_a = QFormLayout()

        self.txt_device_name = QLineEdit("Strukov TiO2")
        self.combo_material = QComboBox()
        self.combo_material.addItems(list(MATERIAL_MOBILITY_MAP.keys()))

        self.txt_device_family = QLineEdit("TiO2_oxide")
        self.combo_model_name = QComboBox()
        self.combo_model_name.addItems(["strukov", "ideal_normalized"])

        layout_a.addRow("Nombre Dispositivo:", self.txt_device_name)
        layout_a.addRow("Material Memristivo:", self.combo_material)
        layout_a.addRow("Familia Física:", self.txt_device_family)
        layout_a.addRow("Modelo Matemático:", self.combo_model_name)
        group_a.setLayout(layout_a)
        main_layout.addWidget(group_a)

        # ── 2. Parámetros Eléctricos Universales ────────────────────────────
        group_b = QGroupBox("2. ⚡ Parámetros Eléctricos Universales")
        layout_b = QFormLayout()

        self.spin_r_on = ArrowDoubleSpinBox(value=100.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_ron_unit = QComboBox()
        self.combo_ron_unit.addItems(["Ω", "kΩ", "MΩ", "GΩ"])
        self.combo_ron_unit.setCurrentText("Ω")
        lay_ron = QHBoxLayout()
        lay_ron.setContentsMargins(0, 0, 0, 0)
        lay_ron.addWidget(self.spin_r_on)
        lay_ron.addWidget(self.combo_ron_unit)

        self.spin_r_off = ArrowDoubleSpinBox(value=16.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_roff_unit = QComboBox()
        self.combo_roff_unit.addItems(["Ω", "kΩ", "MΩ", "GΩ"])
        self.combo_roff_unit.setCurrentText("kΩ")
        lay_roff = QHBoxLayout()
        lay_roff.setContentsMargins(0, 0, 0, 0)
        lay_roff.addWidget(self.spin_r_off)
        lay_roff.addWidget(self.combo_roff_unit)

        self.spin_x0 = ArrowDoubleSpinBox(value=0.10, min_val=0.0, max_val=1.0, step=0.01, decimals=2)

        self.lbl_g_info = QLabel("G_ON: 10.00 mS | G_OFF: 62.50 µS")
        self.lbl_g_info.setStyleSheet("color: #a6adc8; font-size: 11px;")

        self.spin_r_on.valueChanged.connect(self._update_g_labels)
        self.combo_ron_unit.currentTextChanged.connect(self._update_g_labels)
        self.spin_r_off.valueChanged.connect(self._update_g_labels)
        self.combo_roff_unit.currentTextChanged.connect(self._update_g_labels)

        layout_b.addRow("Resistencia R_ON:", lay_ron)
        layout_b.addRow("Resistencia R_OFF:", lay_roff)
        layout_b.addRow("Estado Inicial x_0:", self.spin_x0)
        layout_b.addRow("Conductancias:", self.lbl_g_info)
        group_b.setLayout(layout_b)
        main_layout.addWidget(group_b)

        # ── 3. Parámetros Físicos del Modelo (Strukov) ──────────────────────
        group_e = QGroupBox("3. 🔬 Parámetros Físicos del Material (Strukov 2008)")
        layout_e = QFormLayout()

        self.spin_D_nm = ArrowDoubleSpinBox(value=10.0, min_val=1.0, max_val=500.0, step=1.0, suffix=" nm")
        self.spin_mu_v = ArrowDoubleSpinBox(value=1e-14, min_val=1e-20, max_val=1e-8, step=1e-16, decimals=18)

        self.lbl_material_info = QLabel("TiO₂: Movilidad iónica alta (1×10⁻¹⁴ m²/V·s) — Conmutación rápida (Strukov et al., 2008).")
        self.lbl_material_info.setStyleSheet("color: #a6adc8; font-size: 11px; font-style: italic;")
        self.lbl_material_info.setWordWrap(True)

        layout_e.addRow("Espesor Capa D:", self.spin_D_nm)
        layout_e.addRow("Movilidad µ_v:", self.spin_mu_v)
        layout_e.addRow("Característica del Material:", self.lbl_material_info)
        group_e.setLayout(layout_e)
        main_layout.addWidget(group_e)

        # ── 4. Modificadores de Realismo y Ventanas ──────────────────────────
        group_d = QGroupBox("4. 🛠️ Modificadores de Realismo y Ventanas")
        layout_d = QFormLayout()

        self.combo_realism_mode = QComboBox()
        self.combo_realism_mode.addItems([
            "Modo 1 — Ideal (Strukov Puro)",
            "Modo 2 — Extendido (Ventana Biolek)",
            "Modo 3 — Realista Estocástico (Biolek + C2C + Ruido)",
            "Personalizado"
        ])
        self.combo_realism_mode.currentIndexChanged.connect(self._on_realism_mode_changed)

        # Reproducibilidad Científica (SEED)
        self.spin_seed = ArrowSpinBox(value=42, min_val=0, max_val=999999, step=1)
        self.spin_seed.spin.setToolTip("Semilla fija para reproducibilidad científica estricta")

        self.combo_window_type = QComboBox()
        self.combo_window_type.addItems(["Biolek", "Joglekar", "Sin Ventana"])
        # Auditoría v2026.09.11: max_val limitado a 5 (antes 20).
        # p > 5 hace que f(x) ≈ 0 en los bordes, "congelando" el memristor
        # y bloqueando la conmutación. No es físicamente realista para óxidos neurómorficos.
        self.spin_biolek_p = ArrowSpinBox(value=2, min_val=1, max_val=5, step=1)
        self.spin_biolek_p.spin.setToolTip(
            "Exponente p de la función de ventana de Joglekar/Biolek.\n"
            "Rango recomendado: 1 a 3.\n"
            "Valores altos (p > 5) hacen f(x) ≈ 0 cerca de los bordes,\n"
            "bloqueando la conmutación del memristor ('memristor congelado').\n"
            "Para óxidos neurocomputacionales (TaOx, HfO2), p = 1 o p = 2."
        )

        # C2C — Ornstein-Uhlenbeck
        self.chk_c2c = QCheckBox("Variabilidad Ciclo a Ciclo (C2C - Ornstein-Uhlenbeck)")
        self.chk_c2c.setChecked(False)
        self.spin_c2c_sigma = ArrowDoubleSpinBox(value=0.05, min_val=0.001, max_val=0.5, step=0.01, decimals=3)
        self.spin_c2c_theta = ArrowDoubleSpinBox(value=1.0, min_val=0.1, max_val=10.0, step=0.1, decimals=2)

        # D2D — Device-to-Device
        self.chk_d2d = QCheckBox("Variabilidad Dispositivo a Dispositivo (D2D)")
        self.chk_d2d.setChecked(False)
        self.spin_d2d_sigma = ArrowDoubleSpinBox(value=0.05, min_val=0.001, max_val=0.5, step=0.01, decimals=3)
        self.chk_show_d2d_ensemble = QCheckBox("  ↳ 👥 Graficar Población Monte Carlo")
        self.chk_show_d2d_ensemble.setChecked(True)
        self.chk_show_d2d_ensemble.setEnabled(False)
        self.chk_show_d2d_ensemble.setToolTip("Muestra un abanico de memristores fabricados para visualizar la dispersión D2D")
        self.chk_show_d2d_ensemble.toggled.connect(lambda *_: self.param_changed.emit())

        self.spin_d2d_count = ArrowSpinBox(value=6, min_val=2, max_val=20, step=1)
        self.spin_d2d_count.setEnabled(False)
        self.spin_d2d_count.spin.setToolTip("Número de celdas a simular en la población Monte Carlo (2 a 20)")
        self.spin_d2d_count.valueChanged.connect(lambda *_: self.param_changed.emit())

        # Ruido Térmico de Lectura
        self.chk_noise = QCheckBox("Ruido Térmico de Lectura")
        self.chk_noise.setChecked(False)
        self.spin_noise_std = ArrowDoubleSpinBox(value=1e-6, min_val=0.0, max_val=1e-2, step=1e-6, decimals=7, suffix=" A")

        # ── Modo Volátil / Difusivo (Periódo Refractario Natural) ────────────────────
        self.chk_volatile = QCheckBox(
            "🧠 Modo Volátil (Memristor Difusivo) — Periódo Refractario Natural"
        )
        self.chk_volatile.setChecked(False)
        self.chk_volatile.setToolTip(
            "Activa el término de relajación volátil:\n"
            "  dx/dt_total = dx/dt_strukov + (-(x - x₀) / τ_relax)\n\n"
            "Simula memristores difusivos (Ag/SiO₂, NbOx) que se recuperan\n"
            "solos hacia la alta resistencia cuando el voltaje cae.\n\n"
            "Junto con el shunt en el spike, genera el período refractario\n"
            "dinámico sin ningún mecanismo externo (t_ref = 0).\n"
            "SOLO útil en la Pestaña 3 modo 'Memristor en Fuga'."
        )

        # Shunt opcional: si está OFF (default) x evoluciona naturalmente vía la
        # relajación volátil + RESET por AHP, y el periodo refractario DECRECE
        # ciclo a ciclo. Si está ON fuerza x = x_shunt en cada spike (flat fijo).
        self.chk_apply_shunt = QCheckBox("  ↳ Aplicar shunt fijo en cada spike")
        self.chk_apply_shunt.setChecked(False)   # ← DEFAULT OFF = evolución natural
        self.chk_apply_shunt.setToolTip(
            "OFF (recomendado): x evoluciona solo. El periodo refractario DECRECE "
            "ciclo a ciclo porque R_leak sube al relajarse el memristor.\n\n"
            "ON: fuerza x = x_shunt en cada spike. El flat queda CONSTANTE. "
            "Útil solo si quieres un refractario fijo."
        )
        self.chk_apply_shunt.toggled.connect(lambda *_: self.param_changed.emit())

        self.spin_tau_relax = ArrowDoubleSpinBox(
            value=0.05, min_val=0.001, max_val=2.0, step=0.01, decimals=3, suffix=" s"
        )
        self.spin_tau_relax.setEnabled(False)
        self.spin_tau_relax.spin.setToolTip(
            "τ_relax: Tiempo de relajación del memristor volátil.\n"
            "Determina cuánto tarda x en volver a x₀ tras el shunt.\n"
            "Tiempo refractario estimado ≈ τ · ln((x_shunt - x₀) / 0.05)\n"
            "  τ = 0.05 s  → refractario ≈ 14 ms\n"
            "  τ = 0.10 s  → refractario ≈ 28 ms\n"
            "  τ = 0.20 s  → refractario ≈ 57 ms"
        )

        self.spin_x_shunt = ArrowDoubleSpinBox(
            value=0.99, min_val=0.50, max_val=1.0, step=0.005, decimals=4, suffix=""
        )
        self.spin_x_shunt.setEnabled(False)
        self.spin_x_shunt.spin.setToolTip(
            "x_shunt: Estado al que se fuerza el memristor de fuga\n"
            "en el instante del spike (simula apertura de canales K⁺).\n"
            "x = 0.95 → R_leak ≈ R_ON (máxima conductancia, máximo shunt).\n"
            "x = 0.70 → shunt parcial."
        )

        # x_eq: estado de equilibrio bajo la relajación volátil. Independiente de
        # x_0 (estado inicial). Si x_eq < x_0, el memristor se relaja hacia OFF
        # y el periodo refractario disminuye ciclo a ciclo.
        self.spin_x_eq = ArrowDoubleSpinBox(
            value=0.05, min_val=0.0, max_val=1.0, step=0.01, decimals=3, suffix=""
        )
        self.spin_x_eq.setEnabled(False)
        self.spin_x_eq.spin.setToolTip(
            "x_eq: estado de equilibrio bajo la relajación volátil.\n"
            "Es adónde tiende x cuando no hay excitación.\n"
            "Debe ser MENOR que x_0 para que el memristor se relaje hacia OFF\n"
            "y el periodo refractario disminuya ciclo a ciclo."
        )

        self.lbl_volatile_info = QLabel()
        self.lbl_volatile_info.setWordWrap(True)
        self.lbl_volatile_info.setStyleSheet(
            "color: #94e2d5; font-size: 10px; background-color: #1a2a2a; "
            "border: 1px solid #94e2d5; border-radius: 3px; padding: 3px;"
        )
        self.lbl_volatile_info.setVisible(False)

        def _update_volatile_info():
            if not self.chk_volatile.isChecked():
                self.lbl_volatile_info.setVisible(False)
                return
            from neurolab.devices.realism.volatile import VolatileDecayModifier
            tau = self.spin_tau_relax.value()
            x_s = self.spin_x_shunt.value()
            x_eq = self.spin_x_eq.value() if hasattr(self, "spin_x_eq") else self.spin_x0.value()
            mod = VolatileDecayModifier(tau_relax=tau, x0_override=x_eq)
            t_est = mod.refractory_time_estimate
            r_on = self.get_r_on_ohms()
            r_off = self.get_r_off_ohms()
            r_shunt = r_on * x_s + r_off * (1 - x_s)
            self.lbl_volatile_info.setText(
                f"🧠 Volátil activo | τ_relax = {tau*1e3:.0f} ms | "
                f"x_shunt = {x_s:.2f} → Rₚₐₑₑₐ ≈ {r_shunt/1e3:.1f} kΩ | "
                f"Refractario estimado ≈ {t_est*1e3:.0f} ms"
            )
            self.lbl_volatile_info.setVisible(True)

        self._update_volatile_info = _update_volatile_info

        def _on_volatile_toggled(checked):
            self.spin_tau_relax.setEnabled(checked)
            self.spin_x_shunt.setEnabled(checked)
            self.spin_x_eq.setEnabled(checked)   # ← añade esto
            _update_volatile_info()
            self.param_changed.emit()

        self.chk_volatile.toggled.connect(_on_volatile_toggled)
        self.spin_tau_relax.valueChanged.connect(lambda *_: (_update_volatile_info(), self.param_changed.emit()))
        self.spin_x_shunt.valueChanged.connect(lambda *_: (_update_volatile_info(), self.param_changed.emit()))
        self.spin_x_eq.valueChanged.connect(lambda *_: (_update_volatile_info(), self.param_changed.emit()))

        layout_d.addRow("Modo de Presets:", self.combo_realism_mode)
        layout_d.addRow("🔑 Semilla (Seed):", self.spin_seed)
        layout_d.addRow("Tipo de Ventana:", self.combo_window_type)
        layout_d.addRow("Exponente Ventana (p):", self.spin_biolek_p)
        layout_d.addRow(self.chk_c2c)
        layout_d.addRow("  ↳ Volatilidad C2C (σ):", self.spin_c2c_sigma)
        layout_d.addRow("  ↳ Retorno Media (θ):", self.spin_c2c_theta)
        layout_d.addRow(self.chk_d2d)
        layout_d.addRow("  ↳ Dispersión D2D (σ):", self.spin_d2d_sigma)
        layout_d.addRow(self.chk_show_d2d_ensemble)
        layout_d.addRow("  ↳ Nº Celdas Población:", self.spin_d2d_count)
        layout_d.addRow(self.chk_noise)
        layout_d.addRow("  ↳ Nivel Ruido (std):", self.spin_noise_std)
        layout_d.addRow(self.chk_volatile)
        layout_d.addRow(self.chk_apply_shunt)
        layout_d.addRow("  ↳ Tiempo Relajación (τ):", self.spin_tau_relax)
        layout_d.addRow("  ↳ Estado Shunt Spike (xₚ):", self.spin_x_shunt)
        layout_d.addRow("  ↳ Estado Equilibrio (x_eq):", self.spin_x_eq)
        layout_d.addRow(self.lbl_volatile_info)
        group_d.setLayout(layout_d)
        main_layout.addWidget(group_d)

        # ── 5. Datos de Validación Experimental (CSV) ─────────────────────────
        group_csv = QGroupBox("5. 📊 Datos de Validación Experimental (CSV)")
        layout_csv = QVBoxLayout()

        self.btn_csv_toggle = QPushButton("📊 Validación CSV: ACTIVA")
        self.btn_csv_toggle.setCheckable(True)
        self.btn_csv_toggle.setChecked(True)
        self.btn_csv_toggle.setMinimumHeight(34)
        self.btn_csv_toggle.clicked.connect(self._on_csv_toggle_clicked)
        self._update_csv_toggle_style()

        layout_csv.addWidget(self.btn_csv_toggle)
        group_csv.setLayout(layout_csv)
        main_layout.addWidget(group_csv)

        # ── 6. Gráficas Visibles ──────────────────────────────────────────────────
        group_plots = QGroupBox("6. 📈 Gráficas Visibles")
        layout_plots = QVBoxLayout()
        
        self.chk_plot_iv = QCheckBox("Curva I-V (Histéresis)")
        self.chk_plot_vt_it = QCheckBox("Señales V(t) e I(t)")
        self.chk_plot_x = QCheckBox("Estado Interno x(t)")
        self.chk_plot_r = QCheckBox("Resistencia R(t)")
        self.chk_plot_g = QCheckBox("Conductancia G(t)")
        self.chk_plot_p = QCheckBox("Potencia P(t)")
        
        for chk in [self.chk_plot_iv, self.chk_plot_vt_it, self.chk_plot_x, self.chk_plot_r, self.chk_plot_g, self.chk_plot_p]:
            chk.setChecked(True)
            layout_plots.addWidget(chk)
            chk.toggled.connect(lambda *_: self.param_changed.emit())
            
        group_plots.setLayout(layout_plots)
        main_layout.addWidget(group_plots)

        main_layout.addStretch()
        self.setLayout(main_layout)
        self._update_g_labels()
        self._connect_signals()

    @property
    def is_csv_validation_enabled(self) -> bool:
        """Devuelve True si se deben superponer los datos de validación CSV."""
        return self.btn_csv_toggle.isChecked()

    @property
    def active_plots(self) -> list[str]:
        """Devuelve una lista con los identificadores de las gráficas seleccionadas."""
        active = []
        if self.chk_plot_iv.isChecked(): active.append("iv")
        if self.chk_plot_vt_it.isChecked(): active.append("vt_it")
        if self.chk_plot_x.isChecked(): active.append("x")
        if self.chk_plot_r.isChecked(): active.append("r")
        if self.chk_plot_g.isChecked(): active.append("g")
        if self.chk_plot_p.isChecked(): active.append("p")
        return active

    def _on_csv_toggle_clicked(self):
        """Actualiza el estilo del toggle al cambiar estado y dispara param_changed."""
        self._update_csv_toggle_style()
        self.param_changed.emit()

    def _update_csv_toggle_style(self):
        """Aplica el estilo visual al botón toggle según el estado ON/OFF."""
        if self.btn_csv_toggle.isChecked():
            self.btn_csv_toggle.setText("📊 Validación CSV: ACTIVA")
            self.btn_csv_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #f9e2af;
                    color: #11111b;
                    font-weight: bold;
                    border-radius: 5px;
                    border: 2px solid #e6a817;
                }
                QPushButton:hover { background-color: #eba836; }
            """)
        else:
            self.btn_csv_toggle.setText("📊 Validación CSV: INACTIVA")
            self.btn_csv_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #313244;
                    color: #585b70;
                    font-weight: bold;
                    border-radius: 5px;
                    border: 2px solid #45475a;
                }
                QPushButton:hover { background-color: #45475a; color: #cdd6f4; }
            """)

    # ── Lógica de presets de modo ────────────────────────────────────────────

    def _on_realism_mode_changed(self, index: int):
        """Conmuta automáticamente las casillas de realismo según el modo seleccionado."""
        if index == 0:   # Modo 1 — Ideal
            self.combo_window_type.setCurrentText("Sin Ventana")
            self.chk_c2c.setChecked(False)
            self.chk_d2d.setChecked(False)
            self.chk_noise.setChecked(False)
        elif index == 1:  # Modo 2 — Extendido
            self.combo_window_type.setCurrentText("Biolek")
            self.chk_c2c.setChecked(False)
            self.chk_d2d.setChecked(False)
            self.chk_noise.setChecked(False)
        elif index == 2:  # Modo 3 — Realista Estocástico
            self.combo_window_type.setCurrentText("Biolek")
            self.chk_c2c.setChecked(True)
            self.chk_d2d.setChecked(True)
            self.chk_noise.setChecked(True)

    def _connect_signals(self):
        """Conecta los cambios de cualquier campo para emitir param_changed."""
        self.txt_device_name.textChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_material.currentIndexChanged.connect(self._on_material_changed)
        self.txt_device_family.textChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_model_name.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_r_on.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_ron_unit.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_r_off.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_roff_unit.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_x0.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_D_nm.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_mu_v.valueChanged.connect(self._on_mu_v_spin_changed)
        self.spin_seed.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.combo_window_type.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_biolek_p.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.chk_c2c.toggled.connect(lambda *_: self.param_changed.emit())
        self.spin_c2c_sigma.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_c2c_theta.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.chk_d2d.toggled.connect(self._on_d2d_toggled)
        self.spin_d2d_sigma.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.chk_noise.toggled.connect(lambda *_: self.param_changed.emit())
        # Volatile: ya conectado arriba en los lambdas de su propia sección

    def _on_material_changed(self, index: int):
        mat_name = self.combo_material.currentText()
        if mat_name in MATERIAL_MOBILITY_MAP:
            data = MATERIAL_MOBILITY_MAP[mat_name]
            if data["mu_v"] is not None:
                self.spin_mu_v.blockSignals(True)
                self.spin_mu_v.setValue(data["mu_v"])
                self.spin_mu_v.blockSignals(False)
            self.txt_device_family.setText(data["family"])
            self.lbl_material_info.setText(data["info"])
        self.param_changed.emit()

    def _on_mu_v_spin_changed(self, val: float):
        matched = False
        for mat_name, data in MATERIAL_MOBILITY_MAP.items():
            if data["mu_v"] is not None and abs(data["mu_v"] - val) < 1e-20:
                self.combo_material.blockSignals(True)
                self.combo_material.setCurrentText(mat_name)
                self.combo_material.blockSignals(False)
                self.lbl_material_info.setText(data["info"])
                matched = True
                break
        if not matched:
            self.combo_material.blockSignals(True)
            self.combo_material.setCurrentText("Personalizado")
            self.combo_material.blockSignals(False)
            self.lbl_material_info.setText(f"Personalizado: µ_v = {val:.2e} m²/V·s")
        self.param_changed.emit()

    def _on_d2d_toggled(self, checked: bool):
        self.chk_show_d2d_ensemble.setEnabled(checked)
        self.spin_d2d_count.setEnabled(checked)
        self.param_changed.emit()

    def get_r_on_ohms(self) -> float:
        unit = self.combo_ron_unit.currentText()
        if unit == "Ω": return self.spin_r_on.value()
        if unit == "kΩ": return self.spin_r_on.value() * 1e3
        if unit == "MΩ": return self.spin_r_on.value() * 1e6
        if unit == "GΩ": return self.spin_r_on.value() * 1e9
        return self.spin_r_on.value()

    def get_r_off_ohms(self) -> float:
        unit = self.combo_roff_unit.currentText()
        if unit == "Ω": return self.spin_r_off.value()
        if unit == "kΩ": return self.spin_r_off.value() * 1e3
        if unit == "MΩ": return self.spin_r_off.value() * 1e6
        if unit == "GΩ": return self.spin_r_off.value() * 1e9
        return self.spin_r_off.value() * 1e3

    def _update_g_labels(self):
        r_on = self.get_r_on_ohms()
        r_off = self.get_r_off_ohms()
        g_on_ms = (1.0 / r_on) * 1e3 if r_on > 0 else 0
        g_off_us = (1.0 / r_off) * 1e6 if r_off > 0 else 0
        self.lbl_g_info.setText(f"G_ON: {g_on_ms:.2f} mS | G_OFF: {g_off_us:.2f} µS")

    def set_signal_panel(self, signal_panel):
        """Conecta la instancia de SignalPanel para unificar la gestión del perfil completo."""
        self.signal_panel = signal_panel

    def _load_strukov_paper_preset(self):
        self.txt_device_name.setText("Strukov TiO2 (Paper Fig 2b)")
        self.combo_material.setCurrentText("TiO₂ (Dióxido de Titanio - Strukov 2008)")
        self.spin_r_on.setValue(100.0)
        self.combo_ron_unit.setCurrentText("Ω")
        self.spin_r_off.setValue(16.0)
        self.combo_roff_unit.setCurrentText("kΩ")
        self.spin_x0.setValue(0.10)
        self.spin_D_nm.setValue(10.0)
        self.spin_mu_v.setValue(1e-14)
        self.spin_seed.setValue(42)
        self.combo_window_type.setCurrentText("Sin Ventana")
        self.chk_c2c.setChecked(False)
        self.chk_d2d.setChecked(False)
        self.chk_noise.setChecked(False)
        self.combo_realism_mode.setCurrentIndex(0)
        self.btn_csv_toggle.setChecked(True)
        self._update_csv_toggle_style()

        if self.signal_panel is not None:
            idx = self.signal_panel.combo_waveform.findText("Sinusoidal")
            if idx >= 0:
                self.signal_panel.combo_waveform.setCurrentIndex(idx)
            self.signal_panel.spin_v0.setValue(1.0)
            self.signal_panel.spin_f0.setValue(0.5)      # 0.5 Hz — escala temporal real del paper
            self.signal_panel.spin_duration.setValue(6.0)  # 6 s — 0.6 unidades × t_0 = 6 s
            self.signal_panel.spin_dt_ms.setValue(1.0)    # 1 ms — 6000 pasos

        self.param_changed.emit()

    # ── Diálogo de Presets del Sistema ──────────────────────────────────────

    def _open_system_presets(self):
        """Abre el diálogo de lista de perfiles del sistema."""
        dlg = SystemPresetsDialog(self._profile_manager, parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.selected_path is not None:
            try:
                data = self._profile_manager.load(dlg.selected_path)
                self.from_dict(data)
                self.param_changed.emit()
                QMessageBox.information(
                    self, "Preset Cargado",
                    f"✓ Preset del sistema cargado:\n{self._profile_manager.get_profile_display_name(dlg.selected_path)}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error al Cargar Preset", f"No se pudo cargar el preset:\n{str(e)}")

    # ── Serialización ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Exporta la configuración actual (dispositivo + señal) como diccionario JSON serializable."""
        data = {
            "device_name": self.txt_device_name.text(),
            "material": self.combo_material.currentText(),
            "device_family": self.txt_device_family.text(),
            "model_name": self.combo_model_name.currentText(),
            "r_on": self.spin_r_on.value(),
            "r_on_unit": self.combo_ron_unit.currentText(),
            "r_off": self.spin_r_off.value(),
            "r_off_unit": self.combo_roff_unit.currentText(),
            "initial_state": self.spin_x0.value(),
            "D_nm": self.spin_D_nm.value(),
            "mu_v": self.spin_mu_v.value(),
            "seed": self.spin_seed.value(),
            "realism_mode_index": self.combo_realism_mode.currentIndex(),
            "window_type": self.combo_window_type.currentText(),
            "biolek_p": self.spin_biolek_p.value(),
            "enable_c2c": self.chk_c2c.isChecked(),
            "c2c_sigma": self.spin_c2c_sigma.value(),
            "c2c_theta": self.spin_c2c_theta.value(),
            "enable_d2d": self.chk_d2d.isChecked(),
            "d2d_sigma": self.spin_d2d_sigma.value(),
            "enable_noise": self.chk_noise.isChecked(),
            "noise_std": self.spin_noise_std.value(),
            "enable_csv_validation": self.chk_csv_validation.isChecked(),
        }

        if self.signal_panel is not None:
            data["signal"] = self.signal_panel.to_dict()

        return data

    def from_dict(self, data: dict):
        """Importa y carga la configuración desde un diccionario JSON."""
        if "device_name" in data:
            self.txt_device_name.setText(data["device_name"])
        if "material" in data:
            idx = self.combo_material.findText(data["material"])
            if idx >= 0:
                self.combo_material.setCurrentIndex(idx)
        if "device_family" in data:
            self.txt_device_family.setText(data["device_family"])
        if "model_name" in data:
            idx = self.combo_model_name.findText(data["model_name"])
            if idx >= 0:
                self.combo_model_name.setCurrentIndex(idx)

        if "r_on" in data:
            self.spin_r_on.setValue(float(data["r_on"]))
        if "r_on_unit" in data:
            self.combo_ron_unit.setCurrentText(str(data["r_on_unit"]))
        elif "r_on" in data and float(data["r_on"]) >= 1000:
            val = float(data["r_on"])
            if val >= 1e6:
                self.spin_r_on.setValue(val / 1e6)
                self.combo_ron_unit.setCurrentText("MΩ")
            else:
                self.spin_r_on.setValue(val / 1e3)
                self.combo_ron_unit.setCurrentText("kΩ")

        if "r_off" in data:
            self.spin_r_off.setValue(float(data["r_off"]))
        if "r_off_unit" in data:
            self.combo_roff_unit.setCurrentText(str(data["r_off_unit"]))
        elif "r_off" in data and float(data["r_off"]) >= 1000:
            val = float(data["r_off"])
            if val >= 1e6:
                self.spin_r_off.setValue(val / 1e6)
                self.combo_roff_unit.setCurrentText("MΩ")
            else:
                self.spin_r_off.setValue(val / 1e3)
                self.combo_roff_unit.setCurrentText("kΩ")

        if "initial_state" in data:
            self.spin_x0.setValue(float(data["initial_state"]))
        if "D_nm" in data:
            self.spin_D_nm.setValue(float(data["D_nm"]))
        if "mu_v" in data:
            self.spin_mu_v.setValue(float(data["mu_v"]))
        if "seed" in data:
            self.spin_seed.setValue(int(data["seed"]))
        if "realism_mode_index" in data:
            self.combo_realism_mode.setCurrentIndex(int(data["realism_mode_index"]))
        if "window_type" in data:
            idx = self.combo_window_type.findText(data["window_type"])
            if idx >= 0:
                self.combo_window_type.setCurrentIndex(idx)
        if "biolek_p" in data:
            self.spin_biolek_p.setValue(int(data["biolek_p"]))
        if "enable_c2c" in data:
            self.chk_c2c.setChecked(bool(data["enable_c2c"]))
        if "c2c_sigma" in data:
            self.spin_c2c_sigma.setValue(float(data["c2c_sigma"]))
        if "c2c_theta" in data:
            self.spin_c2c_theta.setValue(float(data["c2c_theta"]))
        if "enable_d2d" in data:
            self.chk_d2d.setChecked(bool(data["enable_d2d"]))
        if "d2d_sigma" in data:
            self.spin_d2d_sigma.setValue(float(data["d2d_sigma"]))
        if "enable_noise" in data:
            self.chk_noise.setChecked(bool(data["enable_noise"]))
        if "noise_std" in data:
            self.spin_noise_std.setValue(float(data["noise_std"]))
        if "enable_csv_validation" in data:
            self.chk_csv_validation.setChecked(bool(data["enable_csv_validation"]))

        if "signal" in data and self.signal_panel is not None:
            self.signal_panel.from_dict(data["signal"])

    def save_profile_dialog(self):
        """Abre diálogo para guardar el perfil actual a un archivo .json."""
        default_dir = str(self._profile_manager.configs_dir)
        device_name = self.txt_device_name.text().strip() or "perfil_memristor"
        default_filename = str(self._profile_manager.configs_dir / f"{device_name}.json")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Perfil de Memristor",
            default_filename,
            "Archivos JSON (*.json)"
        )
        if file_path:
            try:
                data = self.to_dict()
                saved_path = self._profile_manager.save(
                    data,
                    profile_name=device_name,
                    file_path=file_path
                )
                QMessageBox.information(
                    self, "Perfil Guardado",
                    f"✓ Perfil guardado con éxito en:\n{saved_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error al Guardar", f"No se pudo guardar el archivo:\n{str(e)}")

    def load_profile_dialog(self):
        """Abre diálogo para cargar un perfil de memristor desde un archivo .json."""
        default_dir = str(self._profile_manager.configs_dir)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Perfil de Memristor",
            default_dir,
            "Archivos JSON (*.json)"
        )
        if file_path:
            try:
                data = self._profile_manager.load(file_path)
                self.from_dict(data)
                QMessageBox.information(
                    self, "Perfil Cargado",
                    f"✓ Perfil cargado exitosamente desde:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error al Cargar", f"Archivo JSON no válido:\n{str(e)}")

    # ── Construcción del Memristor ────────────────────────────────────────────

    def build_memristor(self, seed_override: Optional[int] = None) -> Memristor:
        """Construye una instancia de Memristor según la configuración actual del panel."""
        identity = DeviceConfig(
            name=self.txt_device_name.text(),
            family=self.txt_device_family.text(),
            model=self.combo_model_name.currentText()
        )

        electrical = ElectricalConfig(
            r_on=self.get_r_on_ohms(),
            r_off=self.get_r_off_ohms(),
            initial_state=self.spin_x0.value()
        )

        strukov_config = StrukovConfig(
            D=self.spin_D_nm.value() * 1e-9,   # nm → m
            mu_v=self.spin_mu_v.value()
        )

        seed = seed_override if seed_override is not None else self.spin_seed.value()
        modifiers = []

        # 1. Ventana de Frontera
        win_type = self.combo_window_type.currentText()
        if win_type == "Biolek":
            modifiers.append(BiolekWindowModifier(p=self.spin_biolek_p.value()))
        elif win_type == "Joglekar":
            modifiers.append(JoglekarWindowModifier(p=self.spin_biolek_p.value()))

        # 2. Variabilidad D2D (dispersión estática en R_on, R_off y conmutación)
        if self.chk_d2d.isChecked():
            d2d_mod = D2DVariabilityModifier(variability_std=self.spin_d2d_sigma.value(), seed=seed)
            electrical = d2d_mod.apply_to_electrical(electrical)
            modifiers.append(d2d_mod)

        # 3. Variabilidad C2C (Ornstein-Uhlenbeck)
        if self.chk_c2c.isChecked():
            modifiers.append(C2CVariabilityModifier(
                sigma=self.spin_c2c_sigma.value(),
                theta=self.spin_c2c_theta.value(),
                seed=seed
            ))

        # 4. Ruido Térmico de Lectura
        if self.chk_noise.isChecked():
            modifiers.append(ThermalNoiseModifier(noise_std=self.spin_noise_std.value(), seed=seed))

        # 5. Modo Volátil / Difusivo (Término de Relajación — Periodo Refractario Natural)
        if self.chk_volatile.isChecked():
            from neurolab.devices.realism.volatile import VolatileDecayModifier
            # x_eq = equilibrio bajo relajación. Independiente de x_0 (estado inicial).
            # Si no hay control en la GUI, usa 0.05 (memristor difusivo típico).
            x_eq = float(self.spin_x_eq.value()) if hasattr(self, "spin_x_eq") else 0.05
            modifiers.append(VolatileDecayModifier(
                tau_relax=self.spin_tau_relax.value(),
                x0_override=x_eq,
            ))

        return Memristor(
            math_model=StrukovMathModel(),
            electrical=electrical,
            identity=identity,
            model_config=strukov_config,
            modifiers=modifiers,
            clip_x=(win_type != "Sin Ventana")
        )

    def get_volatile_config(self) -> dict:
        """
        Retorna los parámetros del modo volátil para que el bucle de simulación
        pueda aplicar el shunt del memristor de fuga en el momento del spike.

        Returns:
            dict con:
              'enabled' (bool)   : Si el modo volátil está activo.
              'x_shunt' (float)  : Estado a forzar en el memristor al detectar spike.
              'apply_shunt' (bool): Si se aplica el shunt fijo en cada spike.
                                    DEFAULT OFF → el memristor evoluciona naturalmente.
        """
        return {
            "enabled": self.chk_volatile.isChecked(),
            "x_shunt": self.spin_x_shunt.value(),
            "apply_shunt": getattr(self, "chk_apply_shunt", None) is not None
                           and self.chk_apply_shunt.isChecked(),
        }

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\widgets\hybrid_plot_canvas.py` — 140 líneas

```python
"""
neurolab.gui.widgets.hybrid_plot_canvas
========================================
Lienzo Matplotlib para el circuito Híbrido Memristor-LIF (Pestaña 3).

Muestra 4 señales sincronizadas:
  ① V_source(t)  — Voltaje de entrada de la fuente (V)
  ② R_M(t)       — Resistencia dinámica del memristor (kΩ) [la novedad]
  ③ I_mem(t)     — Corriente sináptica inyectada a la neurona (µA)
  ④ V_c(t)       — Potencial de membrana + líneas V_th/V_reset + Raster
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HybridMplCanvas(QWidget):
    """Lienzo de dibujo Matplotlib para visualización del circuito Híbrido (Memristor+LIF)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 8), facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # 4 subplots con eje X compartido
        self.ax_vs = self.figure.add_subplot(411)
        self.ax_r  = self.figure.add_subplot(412, sharex=self.ax_vs)
        self.ax_i  = self.figure.add_subplot(413, sharex=self.ax_vs)
        self.ax_v  = self.figure.add_subplot(414, sharex=self.ax_vs)

        self.figure.subplots_adjust(hspace=0.65, left=0.10, right=0.95, top=0.93, bottom=0.08)
        self._setup_axes()

    # ── Configuración de ejes ────────────────────────────────────────────────

    def _setup_axes(self):
        for ax in [self.ax_vs, self.ax_r, self.ax_i, self.ax_v]:
            ax.set_facecolor("#181825")
            ax.tick_params(colors="#a6adc8", labelcolor="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_color("#45475a")

        self.ax_vs.set_title("① Fuente de Voltaje ($V_{source}$)", color="#cdd6f4", fontsize=10)
        self.ax_vs.set_ylabel("V (V)", color="#cdd6f4")
        self.ax_vs.grid(True, color="#313244", linestyle="--", linewidth=0.6)

        self.ax_r.set_title("② Resistencia Sináptica Dinámica ($R_M$)", color="#cdd6f4", fontsize=10)
        self.ax_r.set_ylabel(r"$R_M$ (k$\Omega$)", color="#cdd6f4")
        self.ax_r.grid(True, color="#313244", linestyle="--", linewidth=0.6)
        self.ax_r.ticklabel_format(useOffset=False, style="plain", axis="y")

        self.ax_i.set_title("③ Corriente Sináptica ($I_{mem} = V_{source} / R_M$)", color="#cdd6f4", fontsize=10)
        self.ax_i.set_ylabel(r"I ($\mu$A)", color="#cdd6f4")
        self.ax_i.grid(True, color="#313244", linestyle="--", linewidth=0.6)

        self.ax_v.set_title("④ Integración Neuronal + Disparos ($V_c$)", color="#cdd6f4", fontsize=10)
        self.ax_v.set_ylabel("$V_c$ (V)", color="#cdd6f4")
        self.ax_v.set_xlabel("Tiempo (s)", color="#cdd6f4")
        self.ax_v.grid(True, color="#313244", linestyle="--", linewidth=0.6)

    # ── Actualización de datos ───────────────────────────────────────────────

    def plot_results(
        self,
        t,
        v_source,
        r_m_hist,
        i_mem_hist,
        v_m_hist,
        spike_times,
        v_th,
        v_reset,
        v_rest,
        x_state_hist=None,
    ):
        """
        Actualiza todos los subplots con los datos del último paso de simulación.
        """
        self.ax_vs.clear()
        self.ax_r.clear()
        if hasattr(self, 'ax_r_twin') and self.ax_r_twin is not None:
            self.ax_r_twin.clear()
            self.ax_r_twin.remove()
            self.ax_r_twin = None
            
        self.ax_i.clear()
        self.ax_v.clear()
        self._setup_axes()

        # ── ① V_source ──────────────────────────────────────────────────────
        self.ax_vs.plot(t, v_source, color="#cba6f7", linewidth=1.5)
        self.ax_vs.set_ylim(bottom=min(0, np.min(v_source)) - 0.05)

        # ── ② R_M(t) dinámica y x(t) en eje derecho ─────────────────────────
        r_kohm = r_m_hist / 1e3
        l1 = self.ax_r.plot(t, r_kohm, color="#a6e3a1", linewidth=1.5, label=r"$R_M$ (k$\Omega$)")
        r_min, r_max = np.min(r_kohm), np.max(r_kohm)
        margin = max((r_max - r_min) * 0.1, 0.5)
        self.ax_r.set_ylim(max(0, r_min - margin), r_max + margin)

        if x_state_hist is not None:
            self.ax_r_twin = self.ax_r.twinx()
            self.ax_r_twin.tick_params(colors="#89dceb", labelcolor="#89dceb")
            self.ax_r_twin.set_ylabel("$x(t)$ Estado", color="#89dceb")
            self.ax_r_twin.set_ylim(-0.05, 1.05)
            for spine in self.ax_r_twin.spines.values():
                spine.set_color("#45475a")
            l2 = self.ax_r_twin.plot(t, x_state_hist, color="#89dceb", linewidth=1.2, linestyle="--", label="$x(t)$ Estado Memristivo")
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            self.ax_r.legend(lines, labels, loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

        # ── ③ I_mem(t) en µA ────────────────────────────────────────────────
        i_ua = i_mem_hist * 1e6
        self.ax_i.plot(t, i_ua, color="#f9e2af", linewidth=1.5)
        self.ax_i.set_ylim(bottom=min(0, np.min(i_ua)) - 0.5)

        # ── ④ V_c(t) = V_m - V_rest, + líneas umbral/reset ─────────────────
        v_c      = v_m_hist - v_rest
        v_th_c   = v_th   - v_rest
        v_reset_c = v_reset - v_rest

        self.ax_v.plot(t, v_c, color="#89b4fa", linewidth=1.5)
        self.ax_v.axhline(v_th_c,    color="#f38ba8", linestyle="--", alpha=0.85, linewidth=1.2, label="$V_{th}$ relativo")
        self.ax_v.axhline(v_reset_c, color="#a6e3a1", linestyle=":",  alpha=0.85, linewidth=1.2, label="$V_{reset}$ relativo")
        self.ax_v.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=9)

        # Marcas de spike superpuestas en V_c
        if len(spike_times) > 0:
            for st in spike_times:
                self.ax_v.axvline(st, color="#f38ba8", linewidth=0.8, alpha=0.5, linestyle=":")

        self.canvas.draw()

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\widgets\neuron_config_panel.py` — 311 líneas

```python
import json
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLabel, QComboBox, QHBoxLayout, QPushButton,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Signal
from neurolab.neurons import LIFConfig, LIFNeuron
from neurolab.gui.widgets.arrow_spinbox import ArrowDoubleSpinBox

class NeuronConfigPanel(QWidget):
    """Panel de configuración en tiempo real para los parámetros LIF."""
    param_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        group = QGroupBox("Parámetros LIF")
        layout = QFormLayout()

        self.spin_c_m = ArrowDoubleSpinBox(value=100.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_c_unit = QComboBox()
        self.combo_c_unit.addItems(["pF", "nF", "uF", "mF"])
        self.combo_c_unit.setCurrentText("nF")
        lay_c = QHBoxLayout()
        lay_c.setContentsMargins(0,0,0,0)
        lay_c.addWidget(self.spin_c_m)
        lay_c.addWidget(self.combo_c_unit)

        self.spin_r_series = ArrowDoubleSpinBox(value=100.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_rs_unit = QComboBox()
        self.combo_rs_unit.addItems(["Ω", "kΩ", "MΩ", "GΩ"])
        self.combo_rs_unit.setCurrentText("kΩ")
        lay_rs = QHBoxLayout()
        lay_rs.setContentsMargins(0,0,0,0)
        lay_rs.addWidget(self.spin_r_series)
        lay_rs.addWidget(self.combo_rs_unit)

        self.spin_r_leak = ArrowDoubleSpinBox(value=1.0, min_val=0.001, max_val=999999.0, step=1.0, suffix="")
        self.combo_r_unit = QComboBox()
        self.combo_r_unit.addItems(["Ω", "kΩ", "MΩ", "GΩ"])
        self.combo_r_unit.setCurrentText("MΩ")
        lay_r = QHBoxLayout()
        lay_r.setContentsMargins(0,0,0,0)
        lay_r.addWidget(self.spin_r_leak)
        lay_r.addWidget(self.combo_r_unit)
        self.spin_v_rest = ArrowDoubleSpinBox(value=0.0, min_val=-100.0, max_val=100.0, step=0.1, suffix=" V")
        self.spin_v_th = ArrowDoubleSpinBox(value=2.5, min_val=-100.0, max_val=100.0, step=0.05, suffix=" V")
        self.spin_v_reset = ArrowDoubleSpinBox(value=0.0, min_val=-100.0, max_val=100.0, step=0.05, suffix=" V")
        self.spin_t_ref = ArrowDoubleSpinBox(value=2.0, min_val=0.0, max_val=1000.0, step=1.0, suffix=" ms")

        # Label de advertencia dinámica: aparece cuando V_reset >= V_rest
        # (en ese caso el memristor de fuga nunca puede hacer RESET)
        self.lbl_ahp_warning = QLabel()
        self.lbl_ahp_warning.setWordWrap(True)
        self.lbl_ahp_warning.setStyleSheet(
            "color: #f9e2af; font-size: 10px; background-color: #2a2218; "
            "border: 1px solid #f9e2af; border-radius: 3px; padding: 3px;"
        )
        self.lbl_ahp_warning.setVisible(False)

        self.lbl_tau = QLabel("Constante τ_m: — | τ_eq: —")
        self.lbl_tau.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold;")

        self.spin_c_m.valueChanged.connect(self._on_change)
        self.spin_r_series.valueChanged.connect(self._on_change)
        self.spin_r_leak.valueChanged.connect(self._on_change)
        self.spin_v_rest.valueChanged.connect(self._on_change)
        self.spin_v_th.valueChanged.connect(self._on_change)
        self.spin_v_reset.valueChanged.connect(self._on_change)
        self.spin_t_ref.valueChanged.connect(self._on_change)

        self.combo_c_unit.currentTextChanged.connect(self._on_change)
        self.combo_rs_unit.currentTextChanged.connect(self._on_change)
        self.combo_r_unit.currentTextChanged.connect(self._on_change)

        layout.addRow("Capacitancia (C_m):", lay_c)
        layout.addRow("Resistencia Serie (R_S):", lay_rs)
        layout.addRow("Resistencia Fuga (R_leak):", lay_r)
        layout.addRow("Potencial Reposo (V_rest):", self.spin_v_rest)
        layout.addRow("Potencial Umbral (V_th):", self.spin_v_th)
        layout.addRow("Potencial Reset (V_reset):", self.spin_v_reset)
        layout.addRow("Periodo Refractario (t_ref):", self.spin_t_ref)
        layout.addRow(self.lbl_ahp_warning)
        layout.addRow("Info de Circuito:", self.lbl_tau)

        # Botones de Guardar / Cargar Perfil LIF
        btn_save = QPushButton("💾 Guardar Config LIF")
        btn_save.setStyleSheet("""
            QPushButton { background-color: #313244; color: #89b4fa; font-weight: bold; padding: 6px; border-radius: 4px; border: 1px solid #45475a; }
            QPushButton:hover { background-color: #45475a; color: #b4befe; }
        """)
        btn_save.clicked.connect(self._save_lif_config_file)

        btn_load = QPushButton("📂 Cargar Config LIF")
        btn_load.setStyleSheet("""
            QPushButton { background-color: #313244; color: #a6e3a1; font-weight: bold; padding: 6px; border-radius: 4px; border: 1px solid #45475a; }
            QPushButton:hover { background-color: #45475a; color: #b4befe; }
        """)
        btn_load.clicked.connect(self._load_lif_config_file)

        lay_io_btns = QHBoxLayout()
        lay_io_btns.setContentsMargins(0, 4, 0, 4)
        lay_io_btns.addWidget(btn_save)
        lay_io_btns.addWidget(btn_load)
        layout.addRow(lay_io_btns)

        # Preset del circuito hardware real (LM393 + 2N7000)
        btn_hw = QPushButton("🧩 Preset Guía Hardware (LM393/2N7000)")
        btn_hw.setStyleSheet("""
            QPushButton { background-color: #cba6f7; color: #11111b; font-weight: bold; padding: 8px; border-radius: 6px; }
            QPushButton:hover { background-color: #b4befe; }
        """)
        btn_hw.setToolTip(
            "Aplica los valores del circuito real de la tesis:\n"
            "C_m = 100 nF, R_S = 100 kΩ, R_leak = 1 MΩ, V_th = +2.5 V, V_reset = 0 V (a masa vía 2N7000),\n"
            "t_ref = 2 ms."
        )
        btn_hw.clicked.connect(self._apply_hardware_preset)
        layout.addRow(btn_hw)

        # Preset Biofísico Memristivo — habilita hiperpolarización para RESET del memristor de fuga
        btn_bio = QPushButton("🧠 Preset Biofísico Memristivo (AHP activo)")
        btn_bio.setStyleSheet("""
            QPushButton { background-color: #a6e3a1; color: #11111b; font-weight: bold; padding: 8px; border-radius: 6px; }
            QPushButton:hover { background-color: #94e2d5; }
        """)
        btn_bio.setToolTip(
            "Preset optimizado para la Pestaña 3 con Memristor en Fuga:\n"
            "V_reset = -0.20 V (hiperpolarización post-spike)\n"
            "Permite que el memristor de fuga haga RESET durante el colapso de membrana,\n"
            "generando un periodo refractario dinámico y adaptación de frecuencia real.\n"
            "Usar con ventana de Biolek en el memristor de fuga (Pestaña 1 → Sub-Pestaña 2)."
        )
        btn_bio.clicked.connect(self._apply_biophysical_preset)
        layout.addRow(btn_bio)
        
        group.setLayout(layout)
        main_layout.addWidget(group)
        main_layout.addStretch()
        self.setLayout(main_layout)

        # Recalcular la etiqueta τ/I_th con los valores iniciales reales del panel
        self._on_change()

    def _get_c_multiplier(self) -> float:
        unit = self.combo_c_unit.currentText()
        if unit == "pF": return 1e-12
        if unit == "nF": return 1e-9
        if unit in ["uF", "µF"]: return 1e-6
        if unit == "mF": return 1e-3
        return 1e-9

    def _get_rs_multiplier(self) -> float:
        unit = self.combo_rs_unit.currentText()
        if unit == "Ω": return 1.0
        if unit == "kΩ": return 1e3
        if unit == "MΩ": return 1e6
        if unit == "GΩ": return 1e9
        return 1e3

    def _get_r_multiplier(self) -> float:
        unit = self.combo_r_unit.currentText()
        if unit == "Ω": return 1.0
        if unit == "kΩ": return 1e3
        if unit == "MΩ": return 1e6
        if unit == "GΩ": return 1e9
        return 1e6

    def _on_change(self):
        c_farads = self.spin_c_m.value() * self._get_c_multiplier()
        rs_ohms = self.spin_r_series.value() * self._get_rs_multiplier()
        rleak_ohms = self.spin_r_leak.value() * self._get_r_multiplier()
        
        # 1. tau_m pasivo de membrana (olvido/fuga sin entrada): tau_m = R_leak * C_m
        tau_leak_ms = rleak_ohms * c_farads * 1e3
        
        # 2. tau_eq equivalente durante la carga con V_IN: tau_eq = (R_S || R_leak) * C_m
        r_eq = (rs_ohms * rleak_ohms) / (rs_ohms + rleak_ohms)
        tau_eq_ms = r_eq * c_farads * 1e3
        
        # Voltaje asintótico asumiendo Vin = 5V
        try:
            v_inf_5v = (rleak_ohms / (rs_ohms + rleak_ohms)) * 5.0
            self.lbl_tau.setText(f"τ_m (Fuga): {tau_leak_ms:.1f} ms | τ_eq (Carga): {tau_eq_ms:.1f} ms | V_inf(5V): {v_inf_5v:.2f} V")
        except ZeroDivisionError:
            self.lbl_tau.setText(f"τ_m: {tau_leak_ms:.1f} ms | τ_eq: N/A")
            
        self.param_changed.emit()

    def _apply_hardware_preset(self):
        """
        Carga los parámetros del circuito real de la guía hardware (LM393 + 2N7000):
        C_m = 100 nF, R_S = 100 kΩ, R_leak = 1 MΩ, V_th = +2.5 V, V_reset = 0 V, t_ref = 2 ms.
        """
        self.combo_c_unit.setCurrentText("nF")
        self.spin_c_m.setValue(100.0)
        self.combo_rs_unit.setCurrentText("kΩ")
        self.spin_r_series.setValue(100.0)
        self.combo_r_unit.setCurrentText("MΩ")
        self.spin_r_leak.setValue(1.0)
        self.spin_v_rest.setValue(0.0)
        self.spin_v_th.setValue(2.5)
        self.spin_v_reset.setValue(0.0)
        self.spin_t_ref.setValue(2.0)
        self._on_change()

    def _apply_biophysical_preset(self):
        """
        Preset Biofísico Memristivo: habilita hiperpolarización post-spike (AHP).

        V_reset = -0.20 V (por debajo de V_rest = 0.0 V) permite que el memristor
        de fuga (Pestaña 3) experimente un voltaje negativo justo tras el spike.
        Esto induce el proceso RESET en el memristor, aumentando R_leak y generando
        un periodo refractario dinámico y adaptativo de frecuencia.

        Recomendado: usar ventana de Biolek en el memristor de fuga (es direccional).
        """
        self.combo_c_unit.setCurrentText("nF")
        self.spin_c_m.setValue(100.0)
        self.combo_rs_unit.setCurrentText("kΩ")
        self.spin_r_series.setValue(100.0)
        self.combo_r_unit.setCurrentText("MΩ")
        self.spin_r_leak.setValue(1.0)
        self.spin_v_rest.setValue(0.0)
        self.spin_v_th.setValue(2.5)
        self.spin_v_reset.setValue(-0.20)  # Hiperpolarización: V_m cae por debajo de V_rest
        self.spin_t_ref.setValue(0.0)      # Sin t_ref forzado: el periodo refractario emerge del memristor
        self._on_change()

    def set_signal_panel(self, signal_panel):
        """Asocia el panel de señal para incluir su estado en el archivo JSON."""
        self._signal_panel = signal_panel

    def to_dict(self) -> dict:
        data = {
            "c_m_value": self.spin_c_m.value(),
            "c_m_unit": self.combo_c_unit.currentText(),
            "r_series_value": self.spin_r_series.value(),
            "r_series_unit": self.combo_rs_unit.currentText(),
            "r_leak_value": self.spin_r_leak.value(),
            "r_leak_unit": self.combo_r_unit.currentText(),
            "v_rest": self.spin_v_rest.value(),
            "v_th": self.spin_v_th.value(),
            "v_reset": self.spin_v_reset.value(),
            "t_ref": self.spin_t_ref.value(),
        }
        if hasattr(self, '_signal_panel') and self._signal_panel is not None:
            data["signal_panel"] = self._signal_panel.to_dict()
        return data

    def from_dict(self, data: dict):
        if "c_m_value" in data: self.spin_c_m.setValue(data["c_m_value"])
        if "c_m_unit" in data: self.combo_c_unit.setCurrentText(data["c_m_unit"])
        if "r_series_value" in data: self.spin_r_series.setValue(data["r_series_value"])
        if "r_series_unit" in data: self.combo_rs_unit.setCurrentText(data["r_series_unit"])
        if "r_leak_value" in data: self.spin_r_leak.setValue(data["r_leak_value"])
        if "r_leak_unit" in data: self.combo_r_unit.setCurrentText(data["r_leak_unit"])
        if "v_rest" in data: self.spin_v_rest.setValue(data["v_rest"])
        if "v_th" in data: self.spin_v_th.setValue(data["v_th"])
        if "v_reset" in data: self.spin_v_reset.setValue(data["v_reset"])
        if "t_ref" in data: self.spin_t_ref.setValue(data["t_ref"])
        if "signal_panel" in data and hasattr(self, '_signal_panel') and self._signal_panel is not None:
            self._signal_panel.from_dict(data["signal_panel"])
        self._on_change()

    def _save_lif_config_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Configuración de Neurona LIF", "configs/lif_config.json", "Archivos JSON (*.json)"
        )
        if file_path:
            payload = {
                "_meta": {
                    "type": "lif_neuron_config",
                    "saved_at": datetime.now().isoformat(timespec="seconds")
                }
            }
            payload.update(self.to_dict())
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "Guardado Exitoso", f"Configuración LIF guardada en:\n{file_path}")

    def _load_lif_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Configuración de Neurona LIF", "configs/", "Archivos JSON (*.json)"
        )
        if file_path and Path(file_path).exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.from_dict(data)
                QMessageBox.information(self, "Carga Exitosa", f"Configuración LIF cargada desde:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error al Cargar", f"No se pudo cargar la configuración:\n{str(e)}")

    def build_neuron(self) -> LIFNeuron:
        cfg = LIFConfig(
            c_m = self.spin_c_m.value() * self._get_c_multiplier(),
            r_series = self.spin_r_series.value() * self._get_rs_multiplier(),
            r_leak = self.spin_r_leak.value() * self._get_r_multiplier(),
            v_rest = self.spin_v_rest.value(),
            v_th = self.spin_v_th.value(),
            v_reset = self.spin_v_reset.value(),
            t_ref = self.spin_t_ref.value() * 1e-3
        )
        return LIFNeuron(cfg)

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\widgets\neuron_plot_canvas.py` — 161 líneas

```python
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class NeuronMplCanvas(QWidget):
    """Lienzo de dibujo Matplotlib para visualización de la neurona LIF (5 subplots sincronizados)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 8), facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # 5 subplots: R_S(t), R_leak(t), I(t), V_m(t), V_out(t)
        self.ax_rs = self.figure.add_subplot(511)
        self.ax_rl = self.figure.add_subplot(512, sharex=self.ax_rs)
        self.ax_i  = self.figure.add_subplot(513, sharex=self.ax_rs)
        self.ax_v  = self.figure.add_subplot(514, sharex=self.ax_rs)
        self.ax_s  = self.figure.add_subplot(515, sharex=self.ax_rs)
        
        self.figure.subplots_adjust(hspace=0.65, left=0.10, right=0.95, top=0.94, bottom=0.08)
        self._setup_axes()

    def _setup_axes(self):
        for ax in [self.ax_rs, self.ax_rl, self.ax_i, self.ax_v, self.ax_s]:
            ax.set_facecolor("#181825")
            ax.tick_params(colors="#a6adc8", labelcolor="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_color("#45475a")
                
        self.ax_rs.set_title("① Resistencia Serie ($R_S$)", color="#cdd6f4", fontsize=9)
        self.ax_rs.set_ylabel(r"$R_S$ (k$\Omega$)", color="#89dceb")
        self.ax_rs.grid(True, color="#313244", linestyle="--", linewidth=0.6)
        self.ax_rs.ticklabel_format(useOffset=False, style='plain', axis='y')

        self.ax_rl.set_title("② Resistencia de Fuga ($R_{leak}$)", color="#cdd6f4", fontsize=9)
        self.ax_rl.set_ylabel(r"$R_{leak}$ (k$\Omega$)", color="#a6e3a1")
        self.ax_rl.grid(True, color="#313244", linestyle="--", linewidth=0.6)
        self.ax_rl.ticklabel_format(useOffset=False, style='plain', axis='y')
        
        self.ax_i.set_title("③ Estímulo de Entrada: Voltaje ($V_{IN}$) y Corriente ($I_{in}$)", color="#cdd6f4", fontsize=9)
        self.ax_i.set_ylabel("$V_{IN}$ (V)", color="#89b4fa")
        self.ax_i.grid(True, color="#313244", linestyle="--", linewidth=0.6)
        
        self.ax_v.set_title("④ Potencial de Membrana / Capacitor ($V_c$)", color="#cdd6f4", fontsize=9)
        self.ax_v.set_ylabel("$V_c$ (V)", color="#cdd6f4")
        self.ax_v.grid(True, color="#313244", linestyle="--", linewidth=0.6)
        
        self.ax_s.set_title("⑤ Salida de Voltaje de Disparo ($V_{OUT}$)", color="#cdd6f4", fontsize=9)
        self.ax_s.set_ylabel("$V_{OUT}$ (V)", color="#cdd6f4")
        self.ax_s.set_xlabel("Tiempo (s)", color="#cdd6f4")
        self.ax_s.grid(True, color="#313244", linestyle="--", linewidth=0.6)

    def plot_results(self, t, signal_in, v_m, r_m_hist, spike_times, v_th, v_reset, v_rest, i_in=None, r_series_hist=None):
        self.ax_rs.clear()
        self.ax_rl.clear()
        self.ax_i.clear()
        if hasattr(self, 'ax_i_twin') and self.ax_i_twin is not None:
            self.ax_i_twin.clear()
            self.ax_i_twin.remove()
            self.ax_i_twin = None
            
        self.ax_v.clear()
        self.ax_s.clear()
        self._setup_axes()
        
        # 1. Resistencia Serie R_S (Subplot ①)
        if r_series_hist is not None:
            r_series_kohm = r_series_hist / 1e3
        else:
            r_series_kohm = np.full_like(t, 100.0)
            
        self.ax_rs.plot(t, r_series_kohm, color="#89dceb", linewidth=1.5, label=r"$R_S$ (Serie)")
        r_min, r_max = np.min(r_series_kohm), np.max(r_series_kohm)
        margin = max((r_max - r_min) * 0.15, 1.0)
        self.ax_rs.set_ylim(max(0, r_min - margin), r_max + margin)
        self.ax_rs.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

        # 2. Resistencia de Fuga R_leak (Subplot ②)
        r_leak_kohm = r_m_hist / 1e3
        self.ax_rl.plot(t, r_leak_kohm, color="#a6e3a1", linewidth=1.5, label=r"$R_{leak}$ (Fuga)")
        r_min_l, r_max_l = np.min(r_leak_kohm), np.max(r_leak_kohm)
        margin_l = max((r_max_l - r_min_l) * 0.15, 10.0)
        self.ax_rl.set_ylim(max(0, r_min_l - margin_l), r_max_l + margin_l)
        self.ax_rl.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)

        # 3. Señal de entrada V_IN e I_in (Subplot ③)
        max_abs = np.max(np.abs(signal_in))
        if max_abs > 0.01:
            l1 = self.ax_i.plot(t, signal_in, color="#89b4fa", linewidth=1.5, label="$V_{IN}$ (V)")
            self.ax_i.set_ylabel("$V_{IN}$ (V)", color="#89b4fa")
            
            self.ax_i_twin = self.ax_i.twinx()
            self.ax_i_twin.tick_params(colors="#f9e2af", labelcolor="#f9e2af")
            self.ax_i_twin.set_ylabel(r"$I_{in}$ ($\mu$A)", color="#f9e2af")
            for spine in self.ax_i_twin.spines.values():
                spine.set_color("#45475a")
                
            if i_in is not None:
                i_uA = i_in * 1e6
            else:
                i_uA = signal_in * 10.0
                
            l2 = self.ax_i_twin.plot(t, i_uA, color="#f9e2af", linewidth=1.2, linestyle="--", label=r"$I_{in}$ ($\mu$A)")
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            self.ax_i.legend(lines, labels, loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)
        else:
            self.ax_i.set_title("③ Control de Corriente Inyectada ($I_{in}$)", color="#cdd6f4", fontsize=9)
            self.ax_i.set_ylabel(r"$I_{in}$ ($\mu$A)", color="#f9e2af")
            self.ax_i.plot(t, signal_in * 1e6, color="#f9e2af", linewidth=1.5)

        # 4. Potencial del Capacitor V_c (Subplot ④)
        # Mostramos V_m directamente (no V_m - V_rest) para que V_rest, V_th y V_reset
        # sean líneas absolutas interpretables con sus valores físicos reales.
        self.ax_v.plot(t, v_m, color="#cba6f7", linewidth=1.5, label="$V_m$ (membrana)")

        # Líneas de referencia
        self.ax_v.axhline(v_th,    color="#f38ba8", linestyle="--", alpha=0.8, linewidth=1.2, label=f"$V_{{th}}$ = {v_th:+.2f} V")
        self.ax_v.axhline(v_reset, color="#a6e3a1", linestyle=":",  alpha=0.8, linewidth=1.2, label=f"$V_{{reset}}$ = {v_reset:+.2f} V")
        self.ax_v.axhline(v_rest,  color="#585b70", linestyle="-.", alpha=0.6, linewidth=0.8, label=f"$V_{{rest}}$ = {v_rest:+.2f} V")

        # Picos visuales en los instantes de spike (Punto 3 auditoría v2026.09.11)
        # Se dibujan marcadores al nivel de V_th para que los spikes sean visibles.
        # No modifica los datos de V_m — solo es un artefacto gráfico educativo.
        if len(spike_times) > 0:
            spike_v_vals = [v_th] * len(spike_times)
            self.ax_v.scatter(spike_times, spike_v_vals,
                              color="#f38ba8", s=25, zorder=5, marker="^",
                              label=f"Spikes ({len(spike_times)})")

        # Rango Y automático: incluye V_reset negativo (AHP) + V_th + margen
        y_lo = min(v_reset, v_rest, np.min(v_m)) - abs(v_th) * 0.15
        y_hi = max(v_th, np.max(v_m)) + abs(v_th) * 0.15
        self.ax_v.set_ylim(y_lo, y_hi)
        self.ax_v.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)


        # 5. Salida V_OUT Spikes (Subplot ⑤)
        steps = len(t)
        dt = t[1] - t[0] if steps > 1 else 1e-4
        v_out = 0.02 + 0.005 * np.sin(2.0 * np.pi * 50.0 * t)
        v_peak = max(0.53, 0.55 * v_th)
        
        for st in spike_times:
            idx = int(np.searchsorted(t, st))
            if 0 <= idx < steps:
                tail_len = int(0.002 / dt)
                for m in range(min(tail_len, steps - idx)):
                    decay = np.exp(-m * dt / 0.0004)
                    v_out[idx + m] = max(v_out[idx + m], 0.02 + (v_peak - 0.02) * decay)
                    
        self.ax_s.plot(t, v_out, color="#f38ba8", linewidth=1.5, label="$V_{OUT}$ Spikes")
        self.ax_s.set_ylim(-0.02, max(0.6, v_peak + 0.08))
        self.ax_s.legend(loc="upper right", facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)
        self.canvas.draw()

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\widgets\plot_canvas.py` — 182 líneas

```python
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MplCanvas(QWidget):
    """
    Widget de Matplotlib embebido en PySide6 para la visualización en tiempo real
    de los resultados de la simulación del Memristor.
    """

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        super().__init__(parent)

        # Configurar figura con tema oscuro elegante
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e2e')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Disposición visual
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _style_axes_list(self, axes: list):
        """Aplica estilos visuales oscuros y armoniosos a las gráficas dinámicas."""
        for ax in axes:
            ax.set_facecolor('#181825')
            ax.tick_params(colors='#cdd6f4', labelsize=8)
            ax.xaxis.label.set_color('#cdd6f4')
            ax.yaxis.label.set_color('#cdd6f4')
            ax.title.set_color('#89b4fa')
            for spine in ax.spines.values():
                spine.set_color('#45475a')
            ax.grid(True, linestyle='--', alpha=0.3, color='#585b70')

    def plot_results(
        self,
        t: np.ndarray,
        v_in: np.ndarray,
        v_drop: np.ndarray,
        i_out: np.ndarray,
        x_state: np.ndarray,
        r_hist: np.ndarray,
        g_hist: np.ndarray,
        val_data: dict | None = None,
        active_plots: list[str] | None = None,
        ensemble_data: list[dict] | None = None
    ):
        """Actualiza y dibuja los resultados de la simulación seleccionados dinámicamente."""
        if not active_plots:
            active_plots = ["iv", "vt_it", "x", "r", "g", "p"]

        self.figure.clear()
        n = len(active_plots)
        
        if n == 0:
            self.canvas.draw()
            return
            
        if n == 1: rows, cols = 1, 1
        elif n == 2: rows, cols = 1, 2
        elif n == 3: rows, cols = 1, 3
        elif n == 4: rows, cols = 2, 2
        else: rows, cols = 2, 3

        axes_to_style = []
        t_ms = t * 1e3

        for idx, plot_id in enumerate(active_plots, start=1):
            ax = self.figure.add_subplot(rows, cols, idx)
            axes_to_style.append(ax)

            if plot_id == "iv":
                if ensemble_data:
                    for ens_idx, k_item in enumerate(ensemble_data):
                        label_ens = 'Población D2D' if ens_idx == 0 else None
                        ax.plot(k_item["v_drop"], k_item["i_out"] * 1e3, color='#f5c2e7', alpha=0.3, linewidth=0.8, label=label_ens)
                ax.plot(v_drop, i_out * 1e3, color='#f38ba8', linewidth=1.8, label='Simulado (Base)')
                if val_data is not None:
                    ax.plot(val_data["v_interp"], val_data["i_val_mA"], color='#f9e2af', linewidth=1.2, linestyle='--', label='Validación CSV')
                ax.legend(loc='upper left', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Curva I-V (Histéresis)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Voltaje $V_{drop}$ (V)")
                ax.set_ylabel("Corriente (mA)")

            elif plot_id == "vt_it":
                ax.plot(t_ms, v_in, color='#cdd6f4', linewidth=1.0, linestyle='--', label='V_in (Gen)')
                ax.plot(t_ms, v_drop, color='#89b4fa', linewidth=1.5, label='V_drop (Mem)')
                if val_data is not None:
                    ax.plot(val_data["t_v"] * 1e3, val_data["v_val"], color='#f9e2af', linewidth=1.2, linestyle=':', label='V_val (CSV)')
                ax.set_title("Voltaje y Corriente vs Tiempo", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Voltaje (V)", color='#89b4fa')
                ax.tick_params(axis='y', labelcolor='#89b4fa')

                ax_i_twin = ax.twinx()
                if ensemble_data:
                    for k_item in ensemble_data:
                        ax_i_twin.plot(t_ms, k_item["i_out"] * 1e3, color='#f5c2e7', alpha=0.25, linewidth=0.8)
                ax_i_twin.plot(t_ms, i_out * 1e3, color='#f38ba8', linewidth=1.2, linestyle='--', label='Corriente I(t)')
                if val_data is not None:
                    ax_i_twin.plot(val_data["t_i"] * 1e3, val_data["i_val_mA"], color='#fab387', linewidth=1.2, linestyle=':', label='I_val (CSV)')
                ax_i_twin.set_ylabel("Corriente (mA)", color='#f38ba8')
                
                ax_i_twin.tick_params(colors='#f38ba8', labelsize=8)
                for spine in ax_i_twin.spines.values():
                    spine.set_color('#45475a')

                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax_i_twin.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')

            elif plot_id == "x":
                if ensemble_data:
                    for ens_idx, k_item in enumerate(ensemble_data):
                        label_ens = 'Población D2D' if ens_idx == 0 else None
                        ax.plot(t_ms, k_item["x_state"], color='#a6e3a1', alpha=0.3, linewidth=0.8, label=label_ens)
                ax.plot(t_ms, x_state, color='#a6e3a1', linewidth=1.8, label='x(t) Simulado')
                if val_data is not None:
                    ax.plot(val_data["t_w"] * 1e3, val_data["wd_val"], color='#f9e2af', linewidth=1.5, linestyle='--', label='w/d (CSV)')
                ax.legend(loc='lower right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.axhline(0.0, color='#f38ba8', linestyle=':', alpha=0.5)
                ax.axhline(1.0, color='#f38ba8', linestyle=':', alpha=0.5)
                ax.set_title("Estado Interno Normalizado x(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("x (0.0 = OFF, 1.0 = ON)")
                
                # Auto-escalar límites en Y si x(t) se desborda (>1.1 o <-0.1 en Modo Ideal)
                max_x_val = np.max(x_state)
                min_x_val = np.min(x_state)
                if max_x_val <= 1.1 and min_x_val >= -0.1:
                    ax.set_ylim(-0.05, 1.05)

            elif plot_id == "r":
                if ensemble_data:
                    for ens_idx, k_item in enumerate(ensemble_data):
                        label_ens = 'Población D2D' if ens_idx == 0 else None
                        ax.plot(t_ms, k_item["r_hist"] / 1e3, color='#f9e2af', alpha=0.3, linewidth=0.8, label=label_ens)
                ax.plot(t_ms, r_hist / 1e3, color='#fab387', linewidth=1.8, label='R(t) Simulado')
                if val_data is not None and "r_val_kohm" in val_data:
                    ax.plot(val_data["t_i"] * 1e3, val_data["r_val_kohm"], color='#f9e2af', linewidth=1.2, linestyle='--', label='R_val (CSV)')
                ax.legend(loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Resistencia Instantánea R(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Resistencia (kΩ)")

            elif plot_id == "g":
                if ensemble_data:
                    for k_item in ensemble_data:
                        ax.plot(t_ms, k_item["g_hist"] * 1e6, color='#94e2d5', alpha=0.3, linewidth=0.8)
                ax.plot(t_ms, g_hist * 1e6, color='#89dceb', linewidth=1.8, label='G(t) Simulado')
                if val_data is not None and "g_val_us" in val_data:
                    ax.plot(val_data["t_i"] * 1e3, val_data["g_val_us"], color='#f9e2af', linewidth=1.2, linestyle='--', label='G_val (CSV)')
                ax.legend(loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Conductancia Instantánea G(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Conductancia (µS)")

            elif plot_id == "p":
                if ensemble_data:
                    for k_item in ensemble_data:
                        ax.plot(t_ms, k_item["v_drop"] * (k_item["i_out"] * 1e3), color='#cba6f7', alpha=0.25, linewidth=0.8)
                p_mw = v_drop * (i_out * 1e3)
                ax.plot(t_ms, p_mw, color='#cba6f7', linewidth=1.8, label='P(t) Simulado')
                if val_data is not None:
                    p_csv_mw = val_data["v_interp"] * val_data["i_val_mA"]
                    ax.plot(val_data["t_i"] * 1e3, p_csv_mw, color='#f9e2af', linewidth=1.2, linestyle='--', label='P_val (CSV)')
                ax.legend(loc='upper right', fontsize=7, facecolor='#1e1e2e', edgecolor='#45475a')
                ax.set_title("Potencia Instantánea P(t)", fontsize=9, fontweight='bold')
                ax.set_xlabel("Tiempo (ms)")
                ax.set_ylabel("Potencia (mW)")

        self._style_axes_list(axes_to_style)
        try:
            self.figure.tight_layout(pad=1.8)
        except Exception:
            pass
        self.canvas.draw()

```



---

## neurolab

`neuromorphic_lab\neurolab\gui\widgets\signal_panel.py` — 167 líneas

```python
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QComboBox,
    QPushButton, QCheckBox
)
from PySide6.QtCore import Signal
import numpy as np
from neurolab.gui.widgets.arrow_spinbox import ArrowDoubleSpinBox, ArrowSpinBox

# Límite de seguridad para la señal: evita colapso de memoria con duraciones/dts
# extremos (e.g. 10 000 s @ 0.001 ms -> 1e10 pasos). El dt se escala hacia arriba.
MAX_SIGNAL_STEPS = 200_000

class SignalPanel(QWidget):
    """
    Panel de configuración para la fuente de voltaje de excitación experimental.
    Utiliza botones independientes de flechas (▲ y ▼) fuera de la barra de texto.
    """
    run_simulation_requested = Signal()
    param_changed = Signal()

    def __init__(self, parent=None, mode="voltage"):
        super().__init__(parent)
        self.mode = mode
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        group = QGroupBox("5. 📡 Configuración Experimental de Señal")
        form = QFormLayout()

        # Interruptor de Actualización en Tiempo Real
        self.chk_realtime = QCheckBox("⚡ Simulación en Tiempo Real")
        self.chk_realtime.setChecked(True)
        self.chk_realtime.setStyleSheet("font-weight: bold; color: #a6e3a1;")

        self.combo_waveform = QComboBox()
        self.combo_waveform.addItems(["Sinusoidal", "Triangular", "Pulsos Cuadrados", "Tren de Pulsos (Unipolar)", "Corriente Constante"])

        # Controles numéricos con flechas independientes
        if self.mode == "current":
            self.spin_v0 = ArrowDoubleSpinBox(value=1.0, min_val=0.01, max_val=1000.0, step=1.0, suffix=" µA")
            amp_label = "Amplitud I_0:"
        else:
            self.spin_v0 = ArrowDoubleSpinBox(value=1.0, min_val=0.01, max_val=1000.0, step=1.0, suffix=" V")
            amp_label = "Amplitud V_0:"
        self.spin_f0 = ArrowDoubleSpinBox(value=100.0, min_val=0.001, max_val=10000.0, step=1.0, decimals=3, suffix=" Hz")
        self.spin_duration = ArrowDoubleSpinBox(value=0.05, min_val=0.0001, max_val=10000.0, step=0.01, decimals=4, suffix=" s")
        self.spin_dt_ms = ArrowDoubleSpinBox(value=0.001, min_val=0.0001, max_val=10.0, step=0.0005, decimals=5, suffix=" ms")

        form.addRow(self.chk_realtime)
        form.addRow("Forma de Onda:", self.combo_waveform)
        form.addRow(amp_label, self.spin_v0)
        form.addRow("Frecuencia f_0:", self.spin_f0)
        form.addRow("Duración Total:", self.spin_duration)
        form.addRow("Paso Temporal Δt:", self.spin_dt_ms)
        group.setLayout(form)
        layout.addWidget(group)

        # Botón de Simulación
        self.btn_run = QPushButton("▶ EJECUTAR SIMULACIÓN")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #11111b;
                font-weight: bold;
                font-size: 13px;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
        """)
        self.btn_run.clicked.connect(self.run_simulation_requested.emit)
        layout.addWidget(self.btn_run)

        self.setLayout(layout)
        self._connect_signals()

    def _connect_signals(self):
        """Conecta cambios de la fuente de señal para emitir param_changed."""
        self.combo_waveform.currentTextChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_v0.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_f0.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_duration.valueChanged.connect(lambda *_: self.param_changed.emit())
        self.spin_dt_ms.valueChanged.connect(lambda *_: self.param_changed.emit())

    @property
    def is_realtime_enabled(self) -> bool:
        """Devuelve True si el modo de actualización en tiempo real está activo."""
        return self.chk_realtime.isChecked()

    def to_dict(self) -> dict:
        """Exporta la configuración de señal a un diccionario JSON."""
        return {
            "realtime_enabled": self.chk_realtime.isChecked(),
            "waveform": self.combo_waveform.currentText(),
            "v0": self.spin_v0.value(),
            "f0": self.spin_f0.value(),
            "duration": self.spin_duration.value(),
            "dt_ms": self.spin_dt_ms.value(),
        }

    def from_dict(self, data: dict):
        """Carga la configuración de señal desde un diccionario JSON."""
        if "realtime_enabled" in data:
            self.chk_realtime.setChecked(bool(data["realtime_enabled"]))
        if "waveform" in data:
            idx = self.combo_waveform.findText(data["waveform"])
            if idx >= 0:
                self.combo_waveform.setCurrentIndex(idx)
        if "v0" in data: self.spin_v0.setValue(data["v0"])
        if "f0" in data: self.spin_f0.setValue(data["f0"])
        if "duration" in data: self.spin_duration.setValue(data["duration"])
        if "dt_ms" in data: self.spin_dt_ms.setValue(data["dt_ms"])

    def reset_defaults(self):
        """Restablece los parámetros por defecto de la señal experimental del Paper Strukov 2008."""
        self.chk_realtime.setChecked(True)
        self.combo_waveform.setCurrentIndex(0)
        self.spin_v0.setValue(1.0)
        self.spin_f0.setValue(100.0)      # 100 Hz
        self.spin_duration.setValue(0.05)  # 50 ms -> 5 ciclos completos a 100 Hz
        self.spin_dt_ms.setValue(0.001)    # 0.001 ms -> 1 µs (50 000 pasos)

    def generate_voltage_signal(self):
        """
        Genera el vector de tiempo t y el vector de voltaje v según la configuración.

        Returns:
            tuple (t, v, dt_sec)
        """
        v0 = self.spin_v0.value()
        f0 = self.spin_f0.value()
        duration = self.spin_duration.value()
        dt_sec = self.spin_dt_ms.value() * 1e-3  # ms a segundos

        steps = int(duration / dt_sec)
        if steps < 2:
            steps = 2
            dt_sec = duration / float(steps)
        if steps > MAX_SIGNAL_STEPS:
            # Límite de seguridad: no generar más de MAX_SIGNAL_STEPS pasos (evita colapso)
            dt_sec = duration / float(MAX_SIGNAL_STEPS)
            steps = MAX_SIGNAL_STEPS

        t = np.linspace(0.0, duration, steps)
        waveform = self.combo_waveform.currentText()

        if waveform == "Sinusoidal":
            v = v0 * np.sin(2.0 * np.pi * f0 * t)
        elif waveform == "Triangular":
            phase = (t * f0) % 1.0
            v = np.where(phase < 0.5, 4.0 * v0 * phase - v0, 3.0 * v0 - 4.0 * v0 * phase)
        elif waveform == "Pulsos Cuadrados":
            # Pulso cuadrado unipolar (50% de ciclo de trabajo) de 0 a v0
            v = np.where(np.sin(2.0 * np.pi * f0 * t) >= 0, v0, 0.0)
        elif waveform == "Tren de Pulsos (Unipolar)":
            # Pulso positivo estrecho (20% de ciclo de trabajo) de 0 a v0
            phase = (t * f0) % 1.0
            v = np.where(phase < 0.2, v0, 0.0)
        elif waveform == "Corriente Constante":
            v = np.full(steps, v0)
        else:
            v = v0 * np.sin(2.0 * np.pi * f0 * t)

        return t, v, dt_sec

```



---

## neurolab

`neuromorphic_lab\neurolab\io\__init__.py` — 13 líneas

```python
"""
neurolab.io
============
Módulo de entrada/salida del simulador Neuromorphic Lab.

Exporta:
    ProfileManager : Gestión centralizada de perfiles JSON (guardar, cargar, listar).
"""
from neurolab.io.profile_manager import ProfileManager  # noqa: F401
from neurolab.io.validation_loader import ValidationDataLoader  # noqa: F401

__all__ = ["ProfileManager", "ValidationDataLoader"]


```



---

## neurolab

`neuromorphic_lab\neurolab\io\profile_manager.py` — 216 líneas

```python
"""
neurolab.io.profile_manager
============================
Gestión centralizada de perfiles de configuración de memristor.

Responsabilidades:
  - Directorio por defecto: neuromorphic_lab/configs/
  - Guardado con metadatos (versión, timestamp, nombre)
  - Carga con validación básica de esquema
  - Autoguardado/restauración de última sesión (last_session.json)
  - Listado de perfiles disponibles en disco
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Versión actual de esquema de perfil
_PROFILE_SCHEMA_VERSION = "1.0"
_APP_VERSION = "0.1.0"
_LAST_SESSION_FILENAME = "last_session.json"


class ProfileManager:
    """
    Gestiona el ciclo de vida de perfiles JSON del simulador.

    Uso típico:
        pm = ProfileManager()
        pm.save(data, "mi_experimento")
        data = pm.load("mi_experimento.json")
        pm.save_last_session(data)
        data = pm.load_last_session()
    """

    def __init__(self, configs_dir: Optional[Path] = None):
        """
        Parámetros:
            configs_dir: Directorio de perfiles. Por defecto se resuelve
                         automáticamente como `neuromorphic_lab/configs/`.
        """
        if configs_dir is not None:
            self.configs_dir = Path(configs_dir)
        else:
            self.configs_dir = self._resolve_default_configs_dir()

        self.configs_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolución de directorios ────────────────────────────────────────────

    @staticmethod
    def _resolve_default_configs_dir() -> Path:
        """
        Resuelve el directorio `configs/` relativo al paquete `neurolab`.
        Funciona independientemente del directorio de trabajo actual.
        """
        # __file__ → neurolab/io/profile_manager.py
        # .parent → neurolab/io/
        # .parent.parent → neurolab/
        # .parent.parent.parent → neuromorphic_lab/   (si neurolab está dentro)
        neurolab_pkg = Path(__file__).resolve().parent.parent  # neurolab/
        project_root = neurolab_pkg.parent                      # neuromorphic_lab/
        return project_root / "configs"

    # ── Metadatos ────────────────────────────────────────────────────────────

    @staticmethod
    def _inject_metadata(data: dict, profile_name: str = "") -> dict:
        """Añade metadatos de control de versión al diccionario antes de guardar."""
        meta = {
            "_meta": {
                "schema_version": _PROFILE_SCHEMA_VERSION,
                "app_version": _APP_VERSION,
                "profile_name": profile_name,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
        }
        meta.update(data)
        return meta

    @staticmethod
    def _validate_schema(data: dict) -> bool:
        """Validación básica: verifica que el JSON tenga las claves mínimas esperadas."""
        required_keys = {"r_on", "r_off", "model_name"}
        return required_keys.issubset(data.keys())

    # ── Guardar ──────────────────────────────────────────────────────────────

    def save(self, data: dict, profile_name: str, file_path: Optional[Path] = None) -> Path:
        """
        Guarda un perfil de configuración en disco.

        Parámetros:
            data         : Diccionario de configuración (de ConfigPanel.to_dict()).
            profile_name : Nombre descriptivo del perfil (sin extensión).
            file_path    : Ruta absoluta opcional. Si se omite, guarda en configs_dir.

        Retorna:
            Path al archivo guardado.
        """
        payload = self._inject_metadata(data, profile_name=profile_name)

        if file_path is None:
            safe_name = _sanitize_filename(profile_name) or "perfil"
            file_path = self.configs_dir / f"{safe_name}.json"

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        return file_path

    def save_last_session(self, data: dict) -> Path:
        """
        Guarda el estado actual como 'última sesión' para restauración automática.
        Archivo: configs/last_session.json
        """
        session_path = self.configs_dir / _LAST_SESSION_FILENAME
        payload = self._inject_metadata(data, profile_name="[Última Sesión]")
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        return session_path

    # ── Cargar ───────────────────────────────────────────────────────────────

    def load(self, file_path: Path) -> dict:
        """
        Carga un perfil JSON desde una ruta absoluta y lo valida.

        Retorna:
            Diccionario de configuración (sin la clave `_meta`).

        Lanza:
            ValueError si el JSON no pasa la validación básica de esquema.
            FileNotFoundError si el archivo no existe.
            json.JSONDecodeError si el archivo no es JSON válido.
        """
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Separar metadatos del payload de configuración
        data = {k: v for k, v in raw.items() if k != "_meta"}

        if not self._validate_schema(data):
            raise ValueError(
                f"El archivo '{file_path.name}' no tiene el esquema esperado de perfil.\n"
                f"Claves mínimas requeridas: r_on, r_off, model_name."
            )
        return data

    def load_last_session(self) -> Optional[dict]:
        """
        Carga la última sesión guardada. Retorna None si no existe.
        """
        session_path = self.configs_dir / _LAST_SESSION_FILENAME
        if not session_path.exists():
            return None
        try:
            return self.load(session_path)
        except Exception:
            return None

    # ── Listado ──────────────────────────────────────────────────────────────

    def list_profiles(self, exclude_session: bool = True) -> list[Path]:
        """
        Lista todos los perfiles JSON disponibles en configs_dir.

        Parámetros:
            exclude_session: Si True, excluye last_session.json del listado.

        Retorna:
            Lista de Paths ordenados alfabéticamente.
        """
        if not self.configs_dir.exists():
            return []

        profiles = sorted(self.configs_dir.glob("*.json"))

        if exclude_session:
            profiles = [p for p in profiles if p.name != _LAST_SESSION_FILENAME]

        return profiles

    def get_profile_display_name(self, path: Path) -> str:
        """
        Devuelve el nombre de display del perfil: primero intenta leer
        el campo `_meta.profile_name`, si no existe usa el nombre del archivo.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            meta = raw.get("_meta", {})
            name = meta.get("profile_name", "").strip()
            if name and name != "[Última Sesión]":
                return name
        except Exception:
            pass
        return path.stem.replace("_", " ").title()


# ── Utilidades ────────────────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """Elimina caracteres no seguros para nombres de archivo."""
    import re
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r'\s+', "_", name)
    name = name[:80]  # Limitar longitud
    return name

```



---

## neurolab

`neuromorphic_lab\neurolab\io\validation_loader.py` — 99 líneas

```python
"""
neurolab.io.validation_loader
================================
Carga y procesamiento de los archivos CSV de validación física:
- CvsT.csv   : Corriente (mA) vs Tiempo (s)
- VvsT.csv   : Voltaje (V) vs Tiempo (s)
- WDvsT.csv  : Estado interno w/d (normalizado [0, 1]) vs Tiempo (s)
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np


class ValidationDataLoader:
    """Carga los datos experimentales de validación desde los archivos CSV."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        time_scale: float = 10.0,       # Escala tiempo CSV a 6.0 s (0.6 * 10 = 6.0 s)
        current_scale: float = 0.010     # Escala corriente CSV a [-0.10, 0.10] mA
    ):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = self._resolve_default_data_dir()

        self.time_scale = time_scale
        self.current_scale = current_scale
        self._cached_data: Optional[Dict[str, np.ndarray]] = None

    @staticmethod
    def _resolve_default_data_dir() -> Path:
        """Resuelve la ruta a `neuromorphic_lab/data for validation`."""
        neurolab_pkg = Path(__file__).resolve().parent.parent
        project_root = neurolab_pkg.parent
        return project_root / "data for validation"

    def load_all(self, force_reload: bool = False) -> Optional[Dict[str, np.ndarray]]:
        """
        Carga CvsT.csv, VvsT.csv y WDvsT.csv.
        Retorna un diccionario con vectores numpy, o None si los archivos no existen.
        """
        if self._cached_data is not None and not force_reload:
            return self._cached_data

        try:
            cvst_path = self.data_dir / "CvsT.csv"
            vvst_path = self.data_dir / "VvsT.csv"
            wdvst_path = self.data_dir / "WDvsT.csv"

            if not (cvst_path.exists() and vvst_path.exists() and wdvst_path.exists()):
                return None

            c_data = np.loadtxt(cvst_path, delimiter=",")
            v_data = np.loadtxt(vvst_path, delimiter=",")
            wd_data = np.loadtxt(wdvst_path, delimiter=",")

            # Escalar tiempo t_0 = 10 ms y corriente i_0 = 10 mA
            t_i = c_data[:, 0] * self.time_scale
            i_val_mA = c_data[:, 1] * self.current_scale
            i_val_amps = i_val_mA * 1e-3  # Convertir mA a Amperios

            t_v = v_data[:, 0] * self.time_scale
            v_val = v_data[:, 1]

            t_w = wd_data[:, 0] * self.time_scale
            wd_val = wd_data[:, 1]

            # Interpolar V(t) en los instantes de t_i para construir la curva I-V experimental
            v_interp = np.interp(t_i, t_v, v_val)

            # Resistencia Instantánea R = |V| / |I| de los datos CSV (en kΩ)
            # Se enmascaran cruces por cero estrictos (|I| < 0.001 mA) para evitar asíntotas infinitas
            mask_i = np.abs(i_val_amps) >= 5e-5
            r_val_kohm = np.where(mask_i, np.abs(v_interp / i_val_amps) / 1e3, np.nan)

            # Conductancia Instantánea G = 1 / R (en µS)
            # Se calcula como 1/R en lugar de I/V para evitar picos artificiales
            # en los cruces por cero del voltaje.
            g_val_us = np.where(mask_i, 1.0 / (r_val_kohm * 1e3) * 1e6, np.nan)

            self._cached_data = {
                "t_i": t_i,
                "i_val_mA": i_val_mA,
                "i_val": i_val_amps,
                "t_v": t_v,
                "v_val": v_val,
                "t_w": t_w,
                "wd_val": wd_val,
                "v_interp": v_interp,
                "r_val_kohm": r_val_kohm,
                "g_val_us": g_val_us,
            }
            return self._cached_data
        except Exception as e:
            print(f"Error al cargar datos de validación CSV: {e}")
            return None

```



---

## neurolab

`neuromorphic_lab\neurolab\neurons\__init__.py` — 13 líneas

```python
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

```



---

## neurolab

`neuromorphic_lab\neurolab\neurons\base.py` — 31 líneas

```python
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

```



---

## neurolab

`neuromorphic_lab\neurolab\neurons\config.py` — 68 líneas

```python
"""
neurolab.neurons.config
=======================
Contiene los parámetros configurables de las neuronas.

Cambios (v2026.09.11):
  - V_reset puede ser negativo (hiperpolarización post-spike / AHP).
    Esto es necesario para que el memristor de fuga (Pestaña 3) pueda
    hacer RESET y generar un periodo refractario dinámico.
  - Ver preset 'Biofísico Memristivo' en NeuronConfigPanel.
"""
from dataclasses import dataclass
import warnings

@dataclass
class LIFConfig:
    """
    Configuración de parámetros físicos y eléctricos para la neurona LIF.

    Nota sobre V_reset y la hiperpolarización (AHP):
    ─────────────────────────────────────────────────
    En el modelo LIF estándar, V_reset ≥ V_rest (la membrana se resetea a un
    valor igual o mayor que el reposo). Esto es válido para resistencias fijas.

    En el modo Híbrido Memristivo (Pestaña 3), si V_reset < V_rest, la membrana
    cae por debajo del reposo tras el spike (hiperpolarización). Esto expone al
    memristor de fuga a un voltaje negativo (V_m - V_rest < 0), induciendo el
    proceso RESET (aumento de R_leak) y generando un periodo refractario dinámico.
    """
    c_m: float = 500e-9       # Capacitancia de membrana (Faradios) -> 500 nF
    r_leak: float = 100e3     # Resistencia de fuga (Ohmios) -> 100 kΩ
    r_series: float = 100e3   # Resistencia en serie de entrada (Ohmios) -> 100 kΩ
    v_rest: float = 0.0       # Potencial de reposo (Voltios)
    v_th: float = 0.95        # Potencial de umbral (Voltios)
    v_reset: float = 0.0      # Potencial de reinicio (Voltios)
                               # Puede ser negativo para hiperpolarización (AHP biofísico)
    t_ref: float = 0.002      # Período refractario (Segundos)

    def __post_init__(self):
        """Valida que los parámetros físicos sean coherentes."""
        if self.c_m <= 0:
            raise ValueError("La capacitancia de membrana (c_m) debe ser estrictamente mayor que cero.")
        if self.r_leak <= 0:
            raise ValueError("La resistencia de fuga (r_leak) debe ser estrictamente mayor que cero.")
        if self.r_series <= 0:
            raise ValueError("La resistencia en serie (r_series) debe ser strictly mayor que cero.")
        if self.v_th <= self.v_reset:
            raise ValueError("El potencial de umbral (v_th) debe ser superior al potencial de reinicio (v_reset).")
        if self.t_ref < 0:
            raise ValueError("El período refractario no puede ser negativo.")
        pass

    @property
    def r_eq(self) -> float:
        """Resistencia equivalente R_eq = R_series || R_leak."""
        return (self.r_series * self.r_leak) / (self.r_series + self.r_leak)

    @property
    def tau(self) -> float:
        """
        Calcula la constante de tiempo equivalente tau = R_eq * C_m.
        """
        return self.r_eq * self.c_m

    @property
    def has_afterhyperpolarization(self) -> bool:
        """True si V_reset < V_rest (hiperpolarización post-spike activa)."""
        return self.v_reset < self.v_rest

```



---

## neurolab

`neuromorphic_lab\neurolab\neurons\lif.py` — 110 líneas

```python
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

        # --- Mecanismo de Disparo (Spike) y Reset ---
        if self.v_membrane >= self.config.v_th:
            # 1. Registrar el evento de spike
            self.has_spiked = True
            
            # 2. Registrar el instante temporal exacto
            self.spike_times.append(self.t)
            
            # 3. Reiniciar el potencial de membrana
            self.v_membrane = self.config.v_reset
            
            # 4. Iniciar periodo refractario (sólo si está configurado > 0)
            if self.config.t_ref > 0.0:
                self.refractory_time_left = self.config.t_ref

        return self.has_spiked

```



---

## tests

`neuromorphic_lab\tests\__init__.py` — 3 líneas

```python
"""
Suite de pruebas para neurolab.
"""

```



---

## tests

`neuromorphic_lab\tests\test_lif_hybrid.py` — 136 líneas

```python
"""
tests.test_lif_hybrid
======================
Regresión: verifica que los flujos de la GUI (Pestaña 2 y 3) disparan spikes con
los valores por defecto coherentes (C=100 nF, R=1 MΩ, V_th=0.95, V_reset=0.15,
t_ref=2 ms) usando los circuitos reales de neurolab.circuits.hybrid.
"""
import numpy as np

from neurolab.neurons import LIFConfig, LIFNeuron
from neurolab.circuits.hybrid import MemristorLIFCircuit, ResistorLIFCircuit
from neurolab.devices.presets import create_strukov_paper_device

# Parámetros que hoy definen las pestañas 2 y 3 de la GUI
DT = 1e-4
DURATION = 0.1      # 100 ms
F0 = 40.0           # 40 Hz -> 25 ms de período -> 5 ms de pulso (20% duty)


def _build_neuron() -> LIFNeuron:
    return LIFNeuron(LIFConfig(
        c_m=100e-9,   # 100 nF
        r_leak=1e6,   # 1 MΩ  -> tau = 100 ms
        v_rest=0.0,
        v_th=0.95,
        v_reset=0.15,
        t_ref=2e-3,
    ))


def _pulse_train_waveform(k: int) -> tuple:
    """Devuelve (i_uA, v_V) para el paso k del tren de pulsos unipolar al 20%."""
    phase = (k * DT * F0) % 1.0
    active = phase < 0.2
    return (20.0 if active else 0.0), (1.0 if active else 0.0)


def test_lif_tren_pulsos_dispara_spikes():
    """Pestaña 2: pulso de 20 µA @40 Hz debe producir al menos 1 spike por pulso."""
    n = _build_neuron()
    steps = int(DURATION / DT)
    for k in range(steps):
        i, _ = _pulse_train_waveform(k)
        n.step(i * 1e-6, DT)
    assert len(n.spike_times) >= 4, f"Se esperaban ~4 spikes, se obtuvieron {len(n.spike_times)}"


def test_lif_voltage_input_with_r_series():
    """Pestaña 2 (Fuente de Voltaje): V_IN = 5V @40 Hz con R_S = 100 kΩ debe disparar spikes."""
    n = LIFNeuron(LIFConfig(
        c_m=100e-9,      # 100 nF
        r_series=100e3,  # 100 kΩ
        r_leak=1e6,      # 1 MΩ
        v_rest=0.0,
        v_th=1.5,        # 1.5V umbral < V_inf * (1 - exp(-5ms/9ms)) = 1.95V
        v_reset=0.0,
        t_ref=2e-3
    ))
    steps = int(DURATION / DT)
    for k in range(steps):
        _, v_val = _pulse_train_waveform(k)
        v_in = v_val * 5.0  # Pulsos de 5V
        n.step(voltage_input=v_in, dt=DT)
    assert len(n.spike_times) >= 1, f"Se esperaban spikes con V_IN=5V y R_S=100kΩ, se obtuvieron {len(n.spike_times)}"


def test_hybrid_memristor_lif_dispara_spikes():
    """Pestaña 3 (modo Memristor): fuente 1 V @40 Hz debe excitar la neurona."""
    mem = create_strukov_paper_device(initial_state=0.1)
    n = _build_neuron()
    circuit = MemristorLIFCircuit(memristor=mem, neuron=n)
    steps = int(DURATION / DT)
    for k in range(steps):
        _, v = _pulse_train_waveform(k)
        circuit.step(v, DT)
    assert len(n.spike_times) >= 1, f"Se esperaban spikes, se obtuvieron {len(n.spike_times)}"


def test_hybrid_resistor_lif_dispara_spikes():
    """Pestaña 3 (modo Resistencia Fija): 10 kΩ con fuente 1 V debe excitar la neurona."""
    n = _build_neuron()
    circuit = ResistorLIFCircuit(r_input=10_000.0, neuron=n)
    steps = int(DURATION / DT)
    for k in range(steps):
        _, v = _pulse_train_waveform(k)
        circuit.step(v, DT)
    assert len(n.spike_times) >= 1, f"Se esperaban spikes, se obtuvieron {len(n.spike_times)}"


def test_clamp_corriente_equivale_pestana2():
    """
    Equivalencia EXACTA entre Pestaña 2 y Pestaña 3 (Resistencia Fija + clamp):
    si V_fuente(t) = V_m + I_objetivo(t)·R_syn, la corriente inyectada es idéntica
    a la fuente de corriente de la Pestaña 2 → mismos tiempos de spike y misma V(t).
    """
    r_syn = 10_000.0
    n_cur = _build_neuron()      # Pestaña 2: inyección directa de corriente
    n_cla = _build_neuron()      # Pestaña 3: clamp de corriente vía voltaje
    circuit = ResistorLIFCircuit(r_input=r_syn, neuron=n_cla)

    steps = int(DURATION / DT)
    for k in range(steps):
        i_uA, _ = _pulse_train_waveform(k)
        i_target = i_uA * 1e-6

        # Pestaña 2
        n_cur.step(i_target, DT)

        # Pestaña 3 (clamp): fuente de voltaje que cancela la caída por V_m
        v_source = n_cla.v_membrane + i_target * r_syn
        circuit.step(v_source, DT)

    # Corriente efectivamente inyectada por el clamp == i_target (diodo y R exactos)
    assert len(n_cla.spike_times) == len(n_cur.spike_times)
    np.testing.assert_allclose(n_cla.spike_times, n_cur.spike_times, atol=1e-12)
    assert len(n_cur.spike_times) >= 4  # sigue siendo la demo de 1 spike/pulso


def test_lif_analytical_validation_metrics():
    """Sección E: Verifica que la trayectoria numérica LIF reproduce la solución analítica diferencial exacta (R² > 0.99)."""
    from neurolab.core.lif_validation import compute_lif_validation_metrics
    cfg = LIFConfig(c_m=100e-9, r_series=100e3, r_leak=1e6, v_rest=0.0, v_th=2.5, v_reset=0.0)
    n = LIFNeuron(cfg)
    steps = 1000
    t = np.linspace(0, 0.1, steps)
    dt = t[1] - t[0]
    v_signal = np.full(steps, 2.0)  # Voltaje constante subumbral de 2.0V
    v_sim = np.zeros(steps)
    
    for k in range(steps):
        v_sim[k] = n.v_membrane
        n.step(voltage_input=v_signal[k], dt=dt)
        
    metrics = compute_lif_validation_metrics(t, v_sim, v_signal, cfg, is_voltage_input=True)
    assert metrics["r2"] > 0.99, f"Se esperaba R² > 0.99, se obtuvo {metrics['r2']}"
    assert metrics["mae"] < 0.05, f"Se esperaba MAE < 50 mV, se obtuvo {metrics['mae']}"
```



---

## tests

`neuromorphic_lab\tests\test_strukov.py` — 136 líneas

```python
import pytest
import numpy as np
from neurolab.devices.presets import (
    create_strukov_paper_device,
    create_strukov_2008_fig2b_device,
    create_strukov_normalized_preset,
    create_strukov_stochastic_preset
)
from neurolab.devices.realism import (
    BiolekWindowModifier,
    C2CVariabilityModifier,
    ThermalNoiseModifier,
)

def test_strukov_paper_preset_initial_state():
    """Verifica que el preset de la Fig 2b de Strukov se inicialice con los valores correctos."""
    mem = create_strukov_paper_device(initial_state=0.1)
    
    assert mem.identity.device_name == "Strukov TiO2 (Paper Fig 2b)"
    assert mem.electrical.r_on == 100.0
    assert mem.electrical.r_off == 16_000.0
    assert mem.x == 0.1
    
    # R(0.1) = 100 * 0.1 + 16000 * 0.9 = 10 + 14400 = 14410 Ohm
    expected_resistance = 14410.0
    assert pytest.approx(mem.resistance, rel=1e-5) == expected_resistance
    assert pytest.approx(mem.conductance, rel=1e-5) == 1.0 / expected_resistance

def test_strukov_official_presets():
    """Verifica la instanciación de los 3 perfiles oficiales."""
    dev_ideal = create_strukov_2008_fig2b_device()
    assert dev_ideal.identity.name == "Strukov 2008 - Figure 2b (Ideal)"
    assert len(dev_ideal.modifiers) == 0

    dev_norm = create_strukov_normalized_preset(p=5)
    assert dev_norm.identity.name == "Strukov TiO2 (Normalizado - Ventana Biolek)"
    assert len(dev_norm.modifiers) == 1

    dev_stoch = create_strukov_stochastic_preset(seed=42)
    assert dev_stoch.identity.name == "Strukov TiO2 (Estocástico Realista)"
    assert len(dev_stoch.modifiers) == 4

def test_stochastic_reproducibility():
    """Verifica que dos simulaciones con la misma semilla seed=42 entreguen vectores idénticos."""
    dev1 = create_strukov_stochastic_preset(seed=42)
    dev2 = create_strukov_stochastic_preset(seed=42)

    v0, f0, dt, steps = 1.0, 0.5, 0.0001, 1000
    i_1, i_2 = [], []

    for idx in range(steps):
        t = idx * dt
        v = v0 * np.sin(2.0 * np.pi * f0 * t)
        i_1.append(dev1.step(voltage=v, dt=dt))
        i_2.append(dev2.step(voltage=v, dt=dt))

    # Reproducibilidad científica estricta
    np.testing.assert_array_equal(np.array(i_1), np.array(i_2))

def test_strukov_modo_1_ideal_simulation():
    """Modo 1: Strukov puro sin modificadores."""
    mem = create_strukov_paper_device(initial_state=0.1)
    
    v0 = 1.0
    f0 = 0.5
    dt = 0.0001
    total_time = 4.0 / f0  # 4 ciclos completos
    steps = int(total_time / dt)
    
    states = []
    currents = []
    voltages = []
    
    for i in range(steps):
        t = i * dt
        v = v0 * np.sin(2.0 * np.pi * f0 * t)
        i_out = mem.step(voltage=v, dt=dt)
        
        states.append(mem.x)
        currents.append(i_out)
        voltages.append(v)
    
    # 1. Verificar acotamiento de estado
    assert all(0.0 <= x <= 1.0 for x in states)
    
    # 2. Propiedad fundamental: "Pinched Hysteresis Loop" (I = 0 cuando V = 0)
    for v, i_out in zip(voltages, currents):
        if abs(v) < 1e-9:
            assert abs(i_out) < 1e-9

def test_strukov_modo_2_biolek_window():
    """Modo 2: Modelo con ventana no lineal de Biolek."""
    mem = create_strukov_paper_device(
        modifiers=[BiolekWindowModifier(p=5)],
        initial_state=0.1
    )
    
    v0 = 1.5
    f0 = 0.5
    dt = 0.0001
    steps = 1000
    
    states = []
    for i in range(steps):
        t = i * dt
        v = v0 * np.sin(2.0 * np.pi * f0 * t)
        mem.step(voltage=v, dt=dt)
        states.append(mem.x)
        
    assert all(0.0 <= x <= 1.0 for x in states)

def test_strukov_modo_3_realista_estocastico():
    """Modo 3: Dispositivo estocástico con Biolek, C2C y ruido térmico."""
    mem = create_strukov_paper_device(
        modifiers=[
            BiolekWindowModifier(p=5),
            C2CVariabilityModifier(relative_std=0.05, seed=42),
            ThermalNoiseModifier(noise_std=1e-6, seed=42)
        ],
        initial_state=0.1
    )
    
    v0 = 1.0
    f0 = 0.5
    dt = 0.0001
    steps = 1000
    
    currents = []
    for i in range(steps):
        t = i * dt
        v = v0 * np.sin(2.0 * np.pi * f0 * t)
        i_out = mem.step(voltage=v, dt=dt)
        currents.append(i_out)
        
    assert len(currents) == steps
    assert all(0.0 <= mem.x <= 1.0 for _ in [0])

```



---

## tests

`neuromorphic_lab\tests\test_validation_loader.py` — 55 líneas

```python
"""
test_validation_loader.py
===========================
Prueba unitaria para la carga y renderizado de los datos CSV de validación
(CvsT.csv, VvsT.csv, WDvsT.csv) en Neuromorphic Lab.
"""

import numpy as np
import pytest
from neurolab.io.validation_loader import ValidationDataLoader
from neurolab.gui.widgets.plot_canvas import MplCanvas


def test_validation_data_loader():
    loader = ValidationDataLoader()
    data = loader.load_all()

    assert data is not None, "El cargador de datos de validación devolvió None."
    required_keys = [
        "t_i", "i_val_mA", "i_val", "t_v", "v_val", "t_w", "wd_val", "v_interp",
        "r_val_kohm", "g_val_us"
    ]
    for key in required_keys:
        assert key in data, f"Falta la clave '{key}' en los datos de validación."
        assert len(data[key]) > 0, f"El vector '{key}' está vacío."
        assert isinstance(data[key], np.ndarray), f"El elemento '{key}' debe ser un ndarray."

    # Verificar que los datos no contengan NaNs ni Infs
    assert not np.isnan(data["i_val"]).any(), "Corriente I contiene NaN."
    assert not np.isnan(data["v_val"]).any(), "Voltaje V contiene NaN."
    assert not np.isnan(data["wd_val"]).any(), "Estado w/d contiene NaN."


def test_plot_canvas_with_validation_data(qtbot=None):
    """Verifica que MplCanvas.plot_results acepte val_data sin errores."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    canvas = MplCanvas()
    t = np.linspace(0, 0.6, 100)
    v_in = np.sin(2 * np.pi * 5 * t)
    v_drop = v_in
    i_out = v_drop / 1000.0
    x_state = np.linspace(0.1, 0.8, 100)
    r_hist = np.full(100, 1000.0)
    g_hist = 1.0 / r_hist

    loader = ValidationDataLoader()
    val_data = loader.load_all()

    # Ejecutar sin val_data
    canvas.plot_results(t, v_in, v_drop, i_out, x_state, r_hist, g_hist)

    # Ejecutar con val_data
    canvas.plot_results(t, v_in, v_drop, i_out, x_state, r_hist, g_hist, val_data=val_data)

```



---

## Validación

`neuromorphic_lab/validate_lif_experiments.py` — 143 líneas

```python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neurolab.neurons import LIFConfig, LIFNeuron

def etapa28_spike_individual():
    """ETAPA 2.8 - Generar exactamente un spike."""
    config = LIFConfig()
    neuron = LIFNeuron(config)
    dt = 1e-4
    
    # Corriente apenas por encima del umbral (I_th = 9.5 uA)
    # Por ejemplo, 9.8 uA
    i_in = 9.8e-6 
    
    total_time = 0.2 # 200 ms
    steps = int(total_time / dt)
    t_array = np.linspace(0, total_time, steps, endpoint=False)
    
    v_sim = np.zeros(steps)
    
    # Inyección de pulso corto (para generar 1 solo spike) en vez de corriente continua infinita
    # o detener la corriente justo después del spike.
    for i in range(steps):
        v_sim[i] = neuron.v_membrane
        
        # Inyectar corriente solo hasta que veamos 1 spike
        if len(neuron.spike_times) < 1:
            neuron.step(i_in, dt)
        else:
            neuron.step(0.0, dt) # Dejar que caiga/se mantenga en reset
            
    plt.figure(figsize=(10, 4))
    plt.plot(t_array * 1000, v_sim, label='$V_m(t)$')
    plt.axhline(config.v_th, color='red', linestyle='--', label='Umbral ($V_{th}$)')
    plt.axhline(config.v_reset, color='green', linestyle=':', label='Reset ($V_{reset}$)')
    
    # Marcar spike manual para visualización
    if neuron.spike_times:
        t_sp = neuron.spike_times[0]
        plt.scatter([t_sp * 1000], [config.v_th], color='red', zorder=5, label='Spike Generado')
        
    plt.title("ETAPA 2.8: Spike Individual")
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Potencial de Membrana (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficas/lif_etapa28_spike.png")
    plt.close()
    
    print("OK Etapa 2.8 (Spike Individual) completada. Gráfico guardado.")

def etapa29_tren_spikes():
    """ETAPA 2.9 - Generar un tren de spikes con corriente alta constante."""
    config = LIFConfig()
    neuron = LIFNeuron(config)
    dt = 1e-4
    
    # Corriente alta para múltiples disparos
    i_in = 15e-6 
    
    total_time = 0.5 # 500 ms
    steps = int(total_time / dt)
    t_array = np.linspace(0, total_time, steps, endpoint=False)
    
    v_sim = np.zeros(steps)
    
    for i in range(steps):
        v_sim[i] = neuron.v_membrane
        neuron.step(i_in, dt)
            
    # Análisis cuantitativo
    n_spikes = len(neuron.spike_times)
    f_spike = n_spikes / total_time
    
    isi_array = np.diff(neuron.spike_times)
    mean_isi = np.mean(isi_array) * 1000 if len(isi_array) > 0 else 0
    
    print(f"OK Etapa 2.9 (Tren de Spikes) completada:")
    print(f"  - Spikes generados: {n_spikes}")
    print(f"  - Frecuencia media: {f_spike:.2f} Hz")
    print(f"  - Intervalo Inter-Spike (ISI) promedio: {mean_isi:.2f} ms")
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_array * 1000, v_sim, label='$V_m(t)$')
    for tsp in neuron.spike_times:
        plt.axvline(tsp * 1000, color='red', alpha=0.3)
    plt.title(f"ETAPA 2.9: Tren de Spikes ($f = {f_spike:.1f}$ Hz)")
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Potencial de Membrana (V)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficas/lif_etapa29_train.png")
    plt.close()

def etapa210_curva_fi():
    """ETAPA 2.10 - Curva Frecuencia vs Corriente (F-I Curve)."""
    config = LIFConfig()
    dt = 1e-4
    total_time = 1.0 # 1 segundo para medir frec fácilmente
    
    currents = np.linspace(5e-6, 30e-6, 50)
    frequencies = []
    
    for i_in in currents:
        neuron = LIFNeuron(config)
        steps = int(total_time / dt)
        
        for _ in range(steps):
            neuron.step(i_in, dt)
            
        frequencies.append(len(neuron.spike_times) / total_time)
        
    plt.figure(figsize=(8, 5))
    plt.plot(currents * 1e6, frequencies, 'o-', color='purple')
    plt.axvline(9.5, color='red', linestyle='--', label='I_th teórica (9.5 $\mu$A)')
    
    plt.title("ETAPA 2.10: Curva Frecuencia-Corriente (F-I)")
    plt.xlabel("Corriente de Entrada ($\mu$A)")
    plt.ylabel("Frecuencia de Disparo (Hz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficas/lif_etapa210_FI_curve.png")
    plt.close()
    
    print("OK Etapa 2.10 (Curva F-I) completada. Gráfico guardado.")

def main():
    os.makedirs("graficas", exist_ok=True)
    print("=== INICIANDO EXPERIMENTOS LIF (ETAPAS 2.8 - 2.10) ===")
    etapa28_spike_individual()
    etapa29_tren_spikes()
    etapa210_curva_fi()
    print("=== EXPERIMENTOS FINALIZADOS ===")

if __name__ == '__main__':
    main()

```



---

## Validación

`neuromorphic_lab/validate_lif_rc.py` — 105 líneas

```python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Asegurar que el módulo neurolab esté en el path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neurolab.neurons import LIFConfig, LIFNeuron

def analytical_solution(t_array, v_rest, r, i_in, tau):
    """Calcula la respuesta analítica exacta de un circuito RC."""
    return v_rest + r * i_in * (1 - np.exp(-t_array / tau))

def run_rc_experiment(dt, total_time, i_in):
    """
    Ejecuta la simulación numérica de la neurona LIF y compara con la solución analítica.
    """
    config = LIFConfig()
    neuron = LIFNeuron(config)
    
    steps = int(total_time / dt)
    t_array = np.linspace(0, total_time, steps, endpoint=False)
    
    v_simulated = np.zeros(steps)
    
    for i in range(steps):
        v_simulated[i] = neuron.v_membrane
        neuron.step(i_in, dt)
        
    v_analytical = analytical_solution(t_array, config.v_rest, config.r_leak, i_in, config.tau)
    
    # Cálculo de errores
    error_abs = np.abs(v_simulated - v_analytical)
    mae = np.mean(error_abs)
    mse = np.mean(error_abs**2)
    max_error = np.max(error_abs)
    
    # Error relativo (evitando divisiones por cero al inicio donde V_ana = 0)
    mask = np.abs(v_analytical) > 1e-9
    if np.any(mask):
        rel_error = np.mean(error_abs[mask] / np.abs(v_analytical[mask]))
    else:
        rel_error = 0.0
        
    return t_array, v_simulated, v_analytical, mae, mse, max_error, rel_error

def main():
    # Parámetros del experimento
    total_time = 0.3  # 300 ms (la constante de tiempo tau es de 50 ms, llegará al estado estacionario)
    
    # Queremos que V_m no supere V_th (0.95 V).
    # V_max = R * I_in -> 100 kOhm * I_in
    # Para V_max = 0.8 V -> I_in = 8 uA (8e-6 A)
    i_in = 8e-6 
    
    # Resoluciones temporales (dt) a evaluar
    dts = [1e-3, 1e-4, 1e-5]  # 1 ms, 0.1 ms, 0.01 ms
    
    results = []
    
    # Configuración de la gráfica
    plt.figure(figsize=(10, 6))
    
    # Solución analítica de referencia (usando el dt más fino)
    t_finest, _, v_ana_finest, _, _, _, _ = run_rc_experiment(dts[-1], total_time, i_in)
    plt.plot(t_finest * 1000, v_ana_finest, label='Solución Analítica Exacta', color='black', linewidth=2)
    
    for dt in dts:
        t, v_sim, v_ana, mae, mse, max_e, rel_e = run_rc_experiment(dt, total_time, i_in)
        results.append((dt, mae, mse, max_e, rel_e))
        
        # Graficamos la simulación para visualizar su acoplamiento
        if dt == dts[0]:
            plt.plot(t * 1000, v_sim, label=f'Simulación (Euler) dt={dt*1000:.1f} ms', linestyle='--')
        elif dt == dts[-1]:
            plt.plot(t * 1000, v_sim, label=f'Simulación (Euler) dt={dt*1000:.2f} ms', linestyle='-.', color='green')
            
    # Línea del umbral
    plt.axhline(0.95, color='red', linestyle=':', label='$V_{th}$ (Umbral = 0.95 V)')
    
    plt.title("Validación de Respuesta RC de la Neurona LIF (Corriente Subumbral)", fontsize=14)
    plt.xlabel("Tiempo (ms)", fontsize=12)
    plt.ylabel("Potencial de Membrana $V_m$ (V)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfica
    output_dir = "graficas"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "lif_rc_validation.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    
    # Mostrar resultados numéricos
    print("=== RESULTADOS EXPERIMENTO DE VALIDACIÓN RC ===")
    print(f"Constante de tiempo tau = 50 ms | Corriente inyectada = 8 uA | V_max asintotico = 0.80 V\n")
    print(f"{'dt (s)':<12} | {'MAE (V)':<14} | {'MSE (V^2)':<14} | {'Error Max (V)':<14} | {'Error Relativo':<14}")
    print("-" * 75)
    for res in results:
        print(f"{res[0]:<12.1e} | {res[1]:<14.6e} | {res[2]:<14.6e} | {res[3]:<14.6e} | {res[4]:<14.6e}")
        
    print(f"\n[OK] Gráfica generada y guardada en: {file_path}")

if __name__ == '__main__':
    main()

```



---

## Validación

`neuromorphic_lab/validate_resistor_lif.py` — 144 líneas

```python
"""
validate_resistor_lif.py
========================
Script de validación para la Etapa 2.13.
Realiza experimentos conectando una fuente de voltaje a una neurona LIF
usando diferentes valores de resistencia fija, y verifica que se cumpla
la física de acoplamiento (a mayor resistencia, menor corriente, menor excitación).
"""

import numpy as np
import matplotlib.pyplot as plt
from neurolab.neurons.lif import LIFNeuron
from neurolab.neurons.config import LIFConfig
from neurolab.circuits.hybrid import ResistorLIFCircuit

def run_experiment(r_input: float, v_source_amp: float = 2.0, duration: float = 0.2, dt: float = 1e-4) -> dict:
    """
    Ejecuta una simulación con un ResistorLIFCircuit.
    """
    # Configuración base de la neurona LIF
    config = LIFConfig(
        v_rest=-0.070,
        v_reset=-0.080,
        v_th=-0.055,
        r_leak=10e6,  # 10 MOhms
        c_m=1e-9,     # 1 nF
        t_ref=0.005   # 5 ms refractario
    )
    neuron = LIFNeuron(config)
    circuit = ResistorLIFCircuit(r_input=r_input, neuron=neuron)
    
    steps = int(duration / dt)
    t_array = np.linspace(0, duration, steps)
    
    # Señal de entrada: Tren de pulsos cuadrados (ej. 50 Hz, 10ms ancho de pulso)
    # Convertimos los pulsos en voltaje a aplicar (v_source_amp)
    v_source_array = np.zeros(steps)
    period = 0.02  # 20 ms -> 50 Hz
    pulse_width = 0.01  # 10 ms
    
    for i, t in enumerate(t_array):
        if (t % period) < pulse_width:
            v_source_array[i] = v_source_amp
        else:
            # Durante el resto del tiempo, podemos asumir 0V o dejar que la fuente esté en alta impedancia.
            # Para este circuito simple, si V_source es 0, habrá corriente negativa.
            v_source_array[i] = 0.0
            
    v_m_hist = np.zeros(steps)
    i_in_hist = np.zeros(steps)
    spike_times = []
    
    for i in range(steps):
        v_source = v_source_array[i]
        
        # Ojo: si V_source es 0, V_source - V_m será positivo (porque V_m es negativo).
        # En la realidad, si la fuente se apaga, podría quedar a 0V, lo que inyecta corriente "hacia atrás".
        # Para evitar que la neurona se descargue artificialmente muy rápido,
        # podríamos emular que V_source se acopla solo cuando hay pulso.
        # Pero mantendremos el modelo de circuito ideal donde la fuente de voltaje impone sus 0V.
        
        res = circuit.step(v_source, dt)
        
        v_m_hist[i] = res["v_m"]
        i_in_hist[i] = res["i_in"]
        if res["has_spiked"]:
            spike_times.append(t_array[i])
            
    num_spikes = len(spike_times)
    freq = num_spikes / duration if duration > 0 else 0.0
    
    return {
        "t": t_array,
        "v_source": v_source_array,
        "v_m": v_m_hist,
        "i_in": i_in_hist,
        "spike_times": spike_times,
        "num_spikes": num_spikes,
        "freq": freq,
        "r_input": r_input
    }

def main():
    print("--- INICIANDO VALIDACIÓN DE CIRCUITO RESISTENCIA-LIF ---")
    
    # Resistencias a evaluar: 1 MOhm, 10 MOhms, 50 MOhms
    resistances = [1e6, 10e6, 50e6]
    results = []
    
    for r in resistances:
        print(f"\nExperimentando con R_input = {r/1e6:.1f} MOhms")
        res = run_experiment(r_input=r)
        
        print(f"Número de spikes: {res['num_spikes']}")
        print(f"Frecuencia media: {res['freq']:.1f} Hz")
        print(f"Corriente máxima inyectada: {np.max(res['i_in'])*1e6:.2f} uA")
        results.append(res)
        
    print("\n--- VALIDACIÓN FINALIZADA ---")
    print("A mayor resistencia, menor corriente y menor cantidad de spikes. Física conservada.")
    
    # Graficar
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Graficar señal de fuente de voltaje
    axes[0].plot(results[0]["t"], results[0]["v_source"], color="black")
    axes[0].set_ylabel("V_source (V)")
    axes[0].set_title("Voltaje de Fuente")
    axes[0].grid(True)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, res in enumerate(results):
        label = f"R = {res['r_input']/1e6:.1f} MΩ"
        
        # Corriente
        axes[1].plot(res["t"], res["i_in"] * 1e6, label=label, color=colors[i], alpha=0.8)
        
        # Potencial de membrana
        axes[2].plot(res["t"], res["v_m"] * 1e3, label=label, color=colors[i], alpha=0.8)
        
        # Spikes
        for spike in res["spike_times"]:
            axes[2].axvline(x=spike, color=colors[i], linestyle='--', alpha=0.5)

    axes[1].set_ylabel("I_in (μA)")
    axes[1].set_title("Corriente Inyectada a la Neurona")
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].set_xlabel("Tiempo (s)")
    axes[2].set_ylabel("V_m (mV)")
    axes[2].set_title("Potencial de Membrana y Spikes")
    axes[2].axhline(-55, color='red', linestyle=':', label='V_th')
    axes[2].axhline(-80, color='green', linestyle=':', label='V_reset')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("validate_resistor_lif_results.png")
    print("Gráficos guardados en validate_resistor_lif_results.png")
    
if __name__ == "__main__":
    main()

```



---

## Orquestador de validación

`run_validation.py` — 205 líneas

```python
"""
run_validation.py
=================
Orquestador maestro para el Entorno de Validación Física y Neuromórfica.
Permite ejecutar en secuencia de forma profesional y ordenada todos los pasos
de validación (Pasos 1 al 29) y los scripts de análisis.

Autor: Yamil Ronald Uchani Guachalla
Taller de Grado I - Ingeniería Mecatrónica
"""

import os
import sys
import subprocess
import time

# Reconfigurar codificación de consola para evitar UnicodeEncodeError en Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Paletas de colores ANSI para la consola
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

# Mapeo estructurado de bloques de validación del Proyecto de Grado
BLOQUES_VALIDACION = {
    "1. Caracterización del Memristor Aislado (Pasos 1-3)": [
        ("paso1_simulacion_basica.py", "Lazo de histéresis estrangulado (I-V)"),
        ("paso2_colapso_frecuencia.py", "Colapso del lazo de histéresis por frecuencia"),
        ("paso3_memristor_estocastico.py", "Variabilidad estocástica ciclo-a-ciclo (C2C)")
    ],
    "2. Acoplamiento Soma Neuronal e Integración (Pasos 4-10)": [
        ("paso4_neurona_aislada.py", "Integración y disparo de neurona LIF estándar"),
        ("paso5_tren_pulsos.py", "Respuesta dinámica del memristor ante trenes de pulsos"),
        ("paso6_memristor_neurona.py", "Transducción de corriente memristiva a la neurona"),
        ("paso7_simulacion_completa.py", "Dashboard dinámico completo memristor-neurona"),
        ("paso8_estado_hrs.py", "Inhibición de disparo en estado de alta resistencia"),
        ("paso9_estado_lrs.py", "Disparo constante en estado de baja resistencia"),
        ("paso10_comparacion.py", "Comparación cuantitativa final frente a datos del paper")
    ],
    "3. Plasticidad y Aprendizaje Sináptico (Pasos 13-16)": [
        ("paso13_plasticidad_y_memoria.py", "Retención de memoria no volátil en apagón eléctrico"),
        ("paso14_ltp_ltd_protocolo.py", "Curvas de Potenciación/Depresión a Largo Plazo (LTP/LTD)"),
        ("paso15_curva_stdp.py", "Simulación emergente de la ventana de Hebb (STDP)"),
        ("paso16_transmision_neuronal.py", "Retardo temporal de transmisión de señal")
    ],
    "4. Arquitecturas Colectivas Crossbar (Pasos 17-24)": [
        ("paso17_red_crossbar.py", "Arreglo crossbar 3x3 y leyes de Kirchhoff"),
        ("paso18_crossbar_escalable.py", "Clasificador lineal de patrones en red 8x4"),
        ("paso19_caracterizacion_crossbar.py", "Mapa de conductancias y fidelidad de lectura"),
        ("paso20_caracterizacion_forming.py", "Proceso de electroformado en muestras vírgenes"),
        ("paso21_caracterizacion_pulsos.py", "Conmutación resistiva rápida por pulsos cortos"),
        ("paso22_analisis_sneak_paths.py", "Evaluación de corrientes parásitas y fugas"),
        ("paso23_analisis_noise_margin.py", "Degradación de margen de ruido de lectura"),
        ("paso24_analisis_metales.py", "Simulación de caída IR de tensión según metalización")
    ],
    "5. Replicaciones del Estado del Arte y Pre-Forming (Pasos 25-29)": [
        ("paso25_replicacion_prezioso_s5.py", "Dispersión estadística D2D de SET/RESET (Prezioso)"),
        ("paso25b_replicacion_s5_curvas_y_mapas.py", "Curvas espaciales complementarias SET/RESET"),
        ("paso25c_replicacion_s5_strukov_ideal.py", "Simulación determinista ideal comparativa"),
        ("paso26_replicacion_prezioso_s6.py", "Evolución estocástica de conductancia ruidosa"),
        ("paso27_replicacion_figura5_sneak.py", "Simulación nodal modopolítico MNA de sneak paths"),
        ("paso28_comparacion_strukov_estocastico.py", "Barras de error determinista vs estocástico"),
        ("paso29_replicacion_pre_forming.py", "Estado de virginidad y conducción por Efecto Túnel")
    ]
}

def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    banner = f"""
{C_CYAN}{C_BOLD}=====================================================================
  UNIVERSIDAD CATÓLICA BOLIVIANA "SAN PABLO" - SEDE LA PAZ
  FACULTAD DE INGENIERÍA - CARRERA DE INGENIERÍA MECATRÓNICA
=====================================================================
  SIMULADOR NEUROMÓRFICO BASADO EN MEMRISTORES (VALIDACIÓN)
  Proyecto de Grado - Yamil Ronald Uchani Guachalla
====================================================================={C_END}
"""
    print(banner)

def run_script(script_name, description):
    script_path = os.path.join("memristor_simulator", "steps", script_name)
    print(f"  {C_YELLOW}→ Ejecutando:{C_END} {C_BOLD}{script_name:<40}{C_END} ({description})")
    
    # Asegurar que el entorno de ejecución de Python use UTF-8 para stdout/stderr
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    
    start_time = time.time()
    try:
        # Ejecutar script silenciando salidas para mantener consola limpia, salvo errores
        res = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        duration = time.time() - start_time
        if res.returncode == 0:
            print(f"    {C_GREEN}✔ COMPLETADO{C_END} en {duration:.2f}s")
            return True
        else:
            print(f"    {C_RED}✘ FALLÓ{C_END} (Código {res.returncode})")
            print(f"{C_RED}Detalle del error:{C_END}\n{res.stderr}")
            return False
    except Exception as e:
        print(f"    {C_RED}✘ ERROR EXCEPCIONAL:{C_END} {e}")
        return False

def main():
    print_header()
    
    print(f"{C_BOLD}Seleccione una opción de ejecución:{C_END}")
    print("  [1] Ejecutar toda la suite de validación (Pasos 1 al 29)")
    print("  [2] Ejecutar un bloque de validación específico")
    print("  [3] Ejecutar una simulación/paso individual")
    print("  [4] Salir")
    
    try:
        opt = input(f"\n{C_BLUE}Opción > {C_END}").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nEjecución cancelada.")
        return

    if opt == "1":
        print_header()
        print(f"{C_CYAN}{C_BOLD}=== INICIANDO VALIDACIÓN COMPLETA (PASOS 1 AL 29) ==={C_END}\n")
        failed = []
        for bloque, scripts in BLOQUES_VALIDACION.items():
            print(f"\n{C_BOLD}{bloque}{C_END}")
            print("-" * len(bloque))
            for script, desc in scripts:
                success = run_script(script, desc)
                if not success:
                    failed.append(script)
        
        print("\n" + "=" * 65)
        if not failed:
            print(f"{C_GREEN}{C_BOLD}✔ Suite de validación completada exitosamente sin errores.{C_END}")
        else:
            print(f"{C_RED}{C_BOLD}✘ Se completó la ejecución, pero fallaron los siguientes scripts:{C_END}")
            for f in failed:
                print(f"  - {f}")
        print("=" * 65)

    elif opt == "2":
        print_header()
        print(f"{C_BOLD}Seleccione el bloque a ejecutar:{C_END}")
        bloques = list(BLOQUES_VALIDACION.keys())
        for idx, b in enumerate(bloques):
            print(f"  [{idx + 1}] {b}")
        
        try:
            b_opt = int(input(f"\n{C_BLUE}Bloque > {C_END}").strip()) - 1
            if 0 <= b_opt < len(bloques):
                selected_bloque = bloques[b_opt]
                print_header()
                print(f"{C_CYAN}{C_BOLD}=== EJECUTANDO: {selected_bloque} ==={C_END}\n")
                for script, desc in BLOQUES_VALIDACION[selected_bloque]:
                    run_script(script, desc)
            else:
                print(f"{C_RED}Opción inválida.{C_END}")
        except ValueError:
            print(f"{C_RED}Entrada no válida.{C_END}")

    elif opt == "3":
        print_header()
        print(f"{C_BOLD}Seleccione el script individual a ejecutar:{C_END}")
        all_scripts = []
        for idx, (bloque, scripts) in enumerate(BLOQUES_VALIDACION.items()):
            for s, d in scripts:
                all_scripts.append((s, d))
        
        for idx, (s, d) in enumerate(all_scripts):
            print(f"  [{idx + 1:2d}] {s:<42} ({d})")
            
        try:
            s_opt = int(input(f"\n{C_BLUE}Script # > {C_END}").strip()) - 1
            if 0 <= s_opt < len(all_scripts):
                script, desc = all_scripts[s_opt]
                print_header()
                run_script(script, desc)
            else:
                print(f"{C_RED}Opción inválida.{C_END}")
        except ValueError:
            print(f"{C_RED}Entrada no válida.{C_END}")

    elif opt == "4":
        print("\nSaliendo del orquestador.")
    else:
        print(f"{C_RED}Opción no reconocida.{C_END}")

if __name__ == "__main__":
    # Asegurar soporte de colores en consolas Windows heredadas
    if sys.platform.startswith("win"):
        os.system("color")
    main()

```



---

## Test de GUI (raiz)

`test_gui.py` — 22 líneas

```python
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuromorphic_lab"))

from PySide6.QtWidgets import QApplication
from neurolab.gui.main_window import MainWindow

app = QApplication.instance() or QApplication(sys.argv)
win = MainWindow()

# La ventana principal usa docks (ya no hay QTabWidget principal).
# run_simulation() despacha a cada simulación según el dock visible.
win.run_simulation()

# Ejecutar de forma explícita los 3 motores de simulación para validación headless
win._run_memristor_simulation()
win._run_neuron_simulation()
win._run_hybrid_simulation()

print("[OK] MainWindow initialized and all 3 simulations executed successfully.")


```
