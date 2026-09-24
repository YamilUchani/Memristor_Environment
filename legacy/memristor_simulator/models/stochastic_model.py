"""
models/stochastic_model.py
===========================
Variabilidad ciclo-a-ciclo (C2C) del memristor Strukov (2008).

La variabilidad intrínseca de dispositivos de memoria resistiva surge de:
  - Fluctuaciones en la distribución de dopantes (vacancias de O₂)
  - Ruido térmico-eléctrico en la interfaz metal-óxido
  - Variabilidad en la formación/ruptura del filamento conductor

Implementación
--------------
  - R_on ~ Log-Normal(μ_ln, σ_ln)  con σ/μ = 15 % (literatura: Lee 2012)
  - R_off ~ Log-Normal con σ/μ = 10 %
  - Ruido gaussiano aditivo en x(t) en cada ciclo: σ_x = 0.01–0.05

Referencias
-----------
  [1] Strukov et al., Nature 453 (2008).
  [2] Lee et al., Nano Lett. 12 (2012) — variabilidad en TiO₂.
"""

import numpy as np
import copy
import math
from dataclasses import dataclass, field
from typing import Optional
from .strukov_model import StrukovMemristor
from ..config.parameters import StrukovParameters


# ── Configuración de variabilidad ────────────────────────────────────────────

@dataclass
class C2CConfig:
    """
    Configuración del modelo estocástico de variabilidad C2C.

    Atributos
    ---------
    enabled : bool
        Activa/desactiva toda la variabilidad.
    r_on_cv : float
        Coeficiente de variación de R_on (σ/μ). Default 15 %.
    r_off_cv : float
        Coeficiente de variación de R_off (σ/μ). Default 10 %.
    state_noise_sigma : float
        Desviación estándar del ruido gaussiano en x(t) por ciclo.
    use_lognormal : bool
        Si True, usa distribución Log-Normal para R; si False, Gaussiana.
    seed : int | None
        Semilla para reproducibilidad. None = aleatorio.
    """
    enabled: bool = True
    r_on_cv: float = 0.15         # 15 %
    r_off_cv: float = 0.10        # 10 %
    state_noise_sigma: float = 0.02
    use_lognormal: bool = True
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


# ── Funciones de distribución ─────────────────────────────────────────────────

def _lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """
    Calcula μ_ln y σ_ln de una distribución Log-Normal a partir de
    la media aritmética y el coeficiente de variación cv = σ/μ.

    μ_ln = ln(mean) − σ_ln²/2
    σ_ln = sqrt(ln(1 + cv²))
    """
    sigma_ln = math.sqrt(math.log(1.0 + cv ** 2))
    mu_ln = math.log(mean) - 0.5 * sigma_ln ** 2
    return mu_ln, sigma_ln


def sample_r_on(r_on_nominal: float, cv: float, use_lognormal: bool = True) -> float:
    """Muestrea R_on con variabilidad C2C."""
    if use_lognormal:
        mu, sig = _lognormal_params(r_on_nominal, cv)
        sample = np.random.lognormal(mu, sig)
    else:
        sample = np.random.normal(r_on_nominal, r_on_nominal * cv)
    # Protección física: R_on ≥ 1 Ω
    return max(1.0, sample)


def sample_r_off(r_off_nominal: float, cv: float,
                 r_on_sample: float, use_lognormal: bool = True) -> float:
    """Muestrea R_off con variabilidad C2C (garantiza R_off > R_on)."""
    if use_lognormal:
        mu, sig = _lognormal_params(r_off_nominal, cv)
        sample = np.random.lognormal(mu, sig)
    else:
        sample = np.random.normal(r_off_nominal, r_off_nominal * cv)
    # Protección física: R_off > R_on (ratio mínimo 2:1)
    return max(sample, r_on_sample * 2.0)


# ── Clase principal ────────────────────────────────────────────────────────────

class StochasticMemristor(StrukovMemristor):
    """
    Memristor Strukov con variabilidad ciclo-a-ciclo (C2C).

    Extiende StrukovMemristor: al detectar el inicio de cada nuevo ciclo
    (cruce por cero de dxdt con cambio de signo), re-muestrea los parámetros
    R_on y R_off desde sus distribuciones estadísticas.

    La variable de estado x(t) también recibe un pequeño ruido gaussiano
    al inicio de cada ciclo para modelar fluctuaciones de dopantes.

    Uso::

        cfg = C2CConfig(enabled=True, r_on_cv=0.15, seed=42)
        device = StochasticMemristor(params, cfg)
        for t in t_points:
            i = device.step(v(t), dt, detect_cycle=True)
    """

    def __init__(self, params: StrukovParameters, c2c_config: C2CConfig = None):
        super().__init__(params)
        self.c2c = c2c_config or C2CConfig(enabled=False)
        self._nominal_params = copy.deepcopy(params)
        self._prev_dxdt: float = 0.0
        self._cycle_count: int = 0
        self._r_on_history: list = []
        self._r_off_history: list = []

    @property
    def cycle_count(self) -> int:
        """Número de ciclos completados (semiciclos detectados)."""
        return self._cycle_count

    def step(self, voltage: float, dt: float,
             detect_cycle: bool = True) -> float:
        """
        Paso de simulación con detección de ciclo y re-muestreo de parámetros.

        Parámetros
        ----------
        voltage : float
            Voltaje aplicado (V).
        dt : float
            Paso de tiempo (s).
        detect_cycle : bool
            Si True, detecta cambios de ciclo y aplica variabilidad C2C.

        Retorna
        -------
        float : Corriente I(t) en amperios.
        """
        if self.c2c.enabled and detect_cycle:
            cur_dxdt = self.calculate_state_derivative(voltage)
            if (np.sign(self._prev_dxdt) != np.sign(cur_dxdt)
                    and abs(cur_dxdt) > 1e-12):
                self._apply_c2c_variability()
                self._cycle_count += 1
            self._prev_dxdt = cur_dxdt

        return super().step(voltage, dt)

    def _apply_c2c_variability(self):
        """Re-muestrea parámetros y aplica ruido al estado."""
        nom = self._nominal_params

        # Re-muestrear R_on y R_off
        new_r_on = sample_r_on(nom.R_on, self.c2c.r_on_cv, self.c2c.use_lognormal)
        new_r_off = sample_r_off(nom.R_off, self.c2c.r_off_cv, new_r_on,
                                  self.c2c.use_lognormal)

        self._r_on_history.append(new_r_on)
        self._r_off_history.append(new_r_off)

        new_params = copy.deepcopy(nom)
        new_params.R_on = new_r_on
        new_params.R_off = new_r_off
        self.update_parameters(new_params)

        # Ruido en variable de estado
        if self.c2c.state_noise_sigma > 0:
            noise = np.random.normal(0.0, self.c2c.state_noise_sigma)
            self.x = np.clip(self.x + noise, 0.0, 1.0)
            self._recalculate_resistance()

    def get_variability_stats(self) -> dict:
        """
        Estadísticas de la variabilidad C2C observada durante la simulación.

        Retorna
        -------
        dict con media, std, cv real y coeficiente R² vs nominal para R_on y R_off.
        """
        if not self._r_on_history:
            return {"warning": "No C2C cycles detected yet."}

        r_on_arr = np.array(self._r_on_history)
        r_off_arr = np.array(self._r_off_history)

        def _r2(arr, nominal):
            ss_res = np.sum((arr - nominal) ** 2)
            ss_tot = np.sum((arr - arr.mean()) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        return {
            "n_cycles": self._cycle_count,
            "r_on_mean": float(r_on_arr.mean()),
            "r_on_std": float(r_on_arr.std()),
            "r_on_cv_real": float(r_on_arr.std() / r_on_arr.mean()),
            "r_off_mean": float(r_off_arr.mean()),
            "r_off_std": float(r_off_arr.std()),
            "r_off_cv_real": float(r_off_arr.std() / r_off_arr.mean()),
            "r_on_samples": r_on_arr.tolist(),
            "r_off_samples": r_off_arr.tolist(),
        }
