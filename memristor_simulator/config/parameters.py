"""
config/parameters.py
====================
Parámetros físicos del modelo Strukov et al. (2008), Tabla Anexo 2.

Referencia:
    Strukov, D. B. et al. "The missing memristor found."
    Nature 453, 80–83 (2008). DOI: 10.1038/nature06932

Todas las unidades en el Sistema Internacional (SI).
"""

from dataclasses import dataclass, field


@dataclass
class StrukovParameters:
    """
    Parámetros físicos del dispositivo TiO₂ de Strukov (2008).

    Ecuaciones del modelo:
        V(t) = [R_on · x(t) + R_off · (1 − x(t))] · I(t)     (Ec. 5)
        dx/dt = (μ_v · R_on / D²) · I(t)                       (Ec. 6)
        x ∈ [0, 1]  (variable de estado normalizada w/D)

    Atributos
    ---------
    R_on : float
        Resistencia del estado ON (Ω). Región dopada (TiO₂₋ₓ).
        Paper: i₀ = v₀/R_on ⇒ R_on = 100 Ω para i₀ = 10 mA, v₀ = 1 V.
    R_off : float
        Resistencia del estado OFF (Ω). Región sin dopar (TiO₂).
        Fig 2b: R_off/R_on = 160  → R_off = 16 000 Ω.
        Fig 2c: R_off/R_on = 380  → R_off = 38 000 Ω.
    D : float
        Espesor total de la película (m). Paper: D = 10 nm.
    mu_v : float
        Movilidad iónica media (m²/V·s).
        Paper: μ_v = 10⁻¹⁰ cm²/V·s = 10⁻¹⁴ m²/V·s.
    w_init : float
        Posición inicial de la frontera dopada normalizada (w/D ∈ [0, 1]).
    enable_nonlinear_drift : bool
        Activa la función ventana de Biolek para limitar la deriva iónica
        cerca de los bordes x → 0 y x → 1 (hard-switching).
    enable_hard_switching : bool
        Mantiene x constante en los límites hasta inversión de polaridad.

    Parámetros térmicos (Joule Heating — Objetivo 1, extensión):
    R_th : float
        Resistencia térmica efectiva del dispositivo (K/W).
    C_th : float
        Capacidad térmica efectiva del dispositivo (J/K).
    alpha_T : float
        Coeficiente de temperatura de la resistencia (1/K).
    T_amb : float
        Temperatura ambiente (K). Default = 298.15 K = 25 °C.
    """
    # ── Parámetros eléctricos básicos (Paper, Figura 2b) ──────────────────
    R_on: float = 100.0          # Ω
    R_off: float = 16_000.0      # Ω  (ratio 160:1, Fig 2b)
    D: float = 10e-9             # m  (10 nm)
    mu_v: float = 1e-14          # m²/V·s  (= 10⁻¹⁰ cm²/V·s)
    w_init: float = 0.1          # x₀ = w₀/D (inicio en región baja)

    # ── Extensiones de dinámica no lineal ─────────────────────────────────
    enable_nonlinear_drift: bool = True   # Función ventana Biolek
    enable_hard_switching: bool = True    # Fijación en bordes

    # ── Módulo térmico (Joule Heating) ────────────────────────────────────
    enable_thermal: bool = False
    R_th: float = 1e6            # K/W
    C_th: float = 1e-12          # J/K
    alpha_T: float = 0.002       # 1/K
    T_amb: float = 298.15        # K

    # ── Propiedades derivadas (solo lectura) ──────────────────────────────
    @property
    def ratio(self) -> float:
        """R_off / R_on — relación de contraste del dispositivo."""
        return self.R_off / self.R_on

    @property
    def tau_drift(self) -> float:
        """
        Escala de tiempo de deriva iónica [s].
        t₀ = D² / (μ_v · v₀)  con v₀ = 1 V (Ec. normalización del paper).
        """
        return self.D**2 / (self.mu_v * 1.0)   # v₀ = 1 V

    @property
    def beta(self) -> float:
        """
        Factor de sensibilidad de la memristancia al tamaño (μ_v · R_on / D²).
        Aparece directamente en dx/dt = β · I(t).
        """
        return (self.mu_v * self.R_on) / (self.D**2)

    def __str__(self) -> str:
        lines = [
            "+- StrukovParameters -------------------------------------+",
            f"|  R_on          = {self.R_on:>12.1f}  Ohm                   |",
            f"|  R_off         = {self.R_off:>12.1f}  Ohm                   |",
            f"|  Ratio R/R     = {self.ratio:>12.1f}  (adim.)               |",
            f"|  D             = {self.D*1e9:>12.1f}  nm                    |",
            f"|  mu_v          = {self.mu_v:>12.2e}  m^2/V*s               |",
            f"|  beta (dx/dt/I)= {self.beta:>12.4e}  s^-1*A^-1             |",
            f"|  tau_drift     = {self.tau_drift*1e3:>12.4f}  ms                    |",
            f"|  x0 (w_init)   = {self.w_init:>12.4f}  (adim.)               |",
            f"|  Ventana Biolek: {'ON ' if self.enable_nonlinear_drift else 'OFF'}   |",
            f"|  Hard-switch:   {'ON ' if self.enable_hard_switching else 'OFF'}   |",
            f"|  Termico:       {'ON ' if self.enable_thermal else 'OFF'}   |",
            "+---------------------------------------------------------+",
        ]
        return "\n".join(lines)


# ── Presets de fábrica para reproducción directa del paper ────────────────

def fig2b_params() -> StrukovParameters:
    """
    Parámetros exactos de la Figura 2b de Strukov (2008).
    Voltaje senoidal v₀·sin(ω₀t), R_off/R_on = 160.
    """
    return StrukovParameters(
        R_on=100.0,
        R_off=16_000.0,
        D=10e-9,
        mu_v=1e-14,
        w_init=0.1,
        enable_nonlinear_drift=True,
        enable_hard_switching=True,
    )


def fig2c_params() -> StrukovParameters:
    """
    Parámetros exactos de la Figura 2c de Strukov (2008).
    Voltaje 6·v₀·sin²(ω₀t), R_off/R_on = 380.
    """
    return StrukovParameters(
        R_on=100.0,
        R_off=38_000.0,
        D=10e-9,
        mu_v=1e-14,
        w_init=0.1,
        enable_nonlinear_drift=True,
        enable_hard_switching=True,
    )


def default_params() -> StrukovParameters:
    """Parámetros por defecto para uso general (Fig 2b)."""
    return fig2b_params()
