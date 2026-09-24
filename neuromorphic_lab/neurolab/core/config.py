from dataclasses import dataclass, field
from typing import Dict, Any, Optional

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
    RON: float = 100.0
    ROFF: float = 16000.0
    x0: float = 0.10
    clip_x: bool = True

    def __post_init__(self):
        if self.D <= 0:
            raise ValueError("El grosor D debe ser strictly positivo.")
        if self.mu_v <= 0:
            raise ValueError("La movilidad mu_v debe ser strictly positiva.")
        if self.RON is not None:
            self.r_on = self.RON
        if self.ROFF is not None:
            self.r_off = self.ROFF
        if self.x0 is not None:
            self.initial_state = self.x0


@dataclass
class PreziosoConfig:
    """
    Parámetros del modelo Yakopcic (2011) para dispositivos Al2O3/TiO2-x.
    Reproduce Fig 1b de Prezioso 2014.

    Referencia: C. Yakopcic et al., "A Memristor Device Model",
                IEEE Electron Device Letters, 32(10), 1436-1438, 2011.
    """
    # Resistencia ON en polaridad negativa (RESET) -> permite I_RESET max ~ -600 uA
    r_on_reset: float = 1750.0

    # Umbrales de conmutación (V)
    V_p: float = 0.60       # umbral SET (positivo)
    V_n: float = 0.85       # umbral RESET (negativo)

    # Amplitudes de las tasas g(V)
    A_p: float = 200.0      # SET rate
    A_n: float = 600.0      # RESET rate (mayor → asimetría 3×)

    # Coeficientes de decaimiento exponencial de las ventanas
    alpha_p: float = 1.0    # ventana SET
    alpha_n: float = 1.0    # ventana RESET

    # Umbrales de inicio de ventana
    x_p: float = 0.2        # x > x_p → ventana SET activa
    x_n: float = 0.2        # (1-x) > x_n → ventana RESET activa


@dataclass
class PreziosoVirginConfig:
    """
    Parámetros del dispositivo Yakopcic/Prezioso en estado VIRGEN (pre-forming).

    Reproduce Fig. S3 del Suplemento de Prezioso et al.:
        'Training and operation of an integrated neuromorphic network
         based on metal-oxide memristors', Nature 521, 61-64 (2015).

    El estado virgen se caracteriza por:
      - Conductancias casi resistivas (sin histéresis)
      - G media ≈ 0.45 μS @ V_read = 0.1 V
      - Distribución normal con CV ≈ 17%
      - Comportamiento lineal para |V| < V_p (sin conmutación)
    """
    # --- Resistencias (usadas post-forming, referencias físicas) ---
    R_on_set: float = 4750.0        # Ω resistencia ON tras SET
    R_on_reset: float = 1750.0      # Ω resistencia ON tras RESET
    R_off: float = 1.0e6            # Ω resistencia OFF

    # --- Umbrales de conmutación ---
    V_p: float = 0.60               # V umbral SET (positivo)
    V_n: float = 0.85               # V umbral RESET (negativo)

    # --- Tasas de cambio ---
    A_p: float = 200.0              # s⁻¹ tasa SET
    A_n: float = 600.0              # s⁻¹ tasa RESET (asimetría 3×)

    # --- Estado inicial virgen ---
    x0: float = 0.02                # Estado inicial (casi OFF, pre-forming)
    G_initial: float = 0.45e-6      # S conductancia media virgen @ 0.1V
    G_sigma: float = 0.08e-6        # S desviación estándar D2D (≈ 17% CV)
    V_read: float = 0.1             # V voltaje de lectura no destructiva

    # --- Conducción no lineal estado virgen (Fig. S3a) ---
    a_p: float = 2.67e-7            # A amplitud exponencial rama positiva
    b_p: float = 2.77               # V⁻¹ tasa exponencial positiva
    a_n: float = 9.4e-8             # A amplitud exponencial rama negativa
    b_n: float = 3.32               # V⁻¹ tasa exponencial negativa

    # --- Integración ---
    dt: float = 1e-4                # s paso temporal
