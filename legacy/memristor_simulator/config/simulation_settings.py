"""
config/simulation_settings.py
==============================
Configuración de la simulación numérica: tiempo, forma de onda y resolución.
"""

from dataclasses import dataclass
from enum import Enum, auto


class WaveformType(Enum):
    """Tipos de excitación de voltaje soportados."""
    SINE = auto()        # v₀·sin(ω₀t)             — Fig 2b del paper
    SINE_SQUARED = auto()# 6·v₀·sin²(ω₀t)          — Fig 2c del paper
    TRIANGLE = auto()    # Onda triangular          — Validación metodológica
    SQUARE = auto()      # Onda cuadrada            — Hard-switching
    CUSTOM = auto()      # Función externa           — usuario define generate_v(t)


@dataclass
class SimulationSettings:
    """
    Parámetros numéricos y de señal para una simulación.

    Atributos
    ---------
    dt : float
        Paso de integración (s). Debe satisfacer dt ≪ τ_drift / 10.
        Valor seguro: 1e-4 s para f ≤ 5 Hz.
    duration : float
        Duración total de la simulación (s).
    frequency : float
        Frecuencia de la señal de excitación (Hz).
    amplitude : float
        Amplitud de la señal de voltaje v₀ (V).
    waveform : WaveformType
        Tipo de forma de onda aplicada.
    n_cycles : int | None
        Si se especifica, duration = n_cycles / frequency (ignora `duration`).
    save_dir : str
        Directorio donde se guardarán las figuras y CSV. Default = '.'.
    figure_dpi : int
        Resolución de exportación de figuras (puntos por pulgada).
    show_plots : bool
        Si True, llama plt.show() al final de cada figura.
    """
    dt: float = 1e-4              # s
    duration: float = 8.0         # s
    frequency: float = 0.5        # Hz
    amplitude: float = 1.0        # V
    waveform: WaveformType = WaveformType.SINE
    n_cycles: int = None          # None → usa `duration`
    save_dir: str = "."
    figure_dpi: int = 150
    show_plots: bool = False       # False = batch/headless mode

    def __post_init__(self):
        if self.n_cycles is not None:
            self.duration = self.n_cycles / self.frequency

    @property
    def t_end(self) -> float:
        """Tiempo final efectivo (s)."""
        return self.duration

    @property
    def n_steps(self) -> int:
        """Número de pasos de integración."""
        return int(self.duration / self.dt)

    @property
    def omega(self) -> float:
        """Frecuencia angular ω₀ = 2π·f (rad/s)."""
        import math
        return 2.0 * math.pi * self.frequency

    def __str__(self) -> str:
        n_steps = self.n_steps
        lines = [
            "+- SimulationSettings ------------------------------------+",
            f"|  Waveform      : {self.waveform.name:<38} |",
            f"|  Amplitude v0  : {self.amplitude:>8.3f} V                         |",
            f"|  Frequency f   : {self.frequency:>8.3f} Hz                        |",
            f"|  Duration      : {self.duration:>8.3f} s                         |",
            f"|  dt            : {self.dt:>8.2e} s                         |",
            f"|  N steps       : {n_steps:>8d}                             |",
            f"|  Save dir      : {self.save_dir:<38} |",
            "+---------------------------------------------------------+",
        ]
        return "\n".join(lines)


# ── Presets de escenarios del paper ──────────────────────────────────────

def settings_fig2b() -> SimulationSettings:
    """Configuración para reproducir Figura 2b (senoidal, 3 ciclos, f=5Hz real del paper)."""
    return SimulationSettings(
        waveform=WaveformType.SINE,
        amplitude=1.0,
        frequency=0.5,
        n_cycles=3,
        dt=1e-4,
    )


def settings_fig2c() -> SimulationSettings:
    """Configuración para reproducir Figura 2c (sin², 4 ciclos)."""
    return SimulationSettings(
        waveform=WaveformType.SINE_SQUARED,
        amplitude=4.0,
        frequency=0.5,
        n_cycles=4,
        dt=1e-4,
    )


def settings_frequency_sweep(base_frequency: float = 0.5,
                              n_cycles: int = 3) -> list:
    """
    Devuelve una lista de SimulationSettings para el barrido de frecuencias
    (w0, 2w0, 5w0, 10w0) — reproduce el colapso de histéresis del paper.
    """
    multipliers = {"w0": 1, "2w0": 2, "5w0": 5, "10w0": 10}
    settings_list = []
    for label, mult in multipliers.items():
        s = SimulationSettings(
            waveform=WaveformType.SINE,
            amplitude=1.0,
            frequency=base_frequency * mult,
            n_cycles=n_cycles,
            dt=max(1e-5, (n_cycles / (base_frequency * mult)) / 5000),
        )
        settings_list.append((label, s))
    return settings_list
