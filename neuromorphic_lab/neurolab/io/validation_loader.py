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
        self._cached_prezioso_data: Optional[Dict[str, np.ndarray]] = None

    @staticmethod
    def _resolve_default_data_dir() -> Path:
        """Resuelve la ruta a `neuromorphic_lab/data_validation`."""
        neurolab_pkg = Path(__file__).resolve().parent.parent
        project_root = neurolab_pkg.parent
        return project_root / "data_validation"

    def load_all(self, force_reload: bool = False, dataset: str = "default") -> Optional[Dict[str, np.ndarray]]:
        """
        Carga los datos experimentales de validación.
        dataset: "default" (CvsT, VvsT, WDvsT) o "prezioso" (VvsC_Prezioso.csv).
        """
        if dataset == "prezioso":
            return self.load_prezioso(force_reload=force_reload)

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

    def load_prezioso(self, force_reload: bool = False) -> Optional[Dict[str, np.ndarray]]:
        """
        Carga VvsC_Prezioso.csv (Voltaje V vs Corriente µA de Prezioso et al. 2014 Fig 1b).
        """
        if self._cached_prezioso_data is not None and not force_reload:
            return self._cached_prezioso_data

        try:
            prez_path = self.data_dir / "VvsC_Prezioso.csv"
            if not prez_path.exists():
                return None

            data = np.loadtxt(prez_path, delimiter=",")
            v_val = data[:, 0]          # Voltaje (V)
            i_val_uA = data[:, 1]       # Corriente (µA)
            i_val_mA = i_val_uA / 1e3   # Corriente (mA)
            i_val_amps = i_val_uA * 1e-6 # Corriente (A)

            # Tiempo sintetizado para consistencia en gráficas temporales
            t_i = np.linspace(0, 0.4, len(v_val))
            t_v = t_i
            t_w = t_i
            wd_val = np.full_like(v_val, np.nan)

            v_interp = v_val

            mask_i = np.abs(i_val_amps) >= 1e-7
            r_val_kohm = np.where(mask_i, np.abs(v_val / i_val_amps) / 1e3, np.nan)
            g_val_us = np.where(mask_i, (1.0 / (r_val_kohm * 1e3)) * 1e6, np.nan)

            self._cached_prezioso_data = {
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
            return self._cached_prezioso_data
        except Exception as e:
            print(f"Error al cargar datos de validación Prezioso CSV: {e}")
            return None
