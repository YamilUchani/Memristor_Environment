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
