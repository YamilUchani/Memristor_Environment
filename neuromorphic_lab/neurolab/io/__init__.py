"""
neurolab.io
============
Módulo de entrada/salida del simulador Neuromorphic Lab.

Exporta:
    ProfileManager : Gestión centralizada de perfiles JSON (guardar, cargar, listar).
"""
from neurolab.io.profile_manager import ProfileManager  # noqa: F401

__all__ = ["ProfileManager"]
