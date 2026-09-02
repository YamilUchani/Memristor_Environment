# Neuromorphic Lab

Nueva versión limpia del entorno neuromórfico.

## Ejecución

```bash
# Opción 1: desde la raíz del repositorio (recomendada)
python run_app.py

# Opción 2: desde la carpeta del lab
cd neuromorphic_lab
python run_app.py
```

Dependencias: `pip install -r requirements.txt` (en la raíz o en `neuromorphic_lab/`).

## Pestañas de la GUI

1. **🔬 Simulador de Memristor** — Modelo de Strukov (2008) con ventanas (Biolek/Joglekar),
   variabilidad D2D/C2C y ruido térmico; 6 gráficas (I-V, V/I(t), x(t), R(t), G(t), P(t)).
2. **🧠 Simulador Neurona LIF** — LIF con período refractario; tren de pulsos por defecto
   20 µA @ 40 Hz que genera spikes visibles en el raster.
3. **🧠+🔬 Híbrido Memristor-LIF** — Acopla fuente → sinapsis (memristor dinámico en ventana
   secundaria o resistencia fija 10 kΩ) → neurona LIF, usando `neurolab.circuits.hybrid`.

## Tests

```bash
python -m pytest tests -q
```
