# Entorno de Simulación de Memristores (Modelo Strukov 2008)

Este repositorio contiene el núcleo computacional en Python para la simulación física y matemática de dispositivos memristivos basados en dióxido de titanio ($TiO_2$), como parte del proyecto de **Taller de Grado I - Ingeniería Mecatrónica** de **Yamil Uchani**.

El simulador implementa el modelo físico propuesto por Strukov et al. (2008) en su paper *"The missing memristor found"*, añadiendo correcciones no lineales para operar en rangos de frecuencia bajos sin inestabilidad numérica. Este módulo sirve como la validación teórica (OBJ-1 y OBJ-2) antes de su integración como Gemelo Digital interactivo en Unity 3D.

## Características Principales

1. **Resolución de Ecuaciones Diferenciales:** Integración explícita (Forward Euler) del estado de dopaje del memristor ($w/D$).
2. **Deriva No Lineal (Nonlinear Drift):** Implementación de la función ventana de Biolek para emular correctamente el comportamiento de los iones al acercarse a los electrodos, evitando el "hard clipping" irreal.
3. **Validación Morfológica:** Generación automatizada de las curvas I-V (lazo de histéresis estrangulado) contrastadas estadísticamente con la literatura científica.
4. **Análisis Estocástico (Ruido C2C):** Módulo de variabilidad Ciclo a Ciclo utilizando el modelo de "reversión a la media" de Ornstein-Uhlenbeck, simulando el comportamiento térmico degradante de componentes reales.
5. **Exportación Universal:** Serialización automática a `.json` para rápida ingesta desde motores gráficos en C# (Unity).

## Estructura del Proyecto

* `/memristor_simulator`: Directorio principal del código fuente.
  * `/models`: Clases puras de simulación (`strukov_model.py`, `stochastic_model.py`).
  * `/simulations`: Orquestadores de barrido (análisis de frecuencia, corriente-voltaje).
  * `/visualization`: Módulos de trazado con `matplotlib`.
  * `/config`: Parámetros físicos calibrados ($R_{on}, R_{off}, \mu_v, D$).
* `/output_modular`: Carpeta de artefactos autogenerados, gráficas PNG y datos JSON.
* `main.py`: Punto de entrada del simulador que genera todas las figuras de validación en secuencia.
* `reporte_memristor.tex`: Documento fuente de validación académica en LaTeX.

## Requisitos y Uso

Asegúrate de tener un entorno de Python (>=3.8) con los requisitos instalados:

```bash
pip install -r requirements.txt
```

Para generar todas las simulaciones de validación teórica, simplemente ejecuta:

```bash
python memristor_simulator/main.py
```

Las figuras resultantes (y los archivos JSON para Unity) se depositarán automáticamente en la carpeta `output_modular/`.

## Acerca del Autor
**Yamil Ronald Uchani Guachalla**  
Estudiante de Ingeniería Mecatrónica. Este proyecto es la fundación teórica para el posterior desarrollo de herramientas educativas en Computación Neuromórfica.
