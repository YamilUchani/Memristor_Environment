# Entorno de Simulación de Memristores (Digital Twin & Neuromorphic Testing Framework)

Este repositorio contiene la plataforma computacional en Python para la simulación física y matemática de dispositivos memristivos basados en dióxido de titanio ($TiO_2$), sumado al modelado de somas neuronales bioinspirados y redes en matriz de conmutación transversal (Crossbar Arrays). 

Este desarrollo constituye el respaldo científico, matemático y de validación para el **Proyecto de Grado** presentado para la obtención del título de **Licenciatura en Ingeniería Mecatrónica** por **Yamil Ronald Uchani Guachalla** en la **Universidad Católica Boliviana "San PABLO"** (Sede La Paz), bajo la tutoría del **Ing. Axel Daniela Campero Vega** y la relatoría del **Ing. Roberto Vidal Poma Joaniquina**.

---

## 1. Estructura General del Workspace (Reorganizada)

El espacio de trabajo se divide rigurosamente en dos arquitecturas independientes para mantener ordenados los experimentos históricos de la tesis frente al nuevo desarrollo experimental:

```text
Memristor_Environment/
│
├── memristor_simulator/          # Núcleo Histórico y Suite de Validación del Proyecto
│   ├── analysis/                 # Scripts auxiliares, calibración de LIF y cálculo de métricas (RMSE/MAE)
│   ├── config/                   # Configuración estandarizada de parámetros físicos (R_on, R_off, D, mu_v)
│   ├── data/                     # Datasets CSV experimentales digitalizados del estado del arte
│   ├── models/                   # Clases físicas base (Strukov, Estocástico C2C/D2D, TSM-LIF, Crossbar)
│   ├── simulations/              # Resolvedores y orquestadores de barridos alternos y de frecuencia
│   ├── steps/                    # Historial experimental secuencial de validaciones (Pasos 1 al 29)
│   ├── utils/                    # Resolvedores matemáticos y exportación de telemetría a Unity
│   ├── validation/               # Módulos de comparación de error contra literatura científica
│   ├── visualization/            # Orquestadores gráficos de trazado con Matplotlib
│   └── main.py                   # Orquestador original de 3 Fases del simulador base
│
├── neuromorphic_lab/             # Framework de Simulación General para Nuevos Desarrollos Creativos
│   ├── configs/                  # Ficheros de configuración de nuevos componentes
│   ├── docs/                     # Guías y especificaciones técnicas detalladas
│   ├── neurolab/                 # Paquete modular estructurado (devices, neurons, synapses, crossbar, etc.)
│   └── scripts/                  # Scripts de ejecución de experimentos y prototipados creativos
│
├── output_modular/               # Gráficos y telemetría autogenerada de los pasos de validación
│
├── imagenes_para_el_paper/       # Figuras de alta resolución destinadas a la impresión de la tesis
│
├── run_validation.py             # ORQUESTADOR INTERACTIVO MAESTRO DE LA SUITE DE VALIDACIÓN (Consola)
│
├── run_app.py                    # LANZADOR DE LA GUI NEUROMORPHIC LAB (python run_app.py)
│
├── Documento_Maestro_Tesis...md  # Reporte maestro que analiza y contrasta los resultados
├── informe_neuromorfico.txt      # Código fuente LaTeX del informe final del Taller de Grado I
├── references.bib                # Base de datos BibTeX de referencias bibliográficas de la tesis
└── requirements.txt              # Dependencias de librerías del entorno de Python
```

---

## 2. Orquestador Maestro de Validación (`run_validation.py`)

Para facilitar el análisis y la ejecución de la suite de validaciones científicas, se implementó un script interactivo en la raíz del proyecto. Este orquestador cuenta con soporte de colores ANSI y aislamiento de codificación UTF-8 para prevenir errores en terminales de Windows (`cmd.exe` / `PowerShell`).

Para iniciar el menú de control:
```bash
python run_validation.py
```

### Opciones de Ejecución:
*   **`[1]` Ejecutar suite completa:** Lanza secuencialmente los 29 pasos de simulación y muestra tiempos y estados de ejecución.
*   **`[2]` Ejecutar un bloque específico:** Permite correr únicamente áreas de interés (ej. plasticidad sináptica o redes crossbar).
*   **`[3]` Ejecutar paso individual:** Ejecuta un único script de validación específico por número.
*   **`[4]` Salir.**

---

## 3. Catálogo Secuencial de Pasos de Validación (`memristor_simulator/steps/`)

Los scripts en esta carpeta documentan la validación incremental del simulador, abarcando desde el dispositivo aislado hasta arquitecturas de red complejas:

### Bloque 1: Caracterización del Memristor Aislado (Pasos 1-3)
*   [`paso1_simulacion_basica.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso1_simulacion_basica.py): Lazo de histéresis estrangulado (I-V) y deriva de estado lineal con Biolek.
*   [`paso2_colapso_frecuencia.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso2_colapso_frecuencia.py): Colapso del lazo de histéresis a altas frecuencias (inercia iónica).
*   [`paso3_memristor_estocastico.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso3_memristor_estocastico.py): Inyección de ruido estocástico ciclo-a-ciclo (C2C) en dispositivo aislado.

### Bloque 2: Acoplamiento Soma Neuronal e Integración (Pasos 4-10)
*   [`paso4_neurona_aislada.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso4_neurona_aislada.py): Simulación del integrador biológico Leaky Integrate-and-Fire (LIF).
*   [`paso5_tren_pulsos.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso5_tren_pulsos.py): Respuesta del memristor ante trenes de ondas cuadradas rápidas.
*   [`paso6_memristor_neurona.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso6_memristor_neurona.py): Transmisión y acumulación de carga a través del divisor de tensión memristivo.
*   [`paso7_simulacion_completa.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso7_simulacion_completa.py): Ejecución integral acoplada memristor-neurona.
*   [`paso8_estado_hrs.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso8_estado_hrs.py) y [`paso9_estado_lrs.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso9_estado_lrs.py): Simulación de los estados operativos de alta y baja resistencia sináptica.
*   [`paso10_comparacion.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso10_comparacion.py): Script crítico de validación frente a los datos experimentales digitalizados (`Vc vs Time_ Figure 3.csv`).

### Bloque 3: Plasticidad y Aprendizaje Sináptico (Pasos 13-16)
*   [`paso13_plasticidad_y_memoria.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso13_plasticidad_y_memoria.py): Demostración de persistencia de peso y no-volatilidad durante apagones ($V=0$).
*   [`paso14_ltp_ltd_protocolo.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso14_ltp_ltd_protocolo.py): Fortalecimiento y debilitamiento analógico (LTP/LTD) por trenes de 100 pulsos.
*   [`paso15_curva_stdp.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso15_curva_stdp.py): Ventana asimétrica de Hebb por desfase temporal de disparo ($\Delta t$).
*   [`paso16_transmision_neuronal.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso16_transmision_neuronal.py): Simulación del retraso temporal de transmisión presináptica.

### Bloque 4: Arquitecturas Colectivas Crossbar (Pasos 17-24)
*   [`paso17_red_crossbar.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso17_red_crossbar.py): Red crossbar 3x3 y resolución de corrientes nodales mediante Kirchhoff.
*   [`paso18_crossbar_escalable.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso18_crossbar_escalable.py): Clasificación espacial en red 8x4 mediante Multiplicación Vector-Matriz (VMM).
*   [`paso19_caracterizacion_crossbar.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso19_caracterizacion_crossbar.py): Evaluación del mapa de conductancias resultante.
*   [`paso20_caracterizacion_forming.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso20_caracterizacion_forming.py): Simulación del electroformado irreversible (ruptura inicial del dieléctrico).
*   [`paso21_caracterizacion_pulsos.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso21_caracterizacion_pulsos.py): Respuesta transitoria a pulsos ultra-cortos de conmutación.
*   [`paso22_analisis_sneak_paths.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso22_analisis_sneak_paths.py): Simulación de fugas y corrientes de retorno en nodos inactivos.
*   [`paso23_analisis_noise_margin.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso23_analisis_noise_margin.py): Cuantificación de la pérdida del margen de ruido de lectura.
*   [`paso24_analisis_metales.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso24_analisis_metales.py): Caídas de tensión IR parásitas asociadas a metalizaciones (M3, M5, M6).

### Bloque 5: Replicaciones del Estado del Arte y Pre-Forming (Pasos 25-29)
*   [`paso25_replicacion_prezioso_s5.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso25_replicacion_prezioso_s5.py), [`paso25b_replicacion_s5_curvas_y_mapas.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso25b_replicacion_s5_curvas_y_mapas.py) y [`paso25c_replicacion_s5_strukov_ideal.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso25c_replicacion_s5_strukov_ideal.py): Replicación del benchmark de Prezioso (Nature 2015) para voltajes gaussianos de SET y RESET sobre 1000 celdas virtuales.
*   [`paso26_replicacion_prezioso_s6.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso26_replicacion_prezioso_s6.py): Evolución estocástica ruidosa de conmutaciones sucesivas (ruido iónico C2C).
*   [`paso27_replicacion_figura5_sneak.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso27_replicacion_figura5_sneak.py): Construcción nodal mediante matrices dispersas (Scipy MNA) para evaluar caídas IR reales.
*   [`paso28_comparacion_strukov_estocastico.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso28_comparacion_strukov_estocastico.py): Comparativa de corrientes e incertidumbre del modelo determinista ideal contra el estocástico.
*   [`paso29_replicacion_pre_forming.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso29_replicacion_pre_forming.py): Simulación del estado dieléctrico virgen inicial bajo conducción no óhmica de Efecto Túnel cuántico.

---

## 4. Mapeo de Figuras Académicas (`imagenes_para_el_paper/`)

Las figuras destinadas a la impresión del informe final de la tesis son autogeneradas por los siguientes scripts:

| Figura del Informe | Ruta de Archivo | Script de Generación |
| :--- | :--- | :--- |
| **Figura 1** (Pre-forming/Efecto Túnel) | `12_pre_forming_paneles.png` | [`paso29_replicacion_pre_forming.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso29_replicacion_pre_forming.py) |
| **Figura 2a** (Curva I-V Histéresis) | `01_strukov_iv.png` | [`paso1_simulacion_basica.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso1_simulacion_basica.py) |
| **Figura 2b** (Dashboard Strukov) | `01b_strukov_dashboard_completo.png` | `Fase 3` de [`memristor_simulator/main.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/main.py) |
| **Figura 3** (Colapso por Frecuencia) | `02_strukov_freq.png` | [`paso2_colapso_frecuencia.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso2_colapso_frecuencia.py) |
| **Figura 4** (Dashboard Soma LIF) | `03_lif_validation_dashboard.png` | [`paso10_comparacion.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso10_comparacion.py) |
| **Figura 5a** (Plasticidad LTP/LTD) | `07_plasticidad_ltp_ltd.png` | [`paso14_ltp_ltd_protocolo.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso14_ltp_ltd_protocolo.py) |
| **Figura 5b** (Retención no Volátil) | `07b_retencion_memoria.png` | [`paso13_plasticidad_y_memoria.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso13_plasticidad_y_memoria.py) |
| **Figura 6** (Ventana STDP / Hebb) | `06_aprendizaje_stdp.png` | [`paso15_curva_stdp.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso15_curva_stdp.py) |
| **Figura 7** (Dispersión D2D Prezioso S5) | `04_distribucion_estocastica.png` | [`paso25_replicacion_prezioso_s5.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso25_replicacion_prezioso_s5.py) |
| **Figura 8a** (Evolución SET/RESET S6) | `05_evolucion_set_reset.png` | [`paso26_replicacion_prezioso_s6.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso26_replicacion_prezioso_s6.py) |
| **Figura 8b** (Dispersión C2C Prezioso S6)| `04b_ruido_fisico_c2c.png` | `Fase 2` de [`memristor_simulator/main.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/main.py) |
| **Figura 9** (Caída IR y Fugas Sneak) | `09_fugas_sneak_path.png` | [`paso27_replicacion_figura5_sneak.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso27_replicacion_figura5_sneak.py) |
| **Figura 10** (VMM Ideal vs Estocástico) | `10_comparacion_strukov_estocastico.png`| [`paso28_comparacion_strukov_estocastico.py`](file:///g:/Github/Software%20de%20simulacion/Memristor_Environment/memristor_simulator/steps/paso28_comparacion_strukov_estocastico.py) |

---

## 5. Requisitos y Uso

Asegúrate de tener un entorno de Python (>= 3.8) con los requisitos de paquetes instalados:

```bash
pip install -r requirements.txt
```

Para generar todas las simulaciones de validación teórica, puedes ejecutar el orquestador maestro interactivo en la raíz del proyecto o llamar de forma individual a la suite antigua:

```bash
python run_validation.py
```
