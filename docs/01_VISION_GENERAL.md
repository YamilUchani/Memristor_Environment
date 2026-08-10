# Vision general del proyecto

## Proposito

Este repositorio implementa un entorno de simulacion para dispositivos memristivos y circuitos neuromorficos. El nucleo del proyecto es un modelo fisico de memristor basado en Strukov et al. 2008, extendido con simulaciones de variabilidad, analisis de frecuencia, validacion con metricas y una neurona tipo TSM-LIF.

El proyecto tiene una orientacion academica: sirve para explicar, reproducir y validar fenomenos de computacion neuromorfica usando Python. Tambien genera graficas, datos tabulares y paquetes exportables que pueden alimentar herramientas externas como Unity.

## Que es lo que se simula

El sistema simula principalmente tres niveles:

1. Dispositivo individual: un memristor TiO2 con resistencia dependiente del estado interno `x = w/D`.
2. Circuito funcional: una neurona fisica tipo Leaky Integrate-and-Fire donde un dispositivo de umbral TSM participa en la descarga del capacitor.
3. Sistema o red: arreglos crossbar, plasticidad, efectos de sneak paths y variabilidad estadistica.

## Concepto central

Un memristor no se comporta como una resistencia fija. Su resistencia depende de la historia electrica del dispositivo. En este proyecto, esa historia se resume mediante una variable interna normalizada:

```text
x = w / D
```

Donde:

- `w` es el ancho de la region dopada.
- `D` es el espesor total de la pelicula.
- `x = 0` representa una region completamente en alta resistencia.
- `x = 1` representa una region completamente en baja resistencia.

La resistencia se calcula interpolando entre dos extremos:

```text
R(x) = R_on * x + R_off * (1 - x)
```

Si `x` aumenta, el dispositivo se acerca a `R_on`. Si `x` disminuye, se acerca a `R_off`.

## Estructura general del repositorio

El proyecto combina un paquete modular y scripts de experimentacion.

```text
memristor_simulator/
  config/
  models/
  simulations/
  validation/
  visualization/
  utils/
  main.py

output_modular/
imagenes_para_el_paper/
graficas/
paso*.py
validate_lif.py
calibrate_lif.py
generate_all_metrics.py
README.md
requirements.txt
```

La carpeta `memristor_simulator` contiene el codigo reusable. Los scripts `paso*.py` son experimentos, reproducciones, analisis o etapas de un flujo academico. La carpeta `output_modular` contiene resultados ya generados: PNG, CSV, JSON y paquetes de datos.

## Capas funcionales

### 1. Configuracion

La configuracion esta en:

- `memristor_simulator/config/parameters.py`
- `memristor_simulator/config/simulation_settings.py`

Aqui se definen los parametros fisicos del dispositivo, por ejemplo `R_on`, `R_off`, `D`, `mu_v`, `w_init`, y los parametros de simulacion como `dt`, frecuencia, duracion, amplitud y tipo de forma de onda.

### 2. Modelos

La carpeta `models` contiene las clases que representan dispositivos o circuitos:

- `strukov_model.py`: memristor Strukov.
- `stochastic_model.py`: memristor con variabilidad ciclo-a-ciclo.
- `lif_neuron.py`: neurona fisica TSM-LIF.
- `thermal_model.py`: extension termica.
- `physical_crossbar.py`: arreglo crossbar fisico basado en objetos memristor.

### 3. Simulaciones

La carpeta `simulations` orquesta experimentos completos:

- `iv_characterization.py`: curvas I-V, Figura 2b y 2c, barrido triangular.
- `frequency_analysis.py`: colapso de histeresis con aumento de frecuencia.
- `stochastic_analysis.py`: variabilidad C2C.
- `lif_validation.py`: experimentos de validacion de neurona LIF.

### 4. Validacion

La carpeta `validation` implementa metricas y comparaciones:

- `metrics.py`: R2, RMSE, error relativo, area de histeresis, excursion de estado.
- `comparison.py`: comparacion contra referencias/paper.
- `lif_experimental_data.py`: datos o soporte para validacion LIF.

### 5. Visualizacion

La carpeta `visualization` genera figuras:

- Curvas I-V.
- Evolucion temporal de estado.
- Dashboard consolidado.
- Graficas de validacion LIF.

### 6. Exportacion

La carpeta `utils` incluye herramientas para:

- Generar formas de onda.
- Integrar numericamente.
- Exportar datos a CSV/JSON.
- Preparar paquetes para Unity.

## Flujo tipico de uso

Un flujo tipico es:

1. Elegir parametros fisicos (`fig2b_params`, `fig2c_params`, `default_params`).
2. Elegir configuracion temporal y forma de onda (`settings_fig2b`, `settings_fig2c`).
3. Ejecutar una simulacion (`IVCharacterization`, `FrequencyAnalysis`, `StochasticAnalysis`, `LIFValidation`).
4. Calcular metricas.
5. Generar graficas.
6. Exportar resultados.

## Resultado esperado

El objetivo visual mas importante para el memristor es obtener una curva I-V con lazo de histeresis estrangulado en el origen. Este lazo es una firma clasica de comportamiento memristivo: el dispositivo recuerda su trayectoria previa y no responde igual para el mismo voltaje dependiendo del sentido temporal de la excitacion.

Para la neurona LIF, el objetivo visual es observar:

- Integracion del voltaje en el capacitor.
- Superacion de umbral.
- Conmutacion abrupta del TSM.
- Descarga.
- Aparicion de spikes.
- Tren de disparos bajo entrada sostenida.

## Archivos academicos relevantes

Ademas del codigo modular, hay archivos pensados como soporte de tesis o paper:

- `Documento_Maestro_Tesis_Neuromorfica.md`
- `references.bib`
- `informe_neuromorfico.txt`
- `imagenes_para_el_paper/`
- `output_modular/`
- scripts `paso13` a `paso29`, relacionados con memoria, plasticidad, STDP, crossbar, sneak paths, ruido, replicas de figuras y analisis adicionales.

## Idea fuerza del proyecto

El proyecto no es solo una coleccion de graficas. Es un pipeline de modelado: parte de una ecuacion fisica de dispositivo, la convierte en una simulacion numerica, valida su comportamiento contra literatura, agrega variabilidad realista y escala hacia bloques neuromorficos como neuronas, sinapsis y crossbars.
