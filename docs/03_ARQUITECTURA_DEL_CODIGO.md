# Arquitectura del codigo

## Organizacion principal

El paquete central es `memristor_simulator`. Esta pensado como una libreria reusable: los modelos estan separados de las simulaciones, las simulaciones de las graficas y las graficas de la exportacion de datos.

```text
memristor_simulator/
  config/
  models/
  simulations/
  utils/
  validation/
  visualization/
  main.py
```

Esta separacion es importante porque permite probar y reutilizar cada capa sin depender de notebooks o scripts sueltos.

## `config/`

Contiene configuracion fisica y numerica.

### `parameters.py`

Define `StrukovParameters`, que agrupa los parametros del memristor:

- `R_on`
- `R_off`
- `D`
- `mu_v`
- `v_th_mem`
- `w_init`
- flags de deriva no lineal, hard switching y termica

Tambien incluye presets:

- `fig2b_params()`
- `fig2c_params()`
- `default_params()`

Estos presets son importantes para reproducibilidad. En vez de tener numeros dispersos en varios scripts, los escenarios base quedan centralizados.

### `simulation_settings.py`

Define:

- `WaveformType`
- `SimulationSettings`
- `settings_fig2b()`
- `settings_fig2c()`
- `settings_frequency_sweep()`

`SimulationSettings` controla `dt`, duracion, frecuencia, amplitud, forma de onda, numero de ciclos, carpeta de salida y opciones de figura.

## `models/`

Contiene las clases que representan objetos fisicos o circuitales.

### `strukov_model.py`

Es el nucleo del memristor. Incluye funciones puras:

- `calculate_resistance(x, r_on, r_off)`
- `dxdt_strukov(i_in, R_on, D, mu_v, ...)`
- `window_biolek(x, p=5)`
- `memristance_analytical(q, ...)`

Y la clase:

- `StrukovMemristor`

La clase mantiene estado interno:

- `x`
- `resistance`
- `temperature`
- `_q_accumulated`

El metodo principal es:

```python
i = device.step(voltage, dt)
```

Cada llamada calcula corriente, actualiza el estado, aplica ventana/limites y recalcula resistencia.

### `stochastic_model.py`

Extiende el modelo base para incluir variabilidad. Contiene:

- `C2CConfig`
- `sample_r_on`
- `sample_r_off`
- `StochasticMemristor`

La variabilidad se modela con distribuciones lognormales o gaussianas, protegiendo siempre restricciones fisicas como `R_off > R_on`.

### `lif_neuron.py`

Implementa una neurona fisica TSM-LIF. Define:

- `LIFParameters`
- `default_lif_params()`
- `fast_lif_params()`
- `LIFNeuron`

El metodo principal tambien es `step`, pero aqui recibe `Vin` y devuelve:

```python
fired, V = neuron.step(Vin, dt)
```

Donde `fired` indica si hubo spike y `V` es el potencial del capacitor.

### `physical_crossbar.py`

Define `PhysicalCrossbarArray`, una matriz `NxN` de memristores reales del modelo `StrukovMemristor`. Sirve para calcular resistencia de sneak path a partir del estado real de cada celda.

## `simulations/`

Contiene orquestadores de experimentos.

### `iv_characterization.py`

Define `SimulationResult`, estructura donde se guardan arrays de:

- tiempo
- voltaje
- corriente
- resistencia
- estado interno

Define tambien `IVCharacterization`, que ejecuta un barrido completo.

Funciones clave:

- `run_figure_2b()`
- `run_figure_2c()`
- `run_triangular_sweep()`

### `frequency_analysis.py`

Define `FrequencyAnalysis`. Ejecuta varias simulaciones del mismo dispositivo a distintas frecuencias. Luego reporta:

- frecuencia
- excursion de estado `dx`
- area del lazo
- estado cualitativo de colapso

### `stochastic_analysis.py`

Define:

- `StochasticResult`
- `StochasticAnalysis`

Ejecuta multiples ciclos independientes con parametros remuestreados. Calcula promedios, desviaciones, coeficientes de variacion y R2 entre ciclos.

### `lif_validation.py`

Define experimentos para validar la neurona TSM-LIF:

- rampa de voltaje
- umbral
- spike unico
- tren de spikes
- comparacion experimental

Devuelve `LIFValidationResults`, que empaqueta todos los experimentos.

## `validation/`

### `metrics.py`

Implementa metricas cuantitativas:

- `calculate_r_squared`
- `rmse`
- `relative_error`
- `hysteresis_area`
- `hysteresis_collapse_index`
- `state_excursion`
- `compute_all_metrics`

Estas funciones permiten pasar de una grafica visual a una evaluacion cuantitativa.

### `comparison.py`

Se usa para comparar resultados simulados con referencias del paper o datos experimentales.

## `visualization/`

Contiene graficadores:

- `plot_hysteresis.py`
- `plot_state.py`
- `plot_dashboard.py`
- `plot_lif_validation.py`

Esta capa debe recibir resultados ya calculados y convertirlos en figuras, sin modificar la fisica del modelo.

## `utils/`

### `integrators.py`

Contiene generadores de forma de onda:

- seno
- seno cuadrado
- triangular
- cuadrada

Tambien contiene integradores:

- Euler explicito
- RK4

### `data_export.py`

Contiene exportacion a:

- CSV
- JSON
- paquete para Unity

## `main.py`

Es el punto de entrada principal. Divide la ejecucion en tres fases:

1. Fase 1: caracterizacion I-V.
2. Fase 2: analisis estocastico y frecuencia.
3. Fase 3: dashboard y exportacion Unity.

Comando tipico:

```bash
python memristor_simulator/main.py --no-show
```

Tambien puede ejecutarse por fase:

```bash
python memristor_simulator/main.py --phase 1 --no-show
python memristor_simulator/main.py --phase 2 --no-show
```

## Scripts `paso*.py`

Los scripts `paso*.py` parecen representar una progresion academica. Cubren temas como:

- neurona aislada
- trenes de pulsos
- estados HRS/LRS
- plasticidad y memoria
- LTP/LTD
- STDP
- crossbar
- sneak paths
- noise margin
- forming
- analisis de metales
- reproducciones de figuras de Prezioso
- comparacion Strukov vs estocastico

Estos scripts no sustituyen al paquete modular; lo complementan como experimentos reproducibles y material de tesis.

## Patron de diseno del proyecto

El patron dominante es:

```text
Parametros -> Modelo -> Simulacion -> Resultado -> Metrica -> Figura/Exportacion
```

Ese patron es sano porque evita mezclar demasiadas responsabilidades en un unico archivo.

## Recomendacion de mantenimiento

Cuando se agregue una simulacion nueva, conviene mantener esta estructura:

1. Si hay nueva fisica, agregar o extender un archivo en `models/`.
2. Si solo es un nuevo experimento, agregarlo en `simulations/`.
3. Si requiere metricas, agregarlas en `validation/metrics.py`.
4. Si requiere figura nueva, crear funcion en `visualization/`.
5. Si produce datos reutilizables, exportarlos desde `utils/data_export.py`.
