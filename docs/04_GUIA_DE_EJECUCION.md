# Guia de ejecucion

## Requisitos

El archivo `requirements.txt` indica dependencias principales:

```text
numpy
matplotlib
pandas
```

Se recomienda usar un entorno virtual para evitar conflictos con otras instalaciones de Python.

## Instalacion

Desde la raiz del repositorio:

```bash
pip install -r requirements.txt
```

Si se trabaja en Windows con PowerShell:

```powershell
python -m pip install -r requirements.txt
```

## Ejecutar simulacion principal

El punto de entrada principal es:

```bash
python memristor_simulator/main.py --no-show
```

`--no-show` evita abrir ventanas interactivas de Matplotlib y permite ejecutar en modo batch.

## Ejecutar por fases

### Fase 1: caracterizacion I-V

```bash
python memristor_simulator/main.py --phase 1 --no-show
```

Genera reproducciones de curvas I-V, barridos senoidales/triangulares, evolucion de estado y metricas de validacion.

### Fase 2: frecuencia y variabilidad

```bash
python memristor_simulator/main.py --phase 2 --no-show
```

Ejecuta:

- barrido de frecuencias
- colapso de histeresis
- variabilidad ciclo-a-ciclo
- graficas de ruido fisico

### Fase 3: dashboard y exportacion

```bash
python memristor_simulator/main.py --phase 3 --no-show
```

Advertencia: la fase 3 necesita resultados de fases 1 y 2 disponibles en la misma ejecucion. En el codigo actual, si se ejecuta solo `--phase 3`, el script avisa que requiere datos previos.

## Ejecutar validacion LIF

El repositorio contiene `validate_lif.py` y tambien `memristor_simulator/simulations/lif_validation.py`. Una ejecucion tipica puede ser:

```bash
python validate_lif.py
```

Segun el script, se generan experimentos de:

- rampa
- umbral
- spike unico
- tren de spikes
- comparacion experimental

## Ejecutar scripts por pasos

Los scripts `paso*.py` se ejecutan como archivos Python independientes:

```bash
python paso13_plasticidad_y_memoria.py
python paso14_ltp_ltd_protocolo.py
python paso15_curva_stdp.py
python paso17_red_crossbar.py
python paso22_analisis_sneak_paths.py
```

Antes de ejecutar un script especifico conviene abrirlo y revisar:

- que archivos de entrada espera
- donde guarda salidas
- si depende de datos ya generados
- si crea figuras en `output_modular`

## Carpetas de salida

La carpeta mas importante es:

```text
output_modular/
```

Alli se guardan:

- PNG de curvas y dashboards.
- CSV de datos temporales y metricas.
- JSON exportables.
- carpetas de exportacion tipo `fig2b_unity`.

Tambien existen:

```text
imagenes_para_el_paper/
graficas/
```

Estas carpetas contienen figuras preparadas o derivadas para presentacion, paper o informe.

## Lectura rapida de resultados

Despues de ejecutar el flujo principal, revisar:

- `output_modular/fig2b_reproduced.png`
- `output_modular/fig2c_reproduced.png`
- `output_modular/metodologia_strukov_valid.png`
- `output_modular/frequency_collapse_demonstration.png`
- `output_modular/caracterizacion_estocastica.png`
- `output_modular/ruido_fisico_c2c.png`
- `output_modular/dashboard_completo.png`
- `output_modular/fig2b_data.csv`
- `output_modular/fig2b_complete.json`

## Como interpretar errores comunes

### `ModuleNotFoundError`

Puede ocurrir si se ejecuta desde una carpeta incorrecta. Ejecutar desde la raiz del repo:

```bash
python memristor_simulator/main.py --no-show
```

### No aparecen graficas

Si usaste `--no-show`, es normal. Las figuras se guardan como PNG.

### No se encuentra un CSV experimental

Algunas comparaciones intentan leer CSV externos como:

- `Time-Voltage.csv`
- `Time-Current.csv`
- `Time-WD.csv`
- `Vin vs Time_ Figure 3.csv`
- `Vc vs Time_ Figure 3.csv`
- `Vout vs Time_ Figure 3.csv`

Si no estan presentes, el script puede emitir avisos o saltar la comparacion.

### Simulacion lenta

Reducir:

- numero de ciclos
- cantidad de muestras
- `n_cycles`
- cantidad de corridas estocasticas

Pero no aumentar demasiado `dt`, porque puede afectar estabilidad y precision.

## Orden recomendado para reproducir el proyecto

1. Instalar dependencias.
2. Ejecutar `python memristor_simulator/main.py --phase 1 --no-show`.
3. Confirmar que se generen PNG y CSV en `output_modular`.
4. Ejecutar `python memristor_simulator/main.py --phase 2 --no-show`.
5. Ejecutar validaciones LIF con `python validate_lif.py`.
6. Ejecutar scripts `paso*.py` especificos si se necesita una figura o analisis concreto.

## Comprobacion minima de que todo funciona

Una comprobacion ligera es:

```bash
python memristor_simulator/main.py --phase 1 --no-show
```

Si termina sin excepciones y genera `fig2b_reproduced.png`, el nucleo del memristor, las formas de onda, la integracion y la visualizacion estan funcionando.
