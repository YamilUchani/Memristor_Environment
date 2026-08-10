# Crossbar, plasticidad y sneak paths

## Por que importa el crossbar

Un crossbar memristivo es una matriz de dispositivos conectados en filas y columnas. Cada cruce puede representar una sinapsis programable. En computacion neuromorfica, esto permite implementar operaciones tipo matriz-vector directamente en hardware.

Este repositorio contiene simulaciones relacionadas con:

- estados HRS y LRS
- redes crossbar
- sneak paths
- noise margin
- plasticidad LTP/LTD
- STDP
- transmision neuronal

## Estados HRS y LRS

Los dos estados resistivos fundamentales son:

- HRS: High Resistance State.
- LRS: Low Resistance State.

En el modelo Strukov:

```text
HRS ~ R_off
LRS ~ R_on
```

El estado interno `x` controla la transicion entre ambos.

## Crossbar fisico

`physical_crossbar.py` implementa `PhysicalCrossbarArray`, que crea una matriz `NxN` de objetos `StrukovMemristor`.

Esto es mas fisico que una matriz simple de numeros, porque cada celda conserva:

- estado `x`
- resistencia actual
- parametros propios
- posible historia de conmutacion

## Programacion de patrones

La clase incluye:

```python
set_pattern_all_ones()
set_pattern_all_zeros()
```

Estas funciones fuerzan toda la red a:

- estado ON (`x = 1`)
- estado OFF (`x = 0`)

Son utiles para analizar casos extremos.

## Sneak paths

Un sneak path es un camino de corriente no deseado que aparece en una matriz crossbar. Aunque se quiera leer una celda especifica, la corriente puede circular por rutas alternativas a traves de otras celdas.

Esto degrada:

- lectura
- margen de ruido
- interpretacion de estados
- escalabilidad de la matriz

## Modelo V/2

El metodo `get_sneak_resistance` calcula una resistencia parasita aproximada basada en tres grupos:

- `R2`: celdas no seleccionadas en la fila objetivo.
- `R3`: resto de la matriz no seleccionada.
- `R4`: celdas no seleccionadas en la columna objetivo.

Luego:

```text
R_sneak = R2 + R3 + R4
```

Esto permite estimar cuanto camino alternativo aparece alrededor de la celda leida.

## Plasticidad

La plasticidad es la capacidad de modificar el peso sinaptico. En memristores, el peso puede relacionarse con conductancia:

```text
G = 1 / R
```

Si la conductancia aumenta, se puede interpretar como potenciacion. Si disminuye, como depresion.

## LTP y LTD

LTP significa Long-Term Potentiation. LTD significa Long-Term Depression.

En terminos de memristor:

- LTP: el dispositivo se mueve hacia menor resistencia o mayor conductancia.
- LTD: el dispositivo se mueve hacia mayor resistencia o menor conductancia.

Los scripts del proyecto relacionados son:

- `paso13_plasticidad_y_memoria.py`
- `paso14_ltp_ltd_protocolo.py`
- `paso15_curva_stdp.py`

## STDP

STDP significa Spike-Timing-Dependent Plasticity. Es una regla donde el cambio sinaptico depende de la diferencia temporal entre spike presinaptico y postsinaptico.

Conceptualmente:

```text
Delta t = t_post - t_pre
```

Si `Delta t` es positivo o negativo, el peso puede aumentar o disminuir segun la regla usada.

En hardware memristivo, STDP puede emularse aplicando pulsos temporales cuya superposicion modifica el estado del dispositivo.

## Relacion con el parametro `v_th_mem`

`StrukovParameters` contiene `v_th_mem`, un umbral opcional. Si se usa, el estado del memristor solo cambia cuando el voltaje supera cierto umbral.

Esto es util para plasticidad porque evita que pequenas senales modifiquen la sinapsis todo el tiempo.

## Scripts relevantes

Los nombres de scripts indican una progresion:

- `paso16_transmision_neuronal.py`: propagacion o transmision.
- `paso17_red_crossbar.py`: red crossbar.
- `paso18_crossbar_escalable.py`: escalamiento.
- `paso19_caracterizacion_crossbar.py`: metricas de matriz.
- `paso22_analisis_sneak_paths.py`: caminos parasitos.
- `paso23_analisis_noise_margin.py`: margen de ruido.
- `paso24_analisis_metales.py`: impacto de materiales.

## Como interpretar las salidas

En `output_modular` aparecen figuras como:

- `paso17_red_crossbar.png`
- `paso18_crossbar_escalable.png`
- `paso19_caracterizacion_crossbar.png`
- `paso22_analisis_sneak_paths.png`
- `paso23_analisis_noise_margin.png`
- `paso15_curva_stdp.png`
- `paso14_ltp_ltd.png`

Estas figuras deberian usarse junto con el script que las genero, para entender parametros y supuestos.

## Riesgos tecnicos

Al escalar crossbars aparecen problemas:

- mas sneak paths
- menor margen entre HRS y LRS leidos
- sensibilidad a variabilidad C2C
- acumulacion de errores de programacion
- dependencia de esquema de lectura

Por eso es importante combinar simulacion ideal, simulacion fisica y analisis de ruido.

## Resumen

La parte crossbar conecta el dispositivo memristivo individual con aplicaciones neuromorficas. Muestra como una celda que cambia de resistencia puede convertirse en una sinapsis, y como muchas sinapsis forman una red que trae nuevos problemas fisicos: fugas, ruido, escalabilidad y variabilidad.
