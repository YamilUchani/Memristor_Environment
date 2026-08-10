# Validacion y metricas

## Por que validar

Una simulacion fisica no basta con "verse bonita". Debe mostrar que reproduce propiedades esperadas del fenomeno. En este proyecto la validacion combina:

- forma cualitativa de curvas
- metricas cuantitativas
- comparacion con datos de paper
- estabilidad numerica
- consistencia fisica de estados internos

## Validacion morfologica I-V

La firma principal del memristor es el lazo de histeresis estrangulado. Para validarlo se observa:

- cruce cerca del origen
- dos ramas diferenciadas
- dependencia con la direccion temporal
- colapso del lazo cuando aumenta la frecuencia

La Figura 2b se usa como escenario senoidal bipolar. La Figura 2c usa excitacion tipo seno cuadrado y una relacion `R_off/R_on` distinta.

## Metricas implementadas

Las metricas estan en:

```text
memristor_simulator/validation/metrics.py
```

### R2

```text
R2 = 1 - SS_res / SS_tot
```

Mide que tan bien una prediccion sigue una referencia. Valores cercanos a `1` indican mejor ajuste.

### RMSE

```text
RMSE = sqrt(mean((y_true - y_pred)^2))
```

Mide error absoluto en las mismas unidades de la variable comparada.

### Error relativo

```text
error_relativo = mean(|y_true - y_pred| / (|y_true| + epsilon)) * 100
```

Permite interpretar el error en porcentaje, aunque debe tratarse con cuidado cerca de cero.

### Area de histeresis

```text
A = | integral I dV |
```

Sirve como proxy del tamano del lazo. Si el area disminuye mucho al aumentar frecuencia, se confirma el colapso esperado.

### Indice de colapso de histeresis

Compara el area simulada con un area maxima de referencia. Un valor alto indica que el lazo esta colapsando.

### Excursion de estado

```text
delta_x = x_max - x_min
```

Mide cuanto se movio la frontera dopada. Si `delta_x` es pequeno a alta frecuencia, el dispositivo casi no actualizo memoria.

## Validacion por frecuencia

`FrequencyAnalysis` ejecuta el mismo dispositivo a:

- `w0`
- `2w0`
- `5w0`
- `10w0`

Para cada frecuencia calcula:

- `dx`
- area de histeresis
- estado cualitativo

La expectativa fisica es:

```text
a mayor frecuencia -> menor cambio de x -> menor area de lazo
```

## Validacion estocastica

`StochasticAnalysis` ejecuta multiples ciclos con variaciones en:

- `R_on`
- `R_off`
- corriente maxima

Se calculan:

- media de `R_on`
- desviacion estandar de `R_on`
- coeficiente de variacion
- media de corriente maxima
- desviacion de corriente maxima
- R2 promedio entre ciclos y corriente media

Esto no busca que todos los ciclos sean identicos. Busca cuantificar la dispersion.

## Validacion de LIF

La neurona TSM-LIF se valida con experimentos funcionales:

1. Rampa: el capacitor debe cargar progresivamente.
2. Umbral: una entrada subumbral no debe disparar; una sobreumbral si.
3. Spike unico: un pulso suficientemente largo debe producir descarga controlada.
4. Tren de spikes: una entrada sostenida debe producir disparos repetidos.
5. Comparacion experimental: se usa una onda cuadrada para aproximar datos de referencia.

Metricas utiles:

- numero de spikes
- tiempo del primer spike
- frecuencia media de disparo
- maximo de `Vout`
- rango de `Vc`
- evolucion de `w`

## Validacion con datos externos

El proyecto incluye CSV con datos de figuras:

- `Time-Voltage.csv`
- `Time-Current.csv`
- `Time-WD.csv`
- `Vin vs Time_ Figure 3.csv`
- `Vc vs Time_ Figure 3.csv`
- `Vout vs Time_ Figure 3.csv`

Estos archivos permiten comparar senales simuladas contra curvas digitalizadas o experimentales.

## Riesgos de interpretacion

### Una grafica similar no siempre implica ajuste fisico

Dos curvas pueden parecer similares, pero tener parametros no fisicos. Por eso conviene reportar parametros y metricas junto a la figura.

### R2 puede ser enganoso cerca de senales pequenas

Si la referencia tiene poca variacion o cruces por cero, R2 y error relativo pueden comportarse de forma poco intuitiva. Es mejor combinar varias metricas.

### La escala de corriente importa

El codigo a veces escala corriente para comparacion visual. En reportes academicos se debe explicar si una curva esta normalizada o escalada.

## Checklist de validacion para una figura

Antes de usar una figura en tesis o paper, revisar:

- parametros usados
- forma de onda
- frecuencia
- `dt`
- numero de ciclos
- escala de corriente
- si hay ruido o no
- si los ejes tienen unidades
- si la figura corresponde al archivo CSV/JSON exportado
- si la metrica cuantitativa acompana la afirmacion visual

## Resultado ideal de validacion

Una validacion fuerte deberia poder decir:

```text
Con los parametros X y la configuracion Y, el modelo reproduce la morfologia I-V esperada, mantiene x dentro de [0,1], muestra colapso de histeresis con frecuencia y cuantifica la dispersion C2C mediante medias, desviaciones y R2.
```
