# Modelo LIF neuromorfico

## Que representa

El modulo `lif_neuron.py` implementa una neurona fisica tipo Leaky Integrate-and-Fire basada en un dispositivo de conmutacion de umbral, llamado TSM.

El circuito conceptual es:

```text
Vin(t) -> Rs -> Vc -> TSM -> R0 -> GND
```

Donde:

- `Vin` es el voltaje de entrada.
- `Rs` controla la carga del capacitor.
- `Vc` es el potencial del capacitor.
- `TSM` cambia abruptamente su resistencia al superar un umbral.
- `R0` permite leer una salida `Vout`.

## Variables principales

| Variable | Significado |
|---|---|
| `V` | voltaje del capacitor, tambien llamado `Vc` |
| `w` | estado interno del TSM, entre 0 y 1 |
| `R_tsm` | resistencia instantanea del dispositivo de umbral |
| `Vout` | salida del divisor TSM + R0 |
| `spike_count` | numero de disparos detectados |

## Parametros

`LIFParameters` define:

- `C`: capacitancia.
- `R_s`: resistencia serie de entrada.
- `R_off`: resistencia alta del TSM.
- `R_on`: resistencia baja del TSM.
- `R_0`: resistencia de salida.
- `V_th`: umbral de encendido.
- `V_hold`: voltaje de mantenimiento.
- `alpha`: velocidad de encendido.
- `beta`: velocidad de apagado.
- `w_init`: estado inicial.

## Ecuaciones del circuito

La dinamica del capacitor es:

```text
dVc/dt = ((Vin - Vc)/Rs - Vc/(R_TSM + R0)) / C
```

Esta ecuacion expresa balance de corriente:

- corriente de entrada por `Rs`
- corriente de salida por `TSM + R0`
- acumulacion de carga en `C`

## Dinamica del TSM

El TSM cambia su estado interno `w`.

Si esta apagado y `Vc` supera `V_th`, `w` aumenta rapidamente. Si no supera el umbral, `w` decae.

Si esta encendido, se mantiene mientras `Vc` este por encima de `V_hold`. Cuando `Vc` cae por debajo de `V_hold`, el dispositivo se apaga gradualmente.

Este comportamiento introduce histeresis de umbral:

- `V_th` enciende.
- `V_hold` mantiene o permite apagar.

## Resistencia efectiva

El codigo usa una transicion sigmoide para convertir `w` en conductancia efectiva:

```text
w_eff = 1 / (1 + exp(-50 * (w - 0.2)))
```

Luego:

```text
G_tsm = (1/R_off) * (1 - w_eff) + (1/R_on) * w_eff
R_tsm = 1 / G_tsm
```

Esto permite una transicion abrupta pero numericamente suave.

## Deteccion de spike

Un spike se detecta cuando `w` cruza `0.5` hacia arriba:

```text
fired = (w >= 0.5) and (w_prev < 0.5)
```

Esto evita contar como muchos spikes un unico periodo prolongado en estado encendido.

## Experimentos de validacion

`lif_validation.py` define cinco experimentos.

### EXP-1: rampa de voltaje

Entrada: `Vin` aumenta de `0` a `5 V`.

Objetivo: observar como `Vc` sube hasta que el TSM conmuta.

### EXP-2: umbral

Entrada:

- primero `0.5 V`, subumbral
- pausa
- luego `5 V`, sobreumbral

Objetivo: demostrar que no toda entrada produce spike.

### EXP-3: spike unico

Entrada: pulso corto de `5 V`.

Objetivo: producir un disparo fisico unico y observar descarga.

### EXP-4: tren de spikes

Entrada: `5 V` constante durante mas tiempo.

Objetivo: producir disparos repetidos.

### EXP-5: comparacion experimental

Entrada: onda cuadrada a `100 Hz`.

Objetivo: aproximar datos experimentales con `Vin`, `Vc` y `Vout`.

## Diferencia entre LIF clasico y este modelo

Un LIF matematico clasico suele usar:

```text
dV/dt = -(V - V_rest)/tau + I/C
```

y dispara cuando `V` cruza un umbral ideal. Este proyecto usa un circuito fisico con:

- capacitor real
- resistencias reales
- dispositivo TSM con estado interno
- salida por divisor resistivo
- descarga emergente por conmutacion

Eso hace que el spike no sea solo una condicion logica. Es consecuencia de la dinamica electrica.

## Que mirar en las graficas

Para confirmar que el modelo funciona:

- `Vc` debe integrar la entrada.
- `w` debe subir abruptamente al superar umbral.
- `R_tsm` debe caer cuando el TSM enciende.
- `Vout` debe mostrar un pico asociado al disparo.
- Luego `Vc` debe descargarse y permitir recuperacion.

## Posibles parametros a calibrar

Si el modelo no coincide con datos experimentales, ajustar:

- `C`: cambia tiempos de carga.
- `R_s`: cambia corriente de entrada.
- `R_0`: cambia amplitud de salida.
- `V_th`: cambia punto de disparo.
- `V_hold`: cambia apagado.
- `alpha`: cambia velocidad de encendido.
- `beta`: cambia recuperacion.

## Resumen

La neurona TSM-LIF del proyecto convierte un memristor de umbral en un bloque funcional neuromorfico. Su valor esta en que une fenomenos de dispositivo, circuito y computacion neuronal en un mismo modelo paso a paso.
