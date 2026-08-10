# Glosario y formulas

## Variables del memristor

| Simbolo | En codigo | Significado | Unidad |
|---|---|---|---|
| `R_on` | `R_on` | resistencia de estado ON | ohm |
| `R_off` | `R_off` | resistencia de estado OFF | ohm |
| `D` | `D` | espesor total | m |
| `mu_v` | `mu_v` | movilidad ionica | m^2/(V*s) |
| `w` | implicita | ancho de region dopada | m |
| `x` | `x`, `state` | estado normalizado `w/D` | adimensional |
| `V` | `voltage` | voltaje aplicado | V |
| `I` | `current` | corriente | A |
| `q` | `charge` | carga acumulada | C |
| `Phi` | `flux` | flujo acumulado | V*s |

## Formulas del memristor

### Estado normalizado

```text
x = w / D
```

### Resistencia instantanea

```text
R(x) = R_on * x + R_off * (1 - x)
```

### Corriente

```text
I(t) = V(t) / R(x)
```

### Dinamica de estado

```text
dx/dt = (mu_v * R_on / D^2) * I(t)
```

### Factor beta

```text
beta = mu_v * R_on / D^2
```

### Funcion ventana

```text
f(x) = 1 - (2x - 1)^(2p)
```

### Dinamica con ventana

```text
dx/dt = beta * I(t) * f(x)
```

### Memristancia analitica

```text
M(q) = R_off * [1 - (mu_v * R_on / D^2) * q]
```

## Variables LIF

| Simbolo | En codigo | Significado | Unidad |
|---|---|---|---|
| `C` | `C` | capacitancia | F |
| `R_s` | `R_s` | resistencia serie de entrada | ohm |
| `R_0` | `R_0` | resistencia de salida | ohm |
| `R_TSM` | `R_tsm` | resistencia del TSM | ohm |
| `V_th` | `V_th` | umbral de encendido | V |
| `V_hold` | `V_hold` | voltaje de mantenimiento | V |
| `alpha` | `alpha` | tasa de encendido | 1/(V*s) |
| `beta` | `beta` | tasa de apagado | 1/s |
| `Vc` | `V` | voltaje del capacitor | V |
| `Vout` | `Vout` | salida medida | V |
| `w` | `w` | estado interno TSM | adimensional |

## Formula LIF principal

```text
dVc/dt = ((Vin - Vc)/Rs - Vc/(R_TSM + R0)) / C
```

## Conductancia TSM

```text
w_eff = 1 / (1 + exp(-50 * (w - 0.2)))
G_tsm = (1/R_off) * (1 - w_eff) + (1/R_on) * w_eff
R_tsm = 1 / G_tsm
```

## Metricas

### R2

```text
R2 = 1 - SS_res / SS_tot
```

### RMSE

```text
RMSE = sqrt(mean((y_true - y_pred)^2))
```

### Error relativo

```text
error_relativo = mean(|y_true - y_pred| / (|y_true| + epsilon)) * 100
```

### Area de histeresis

```text
A = | integral I dV |
```

### Excursion de estado

```text
delta_x = x_max - x_min
```

## Terminos

### Memristor

Dispositivo cuya resistencia depende de su historia electrica.

### Histeresis

Dependencia de la salida no solo del valor actual de entrada, sino de la trayectoria previa.

### Lazo estrangulado

Curva I-V que cruza o se estrecha cerca del origen. Es una firma tipica memristiva.

### HRS

High Resistance State. Estado de alta resistencia.

### LRS

Low Resistance State. Estado de baja resistencia.

### C2C

Cycle-to-Cycle variability. Variabilidad entre ciclos de operacion.

### STDP

Spike-Timing-Dependent Plasticity. Regla de plasticidad dependiente del tiempo relativo entre spikes.

### LTP

Long-Term Potentiation. Aumento persistente de peso sinaptico.

### LTD

Long-Term Depression. Disminucion persistente de peso sinaptico.

### Sneak path

Camino electrico parasitario en una matriz crossbar que altera la lectura de una celda objetivo.

### TSM

Threshold Switching Memristor. Dispositivo memristivo que conmuta al superar un umbral de voltaje.

### LIF

Leaky Integrate-and-Fire. Modelo neuronal que integra entrada, pierde carga y dispara al superar umbral.

## Unidades frecuentes

| Unidad | Significado |
|---|---|
| `V` | voltio |
| `A` | amperio |
| `mA` | miliamperio |
| `ohm` | resistencia |
| `F` | faradio |
| `s` | segundo |
| `Hz` | ciclos por segundo |
| `K` | kelvin |

## Conversiones utiles

```text
1 mA = 1e-3 A
1 uF = 1e-6 F
1 nF = 1e-9 F
1 nm = 1e-9 m
1 kOhm = 1e3 ohm
1 MOhm = 1e6 ohm
```
