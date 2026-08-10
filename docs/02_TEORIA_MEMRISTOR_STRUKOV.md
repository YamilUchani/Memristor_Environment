# Teoria del memristor Strukov

## Que es un memristor

Un memristor es un dispositivo cuya resistencia depende de su historia electrica. A diferencia de una resistencia ideal, que cumple `V = R * I` con `R` constante, en un memristor la resistencia cambia por el paso de corriente y por la redistribucion interna de especies ionicas.

En el modelo usado por este proyecto, el memristor se representa como una pelicula delgada de TiO2 con dos regiones:

- Una region dopada, mas conductora.
- Una region no dopada, menos conductora.

La frontera entre ambas regiones se mueve cuando circula corriente.

## Variable de estado

La variable central es:

```text
x = w / D
```

Donde:

- `w` es la longitud de la region dopada.
- `D` es el espesor total del dispositivo.
- `x` esta normalizada entre `0` y `1`.

Esta normalizacion permite trabajar con una variable adimensional y facil de limitar fisicamente.

## Resistencia dependiente del estado

La resistencia instantanea es:

```text
R(x) = R_on * x + R_off * (1 - x)
```

Interpretacion:

- Si `x = 1`, entonces `R(x) = R_on`.
- Si `x = 0`, entonces `R(x) = R_off`.
- Si `0 < x < 1`, la resistencia es una mezcla lineal entre ambos extremos.

`R_on` representa el estado de baja resistencia. `R_off` representa el estado de alta resistencia.

## Relacion corriente-voltaje

En cada paso temporal, el simulador calcula:

```text
I(t) = V(t) / R(x)
```

Como `R(x)` cambia con el tiempo, la relacion entre `V` e `I` no es una recta fija. Esto genera una curva I-V con histeresis.

## Dinamica de la frontera dopada

El modelo de Strukov propone:

```text
dx/dt = (mu_v * R_on / D^2) * I(t)
```

Donde:

- `mu_v` es la movilidad ionica.
- `R_on` escala el acoplamiento electrico.
- `D` aparece al cuadrado, por lo que el espesor del dispositivo tiene una influencia muy fuerte.
- `I(t)` determina direccion y velocidad de movimiento.

El factor:

```text
beta = mu_v * R_on / D^2
```

aparece en el codigo como una propiedad derivada de `StrukovParameters`.

## Por que aparece la histeresis

La histeresis aparece porque la corriente no depende solo del voltaje instantaneo. Tambien depende del estado `x`, y `x` depende de la corriente anterior.

Dos instantes pueden tener el mismo voltaje pero diferente corriente si el dispositivo llego a ese voltaje desde trayectorias distintas.

Esto se ve en la curva I-V como un lazo. En memristores ideales el lazo suele estar estrangulado en el origen: cuando `V = 0`, la corriente tambien tiende a `0`.

## Funcion ventana

Si se integrara `dx/dt` sin restricciones, `x` podria salir del intervalo fisico `[0, 1]`. Para evitarlo se usa una funcion ventana:

```text
f(x) = 1 - (2x - 1)^(2p)
```

En el codigo esta funcion se llama `window_biolek`.

Efecto:

- Cerca del centro (`x = 0.5`), la ventana vale cerca de `1`.
- Cerca de los bordes (`x = 0` o `x = 1`), la ventana tiende a `0`.
- La derivada se reduce al aproximarse a los limites.

El parametro `p` controla cuan abrupta es esa reduccion.

## Hard switching y limites

El codigo tambien aplica `np.clip` para asegurar:

```text
0 <= x <= 1
```

Esto es una proteccion numerica y fisica. Aunque la ventana reduce la velocidad en los bordes, el clipping evita que errores de integracion acumulados saquen al modelo de rango.

## Memristancia analitica

El archivo `strukov_model.py` incluye:

```text
M(q) = R_off * [1 - (mu_v * R_on / D^2) * q]
```

Esta forma depende de la carga acumulada `q`, pero solo es valida en el regimen lineal sin funcion ventana. Sirve como referencia teorica, no como sustituto completo del modelo paso a paso cuando se usan extensiones no lineales.

## Integracion numerica

El modelo principal usa Euler explicito:

```text
x_new = x_old + dxdt * dt
```

Ventajas:

- Es simple.
- Es facil de explicar.
- Permite registrar cada paso del estado.
- Funciona bien con `dt` suficientemente pequeno.

Limitaciones:

- Puede acumular error si `dt` es grande.
- Puede ser inestable en dinamicas muy rapidas.

Por eso `utils/integrators.py` tambien incluye `rk4_step`, util para simulaciones que requieran mayor precision.

## Parametros fisicos principales

En `StrukovParameters` se definen:

| Parametro | Significado | Unidad |
|---|---|---|
| `R_on` | resistencia de baja resistencia | ohm |
| `R_off` | resistencia de alta resistencia | ohm |
| `D` | espesor de pelicula | m |
| `mu_v` | movilidad ionica | m^2/(V*s) |
| `w_init` | estado inicial normalizado | adimensional |
| `v_th_mem` | umbral de conmutacion opcional | V |
| `enable_nonlinear_drift` | activa funcion ventana | booleano |
| `enable_thermal` | activa correccion termica | booleano |

## Presets del paper

El proyecto incluye:

- `fig2b_params()`: usa `R_off/R_on = 160`.
- `fig2c_params()`: usa `R_off/R_on = 380`.
- `default_params()`: apunta a la configuracion base de Figura 2b.

Estos presets permiten reproducir escenarios de referencia sin reescribir parametros a mano.

## Colapso de histeresis con frecuencia

Una propiedad importante del memristor es que el lazo I-V se reduce al aumentar la frecuencia. La explicacion es intuitiva:

- A baja frecuencia, los iones tienen tiempo de desplazarse.
- A alta frecuencia, el voltaje cambia demasiado rapido.
- La variable `x` apenas cambia.
- El dispositivo se parece mas a una resistencia fija.

Por eso `frequency_analysis.py` compara `w0`, `2w0`, `5w0` y `10w0`.

## Variabilidad fisica

Los dispositivos reales no cambian exactamente igual en cada ciclo. Por eso se introduce variabilidad ciclo-a-ciclo:

- `R_on` se muestrea con dispersion estadistica.
- `R_off` tambien se muestrea.
- Se puede agregar ruido en la variable de estado.

Esta extension no contradice el modelo base. Lo vuelve mas cercano a un dispositivo fisico fabricado, donde existen fluctuaciones de vacancias, interfaces y filamentos.

## Resumen conceptual

El modelo completo se puede leer como este ciclo:

```text
V(t) -> I(t) = V/R(x) -> dx/dt -> x(t + dt) -> R(x) -> nueva corriente
```

Ese bucle de retroalimentacion es la fuente de memoria. El dispositivo no solo responde: conserva una huella de lo ocurrido antes.
