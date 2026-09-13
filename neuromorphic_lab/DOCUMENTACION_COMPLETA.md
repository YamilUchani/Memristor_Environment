# Neuromorphic Lab: documentacion tecnica completa

## 1. Proposito y alcance

`neuromorphic_lab` es un entorno de simulacion neuromorfica escrito en Python. Su objetivo es permitir experimentar, desde una interfaz grafica Qt, con tres niveles de complejidad:

1. Un dispositivo memristivo basado en el modelo fisico de Strukov (2008).
2. Una neurona Leaky Integrate-and-Fire (LIF) impulsada por corriente o por voltaje.
3. Un circuito hibrido en el que uno o dos memristores interactuan con una neurona LIF.

El proyecto combina:

- Modelos fisicos deterministas.
- Modificadores de realismo y variabilidad estadistica.
- Integracion temporal explicita mediante Euler.
- Generacion de señales experimentales.
- Graficas de magnitudes electricas y de estado interno.
- Comparacion contra datos CSV de referencia.
- Metricas cuantitativas de error.
- Perfiles JSON reproducibles.
- Pruebas automaticas de regresion.

Esta guia describe el funcionamiento real del codigo que existe en el subproyecto. No debe confundirse con una especificacion ideal de un simulador de SPICE: el laboratorio es un simulador discreto de modelos compactos, con pasos de tiempo uniformes y componentes calculados mediante ecuaciones de estado.

---

## 2. Estructura del subproyecto

La carpeta tiene esta forma conceptual:

```text
neuromorphic_lab/
|
|-- run_app.py
|-- pyproject.toml
|-- requirements.txt
|-- README.md
|-- DOCUMENTACION_COMPLETA.md
|
|-- configs/
|   |-- strukov_ideal.json
|   |-- strukov_biolek.json
|   |-- strukov_stochastic.json
|   |-- lif_config.json
|   `-- last_session.json          # aparece despues de cerrar la GUI
|
|-- data for validation/
|   |-- CvsT.csv
|   |-- VvsT.csv
|   `-- WDvsT.csv
|
|-- neurolab/
|   |-- core/
|   |-- devices/
|   |-- neurons/
|   |-- circuits/
|   |-- io/
|   `-- gui/
|
`-- tests/
    |-- test_strukov.py
    |-- test_lif_hybrid.py
    `-- test_validation_loader.py
```

### 2.1. Punto de entrada

`run_app.py` prepara el `PYTHONPATH` para que Python encuentre el paquete local `neurolab` y llama a `neurolab.gui.app.main`.

El orden general es:

```text
run_app.py
    -> neurolab.gui.app.main()
        -> QApplication
        -> MainWindow
            -> construccion de paneles
            -> restauracion de last_session.json
            -> simulacion inicial
            -> bucle de eventos Qt
```

La consola permanece ocupada mientras la ventana esta abierta. Eso es normal: el proceso principal esta ejecutando el bucle de eventos de Qt.

### 2.2. Modulos principales

| Modulo | Responsabilidad |
|---|---|
| `neurolab.core.config` | Configuraciones electricas, identidad del dispositivo y parametros fisicos de Strukov. |
| `neurolab.core.memristor` | Estado del memristor, resistencia, conductancia e integracion temporal. |
| `neurolab.core.base_device` | Interfaces para modelos matematicos y modificadores de realismo. |
| `neurolab.devices.models.strukov` | Ecuacion diferencial base del modelo de Strukov. |
| `neurolab.devices.presets` | Construccion de dispositivos preconfigurados. |
| `neurolab.devices.realism` | Ventanas, D2D, C2C, ruido y relajacion volatil. |
| `neurolab.neurons.config` | Parametros fisicos y validaciones del modelo LIF. |
| `neurolab.neurons.lif` | Integracion de la neurona LIF y generacion de spikes. |
| `neurolab.circuits.hybrid` | Acoplamiento de una fuente, un memristor o una resistencia y una LIF. |
| `neurolab.io.validation_loader` | Carga, escalado e interpolacion de CSV de referencia. |
| `neurolab.io.profile_manager` | Guardado, carga y restauracion de perfiles JSON. |
| `neurolab.gui.main_window` | Orquestacion de pestañas, simulaciones, graficas y metricas. |
| `neurolab.gui.widgets` | Paneles de configuracion y canvas Matplotlib. |

---

## 3. Instalacion y ejecucion

### 3.1. Requisitos

El proyecto declara Python 3.10 o superior y estas dependencias:

```text
numpy>=1.20.0
matplotlib>=3.3.0
pandas>=1.2.0
PySide6>=6.4.0
```

`pandas` forma parte de las dependencias declaradas aunque los flujos principales de validacion mostrados en este subproyecto usan `numpy.loadtxt` para los CSV.

### 3.2. Instalacion recomendada

Desde la carpeta `neuromorphic_lab`:

```powershell
cd "g:\Github\Software de simulacion\Memristor_Environment\neuromorphic_lab"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Tambien puede instalarse como proyecto local mediante:

```powershell
python -m pip install -e .
```

### 3.3. Arrancar la GUI

```powershell
python run_app.py
```

Desde la raiz del repositorio tambien se puede usar el ejecutor que apunte al laboratorio, siempre que el directorio de trabajo y las rutas sean correctas. La forma mas directa y menos ambigua es ejecutar `neuromorphic_lab/run_app.py`.

### 3.4. Ejecutar las pruebas

```powershell
python -m pytest tests -q
```

Para una prueba concreta:

```powershell
python -m pytest tests/test_strukov.py -q
python -m pytest tests/test_lif_hybrid.py -q
python -m pytest tests/test_validation_loader.py -q
```

En Windows, si la prueba grafica necesita un backend sin ventana, puede ser necesario configurar Qt en modo headless antes de iniciar pytest. El propio arranque de la aplicacion trata `QT_QPA_PLATFORM=offscreen` con cuidado para no abrir accidentalmente una GUI invisible en una sesion de escritorio.

---

## 4. Modelo de memristor

### 4.1. Estado interno normalizado

El dispositivo utiliza una variable interna `x` que representa la fraccion normalizada de la region dopada:

```text
x = w / D
```

Donde:

- `w` es el ancho de la region dopada.
- `D` es el espesor total de la capa activa.
- `x = 0` corresponde al estado OFF, de alta resistencia.
- `x = 1` corresponde al estado ON, de baja resistencia.

Por defecto el objeto mantiene `x` acotado al intervalo `[0, 1]`. Esto evita que una integracion numerica agresiva produzca estados no fisicos. La propiedad `clip_x` permite desactivar ese recorte para experimentos especificos, aunque los presets normales lo mantienen activo.

### 4.2. Interpolacion de resistencia

La resistencia instantanea se calcula como una interpolacion lineal entre `R_ON` y `R_OFF`:

$$
R(x) = R_{ON} x + R_{OFF}(1-x)
$$

Equivalentemente:

$$
R(x) = R_{OFF} - (R_{OFF}-R_{ON})x
$$

Por tanto:

- Si `x = 0`, `R = R_OFF`.
- Si `x = 1`, `R = R_ON`.
- Para `x = 0.1`, `R_ON = 100 ohm` y `R_OFF = 16000 ohm`, se obtiene `R = 14410 ohm`.

La conductancia es:

$$
G(x) = \frac{1}{R(x)}
$$

El codigo aplica ademas un piso numerico de `1 ohm` a la resistencia calculada. Su objetivo es impedir corrientes artificialmente enormes cuando un experimento usa valores extremos de `R_ON` o un paso temporal demasiado grande.

### 4.3. Corriente instantanea

En cada paso, la corriente base se calcula de forma ohmica:

$$
I(t) = \frac{V(t)}{R(x(t))}
$$

Este valor se usa para calcular la evolucion del estado. Despues, algunos modificadores pueden alterar la corriente devuelta al usuario, por ejemplo para representar ruido de lectura.

### 4.4. Ecuacion de Strukov

El modelo determinista implementado es:

$$
\frac{dx}{dt} = \frac{\mu_v R_{ON}}{D^2} I(t)
$$

Los parametros son:

| Simbolo | Nombre | Unidad |
|---|---|---|
| `x` | estado interno normalizado | adimensional |
| `mu_v` | movilidad de vacancias de oxigeno | `m^2/(V s)` |
| `R_ON` | resistencia del estado ON | `ohm` |
| `D` | espesor de capa activa | `m` |
| `I` | corriente instantanea | `A` |

En el preset de Strukov se usan tipicamente:

```text
R_ON = 100 ohm
R_OFF = 16000 ohm
x_0 = 0.10
D = 10 nm = 10e-9 m
mu_v = 1e-14 m^2/(V s)
```

La direccion del cambio depende del signo de la corriente:

- Corriente positiva: normalmente aumenta `x`, es decir, realiza SET.
- Corriente negativa: normalmente disminuye `x`, es decir, realiza RESET.

La ecuacion base no tiene por si misma una saturacion suave en los bordes. Esa limitacion se agrega mediante ventanas o mediante el recorte de `x` despues de la integracion.

### 4.5. Integracion Euler explicita

El objeto `Memristor` ejecuta un paso de la siguiente forma:

```text
1. Leer x y calcular R(x).
2. Calcular I = V / R(x).
3. Obtener dx/dt del modelo matematico.
4. Pasar dx/dt por todos los modificadores.
5. Calcular x_nuevo = x_actual + dx/dt * dt.
6. Recortar x_nuevo a [0, 1] si clip_x esta activo.
7. Pasar la corriente por los hooks de salida de los modificadores.
8. Devolver la corriente resultante.
```

La formula numerica es:

$$
 x_{n+1} = x_n + \Delta t \left(\frac{dx}{dt}\right)_n
$$

El orden es importante. La corriente se calcula usando el estado anterior, el estado se actualiza despues, y la resistencia almacenada en el historial corresponde al estado actualizado una vez terminado el paso.

### 4.6. Significado de `dt`

El paso `dt` siempre se expresa en segundos cuando llega al modelo. La GUI permite introducirlo en milisegundos y lo convierte mediante:

$$
\Delta t_{s} = \Delta t_{ms} \times 10^{-3}
$$

Un paso demasiado grande puede producir:

- Saltos bruscos de `x`.
- Saturacion prematura en 0 o 1.
- Perdida de la forma de lazo de histeresis.
- Inestabilidad en la neurona o en el circuito hibrido.
- Resultados que parecen fisicos pero son artefactos numericos.

La GUI limita el numero maximo de muestras a `200000`. Si `duration / dt` supera ese limite, aumenta automaticamente el `dt` efectivo para evitar un consumo excesivo de memoria.

---

## 5. Modificadores de realismo

El modelo matematico base se mantiene separado de los efectos adicionales. Cada modificador implementa uno o ambos hooks:

```python
modify_dxdt(...)
modify_current(...)
```

Durante `Memristor.step`, los modificadores de `dx/dt` se aplican en el orden en que fueron registrados. Luego se aplican los modificadores de corriente.

### 5.1. Ventana de Biolek

La ventana de Biolek implementada es:

$$
 f(x,i) = 1 - [x - stp(-i)]^{2p}
$$

con:

```text
stp(-i) = 1 si i < 0
stp(-i) = 0 si i >= 0
```

El resultado final es:

$$
\left(\frac{dx}{dt}\right)_{Biolek} = \frac{dx}{dt} f(x,i)
$$

La funcion depende del signo de la corriente. Esto permite que la atenuacion se aplique cerca del borde correcto segun se este haciendo SET o RESET.

El parametro `p` controla la forma de la ventana:

- `p` bajo: transicion mas ancha y menos abrupta.
- `p` alto: ventana mas concentrada cerca del borde.
- El preset extendido usa normalmente `p = 5`.

### 5.2. Ventana de Joglekar

La ventana de Joglekar implementada es:

$$
 f(x) = 1 - (2x-1)^{2p}
$$

y se multiplica por la derivada base. A diferencia de Biolek, esta forma depende solo de `x`, no del signo de la corriente. Es util para comparar como distintas funciones de ventana cambian la dinamica cerca de los limites.

### 5.3. Variabilidad D2D

D2D significa `device-to-device`: dispersion entre dispositivos fabricados de forma nominalmente identica.

El modificador genera una muestra aleatoria una sola vez durante la construccion del objeto y crea un factor fijo:

$$
 f_{D2D} = 1 + \epsilon
$$

donde `epsilon` procede de una normal truncada. La derivada queda:

$$
\left(\frac{dx}{dt}\right)_{D2D} = f_{D2D}\frac{dx}{dt}
$$

Ese factor no cambia ciclo a ciclo. Por eso representa una diferencia entre chips o dispositivos, no ruido temporal.

El codigo tambien ofrece `apply_to_electrical`, capaz de modificar `R_ON` y `R_OFF` para estudios de poblacion. La simulacion principal usa el escalado de la derivada; la opcion de conjunto Monte Carlo construye dispositivos separados con semillas desplazadas.

### 5.4. Variabilidad C2C

C2C significa `cycle-to-cycle`: variacion que evoluciona durante la simulacion.

Se usa un proceso de Ornstein-Uhlenbeck discreto:

$$
 d\eta = -\theta\eta\,dt_{OU} + \sigma\sqrt{dt_{OU}}\,N(0,1)
$$

El factor aplicado es:

$$
 f_{C2C} = \max(0.1, 1+\eta)
$$

y:

$$
\left(\frac{dx}{dt}\right)_{C2C} = f_{C2C}\frac{dx}{dt}
$$

El paso del proceso OU no esta pensado para reemplazar el paso del integrador. El codigo usa una escala relativa `_OU_DT_SCALE = 20` y calcula:

$$
 dt_{OU} = 20\,dt_{sim}
$$

La GUI inyecta el `dt` real en cada modificador C2C antes del bucle de simulacion mediante `set_simulation_dt`. Esto es importante para que la variabilidad dependa de la escala temporal elegida y no de un valor fijo oculto.

### 5.5. Ruido termico de lectura

El ruido termico se implementa como una perturbacion de la corriente devuelta:

$$
 I_{salida} = I_{ideal} + N(0,\sigma_{noise})
$$

No altera directamente `x`. Por tanto, modela ruido de medida o lectura, no una modificacion de la fisica interna del dispositivo.

Con `noise_std = 1e-6`, la desviacion estandar se interpreta en amperios.

### 5.6. Decaimiento volatil

`VolatileDecayModifier` convierte el comportamiento no volatil de Strukov en uno relajante. Agrega:

$$
\frac{dx}{dt} = \frac{\mu_vR_{ON}}{D^2}I(t) - \frac{x-x_0}{\tau_{relax}}
$$

El segundo termino lleva gradualmente el estado de vuelta hacia `x_0` cuando la excitacion desaparece.

Este modulo es especialmente relevante para representar memristores difusivos y periodos refractarios emergentes. El shunt inmediato a `x_shunt` se realiza en el bucle hibrido cuando ocurre un spike; el modificador se encarga de la relajacion posterior.

La estimacion documentada del periodo refractario es:

$$
 t_{ref} \approx \tau_{relax}\ln\left(\frac{x_{shunt}-x_0}{0.05}\right)
$$

---

## 6. Presets de Strukov

### 6.1. Preset ideal

`create_strukov_2008_fig2b_device` crea un dispositivo con:

```text
R_ON = 100 ohm
R_OFF = 16 kohm
x_0 = 0.1
D = 10 nm
mu_v = 1e-14 m^2/(V s)
modificadores = ninguno
```

Es el preset apropiado para estudiar la ecuacion original y la respuesta ideal de la Fig. 2b de Strukov.

### 6.2. Preset normalizado con Biolek

`create_strukov_normalized_preset(p=5)` parte del preset ideal y agrega una ventana de Biolek. Sirve para estudiar el efecto de las condiciones de frontera sin introducir aun variabilidad aleatoria.

### 6.3. Preset estocastico

`create_strukov_stochastic_preset(seed=42)` agrega:

1. Biolek, `p=5`.
2. D2D con desviacion relativa de `0.05`.
3. C2C con `sigma=0.05`, `theta=1.0`.
4. Ruido de lectura con `noise_std=1e-6 A`.

La semilla permite reproducir exactamente la secuencia de muestras para una misma construccion y un mismo orden de llamadas.

### 6.4. Perfiles JSON incluidos

Los archivos de `configs/` representan perfiles de interfaz, no solo presets de Python.

| Archivo | Proposito |
|---|---|
| `strukov_ideal.json` | Strukov puro, sin ventana ni ruido, senoidal rapida. |
| `strukov_biolek.json` | Biolek determinista con `p=5`. |
| `strukov_stochastic.json` | Biolek, C2C, D2D y ruido termico. |
| `lif_config.json` | Parametros guardados de una configuracion LIF. |
| `last_session.json` | Sesion automatica, si existe. |

Cada perfil de memristor contiene metadatos `_meta` con version de esquema, version de aplicacion, nombre de perfil y fecha de guardado.

---

## 7. Modelo de neurona LIF

### 7.1. Variables de estado

La neurona mantiene:

- `v_membrane`: potencial de membrana en voltios.
- `t`: reloj interno en segundos.
- `refractory_time_left`: tiempo restante del periodo refractario.
- `spike_times`: lista de tiempos en los que hubo disparos.
- `has_spiked`: indicador del paso actual.

### 7.2. Configuracion

`LIFConfig` contiene:

| Parametro | Significado | Unidad |
|---|---|---|
| `c_m` | capacitancia de membrana | faradios |
| `r_leak` | resistencia de fuga | ohmios |
| `r_series` | resistencia de entrada | ohmios |
| `v_rest` | potencial de reposo | voltios |
| `v_th` | umbral de disparo | voltios |
| `v_reset` | potencial despues del spike | voltios |
| `t_ref` | periodo refractario | segundos |

El constructor valida que `c_m`, `r_leak` y `r_series` sean positivos, que `v_th > v_reset` y que `t_ref` no sea negativo.

### 7.3. Constante de tiempo

La resistencia equivalente de la rama de entrada y fuga es:

$$
R_{eq} = R_{series}\parallel R_{leak}
      = \frac{R_{series}R_{leak}}{R_{series}+R_{leak}}
$$

La constante de tiempo correspondiente es:

$$
\tau = R_{eq}C_m
$$

En entrada de corriente directa, la constante de tiempo de la fuga es `R_leak * C_m`. En entrada de voltaje, la presencia de `R_series` cambia el equilibrio y la constante efectiva.

### 7.4. Ecuacion con entrada de voltaje

El modelo implementa una fuente de voltaje conectada mediante una resistencia serie y un diodo ideal excitatorio. La corriente efectiva es:

$$
I_{in} = \frac{\max(V_{in}-V_m,0)}{R_{series}}
$$

La corriente de fuga es:

$$
I_{leak} = \frac{V_m-V_{rest}}{R_{leak}}
$$

La ecuacion de membrana es:

$$
C_m\frac{dV_m}{dt} = I_{in}-I_{leak}
$$

El `max` evita que la neurona devuelva corriente hacia la fuente cuando `V_in` cae por debajo de `V_m`. Esto modela una sinapsis excitatoria unidireccional ideal.

### 7.5. Ecuacion con entrada de corriente

Cuando se usa corriente directa:

$$
C_m\frac{dV_m}{dt} = I_{in} - \frac{V_m-V_{rest}}{R_{leak}}
$$

En el codigo, `current_input` se interpreta directamente en amperios.

### 7.6. Paso temporal LIF

Cada llamada a `LIFNeuron.step` hace lo siguiente:

```text
1. Avanza el reloj t.
2. Limpia has_spiked del paso anterior.
3. Reduce el tiempo refractario pendiente.
4. Calcula la corriente de fuga.
5. Si esta en refractario, bloquea la entrada.
6. Si no, calcula la corriente por voltaje o usa la corriente directa.
7. Integra V_m con Euler explicito.
8. Si V_m >= V_th, registra spike.
9. Aplica V_reset.
10. Inicia t_ref.
```

La actualizacion numerica es:

$$
V_{m,n+1}=V_{m,n}+\frac{I_{in,n}-I_{leak,n}}{C_m}\Delta t
$$

### 7.7. Spike y reset

Cuando `V_m` alcanza `v_th`:

1. `has_spiked` pasa a `True`.
2. El tiempo actual se agrega a `spike_times`.
3. `V_m` se fija en `v_reset`.
4. `refractory_time_left` se fija en `t_ref` si es positivo.

Durante el periodo refractario la entrada no se integra. La fuga sigue teniendo significado fisico, especialmente en el flujo hibrido, donde la membrana debe recuperarse desde un reset hiperpolarizado.

### 7.8. Hiperpolarizacion y reset negativo

`v_reset` puede ser menor que `v_rest`. Esto representa afterhyperpolarization (AHP). Es importante para el modo hibrido con memristor de fuga: si la membrana cae por debajo de reposo, el voltaje aplicado al memristor de fuga es negativo y permite la transicion RESET.

Cuando `v_reset >= v_rest`, la configuracion sigue siendo valida, pero el codigo emite una advertencia indicando que el memristor de fuga no recibira el voltaje negativo necesario para ese mecanismo.

---

## 8. Circuitos hibridos

El laboratorio tiene dos niveles de circuito hibrido:

1. Las clases reutilizables `ResistorLIFCircuit` y `MemristorLIFCircuit`.
2. El flujo especializado de la tercera pestaña, que puede usar memristor en serie, en fuga o en ambas ramas.

### 8.1. Resistencia fija y LIF

`ResistorLIFCircuit.step(v_source, dt)` calcula:

$$
I_{in}=\begin{cases}
\frac{V_{source}-V_m}{R_{input}}, & V_{source}\ge V_m\\
0, & V_{source}<V_m
\end{cases}
$$

Luego llama a `neuron.step(i_in, dt)` y devuelve un diccionario con:

```text
v_source
v_m
i_in
has_spiked
```

### 8.2. Memristor en serie con la LIF

`MemristorLIFCircuit` calcula primero la caida sobre el memristor:

$$
V_M = \max(V_{source}-V_m,0)
$$

Despues ejecuta:

```text
I_M = memristor.step(V_M, dt)
neuron.step(I_M, dt)
```

El diccionario de salida contiene:

```text
v_source
v_m
v_M
i_M
x
r_M
has_spiked
```

La secuencia usa el `V_m` previo para obtener la caida y solo despues actualiza la neurona.

### 8.3. Modos de la tercera pestaña

La GUI permite tres configuraciones:

#### Modo serie

- El Memristor 1 seleccionado sustituye `R_series`.
- La fuga permanece como resistencia fija `R_leak`.
- La corriente de entrada procede de `mem1.step(max(V_in - V_m, 0), dt)`.
- El memristor de serie cambia su resistencia durante la simulacion.

#### Modo fuga

- `R_series` permanece fija.
- El Memristor 2 seleccionado sustituye `R_leak`.
- El memristor de fuga recibe el voltaje real:

$$
V_{leak}=V_m-V_{rest}
$$

No se aplica `max` a este voltaje. Por ello puede ser negativo despues de una hiperpolarizacion y provocar RESET.

La entrada a la neurona sigue siendo unidireccional:

$$
I_{in}=\frac{\max(V_{in}-V_m,0)}{R_{series}}
$$

#### Modo ambos

- El Memristor 1 actua como rama serie.
- El Memristor 2 actua como rama de fuga.
- Cada uno avanza con su propio voltaje y su propio estado.
- Si ambos indices coinciden, la GUI crea instancias separadas usando un desplazamiento de semilla para evitar compartir la misma secuencia aleatoria.

### 8.4. Orden temporal del hibrido

En cada muestra, el flujo real es aproximadamente:

```text
1. Leer V_in y V_m.
2. Calcular la tension del memristor serie, si existe.
3. Avanzar el memristor serie.
4. Calcular la tension del memristor de fuga, si existe.
5. Avanzar el memristor de fuga.
6. Calcular la corriente de fuga de la membrana.
7. Guardar historiales con el V_m previo.
8. Integrar V_m.
9. Comprobar el umbral.
10. Si hay spike, resetear V_m.
11. Iniciar refractario.
12. Si el modo volatil esta activo, forzar x del memristor de fuga a x_shunt.
```

La memoria de un dispositivo se actualiza antes de que la neurona determine si el paso produce spike. El shunt volatil, en cambio, ocurre despues de detectar el spike porque solo entonces se conoce el evento que representa la apertura abrupta del canal conductor.

### 8.5. Refractario hibrido

Durante el refractario:

- La entrada excitatoria se bloquea.
- La fuga sigue activa.
- La membrana se recupera hacia `V_rest`.
- El memristor volatil puede relajarse desde `x_shunt` hacia `x_0`.

Esto produce un periodo refractario que puede ser la suma de dos efectos:

1. El `t_ref` explicito de la LIF.
2. La limitacion dinamica producida por el estado y la resistencia del memristor de fuga.

---

## 9. Generacion de senales

`SignalPanel` construye los vectores `t`, `v` y `dt`.

### 9.1. Parametros

- Amplitud `v0`.
- Frecuencia `f0`.
- Duracion total.
- Paso temporal introducido en milisegundos.
- Forma de onda.

### 9.2. Senoidal

$$
V(t)=V_0\sin(2\pi f_0t)
$$

Es la senal usada habitualmente para observar el lazo de histeresis pinzado del memristor.

### 9.3. Triangular

La implementacion usa la fase `phase = (t*f0) mod 1` y genera una onda entre `-v0` y `+v0`.

### 9.4. Pulsos cuadrados

Se genera una onda unipolar entre `0` y `v0`, con ciclo de trabajo aproximado del 50 por ciento.

### 9.5. Tren de pulsos unipolar

La fase se calcula de la misma forma, pero la señal es activa durante el primer 20 por ciento del ciclo:

```text
V(t) = v0 si phase < 0.2
       0   en otro caso
```

Es la senal por defecto de las demostraciones LIF e hibridas.

### 9.6. Corriente constante

El panel conserva una interfaz de senal unificada y devuelve un valor constante igual a `v0`. En el flujo de memristor se interpreta como la amplitud de la excitacion que se aplica a la entrada implementada por ese panel; para experimentos de corriente directa conviene revisar el panel y la ruta concreta que consume la senal.

### 9.7. Vector de tiempo

La GUI genera `steps = int(duration/dt)` y usa `numpy.linspace(0, duration, steps)`. Si se ajusta el numero de muestras por el limite de seguridad, el `dt` efectivo se recalcula para cubrir la duracion completa.

---

## 10. Interfaz grafica

La ventana principal utiliza PySide6 y Matplotlib.

### 10.1. Pestaña Simulador de Memristor

Incluye:

- Tres subpestañas de dispositivos.
- Identidad y material.
- `R_ON`, `R_OFF` y `x_0`.
- `D` y `mu_v`.
- Ventana de Biolek o Joglekar.
- D2D, C2C y ruido.
- Semilla aleatoria.
- Ensemble Monte Carlo D2D.
- Forma de onda y parametros temporales.
- Carga y guardado JSON.
- Validacion CSV.
- Graficas de corriente, voltaje, estado, resistencia, conductancia y potencia.

Los valores iniciales de las tres subpestañas son distintos para permitir comparaciones rapidas entre TiO2, HfO2 y TaOx. El nombre del material es una configuracion del panel; la ecuacion activa sigue siendo la arquitectura de Strukov con el valor de movilidad seleccionado.

### 10.2. Pestaña Simulador LIF

Construye una `LIFNeuron`, genera una senal de voltaje y guarda:

- Voltaje de entrada.
- Voltaje de membrana.
- Corriente de entrada.
- Resistencia serie.
- Resistencia de fuga.
- Tiempos de spike.
- Umbral, reset y reposo.

Tambien calcula la comparacion contra la solucion analitica discreta del modelo subumbral.

### 10.3. Pestaña Hibrida

Permite seleccionar que memristor ocupa cada rama:

- Memristor en serie.
- Memristor en fuga.
- Ambos.

La GUI muestra un resumen de los dispositivos y ejecuta el bucle especializado descrito en la seccion de circuitos.

### 10.4. Actualizacion en tiempo real

Los paneles emiten `param_changed`. Si el modo de tiempo real esta activo, `MainWindow` arranca un temporizador debounce de aproximadamente 180 ms. Esto evita ejecutar una simulacion completa por cada evento intermedio mientras el usuario cambia un control numerico.

El boton de ejecutar dispara la simulacion inmediatamente.

### 10.5. Restauracion automatica

Al iniciar:

1. Se crea `ProfileManager`.
2. Se intenta cargar `configs/last_session.json`.
3. Si es valido, se cargan sus parametros.
4. Se ejecuta una simulacion inicial una sola vez.

Al cerrar:

1. Se serializa el panel activo.
2. Se escribe `last_session.json`.
3. Los errores de autoguardado se ignoran para no bloquear el cierre.

---

## 11. Perfiles JSON

### 11.1. Esquema general

Los perfiles de memristor contienen normalmente:

```json
{
    "_meta": {
        "schema_version": "1.0",
        "app_version": "0.1.0",
        "profile_name": "Nombre legible",
        "saved_at": "fecha ISO"
    },
    "device_name": "Strukov TiO2",
    "device_family": "oxide_memristor",
    "model_name": "strukov",
    "r_on": 100.0,
    "r_off": 16000.0,
    "initial_state": 0.1,
    "D_nm": 10.0,
    "mu_v": 1e-14,
    "seed": 42,
    "window_type": "Biolek",
    "biolek_p": 5,
    "enable_c2c": true,
    "c2c_sigma": 0.05,
    "c2c_theta": 1.0,
    "enable_d2d": true,
    "d2d_sigma": 0.05,
    "enable_noise": true,
    "noise_std": 1e-6,
    "signal": {
        "waveform": "Sinusoidal",
        "v0": 1.0,
        "f0": 0.5,
        "duration": 8.0,
        "dt_ms": 0.1
    }
}
```

### 11.2. Validacion minima

`ProfileManager` exige al cargar al menos:

```text
r_on
r_off
model_name
```

La validacion del archivo es deliberadamente basica. Las restricciones fisicas mas especificas se validan al construir `ElectricalConfig`, `StrukovConfig` o los paneles.

### 11.3. Seguridad de nombres

Los nombres de perfil se limpian para eliminar caracteres no seguros para nombres de archivo. Los espacios se convierten en guiones bajos y la longitud se limita.

---

## 12. Validacion con datos CSV

### 12.1. Archivos de referencia

`data for validation/` contiene:

- `CvsT.csv`: corriente frente al tiempo.
- `VvsT.csv`: voltaje frente al tiempo.
- `WDvsT.csv`: estado normalizado `w/D` frente al tiempo.

Cada archivo contiene dos columnas numericas sin cabecera.

### 12.2. Carga y escalado

`ValidationDataLoader` usa por defecto:

```text
time_scale = 10.0
current_scale = 0.010
```

La transformacion es:

```text
t = columna_t * 10
I_mA = columna_I * 0.010
I_A = I_mA * 1e-3
V = columna_V
x = columna_WD
```

El voltaje se interpola en los instantes de la corriente:

```text
V_interp = interp(t_i, t_v, V)
```

Despues se calculan:

$$
R = \frac{|V|}{|I|}
$$

expresada en kiloohmios, y:

$$
G = \frac{1}{R}
$$

expresada en microsiemens.

Los puntos cercanos a cruces por cero de corriente se enmascaran para no generar asintotas artificiales. Esos puntos pueden aparecer como `NaN` en resistencia y conductancia y se excluyen de las metricas respectivas.

### 12.3. Metricas

Las magnitudes comparadas son:

- `x(t)` o `w/D`.
- `I(t)` en mA.
- `R(t)` en kiloohmios.
- `G(t)` en microsiemens.

Para cada magnitud se calculan:

#### MAE

$$
MAE = \frac{1}{N}\sum_{k=1}^{N}|y_k^{ref}-y_k^{sim}|
$$

#### RMSE

$$
RMSE = \sqrt{\frac{1}{N}\sum_{k=1}^{N}(y_k^{ref}-y_k^{sim})^2}
$$

#### Error relativo maximo

El codigo normaliza por la amplitud pico de la referencia para evitar singularidades en cruces por cero:

$$
Err_{rel,max}=\frac{\max |y^{ref}-y^{sim}|}{\max |y^{ref}|}\times100
$$

#### R2 de Pearson

El codigo calcula la correlacion de Pearson y la eleva al cuadrado:

$$
R^2 = corr(y^{ref}, y^{sim})^2
$$

Esto mide similitud de forma y no equivale exactamente al `R2` de todos los modelos de regresion. Debe interpretarse como coeficiente de correlacion al cuadrado.

### 12.4. Flujo de validacion en la GUI

1. El usuario activa Validacion CSV.
2. Se ejecuta la simulacion.
3. Se cargan los tres archivos.
4. Se interpolan las trayectorias simuladas en los tiempos de referencia.
5. Se calculan las cuatro familias de metricas.
6. Se muestran MAE, RMSE, error relativo maximo y R2.
7. Los colores solo proporcionan una lectura orientativa: excelente, aceptable o revisar.

La validacion no modifica el estado del dispositivo ni corrige la simulacion; solamente compara resultados.

---

## 13. Validacion analitica de la LIF

La neurona tiene una comprobacion independiente de los CSV de memristor. Para entrada de voltaje, el equilibrio instantaneo se calcula como:

$$
V_{inf}(t) = \frac{R_{leak}}{R_{series}+R_{leak}}V_{in}(t)
+ \frac{R_{series}}{R_{series}+R_{leak}}V_{rest}
$$

La solucion analitica por tramo constante de entrada se aproxima con:

$$
V_{teorico,k}=V_{inf,k}+
(V_{teorico,k-1}-V_{inf,k})e^{-\Delta t/\tau}
$$

La funcion `compute_lif_validation_metrics` compara `v_sim` con esa trayectoria y devuelve:

```text
mae
rmse
rel_err_max
r2
v_analytical
```

Esta validacion comprueba principalmente que el integrador, las unidades y la ecuacion subumbral sean coherentes. Los spikes y resets requieren una interpretacion adicional porque introducen discontinuidades que no pertenecen a la solucion lineal subumbral pura.

---

## 14. Pruebas automaticas

### 14.1. `test_strukov.py`

Comprueba:

- Valores iniciales del preset de Strukov.
- Resistencia y conductancia iniciales.
- Creacion de los tres perfiles oficiales.
- Numero esperado de modificadores.
- Reproducibilidad con una misma semilla.
- Acotamiento de `x` entre 0 y 1.
- Propiedad `I=0` cuando `V=0`.
- Ejecucion de Biolek.
- Ejecucion de la combinacion estocastica.

### 14.2. `test_lif_hybrid.py`

Comprueba:

- Que un tren de pulsos de 20 microamperios a 40 Hz dispara la LIF.
- Que una entrada de 5 V con resistencia serie dispara la LIF.
- Que el circuito memristor-LIF produce spikes.
- Que el circuito resistencia-LIF produce spikes.
- Que el clamp de corriente reproduce exactamente el flujo de corriente directa.
- Que la trayectoria LIF alcanza `R2 > 0.99` y un error absoluto bajo en la validacion analitica.

### 14.3. `test_validation_loader.py`

Comprueba:

- Que los tres CSV se encuentran.
- Que el cargador devuelve todas las claves esperadas.
- Que los vectores no estan vacios.
- Que corriente, voltaje y `w/D` no contienen NaN.
- Que el canvas puede graficar con y sin datos de validacion.

### 14.4. Que significa que pasen las pruebas

Las pruebas confirman contratos locales y regresiones conocidas. No prueban por si solas:

- Validez experimental del material real.
- Convergencia para todos los valores de `dt`.
- Validez de la discretizacion fuera de los rangos usados.
- Exactitud de un circuito electrico general.
- Equivalencia universal con un simulador SPICE.

---

## 15. Procedimiento recomendado para un experimento

### Paso 1: elegir el objetivo

Decidir si se quiere estudiar:

- La ecuacion ideal.
- Efectos de frontera.
- Variabilidad.
- Histeresis.
- Dinamica LIF.
- Acoplamiento sinaptico.
- Periodo refractario volatil.

### Paso 2: elegir una escala temporal

Estimar la duracion de los ciclos y escoger `dt` suficientemente pequeno. Como regla practica, usar muchas muestras por periodo de la señal y repetir el experimento reduciendo `dt` para comprobar estabilidad.

### Paso 3: comenzar con el preset ideal

Primero ejecutar sin ventanas ni ruido. Esto permite separar errores de configuracion de efectos estocasticos.

### Paso 4: agregar una complejidad cada vez

Orden sugerido:

1. Strukov ideal.
2. Biolek o Joglekar.
3. D2D.
4. C2C.
5. Ruido de lectura.
6. Volatilidad.
7. Circuito hibrido.

### Paso 5: guardar el perfil

Guardar el JSON junto con:

- Fecha.
- Semilla.
- Paso temporal efectivo.
- Duracion.
- Forma de onda.
- Version del proyecto.
- Preset usado.

### Paso 6: activar validacion

Comparar `I`, `x`, `R` y `G`, pero interpretar cada metrica junto con su unidad y su escala. Un R2 alto no garantiza un error absoluto bajo si la magnitud tiene amplitud grande.

### Paso 7: verificar reproducibilidad

Repetir con la misma semilla. Para un perfil estocastico, no cambiar el orden de llamadas ni el numero de muestras entre ejecuciones si se espera igualdad muestra a muestra.

---

## 16. Ejemplos conceptuales de uso por Python

### 16.1. Simulacion ideal

```python
import numpy as np
from neurolab.devices.presets.strukov_2008 import create_strukov_2008_fig2b_device

mem = create_strukov_2008_fig2b_device()
dt = 1e-4
t = np.arange(0.0, 1.0, dt)
v = np.sin(2.0 * np.pi * 1.0 * t)

current = np.empty_like(t)
state = np.empty_like(t)

for k, voltage in enumerate(v):
    current[k] = mem.step(voltage, dt)
    state[k] = mem.x
```

### 16.2. Simulacion estocastica reproducible

```python
from neurolab.devices.presets.strukov_2008 import create_strukov_stochastic_preset
from neurolab.devices.realism.c2c import C2CVariabilityModifier

mem = create_strukov_stochastic_preset(seed=42)
for modifier in mem.modifiers:
    if isinstance(modifier, C2CVariabilityModifier):
        modifier.set_simulation_dt(dt)
```

### 16.3. LIF por corriente

```python
from neurolab.neurons.config import LIFConfig
from neurolab.neurons.lif import LIFNeuron

neuron = LIFNeuron(LIFConfig(
    c_m=100e-9,
    r_leak=1e6,
    v_rest=0.0,
    v_th=0.95,
    v_reset=0.15,
    t_ref=2e-3,
))

for current_amps in input_current:
    neuron.step(current_input=current_amps, dt=1e-4)
```

### 16.4. Circuito memristor-LIF

```python
from neurolab.circuits.hybrid import MemristorLIFCircuit

circuit = MemristorLIFCircuit(memristor=mem, neuron=neuron)
for voltage in input_voltage:
    result = circuit.step(voltage, dt)
    if result["has_spiked"]:
        print("Spike en", neuron.t)
```

---

## 17. Limitaciones y decisiones de modelado

### 17.1. Euler explicito

Euler es simple y transparente, pero no es el integrador mas robusto para todos los regímenes. La precision depende de `dt` y puede degradarse con constantes de tiempo muy cortas o parametros extremos.

### 17.2. Modelo resistivo instantaneo

La corriente usa `I=V/R(x)` en cada paso. No hay una solucion nodal general para redes arbitrarias ni parasiticos distribuidos.

### 17.3. Ventanas compactas

Biolek y Joglekar son aproximaciones de borde. No constituyen por si mismas una descripcion completa de mecanismos de transporte, temperatura, filamentos o interfaces.

### 17.4. Ruido termico simplificado

El ruido se agrega directamente a la lectura de corriente con una desviacion fija. No se calcula a partir de temperatura, ancho de banda, resistencia y constante de Boltzmann.

### 17.5. D2D y C2C no son lo mismo

- D2D se fija al crear el dispositivo.
- C2C cambia durante la simulacion.
- Una semilla compartida no implica que dos dispositivos recorran exactamente la misma trayectoria si el orden de llamadas cambia.

### 17.6. Diodo ideal

La entrada excitatoria usa `max(V_source - V_m, 0)`. Esto es una abstraccion de sinapsis unidireccional y no un modelo detallado de un diodo semiconductor.

### 17.7. Metricas R2

El R2 usado para validacion de memristor es correlacion al cuadrado. Debe acompañarse de MAE y RMSE.

### 17.8. Datos CSV

Los factores de escala de los CSV son parte del pipeline de validacion. Si los archivos cambian de unidad, hay que revisar `ValidationDataLoader` antes de interpretar las graficas.

---

## 18. Como extender el proyecto

### 18.1. Agregar un modelo matematico

Crear una clase que herede de `BaseMathModel` e implemente `compute_dxdt`. El modelo debe documentar:

- Variables de entrada.
- Unidades.
- Parametros requeridos.
- Condiciones de validez.
- Signo de la dinamica.

Despues crear un `model_config` especifico, si es necesario, y registrar el modelo en el constructor correspondiente.

### 18.2. Agregar un modificador

Heredar de `BaseRealismModifier`. Usar `modify_dxdt` si el efecto cambia la dinamica interna y `modify_current` si solo cambia la lectura.

Debe decidirse explicitamente si el aleatorio es:

- Una muestra por dispositivo.
- Una muestra por paso.
- Un proceso temporal correlacionado.

La semilla debe recibirse como parametro y el comportamiento debe probarse con una prueba de reproducibilidad.

### 18.3. Agregar una neurona

Heredar de `BaseNeuron` e implementar:

```python
reset() -> None
step(current_input, dt) -> bool
```

Si el modelo admite entrada de voltaje, puede ampliar la firma con un parametro opcional, pero debe conservar la compatibilidad con el circuito que lo utilice.

### 18.4. Agregar una senal

Extender `SignalPanel.generate_voltage_signal` y mantener:

- Vectores de igual longitud.
- Tiempo en segundos.
- `dt` en segundos al salir del panel.
- Limite de muestras.
- Nombre serializable en `to_dict` y `from_dict`.

### 18.5. Agregar una metrica

Ampliar `QuantityMetrics` o crear una funcion paralela. Es necesario definir:

- Unidad.
- Mascara de valores invalidos.
- Umbral de normalizacion.
- Interpolacion temporal.
- Interpretacion del resultado.

### 18.6. Agregar una prueba

Una extension de fisica deberia incluir como minimo:

1. Prueba de construccion.
2. Prueba de limites.
3. Prueba de unidades o magnitud esperada.
4. Prueba de reproducibilidad si hay aleatoriedad.
5. Prueba de integracion si participa en la GUI o en un circuito.

---

## 19. Diagnostico de problemas frecuentes

### La ventana no aparece

Comprobar que no se haya dejado `QT_QPA_PLATFORM=offscreen` activo. Para una prueba headless intencionada puede usarse `NEUROLAB_HEADLESS=1`, pero en una sesion grafica normal hay que permitir el backend de escritorio.

### La simulacion parece congelada

La duracion de la simulacion es aproximadamente proporcional a:

$$
N = \frac{duration}{dt}
$$

Revisar la barra de estado y reducir la duracion o aumentar `dt`. La GUI limita `N` a 200000, pero ese numero aun puede requerir tiempo si se usan varios dispositivos o un ensemble Monte Carlo.

### El estado se pega en 0 o 1

Puede ser un efecto fisico esperado de saturacion, pero tambien puede indicar que `dt`, `mu_v`, `V_0` o la frecuencia no son adecuados. Repetir con un `dt` menor y comparar.

### No aparecen spikes

Revisar:

- `V_th`.
- `V_reset`.
- `C_m`.
- `R_series`.
- `R_leak`.
- Amplitud de la fuente.
- Duracion y duty cycle.
- Si se esta usando entrada de corriente en amperios y no microamperios sin convertir.

### El modo de fuga no hace RESET

Para que el memristor de fuga reciba voltaje negativo despues de un spike, normalmente se necesita `v_reset < v_rest`. Si el reset es igual o mayor que reposo, el voltaje `V_m - V_rest` no sera negativo en la fase de recuperacion.

### La validacion muestra errores enormes

Comprobar primero:

1. Unidades del CSV.
2. Factores `time_scale` y `current_scale`.
3. Duracion simulada frente al rango temporal de referencia.
4. `dt` efectivo despues del limite de muestras.
5. Preset utilizado.
6. Si se comparan datos ideales con un dispositivo estocastico.

### Dos ejecuciones estocasticas no coinciden

Comprobar que:

- Usan la misma semilla.
- Construyen de nuevo el dispositivo.
- Llaman a `set_simulation_dt` con el mismo `dt`.
- Tienen exactamente el mismo numero de pasos.
- No se comparte una instancia entre simulaciones.

---

## 20. Resumen del flujo completo

```text
Usuario cambia parametros
        |
        v
Panel Qt emite param_changed
        |
        v
QTimer debounce o boton Ejecutar
        |
        v
MainWindow.run_simulation()
        |
        +--> Memristor:
        |       construir Memristor
        |       generar V(t)
        |       actualizar C2C con dt
        |       repetir Memristor.step
        |       guardar I, x, R, G
        |       opcionalmente validar CSV
        |
        +--> LIF:
        |       construir LIFNeuron
        |       generar V_IN(t)
        |       repetir LIFNeuron.step
        |       guardar V_m e impulsos
        |       validar contra trayectoria analitica
        |
        `--> Hibrido:
                construir uno o dos memristores
                construir LIFNeuron
                calcular ramas serie/fuga
                integrar memristores
                integrar membrana
                registrar spikes
                aplicar shunt volatil
                graficar circuito completo
```

La idea central del proyecto es mantener separadas tres capas:

1. **Fisica del dispositivo:** ecuacion de `dx/dt`, resistencia y conductancia.
2. **Dinamica neuromorfica:** integracion de la membrana y eventos de spike.
3. **Experimentacion:** GUI, perfiles, senales, graficas y validacion.

Esa separacion permite cambiar el nivel de realismo sin reescribir la neurona, comparar varias instancias del dispositivo y conectar el mismo modelo memristivo tanto a una grafica electrica como a una sinapsis de una neurona LIF.
