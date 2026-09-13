# 🔬 Fundamentos Físicos y Modelo Memristivo de Strukov (2008)

---

> [!NOTE]
> **Marco Teórico y Especificación Académica**  
> Este documento detalla la implementación físico-matemática del dispositivo memristivo de dos capas de $TiO_2$ (HP Labs, Nature 2008), incluyendo la dinámica de frontera, no-linealidades, funciones ventana y la jerarquía de realismo estocástico (D2D y C2C).

---

## 1. 📐 Modelo Físico Fundamental (HP Labs, Nature 2008)

El dispositivo memristivo se estructura físicamente como una película delgada de dióxido de titanio ($TiO_2$) de espesor total $D = 10\text{ nm}$, dispuesta entre dos electrodos metálicos de Platino ($Pt$). La película activa comprende dos regiones distintas:

1. **Región Dopada ($TiO_{2-x}$):** Alta concentración de vacancias de oxígeno cargadas ($V_O^{2+}$), presentando baja resistencia (**baja resistividad $\rho_{\text{ON}}$**).
2. **Región Estequiométrica ($TiO_2$):** Aislante virgen de alta resistencia (**alta resistividad $\rho_{\text{OFF}}$**).

---

### 1.1 Ecuación de Memristancia (Transporte de Electrones)

La resistencia equivalente instantánea del componente, denominada **memristancia $M(x)$**, resulta de la conexión en serie de ambas capas:

$$ M(x) = R_{\text{ON}} \cdot x + R_{\text{OFF}} \cdot (1 - x) $$

*   $M(x)$: Memristancia instantánea en Ohmios ($\Omega$).
*   $R_{\text{ON}}$: Resistencia mínima en el estado completamente encendido (LRS).
*   $R_{\text{OFF}}$: Resistencia máxima en el estado apagado (HRS).
*   $x$: **Variable de estado interno normalizado**, definida como la relación entre el ancho de la capa dopada ($w$) y el espesor total ($D$):

$$ x = \frac{w}{D} \quad \text{con} \quad 0 \leq x \leq 1 $$

La relación voltaje-corriente instantánea se rige por la **Ley de Ohm generalizada**:

$$ V(t) = M(x(t)) \cdot I(t) $$

---

### 1.2 Ecuación de Evolución de Estado (Deriva Iónica)

Al circular una corriente $I(t)$, el campo eléctrico interno impulsa la migración de las vacancias de oxígeno, desplazando la frontera $w$. La tasa de cambio del estado interno sigue la ecuación diferencial de deriva iónica:

$$ \frac{dx}{dt} = \frac{\mu_v \cdot R_{\text{ON}}}{D^2} \cdot I(t) \cdot f(x) $$

Donde $\mu_v = 10^{-14}\text{ m}^2/(\text{V}\cdot\text{s})$ representa la movilidad iónica empírica de las vacancias $V_O^{2+}$ y $f(x)$ es la función ventana.

---

## 2. ⚡ Parámetros de Simulación (Preset Nominal Ideal)

| Parámetro | Símbolo | Valor Nominal | Unidad | Significado Físico |
| :--- | :---: | :---: | :---: | :--- |
| **Resistencia Mínima (LRS)** | $R_{\text{ON}}$ | $100.0$ | $\Omega$ | Resistencia con la capa dopada totalmente extendida |
| **Resistencia Máxima (HRS)** | $R_{\text{OFF}}$ | $16\,000.0$ | $\Omega$ | Resistencia de la matriz estequiométrica virgen |
| **Espesor Activo** | $D$ | $10 \times 10^{-9}$ | $\text{m}$ | Distancia física entre los electrodos de Platino ($10\text{ nm}$) |
| **Movilidad Iónica** | $\mu_v$ | $10^{-14}$ | $\text{m}^2/(\text{V}\cdot\text{s})$ | Constante de deriva empírica de vacancias de $O_2$ |
| **Estado Inicial** | $x_0$ | $0.10$ | Adim. | Estado interno al inicio de la simulación ($w_0/D$) |
| **Paso Temporal** | $dt$ | Configurable | $\text{s}$ | Intervalo de integración del solver numérico |

---

## 3. 🚗 Dinámica de Frontera y Funciones Ventana ($f(x)$)

### 3.1 El Problema de Contorno (Límites Físicos)
La integración diferencial pura de la deriva iónica $\frac{dx}{dt} = k \cdot I(t)$ permitiría que $x$ exceda $1.0$ o caiga por debajo de $0.0$. Físicamente, la frontera $w$ no puede atravesar los electrodos de Platino ($0 \le w \le D$).

> [!TIP]
> **Analogía Didáctica del Coche en el Garaje**  
> Imaginemos la variable de estado $x$ como un coche moviéndose entre dos paredes opuestas ($0$ y $1$):
> *   **Sin Ventana / Hard Clip:** El coche avanza e intenta atravesar la pared a máxima velocidad ($\text{Choque}$). El software interviene artificialmente forzando `x = max(0, min(1, x))`.
> *   **Ventana de Joglekar ($f(x) = 1 - (2x-1)^{2p}$):** El coche detecta la pared cercana y desacelera suavemente. **Problema de Bloqueo (*State-Lock*):** Al tocar exactamente el límite ($f(1)=0$), la velocidad colapsa a cero. Si pones marcha atrás (inversión de corriente), la velocidad sigue siendo cero y el coche queda atascado en la pared.
> *   **Ventana de Biolek ($f(x) = 1 - (x - \text{stp}(-I))^{2p}$):** El sistema evalúa tanto la posición como la **dirección de la corriente** ($\text{stp}(-I)$). Si el coche está pegado a la pared pero inviertes la marcha, Biolek desactiva el freno y **permite la marcha atrás inmediata**.

---

### 3.2 Tabla Comparativa de Funciones Ventana

| Modelo | Ecuación Matemática $f(x)$ | Comportamiento en los Bordes | Evaluación Académica |
| :--- | :--- | :--- | :--- |
| **Sin Ventana (Ideal)** | $f(x) = 1$ | Deriva lineal pura | Simplicidad analítica. Requiere *Hard Clip* que discontinúa la derivada. |
| **Joglekar** | $f(x) = 1 - (2x - 1)^{2p}$ | Frenado simétrico hacia $0$ y $1$ | Transición suave. **Sufre de bloqueo de estado (*State Lock*)** al tocar bordes. |
| **Biolek (Implementado)** | $f(x) = 1 - (x - \text{stp}(-I))^{2p}$ | Dependiente del sentido de $I(t)$ | **Estándar académico actual.** Despega el estado al invertir corriente y modela la no-linealidad. |

---

## 4. 👥 Realismo Estocástico: Jerarquía de Variabilidad D2D y C2C

Para simular redes neuromórficas reales, el simulador organiza la variabilidad en **3 niveles jerárquicos bien definidos**:

```text
                  1. PARÁMETROS NOMINALES (VALOR PROMEDIO)
                  R_ON = 100 Ω  |  R_OFF = 16 kΩ  |  μ_v = 1×10⁻¹⁴
                                     │
                                     ▼
                  2. D2D (DEVICE-TO-DEVICE) — FABRICACIÓN
                  • Se calcula UNA SOLA VEZ al instanciar cada celda k.
                  • Distribución Normal Truncada en [-2σ_D2D, +2σ_D2D].
                  • Modifica permanentemente:
                    - R_ON^(k) = R_ON × (1 + ε_ON,k)
                    - R_OFF^(k) = R_OFF × (1 + ε_OFF,k)
                    - Factor de deriva constante f_drift^(k)
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
       Memristor A              Memristor B              Memristor C
    (R_ON=97.2 Ω, 1.02x)     (R_ON=103.5 Ω, 0.97x)     (R_ON=99.1 Ω, 1.01x)
            │                        │                        │
            ▼                        ▼                        ▼
                  3. C2C (CYCLE-TO-CYCLE) — OPERACIÓN
                  • Fluctúa TEMPORALMENTE en cada paso/ciclo de conmutación.
                  • Proceso de Ornstein-Uhlenbeck (retorno a la media):
                    dη_t = -θ η_t dt + σ_C2C √dt N(0, 1)
                  • Modula dinámicamente la derivada instantánea:
                    (dx/dt)_t = (dx/dt)_base × max(0.1, 1 + η_t)
```

---

### 4.1 Device-to-Device (D2D): Variabilidad de Fabricación
*   **Origen Físico:** Tolerancias litográficas e imperfecciones estructurales quemadas durante la fabricación del chip.
*   **Formulación:** Perturbación Gaussiana Truncada en $[-2\sigma, +2\sigma]$ aplicada una sola vez por celda:
    $$ \epsilon_{\text{ON}} = \text{clip}\Big(\mathcal{N}(0, \sigma_{\text{D2D}}^2), \;-2\sigma_{\text{D2D}}, \;+2\sigma_{\text{D2D}}\Big) $$
    $$ R_{\text{ON}}^{(k)} = R_{\text{ON}} \cdot (1 + \epsilon_{\text{ON}}), \quad R_{\text{OFF}}^{(k)} = R_{\text{OFF}} \cdot (1 + \epsilon_{\text{OFF}}) $$
*   **Visualización:** Genera la **banda/abanico de dispersión suave** entre dispositivos de la población Monte Carlo.

---

### 4.2 Cycle-to-Cycle (C2C): Variabilidad Temporal Dinámica
*   **Origen Físico:** Naturaleza estocástica de la deriva de vacancias de oxígeno a lo largo del tiempo.
*   **Formulación:** Proceso estocástico de Ornstein-Uhlenbeck (reversión a la media) en cada paso de integración:
    $$ d\eta_t = -\theta \cdot \eta_t \cdot dt_{\text{ou}} + \sigma_{\text{C2C}} \sqrt{dt_{\text{ou}}} \cdot \mathcal{N}(0, 1) $$
    $$ \left(\frac{dx}{dt}\right)_t = \left(\frac{dx}{dt}\right)_{\text{base}} \cdot \max(0.1, \; 1.0 + \eta_t) $$
*   **Visualización:** Genera **ondulaciones y rugosidad estocástica continua** sobre la trayectoria $x(t)$ de una misma celda.

---

### 4.3 Resumen Comparativo D2D vs. C2C

| Dimensión | Device-to-Device (D2D) | Cycle-to-Cycle (C2C) |
| :--- | :--- | :--- |
| **Escala Temporal** | **Inmutable** (calculado al fabricar/instanciar) | **Dinámica** (fluctúa en cada paso $dt$) |
| **Dominio de Estudio** | Comparación entre *múltiples memristores* | Evolución temporal de un *mismo memristor* |
| **Parámetros Afectados** | $R_{\text{ON}}^{(k)}, R_{\text{OFF}}^{(k)}$ y velocidad de deriva base | Deriva instantánea $\left(\frac{dx}{dt}\right)_t$ vía $\eta_t$ |
| **Analogía Didáctica** | Diferente estatura entre distintos estudiantes | Variación del nivel de energía de un estudiante cada día |
| **Analogía Videojuegos** | *Vida inicial de distintos enemigos* al nacer | *Daño crítico aleatorio* por cada golpe del mismo personaje |

---

## 5. 📚 Referencias Bibliográficas y Limitaciones

### 5.1 Citas Científicas Principales
> 1. **Strukov, D. B., Snider, G. S., Stewart, D. R., & Williams, R. S. (2008).** *The missing memristor found.* **Nature**, 453(7191), 80-83.
> 2. **Biolek, Z., Biolek, D., & Biolková, V. (2009).** *SPICE model of memristor with window function.* **Radioengineering**, 18(2), 210-214.
> 3. **Joglekar, Y. N., & Wolf, S. J. (2009).** *The elusive memristor: properties of basic electrical circuits.* **European Journal of Physics**, 30(4), 661.

### 5.2 Simplificaciones Físicas del Modelo Base
*   **Deriva Lineal Uniforme:** Asume movilidad iónica ($\mu_v$) constante a lo largo del volumen del óxido.
*   **Frontera Unidimensional Plana:** Ignora geometrías 3D complejas y micro-filamentos percolativos.
*   **Ausencia de Voltaje Umbral:** Asume conmutación para cualquier voltaje $\gt 0$ (ignora energía de activación $E_a$).
*   **Sin Acoplamiento Térmico:** El calentamiento por efecto Joule no modula $\mu_v$.
