# Modelo Constitutivo Seleccionado: VTEAM (Voltage ThrEshold Adaptive Memristor)

Para el entorno de simulación se ha seleccionado el modelo **VTEAM** propuesto por S. Kvatinsky et al. (2015). Este modelo ha sido ampliamente validado en la literatura ya que describe de manera precisa la dinámica del memristor controlado por voltaje, abarcando asimetría, no linealidad, y comportamientos umbrales comunes en dispositivos físicos reales.

## 1. Ecuaciones Principales

La evolución de la variable de estado interna $x$ (que puede representar la longitud de la región dopada en un modelo HP) en función del voltaje aplicado $v(t)$ está dada por la siguiente ecuación diferencial por partes:

$$
\frac{dx}{dt} = 
\begin{cases} 
k_{\text{off}} \left( \frac{v(t)}{v_{\text{off}}} - 1 \right)^{\alpha_{\text{off}}} f_{\text{off}}(x), & \text{si } v(t) > v_{\text{off}} \\ 
0, & \text{si } v_{\text{on}} \le v(t) \le v_{\text{off}} \\ 
k_{\text{on}} \left( \frac{v(t)}{v_{\text{on}}} - 1 \right)^{\alpha_{\text{on}}} f_{\text{on}}(x), & \text{si } v(t) < v_{\text{on}} 
\end{cases}
$$

La resistencia instantánea $R(t)$ se relaciona linealmente con la variable de estado mediante:

$$
R(x) = R_{\text{on}} + \frac{R_{\text{off}} - R_{\text{on}}}{x_{\text{off}} - x_{\text{on}}} (x - x_{\text{on}})
$$

La corriente a través del dispositivo entonces sigue la ley de Ohm generalizada:
$$ i(t) = \frac{v(t)}{R(t)} $$

Donde $f_{\text{on}}(x)$ y $f_{\text{off}}(x)$ son funciones de ventana (por ejemplo, ventana de Biolek o Joglekar) añadidas para limitar el crecimiento de $x$ estrictamente al dominio $[x_{\text{on}}, x_{\text{off}}]$.

## 2. Parámetros Físicos Típicos

Para un comportamiento característico estilo HP TiO2 u óxidos de metales de transición (TMO), se contemplan los siguientes valores de referencia para la paramétrica del modelo:

| Parámetro | Descripción | Valor Típico |
| :--- | :--- | :--- |
| **$R_{\text{on}}$** | Resistencia de estado bajo (Encendido) | $100 \ \Omega - 1 \ \text{k}\Omega$ |
| **$R_{\text{off}}$** | Resistencia de estado alto (Apagado) | $16 \ \text{k}\Omega - 100 \ \text{k}\Omega$ |
| **$v_{\text{on}}$** | Voltaje umbral de encendido (SET) | $-0.5 \ \text{V}$ a $-1.5 \ \text{V}$ |
| **$v_{\text{off}}$** | Voltaje umbral de apagado (RESET) | $0.5 \ \text{V}$ a $1.5 \ \text{V}$ |
| **$\alpha_{\text{on}}, \alpha_{\text{off}}$** | Parámetros de aceleración no lineal | $3 - 4$ (adimensional) |
| **$k_{\text{on}}, k_{\text{off}}$** | Constantes de velocidad de migración | $\sim -8 \times 10^{-5} \ \text{m/s}$ y $8 \times 10^{-5} \ \text{m/s}$ |
| **$x_{\text{on}}$** | Límite inferior de la variable de estado | $0 \ \text{nm}$ |
| **$x_{\text{off}}$** | Límite superior de la variable de estado | $10 \ \text{nm} - 100 \ \text{nm}$ ($w$ total) |
| **$x_0$** | Estado inicial | $x_{\text{off}} / 2$ (Variable, según el dopaje inicial) |

## 3. Referencias
1. Kvatinsky, S., Ramadan, M., Friedman, E. G., & Kolodny, A. (2015). **VTEAM: A General Model for Voltage-Controlled Memristors**. *IEEE Transactions on Circuits and Systems II: Express Briefs*, 62(8), 786-790. https://doi.org/10.1109/TCSII.2015.2433536
2. Biolek, Z., Biolek, D., & Biolková, V. (2009). **SPICE model of memristor with nonlinear dopant drift**. *Radioengineering*, 18(2), 210-214. (Relevante para su implementación de función de ventana).
