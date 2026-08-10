# PASO 12: Conclusión de la Validación Neuromórfica

## 1. Introducción y Arquitectura Evaluada
La primera fase del simulador ha acoplado con éxito un **Memristor basado en el modelo físico de Strukov** con una **Neurona Leaky Integrate-and-Fire (LIF) oscilatoria basada en un Modelo de Conmutación Umbral (TSM)**. El objetivo de esta fase era validar si la combinación de estos dispositivos de estado sólido puede emular fielmente la dinámica de los circuitos biológicos.

## 2. El Memristor: Más allá de un dispositivo electrónico
En la electrónica tradicional, los componentes son estáticos o carecen de memoria inherente sin un circuito de retroalimentación complejo. Los experimentos realizados (Pasos 7 a 10) han demostrado empíricamente que el memristor de TiO₂ posee **memoria analógica no volátil** impulsada por la deriva iónica. 

Su conductancia no depende únicamente del voltaje instantáneo que se le aplica, sino de la **integral histórica de la corriente** que lo ha atravesado ($\frac{dx}{dt} \propto I$). Esta dinámica interna le permite "recordar" experiencias pasadas sin requerir energía para mantener dicho estado, desmarcándolo radicalmente de un simple resistor variable y otorgándole un comportamiento físico con "memoria".

## 3. Integración en Procesamiento Basado en Eventos (Spiking)
La arquitectura implementada no opera con voltajes continuos tradicionales, sino mediante un esquema de codificación basado en picos (Spiking Neural Networks). 
Como se observó en la simulación:
1. La neurona TSM-LIF recibe estimulación en forma de **eventos discretos** (pulsos erráticos a 100 Hz).
2. El memristor, actuando como puerta de entrada, modula la magnitud de la corriente inyectada al capacitor de membrana ($V_c$).
3. La neurona integra esta carga en el tiempo y dispara un evento de salida (*Spike*) solo cuando se supera un umbral físico crítico ($V_{th} = 0.95\text{ V}$).

Esto demuestra que el sistema es capaz de participar en el **procesamiento de eventos asíncronos**, que es el pilar fundamental de la eficiencia energética del cerebro humano.

## 4. El Memristor como Sinapsis Artificial
La validación comparativa final (Paso 10) proporciona la evidencia definitiva de su función sináptica:
* **Estado HRS (Alta Resistencia - Conexión Débil):** Bloquea la integración de carga, inhibiendo por completo el disparo neuronal (0 Spikes).
* **Estado LRS (Baja Resistencia - Conexión Fuerte):** Facilita la carga rápida de la membrana, induciendo una alta tasa de disparo (4 Spikes).
* **Transición de Aprendizaje (Plasticidad):** Se demostró que estímulos eléctricos repetidos fuerzan un descenso gradual en la resistencia ($500\text{ k}\Omega \rightarrow 1.7\text{ k}\Omega$). Esta transición escalonada modula el *Firing Rate* de la neurona post-sináptica en tiempo real, replicando de forma precisa el fenómeno biológico de **Potenciación a Largo Plazo (LTP - Long-Term Potentiation)**.

## 5. Defensa del término "Neuromórfico"
Podemos defender formalmente el uso del término **"Neuromórfico"** en este proyecto porque el sistema desarrollado cumple con los tres requisitos fundamentales de la disciplina acuñada por Carver Mead:
1. **Morfología de Hardware Inspirada en el Cerebro:** El procesamiento (neurona TSM) y la memoria (memristor) están físicamente acoplados (Computación en memoria), rompiendo el cuello de botella de Von Neumann.
2. **Dinámica Sináptica:** El componente de interconexión exhibe plasticidad analógica y autónoma impulsada por leyes físicas locales.
3. **Comunicación por Eventos:** La transferencia de información ocurre mediante trenes de pulsos (Spikes), no por niveles lógicos estáticos.

**Conclusión:**
Los resultados gráficos e integraciones matemáticas de esta simulación confirman que no estamos ante un simple circuito de acondicionamiento de señales, sino ante **un bloque constructivo fundamental de hardware neuromórfico**, capaz de emular fielmente la interacción dinámica entre una sinapsis (Memristor) y un soma neuronal (TSM-LIF).
