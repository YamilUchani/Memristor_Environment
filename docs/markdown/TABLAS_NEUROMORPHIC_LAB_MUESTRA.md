# Muestra de tablas y verificacion — Neuromorphic Lab

Generado a partir de la ejecucion local del laboratorio.

## Resultado de verificacion

| Comprobacion | Resultado |
|---|---:|
| Tests de `neuromorphic_lab/tests` | 14 passed |
| `validate_lif_rc.py` | OK, exit 0 |
| `validate_lif_experiments.py` | OK, exit 0 |
| `validate_resistor_lif.py` | OK, exit 0 |
| Archivos Python incluidos en el bundle | 51 |

## Tablas CSV de validacion

Los CSV originales no tienen fila de encabezados. En todos ellos, la columna 1 representa tiempo y la columna 2 representa la magnitud medida.

### `neuromorphic_lab/data for validation/CvsT.csv`

Filas: 136, columnas: 2. Interpretacion: capacitancia frente a tiempo.

| tiempo | capacitancia |
|---:|---:|
| 0.002770862 | 0.403983797 |
| 0.008231027 | 1.544402683 |
| 0.013008671 | 2.556040585 |
| 0.017786316 | 3.581077665 |
| 0.022905220 | 4.637100342 |

### `neuromorphic_lab/data for validation/VvsT.csv`

Filas: 111, columnas: 2. Interpretacion: voltaje frente a tiempo.

| tiempo | voltaje |
|---:|---:|
| 0.004268146 | 0.122512299 |
| 0.008744273 | 0.254695885 |
| 0.012406559 | 0.360353309 |
| 0.016068845 | 0.471042039 |
| 0.019731131 | 0.569152504 |

### `neuromorphic_lab/data for validation/WDvsT.csv`

Filas: 107, columnas: 2. Interpretacion: ancho de dominio frente a tiempo.

| tiempo | ancho_dominio |
|---:|---:|
| 0.001672646 | 0.102118735 |
| 0.009202784 | 0.112301383 |
| 0.016732922 | 0.134543110 |
| 0.024263059 | 0.169048310 |
| 0.031450918 | 0.212037423 |

## Tabla de validacion RC-LIF

Parametros: corriente de entrada 8 uA, tau 50 ms, duracion 300 ms.

| dt (s) | MAE (V) | MSE (V^2) | error max (V) | error relativo |
|---:|---:|---:|---:|---:|
| 1.0e-03 | 6.501386e-02 | 8.592904e-03 | 1.971821e-01 | 1.126220e-01 |
| 1.0e-04 | 6.624878e-02 | 8.868802e-03 | 1.998148e-01 | 1.149313e-01 |
| 1.0e-05 | 6.632781e-02 | 8.886587e-03 | 1.999815e-01 | 1.150939e-01 |

## Tabla de tren de spikes LIF

| experimento | corriente | duracion | spikes | frecuencia media | ISI medio |
|---|---:|---:|---:|---:|---:|
| Etapa 2.9 | 15 uA | 500 ms | 9 | 18.00 Hz | 52.20 ms |

La etapa 2.8 tambien genero un spike individual y la etapa 2.10 genero la curva frecuencia-corriente.

## Tabla de circuito resistencia-LIF

Duracion: 200 ms. Fuente: pulsos de 2 V, 50 Hz, ancho 10 ms.

| resistencia de entrada | spikes | frecuencia media | corriente maxima |
|---:|---:|---:|---:|
| 1 MOhm | 38 | 190.0 Hz | 2.08 uA |
| 10 MOhm | 31 | 155.0 Hz | 0.21 uA |
| 50 MOhm | 20 | 100.0 Hz | 0.04 uA |

La tendencia observada es consistente con el modelo: al aumentar la resistencia disminuyen la corriente inyectada y la frecuencia de disparo.

## Archivos generados por la verificacion

- `neuromorphic_lab/graficas/lif_rc_validation.png`
- `neuromorphic_lab/graficas/lif_etapa28_spike.png`
- `neuromorphic_lab/graficas/lif_etapa29_train.png`
- `neuromorphic_lab/graficas/lif_etapa210_FI_curve.png`
- `neuromorphic_lab/validate_resistor_lif_results.png`

Para consultar todos los fuentes, tests y validadores en un solo archivo, abre `CODIGO_NEUROLAB_COMPLETO.md` en la raiz del repositorio.
