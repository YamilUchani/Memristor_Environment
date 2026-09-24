"""
neurolab.gui.tests.catalog
==========================
Catálogo oficial de pruebas de validación en tiempo real para la interfaz gráfica neurolab.
"""

TESTS_CATALOG = [
    # ── LECTURA ─────────────────────────────────────────────────────────────
    {
        'id': '03', 'name': 'Barrido de Voltaje', 'category': 'Lectura',
        'params': [
            {'key': 'V_min', 'label': 'V min (V)', 'type': 'float', 'default': -0.1, 'min': -2.0, 'max': 0.0, 'step': 0.05},
            {'key': 'V_max', 'label': 'V max (V)', 'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 2.0, 'step': 0.05},
            {'key': 'n_points', 'label': 'N Puntos', 'type': 'int', 'default': 41, 'min': 10, 'max': 500, 'step': 10},
            {'key': 'G_target', 'label': 'G (μS)', 'type': 'float', 'default': 200.0, 'min': 1.0, 'max': 5000.0, 'step': 10.0},
        ],
        'func': 'run_v_sweep'
    },
    {
        'id': '04', 'name': 'Barrido de Conductancia', 'category': 'Lectura',
        'params': [
            {'key': 'G_min', 'label': 'G min (μS)', 'type': 'float', 'default': 62.5, 'min': 1.0, 'max': 500.0, 'step': 10.0},
            {'key': 'G_max', 'label': 'G max (μS)', 'type': 'float', 'default': 500.0, 'min': 50.0, 'max': 2000.0, 'step': 50.0},
            {'key': 'V_in', 'label': 'V_in (V)', 'type': 'float', 'default': 0.5, 'min': 0.01, 'max': 2.0, 'step': 0.05},
            {'key': 'n_points', 'label': 'N Puntos', 'type': 'int', 'default': 41, 'min': 10, 'max': 500, 'step': 10},
        ],
        'func': 'run_g_sweep'
    },
    {
        'id': '05', 'name': 'Lectura No Destructiva', 'category': 'Lectura',
        'params': [
            {'key': 'V_pico', 'label': 'V pico (mV)', 'type': 'float', 'default': 100.0, 'min': 10.0, 'max': 1000.0, 'step': 10.0},
            {'key': 'freq', 'label': 'Frecuencia (Hz)', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 100.0, 'step': 0.5},
            {'key': 'T', 'label': 'Duración (s)', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.5},
        ],
        'func': 'run_non_destructive'
    },
    {
        'id': '06', 'name': 'Retención de Peso', 'category': 'Lectura',
        'params': [
            {'key': 'G_prog', 'label': 'G programado (μS)', 'type': 'float', 'default': 200.0, 'min': 10.0, 'max': 2000.0, 'step': 10.0},
            {'key': 'T_espera', 'label': 'Duración (s)', 'type': 'float', 'default': 10.0, 'min': 1.0, 'max': 100.0, 'step': 1.0},
        ],
        'func': 'run_retention'
    },
    # ── ESCRITURA ────────────────────────────────────────────────────────────
    {
        'id': '07', 'name': 'Programación LTP (+1V)', 'category': 'Escritura',
        'params': [
            {'key': 'V_pulse', 'label': 'V_pulse (V)', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 3.0, 'step': 0.1},
            {'key': 'n_pulses', 'label': 'N Pulsos', 'type': 'int', 'default': 40, 'min': 5, 'max': 200, 'step': 5},
        ],
        'func': 'run_ltp'
    },
    {
        'id': '08', 'name': 'Programación LTD (−1V)', 'category': 'Escritura',
        'params': [
            {'key': 'V_pulse', 'label': 'V_pulse (V)', 'type': 'float', 'default': -1.0, 'min': -3.0, 'max': -0.1, 'step': 0.1},
            {'key': 'n_pulses', 'label': 'N Pulsos', 'type': 'int', 'default': 40, 'min': 5, 'max': 200, 'step': 5},
        ],
        'func': 'run_ltd'
    },
    {
        'id': '09', 'name': 'Ciclo Reversible (LTP → LTD → LTP)', 'category': 'Escritura',
        'params': [
            {'key': 'n_ltp', 'label': 'N Pulsos LTP', 'type': 'int', 'default': 50, 'min': 10, 'max': 200, 'step': 10},
            {'key': 'n_ltd', 'label': 'N Pulsos LTD', 'type': 'int', 'default': 50, 'min': 10, 'max': 200, 'step': 10},
        ],
        'func': 'run_cycle'
    },
    {
        'id': '15', 'name': 'vs Resistencia Pura (Histéresis)', 'category': 'Escritura',
        'params': [
            {'key': 'V_max', 'label': 'V max (V)', 'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 5.0, 'step': 0.5},
            {'key': 'n_points', 'label': 'N Puntos', 'type': 'int', 'default': 300, 'min': 50, 'max': 1000, 'step': 50},
        ],
        'func': 'run_vs_resistor'
    },
    {
        'id': '16', 'name': 'Ventana de Lectura (Read Window)', 'category': 'Escritura',
        'params': [
            {'key': 'V_max', 'label': 'V max (V)', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 3.0, 'step': 0.1},
            {'key': 'n_points', 'label': 'N Puntos', 'type': 'int', 'default': 50, 'min': 10, 'max': 200, 'step': 10},
        ],
        'func': 'run_read_window'
    },
    {
        'id': '17', 'name': 'Ventana de Escritura (Write Window)', 'category': 'Escritura',
        'params': [
            {'key': 'V_max', 'label': 'V max (V)', 'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 5.0, 'step': 0.1},
            {'key': 'n_points', 'label': 'N Puntos', 'type': 'int', 'default': 50, 'min': 10, 'max': 200, 'step': 10},
        ],
        'func': 'run_write_window'
    },
    {
        'id': '18', 'name': 'Programación Analógica Precisa', 'category': 'Escritura',
        'params': [
            {'key': 'G_target', 'label': 'G_target (μS)', 'type': 'float', 'default': 150.0, 'min': 10.0, 'max': 1000.0, 'step': 10.0},
            {'key': 'max_iter', 'label': 'Máx Iteraciones', 'type': 'int', 'default': 100, 'min': 10, 'max': 500, 'step': 10},
        ],
        'func': 'run_closed_loop_prog'
    },
    # ── INTEGRACIÓN ─────────────────────────────────────────────────────────
    {
        'id': '10', 'name': 'Crossbar → Neurona LIF (4 Paneles)', 'category': 'Integración',
        'params': [
            {'key': 'V_in', 'label': 'V_in (V)', 'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 2.0, 'step': 0.1},
            {'key': 'G_target', 'label': 'G_11 (μS)', 'type': 'float', 'default': 200.0, 'min': 10.0, 'max': 1000.0, 'step': 10.0},
            {'key': 'T_ms', 'label': 'Duración (ms)', 'type': 'float', 'default': 100.0, 'min': 10.0, 'max': 500.0, 'step': 10.0},
        ],
        'func': 'run_lif_integration'
    },
    {
        'id': '11', 'name': 'Integración con STDP', 'category': 'Integración',
        'params': [
            {'key': 'A_plus', 'label': 'A+', 'type': 'float', 'default': 0.05, 'min': 0.001, 'max': 0.5, 'step': 0.01},
            {'key': 'A_minus', 'label': 'A−', 'type': 'float', 'default': -0.025, 'min': -0.5, 'max': -0.001, 'step': 0.01},
            {'key': 'tau_plus', 'label': 'τ+ (ms)', 'type': 'float', 'default': 17.0, 'min': 1.0, 'max': 100.0, 'step': 1.0},
            {'key': 'tau_minus', 'label': 'τ− (ms)', 'type': 'float', 'default': 34.0, 'min': 1.0, 'max': 100.0, 'step': 1.0},
        ],
        'func': 'run_stdp'
    },
    # ── ADICIONALES ─────────────────────────────────────────────────────────
    {
        'id': '12', 'name': 'Escalabilidad Matricial N×N', 'category': 'Adicionales',
        'params': [],
        'func': 'run_matrix_api'
    },
    {
        'id': '19', 'name': 'Variabilidad D2D (Monte Carlo)', 'category': 'Adicionales',
        'params': [
            {'key': 'N_devices', 'label': 'N Dispositivos', 'type': 'int', 'default': 100, 'min': 10, 'max': 500, 'step': 10},
            {'key': 'sigma_pct', 'label': 'Variabilidad σ (%)', 'type': 'float', 'default': 5.0, 'min': 0.5, 'max': 30.0, 'step': 0.5},
        ],
        'func': 'run_d2d'
    },
    {
        'id': '20', 'name': 'Endurance (1000 Ciclos)', 'category': 'Adicionales',
        'params': [
            {'key': 'N_cycles', 'label': 'N Ciclos', 'type': 'int', 'default': 1000, 'min': 50, 'max': 5000, 'step': 50},
        ],
        'func': 'run_endurance'
    },
    # ── CROSSBAR 2×2 ────────────────────────────────────────────────────────
    {
        'id': '2x2_01', 'name': 'Operación Matricial 2×2', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'V1', 'label': 'V1 Sensor S1 (V)', 'type': 'float', 'default': 0.8, 'min': 0.0, 'max': 5.0, 'step': 0.1},
            {'key': 'V2', 'label': 'V2 Sensor S2 (V)', 'type': 'float', 'default': 0.4, 'min': 0.0, 'max': 5.0, 'step': 0.1},
        ],
        'func': 'run_2x2_01'
    },
    {
        'id': '2x2_02', 'name': 'Barrido de Voltaje (Fila 1)', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'V_min', 'label': 'V min (V)', 'type': 'float', 'default': -0.5, 'min': -2.0, 'max': 0.0, 'step': 0.05},
            {'key': 'V_max', 'label': 'V max (V)', 'type': 'float', 'default': 0.5, 'min': 0.0, 'max': 2.0, 'step': 0.05},
            {'key': 'n_points', 'label': 'N Puntos', 'type': 'int', 'default': 41, 'min': 10, 'max': 500, 'step': 10},
        ],
        'func': 'run_2x2_02'
    },
    {
        'id': '2x2_03', 'name': 'Barrido de Conductancia G11', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'G_min', 'label': 'G min (μS)', 'type': 'float', 'default': 62.5, 'min': 1.0, 'max': 500.0, 'step': 10.0},
            {'key': 'G_max', 'label': 'G max (μS)', 'type': 'float', 'default': 500.0, 'min': 50.0, 'max': 2000.0, 'step': 50.0},
            {'key': 'V1', 'label': 'V1 (V)', 'type': 'float', 'default': 0.8, 'min': 0.0, 'max': 5.0, 'step': 0.1},
            {'key': 'V2', 'label': 'V2 (V)', 'type': 'float', 'default': 0.4, 'min': 0.0, 'max': 5.0, 'step': 0.1},
        ],
        'func': 'run_2x2_03'
    },
    {
        'id': '2x2_04', 'name': 'Lectura No Destructiva 2×2', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'V_pico', 'label': 'V pico (mV)', 'type': 'float', 'default': 50.0, 'min': 10.0, 'max': 1000.0, 'step': 10.0},
            {'key': 'freq', 'label': 'Frecuencia (Hz)', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 100.0, 'step': 0.5},
            {'key': 'T', 'label': 'Duración (s)', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.5},
        ],
        'func': 'run_2x2_04'
    },
    {
        'id': '2x2_05', 'name': 'Programación LTP Selectiva', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'n_pulses', 'label': 'N Pulsos LTP', 'type': 'int', 'default': 40, 'min': 5, 'max': 200, 'step': 5},
            {'key': 'V_pulse', 'label': 'V Pulsos (V)', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 3.0, 'step': 0.1},
        ],
        'func': 'run_2x2_05'
    },
    {
        'id': '2x2_06', 'name': 'Programación LTD Selectiva', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'n_pulses', 'label': 'N Pulsos LTD', 'type': 'int', 'default': 40, 'min': 5, 'max': 200, 'step': 5},
            {'key': 'V_pulse', 'label': 'V Pulsos (V)', 'type': 'float', 'default': -1.0, 'min': -3.0, 'max': -0.1, 'step': 0.1},
        ],
        'func': 'run_2x2_06'
    },
    {
        'id': '2x2_07', 'name': 'Ciclo Reversible en 2×2', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'n_ltp', 'label': 'N Pulsos LTP', 'type': 'int', 'default': 50, 'min': 10, 'max': 200, 'step': 10},
            {'key': 'n_ltd', 'label': 'N Pulsos LTD', 'type': 'int', 'default': 50, 'min': 10, 'max': 200, 'step': 10},
        ],
        'func': 'run_2x2_07'
    },
    {
        'id': '2x2_08', 'name': 'Efecto Sneak Paths 2×2', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'seed', 'label': 'Semilla Aleatoria', 'type': 'int', 'default': 42, 'min': 1, 'max': 1000, 'step': 1},
        ],
        'func': 'run_2x2_08'
    },
    {
        'id': '2x2_09', 'name': 'Resistencias de Línea 2×2', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'line_R', 'label': 'R línea (Ω)', 'type': 'float', 'default': 50.0, 'min': 0.0, 'max': 500.0, 'step': 5.0},
        ],
        'func': 'run_2x2_09'
    },
    {
        'id': '2x2_10', 'name': '2 Neuronas LIF Dinámicas', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'T', 'label': 'Duración (ms)', 'type': 'float', 'default': 100.0, 'min': 10.0, 'max': 500.0, 'step': 10.0},
            {'key': 'freq', 'label': 'Frecuencia (Hz)', 'type': 'float', 'default': 10.0, 'min': 1.0, 'max': 100.0, 'step': 1.0},
        ],
        'func': 'run_2x2_10'
    },
    {
        'id': '2x2_11', 'name': 'Ventana STDP en 2×2', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'max_percent', 'label': 'Máx ΔG (%)', 'type': 'float', 'default': 0.5, 'min': 0.01, 'max': 10.0, 'step': 0.1},
        ],
        'func': 'run_2x2_11'
    },
    {
        'id': '2x2_12', 'name': 'Cross-talk e Interferencia', 'category': 'Crossbar 2×2',
        'params': [],
        'func': 'run_2x2_12'
    },
    {
        'id': '2x2_13', 'name': 'Escalabilidad 1×1 → 2×2', 'category': 'Crossbar 2×2',
        'params': [],
        'func': 'run_2x2_13'
    },
    {
        'id': '2x2_14', 'name': 'Uniformidad de Celdas 2×2', 'category': 'Crossbar 2×2',
        'params': [
            {'key': 'G_target', 'label': 'G Target (μS)', 'type': 'float', 'default': 300.0, 'min': 10.0, 'max': 1000.0, 'step': 10.0},
        ],
        'func': 'run_2x2_14'
    },
    {
        'id': '2x2_15', 'name': 'Comparación 1×1 vs 2×2', 'category': 'Crossbar 2×2',
        'params': [],
        'func': 'run_2x2_15'
    },
    # ── CROSSBAR 4×4 ────────────────────────────────────────────────────────
    {
        'id': '4x4_01', 'name': 'Operación Matricial I = G^T·V (4×4)', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_01'
    },
    {
        'id': '4x4_02', 'name': 'Barrido de Voltaje (4×4)', 'category': 'Crossbar 4×4',
        'params': [
            {'key': 'V_min', 'label': 'V_min (V)', 'type': 'float', 'default': -0.5, 'min': -2.0, 'max': 0.0, 'step': 0.05},
            {'key': 'V_max', 'label': 'V_max (V)', 'type': 'float', 'default': 0.5, 'min': 0.0, 'max': 2.0, 'step': 0.05},
        ],
        'func': 'run_4x4_02'
    },
    {
        'id': '4x4_03', 'name': 'Barrido de G₁₁ (4×4)', 'category': 'Crossbar 4×4',
        'params': [
            {'key': 'G_min', 'label': 'G_min (μS)', 'type': 'float', 'default': 62.5, 'min': 10.0, 'max': 500.0, 'step': 10.0},
            {'key': 'G_max', 'label': 'G_max (μS)', 'type': 'float', 'default': 500.0, 'min': 50.0, 'max': 1000.0, 'step': 50.0},
        ],
        'func': 'run_4x4_03'
    },
    {
        'id': '4x4_04', 'name': 'Lectura No Destructiva (4×4)', 'category': 'Crossbar 4×4',
        'params': [
            {'key': 'V_pico', 'label': 'V pico (mV)', 'type': 'float', 'default': 50.0, 'min': 10.0, 'max': 500.0, 'step': 10.0},
        ],
        'func': 'run_4x4_04'
    },
    {
        'id': '4x4_05', 'name': 'LTP Selectivo (4×4)', 'category': 'Crossbar 4×4',
        'params': [
            {'key': 'n_pulses', 'label': 'N Pulsos LTP', 'type': 'int', 'default': 40, 'min': 5, 'max': 200, 'step': 5},
        ],
        'func': 'run_4x4_05'
    },
    {
        'id': '4x4_06', 'name': 'LTD Selectivo (4×4)', 'category': 'Crossbar 4×4',
        'params': [
            {'key': 'n_pulses', 'label': 'N Pulsos LTD', 'type': 'int', 'default': 40, 'min': 5, 'max': 200, 'step': 5},
        ],
        'func': 'run_4x4_06'
    },
    {
        'id': '4x4_07', 'name': 'Ciclo Reversible (4×4)', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_07'
    },
    {
        'id': '4x4_08', 'name': 'Efecto Sneak Paths (4×4)', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_08'
    },
    {
        'id': '4x4_09', 'name': 'Resistencias de Línea (4×4)', 'category': 'Crossbar 4×4',
        'params': [
            {'key': 'line_R', 'label': 'R línea (Ω)', 'type': 'float', 'default': 50.0, 'min': 0.0, 'max': 500.0, 'step': 5.0},
        ],
        'func': 'run_4x4_09'
    },
    {
        'id': '4x4_10', 'name': '4 Neuronas LIF Dinámicas', 'category': 'Crossbar 4×4',
        'params': [
            {'key': 'T', 'label': 'Duración (ms)', 'type': 'float', 'default': 100.0, 'min': 10.0, 'max': 500.0, 'step': 10.0},
        ],
        'func': 'run_4x4_10'
    },
    {
        'id': '4x4_11', 'name': 'Ventana STDP en 4×4', 'category': 'Crossbar 4×4',
        'params': [
            {'key': 'max_percent', 'label': 'Máx ΔG (%)', 'type': 'float', 'default': 0.5, 'min': 0.01, 'max': 10.0, 'step': 0.1},
        ],
        'func': 'run_4x4_11'
    },
    {
        'id': '4x4_12', 'name': 'Cross-talk e Interferencia (4×4)', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_12'
    },
    {
        'id': '4x4_13', 'name': 'Escalabilidad N×N', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_13'
    },
    {
        'id': '4x4_14', 'name': 'Uniformidad de Celdas (4×4)', 'category': 'Crossbar 4×4',
        'params': [
            {'key': 'G_target', 'label': 'G Target (μS)', 'type': 'float', 'default': 300.0, 'min': 10.0, 'max': 1000.0, 'step': 10.0},
        ],
        'func': 'run_4x4_14'
    },
    {
        'id': '4x4_15', 'name': 'Matriz Identidad', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_15'
    },
    {
        'id': '4x4_16', 'name': 'Matriz Diagonal', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_16'
    },
    {
        'id': '4x4_17', 'name': 'Patrón X (4×4)', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_17'
    },
    {
        'id': '4x4_18', 'name': 'Patrón T (4×4)', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_18'
    },
    {
        'id': '4x4_19', 'name': 'Clasificación Patrón 4×4 ("H")', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_19'
    },
    {
        'id': '4x4_20', 'name': 'Comparación 1×1 vs 2×2 vs 4×4', 'category': 'Crossbar 4×4',
        'params': [],
        'func': 'run_4x4_20'
    },
    # ── PREZIOSO 2015 (Validación Experimental) ─────────────────────────────
    {
        'id': 'prez_s3a',
        'name': 'Fig. S3a — Curvas I-V Virgen',
        'category': 'Prezioso 2015',
        'params': [
            {'key': 'n_devices', 'label': 'N Dispositivos', 'type': 'int',
             'default': 5, 'min': 2, 'max': 20, 'step': 1},
            {'key': 'V_max', 'label': 'V max (V)', 'type': 'float',
             'default': 1.0, 'min': 0.5, 'max': 2.0, 'step': 0.05},
            {'key': 'n_points', 'label': 'N Puntos', 'type': 'int',
             'default': 100, 'min': 50, 'max': 500, 'step': 50},
            {'key': 'G_mean_uS', 'label': 'G media (μS)', 'type': 'float',
             'default': 0.45, 'min': 0.1, 'max': 2.0, 'step': 0.05},
            {'key': 'G_sigma_uS', 'label': 'G sigma (μS)', 'type': 'float',
             'default': 0.08, 'min': 0.01, 'max': 0.5, 'step': 0.01},
            {'key': 'seed', 'label': 'Semilla', 'type': 'int',
             'default': 42, 'min': 1, 'max': 9999, 'step': 1},
        ],
        'func': 'run_prez_s3a'
    },
    {
        'id': 'prez_s3b',
        'name': 'Fig. S3b — Mapa Conductancias 4×4',
        'category': 'Prezioso 2015',
        'params': [
            {'key': 'use_active_crossbar', 'label': 'Usar Matriz Crossbar 4×4 Activa', 'type': 'bool', 'default': True},
            {'key': 'n_rows', 'label': 'N Filas (si no activa)', 'type': 'int',
             'default': 4, 'min': 2, 'max': 20, 'step': 1},
            {'key': 'n_cols', 'label': 'N Columnas (si no activa)', 'type': 'int',
             'default': 4, 'min': 2, 'max': 20, 'step': 1},
            {'key': 'G_mean_uS', 'label': 'G media sim (μS)', 'type': 'float',
             'default': 0.45, 'min': 0.1, 'max': 2.0, 'step': 0.05},
            {'key': 'G_sigma_uS', 'label': 'G sigma sim (μS)', 'type': 'float',
             'default': 0.08, 'min': 0.01, 'max': 0.5, 'step': 0.01},
            {'key': 'seed', 'label': 'Semilla', 'type': 'int',
             'default': 42, 'min': 1, 'max': 9999, 'step': 1},
        ],
        'func': 'run_prez_s3b'
    },
    {
        'id': 'prez_s3c',
        'name': 'Fig. S3c — Histograma de Conductancias',
        'category': 'Prezioso 2015',
        'params': [
            {'key': 'use_active_crossbar', 'label': 'Usar Matriz Crossbar 4×4 Activa', 'type': 'bool', 'default': True},
            {'key': 'n_rows', 'label': 'N Filas (si no activa)', 'type': 'int',
             'default': 4, 'min': 2, 'max': 20, 'step': 1},
            {'key': 'n_cols', 'label': 'N Columnas (si no activa)', 'type': 'int',
             'default': 4, 'min': 2, 'max': 20, 'step': 1},
            {'key': 'G_mean_uS', 'label': 'G media sim (μS)', 'type': 'float',
             'default': 0.45, 'min': 0.1, 'max': 2.0, 'step': 0.05},
            {'key': 'G_sigma_uS', 'label': 'G sigma sim (μS)', 'type': 'float',
             'default': 0.08, 'min': 0.01, 'max': 0.5, 'step': 0.01},
            {'key': 'n_bins', 'label': 'N Bins', 'type': 'int',
             'default': 12, 'min': 5, 'max': 30, 'step': 1},
            {'key': 'seed', 'label': 'Semilla', 'type': 'int',
             'default': 42, 'min': 1, 'max': 9999, 'step': 1},
        ],
        'func': 'run_prez_s3c'
    },
]
