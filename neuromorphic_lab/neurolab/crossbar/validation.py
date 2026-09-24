"""
neurolab/crossbar/validation.py
================================
Scripts de validación del crossbar.

Valida:
1. I = G^T · V (operación ideal)
2. Selectividad de programación (1T1R, V/2, V/3)
3. Resistencias de línea (error vs R)
4. Escalabilidad (4×4 a 128×128)
5. Variabilidad D2D y C2C
"""

import numpy as np
import time
from .core import Crossbar


def validate_read_operation(crossbar, n_tests=100, verbose=True):
    """Valida I = G^T · V."""
    errors = []
    for _ in range(n_tests):
        V_rows = np.random.uniform(-0.5, 0.5, crossbar.n_rows)
        I_sim = crossbar.read_ideal(V_rows)
        I_ref = crossbar.G_matrix.T @ V_rows
        errors.append(np.max(np.abs(I_sim - I_ref)))
    
    errors = np.array(errors)
    
    if verbose:
        print("\n" + "="*70)
        print("  VALIDACIÓN 1: I = G^T · V")
        print("="*70)
        print(f"  Tests:            {n_tests}")
        print(f"  Error máx:        {errors.max():.2e} A")
        print(f"  Error medio:      {errors.mean():.2e} A")
        print(f"  Resultado:        {'[PASS]' if errors.max() < 1e-12 else '[FAIL]'}")
        print("="*70)
    
    return errors


def validate_programming_selectivity(crossbar, mode='V2', verbose=True):
    """Valida selectividad de programación."""
    G_before = crossbar.G_matrix.copy()
    
    # Aplicar programación
    if mode == '1T1R':
        crossbar.program_1T1R(1, 1, V_program=2.0, dt=1e-3)
    elif mode == 'V2':
        crossbar.program_V2(1, 1, V_program=2.0, dt=1e-3)
    elif mode == 'V3':
        crossbar.program_V3(1, 1, V_program=2.0, dt=1e-3)
    
    G_after = crossbar.G_matrix.copy()
    delta = (G_after - G_before) * 1e6  # uS
    
    if verbose:
        print("\n" + "="*70)
        print(f"  VALIDACIÓN 2: Selectividad ({mode})")
        print("="*70)
        print("  dG (uS):")
        for i in range(crossbar.n_rows):
            row_str = f"    Fila {i+1}: "
            for j in range(crossbar.n_cols):
                row_str += f"{delta[i,j]:+7.3f} "
            print(row_str)
        
        delta_target = abs(delta[1, 1])
        mask_half = np.ones_like(delta, dtype=bool)
        mask_half[1, :] = False
        mask_half[:, 1] = False
        mask_half[1, 1] = True
        delta_half = np.abs(delta[mask_half])
        delta_half = delta_half[delta_half > 0]
        
        mask_idle = ~mask_half
        delta_idle = np.abs(delta[mask_idle])
        
        print(f"\n  dG objetivo (M22):    {delta_target:.4f} uS")
        if len(delta_half) > 0:
            print(f"  dG half-selected:     {delta_half.mean():.4f} uS")
            ratio = delta_target / delta_half.mean() if delta_half.mean() > 0 else np.inf
            print(f"  Ratio obj/half:       {ratio:.2f}x")
        print(f"  dG idle (max):        {delta_idle.max():.4f} uS")
        print(f"  Resultado:            {'[PASS]' if delta_target > 0.1 else '[FAIL]'}")
        print("="*70)
    
    return delta


def validate_sneak_paths(crossbar, V_rows=None, verbose=True):
    """Valida el efecto de sneak paths."""
    if V_rows is None:
        V_rows = np.ones(crossbar.n_rows) * 0.1
    I_ideal = crossbar.read_ideal(V_rows)
    
    crossbar.cfg.enable_sneak_paths = True
    I_sneak = crossbar.read(V_rows)
    crossbar.cfg.enable_sneak_paths = False
    
    error = np.abs(I_sneak - I_ideal) / np.maximum(np.abs(I_ideal), 1e-12) * 100
    
    if verbose:
        print("\n" + "="*70)
        print("  VALIDACIÓN: Sneak Paths")
        print("="*70)
        print(f"  V_rows:         {V_rows}")
        print(f"  I_ideal (uA):   {I_ideal * 1e6}")
        print(f"  I_sneak (uA):   {I_sneak * 1e6}")
        print(f"  Error (%):      {error}")
        print(f"  Error medio:    {error.mean():.2f} %")
        print("="*70)
    
    return I_ideal, I_sneak, error


def validate_line_resistance(crossbar, V_rows=None, R_H_values=None, verbose=True):
    """Valida efecto de resistencias de línea."""
    if V_rows is None:
        V_rows = np.array([0.1, 0.2, 0.1, 0.2])
    if R_H_values is None:
        R_H_values = [0.0, 0.1, 1.0, 5.0, 10.0, 50.0]
        
    results = []
    for R_H in R_H_values:
        crossbar.cfg.R_line_H = R_H
        crossbar.cfg.enable_line_resistance = (R_H > 0)
        if hasattr(crossbar, 'line_model'):
            crossbar.line_model.R_H = R_H
        
        I_ideal = crossbar.read_ideal(V_rows)
        I_real = crossbar.read(V_rows)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.mean(np.abs(I_real - I_ideal) / np.maximum(np.abs(I_ideal), 1e-12) * 100)
        
        results.append({'R_H': R_H, 'error_percent': error})
    
    crossbar.cfg.enable_line_resistance = False
    crossbar.cfg.R_line_H = 0.0
    if hasattr(crossbar, 'line_model'):
        crossbar.line_model.R_H = 0.0
    
    if verbose:
        print("\n" + "="*70)
        print("  VALIDACIÓN 3: Resistencias de línea")
        print("="*70)
        print(f"  {'R_H (Ohm)':<15} {'Error (%)':<15}")
        print("-"*70)
        for r in results:
            print(f"  {r['R_H']:<15.3f} {r['error_percent']:<15.3f}")
        print("="*70)
    
    return results


def validate_scalability(sizes=None, verbose=True):
    """Valida escalabilidad 4x4 a 128x128."""
    if sizes is None:
        sizes = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
    
    results = []
    for n_rows, n_cols in sizes:
        cb = Crossbar(n_rows=n_rows, n_cols=n_cols)
        
        V = np.ones(n_rows) * 0.1
        
        t0 = time.perf_counter()
        for _ in range(10):
            I = cb.read_ideal(V)
        t_read = (time.perf_counter() - t0) / 10 * 1000  # ms
        
        t0 = time.perf_counter()
        cb.program_V2(n_rows // 2, n_cols // 2, V_program=2.0, dt=1e-3)
        t_prog = (time.perf_counter() - t0) * 1000  # ms
        
        results.append({
            'size': f'{n_rows}×{n_cols}',
            'n_cells': n_rows * n_cols,
            't_read_ms': t_read,
            't_program_ms': t_prog,
        })
    
    if verbose:
        print("\n" + "="*70)
        print("  VALIDACIÓN 4: Escalabilidad")
        print("="*70)
        print(f"  {'Size':<12} {'Celdas':<10} {'t_read (ms)':<15} {'t_prog (ms)':<12}")
        print("-"*70)
        for r in results:
            print(f"  {r['size']:<12} {r['n_cells']:<10} "
                  f"{r['t_read_ms']:<15.4f} {r['t_program_ms']:<12.4f}")
        print("="*70)
    
    return results


def validate_d2d_c2c_variability(crossbar, verbose=True):
    """Valida variabilidad D2D y C2C."""
    G = crossbar.G_matrix * 1e6
    cv_d2d = G.std() / max(1e-12, G.mean()) * 100.0
    
    g_pulses = []
    cell_00 = crossbar.cells[0][0]
    for _ in range(10):
        cell_00.update(V_applied=1.2, dt=1e-4)
        g_pulses.append(cell_00.conductance * 1e6)
    
    g_pulses = np.array(g_pulses)
    cv_c2c = g_pulses.std() / max(1e-12, g_pulses.mean()) * 100.0
    
    if verbose:
        print("\n" + "="*70)
        print("  VALIDACIÓN: Variabilidad D2D y C2C")
        print("="*70)
        print(f"  Dispersión D2D (CV):    {cv_d2d:.2f} %")
        print(f"  Fluctuación C2C (CV):   {cv_c2c:.2f} %")
        print("="*70)
    
    return {'cv_d2d': cv_d2d, 'cv_c2c': cv_c2c}


def validate_all(verbose=True):
    """Ejecuta todas las validaciones."""
    if verbose:
        print("\n" + "█"*70)
        print("  VALIDACIÓN COMPLETA DEL CROSSBAR")
        print("█"*70)
    
    cb = Crossbar(n_rows=4, n_cols=4)
    
    validate_read_operation(cb, verbose=verbose)
    
    for mode in ['1T1R', 'V2', 'V3']:
        cb.reset()
        validate_programming_selectivity(cb, mode=mode, verbose=verbose)
    
    V_rows = np.array([0.1, 0.2, 0.1, 0.2])
    validate_line_resistance(
        cb, V_rows,
        R_H_values=[0.0, 0.1, 1.0, 5.0, 10.0, 50.0],
        verbose=verbose,
    )
    
    validate_scalability(verbose=verbose)
    
    if verbose:
        print("\n" + "█"*70)
        print("  VALIDACIÓN COMPLETADA")
        print("█"*70)
