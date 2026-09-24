"""
physical_crossbar.py
====================
Clase envoltorio para instanciar físicamente un arreglo de crossbar
utilizando objetos StrukovMemristor genuinos. 

Permite extraer propiedades circuitales a nivel de red, como la 
resistencia de Sneak Path basada en el estado resistivo actual
de todos los dispositivos físicos.
"""

import numpy as np
from typing import List
from memristor_simulator.models.strukov_model import StrukovMemristor
from memristor_simulator.config.parameters import StrukovParameters

class PhysicalCrossbarArray:
    def __init__(self, size: int, base_params: StrukovParameters, on_off_ratio: float = 10000.0):
        self.size = size
        self.grid: List[List[StrukovMemristor]] = []
        self.on_off_ratio = on_off_ratio
        
        # Instanciar la matriz física NxN
        for i in range(size):
            row = []
            for j in range(size):
                p_copy = StrukovParameters(
                    R_on=base_params.R_on,
                    R_off=base_params.R_on * on_off_ratio,
                    D=base_params.D,
                    mu_v=base_params.mu_v,
                    w_init=0.0 # Todos inicializados en OFF (w=0)
                )
                row.append(StrukovMemristor(p_copy))
            self.grid.append(row)

    def set_pattern_all_ones(self):
        """Programa toda la red físicamente a estado ON (HRS->LRS)"""
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i][j].x = 1.0
                self.grid[i][j]._recalculate_resistance()

    def set_pattern_all_zeros(self):
        """Programa toda la red físicamente a estado OFF (LRS->HRS)"""
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i][j].x = 0.0
                self.grid[i][j]._recalculate_resistance()

    def get_sneak_resistance(self, target_row: int = 0, target_col: int = 0) -> float:
        """
        Calcula la resistencia del camino de fuga parásito (Sneak Path)
        LEYENDO FÍSICAMENTE la 'memristance' de cada objeto StrukovMemristor
        en la red, según el modelo V/2 de lectura.
        """
        if self.size <= 1:
            return float('inf')

        # G2: Conductancia total de celdas no seleccionadas en la fila objetivo
        G2 = 0.0
        for j in range(self.size):
            if j != target_col:
                G2 += 1.0 / self.grid[target_row][j].memristance
        R2 = 1.0 / G2 if G2 > 0 else float('inf')

        # G3: Conductancia total del resto de la matriz
        G3 = 0.0
        for i in range(self.size):
            if i != target_row:
                for j in range(self.size):
                    if j != target_col:
                        G3 += 1.0 / self.grid[i][j].memristance
        R3 = 1.0 / G3 if G3 > 0 else float('inf')

        # G4: Conductancia total de celdas no seleccionadas en la columna objetivo
        G4 = 0.0
        for i in range(self.size):
            if i != target_row:
                G4 += 1.0 / self.grid[i][target_col].memristance
        R4 = 1.0 / G4 if G4 > 0 else float('inf')

        return R2 + R3 + R4
