"""
neuromorphic_lab/tests/test_addressing.py
===========================================
Pruebas unitarias para decodificadores de dirección, drivers de línea y configs explícitas.
"""

import pytest
import numpy as np
from neurolab.crossbar.addressing import (
    AddressDecoder, RowDriver, ColumnDriver,
    ReadConfig, WriteConfig
)


def test_address_decoder():
    decoder = AddressDecoder(n_lines=4)
    assert decoder.n_bits == 2
    
    # Decodificar dirección 1 → [0, 1, 0, 0]
    one_hot = decoder.decode(1)
    np.testing.assert_array_equal(one_hot, [0.0, 1.0, 0.0, 0.0])
    assert decoder.last_address == 1
    assert decoder.address_to_binary(1) == "01"
    
    # Decodificar dirección 3 → [0, 0, 0, 1]
    one_hot3 = decoder.decode(3)
    np.testing.assert_array_equal(one_hot3, [0.0, 0.0, 0.0, 1.0])
    assert decoder.address_to_binary(3) == "11"
    
    # Dirección fuera de rango lanza ValueError
    with pytest.raises(ValueError):
        decoder.decode(4)
    with pytest.raises(ValueError):
        decoder.decode(-1)


def test_row_driver():
    driver = RowDriver(n_rows=4)
    driver.V_active = 2.0
    driver.V_inactive = 0.0
    
    one_hot = np.array([0.0, 1.0, 0.0, 0.0])
    V_rows = driver.drive(one_hot)
    
    np.testing.assert_array_equal(V_rows, [0.0, 2.0, 0.0, 0.0])
    assert driver.active_row == 1
    
    driver.reset()
    assert driver.active_row == -1


def test_column_driver():
    driver = ColumnDriver(n_cols=4)
    driver.V_active = -1.0
    driver.V_inactive = 0.0
    
    one_hot = np.array([0.0, 0.0, 1.0, 0.0])
    V_cols = driver.drive(one_hot)
    
    np.testing.assert_array_equal(V_cols, [0.0, 0.0, -1.0, 0.0])
    assert driver.active_col == 2


def test_read_config():
    cfg = ReadConfig(V_read=0.20, V_th=0.50)
    assert cfg.is_safe() is True
    assert "📖 Lectura" in cfg.describe()
    
    unsafe_cfg = ReadConfig(V_read=0.60, V_th=0.50)
    assert unsafe_cfg.is_safe() is False


def test_write_config():
    # Esquema V/2 con V_program = 2.0V → V_row = +1.0V, V_col = -1.0V
    cfg = WriteConfig(V_program=2.0, scheme='V2')
    assert cfg.V_row_active() == 1.0
    assert cfg.V_col_active() == -1.0
    assert "⚡ Escritura [V2]" in cfg.describe()
    
    # Esquema V/3 con V_program = 3.0V → V_row = +2.0V, V_col = -1.0V
    cfg_v3 = WriteConfig(V_program=3.0, scheme='V3')
    assert cfg_v3.V_row_active() == 2.0
    assert cfg_v3.V_col_active() == -1.0
    
    # Esquema 1T1R
    cfg_1t1r = WriteConfig(V_program=2.5, scheme='1T1R')
    assert cfg_1t1r.V_row_active() == 2.5
    assert cfg_1t1r.V_col_active() == 0.0
