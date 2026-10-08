# Notas sobre la estructura de los JSON de configuración

## 1. Sección: Dispositivo (Memristor)
Actualmente, esta configuración se guarda en la raíz de `neuromorphic_lab/configs/last_session.json` (y perfiles como `strukov_ideal.json`).
Claves exactas:
- `_meta` (objeto con `schema_version`, `app_version`, `profile_name`, `saved_at`)
- `device_name`
- `material`
- `device_family`
- `model_name`
- `r_on`
- `r_on_unit`
- `r_off`
- `r_off_unit`
- `initial_state`
- `D_nm`
- `mu_v`
- `seed`
- `realism_mode_index`
- `window_type`
- `biolek_p`
- `enable_c2c`
- `c2c_sigma`
- `c2c_theta`
- `enable_d2d`
- `d2d_sigma`
- `enable_noise`
- `noise_std`
- `enable_volatile`
- `tau_relax`
- `x_eq`
- `enable_csv_validation`
- `signal` (objeto con `realtime_enabled`, `waveform`, `v0`, `f0`, `t_on`, `t_off`, `duration`, `dt_ms`)

## 2. Sección: Neurona (LIF)
Se encuentra en `neuromorphic_lab/configs/lif_config.json`.
Claves exactas:
- `_meta` (objeto con `type`, `saved_at`)
- `c_m_value`
- `c_m_unit`
- `r_series_value`
- `r_series_unit`
- `r_leak_value`
- `r_leak_unit`
- `v_rest`
- `v_th`
- `v_reset`
- `t_ref`
- `signal_panel` (objeto con `realtime_enabled`, `waveform`, `v0`, `f0`, `duration`, `dt_ms`)

## 3. Sección: Crossbar
*Nota: Actualmente no existe un archivo de configuración separado ni una sección en `last_session.json` para el Crossbar.* Las simulaciones de crossbar toman los parámetros del memristor activo. Si se añade en el futuro, se deberá definir su estructura.

## 4. Sección: WTA (Winner-Takes-All)
*Nota: Actualmente no existe una sección WTA en los JSON inspeccionados.* De implementarse, se deberá definir un esquema JSON.

## Conclusión de Verificación
- La GUI carga perfiles y sobreescribe/crea `last_session.json` correctamente al cerrarse.
- Se corrió el script de validación `24_jo2010_white.py`, lo que generó con éxito las figuras en `validaciones/figuras/`. El $R^2$ reportado en consola para LTP/LTD fue ~0.9242 (cumpliendo aproximadamente la cota exigida, demostrando que el lab base funciona).
