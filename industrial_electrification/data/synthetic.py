"""Generador de perfiles de ejemplo sintéticos para pruebas y demostraciones."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_pv_profile_csv(filepath: str, n_hours: int = 8760) -> None:
    """Genera una curva solar sintética P50 (MWh/MWp·h) normalizada para España."""
    hours = np.arange(n_hours)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    # Variación estacional (pico en verano)
    season_factor = 0.6 + 0.4 * np.sin((day_of_year - 80) * 2 * np.pi / 365.0)

    # Campana diurna solar (de 6h a 20h)
    daily = np.maximum(0.0, np.sin((hour_of_day - 6) * np.pi / 14.0))
    daily[hour_of_day < 6] = 0.0
    daily[hour_of_day > 20] = 0.0

    # Perfil específico normalizado (pico ~0.85 MWh/MWp)
    pv_profile = daily * season_factor * 0.85

    df = pd.DataFrame({"pv_generation_mwh_e_per_mwp": pv_profile})
    df.to_csv(filepath, index=False, sep=";", decimal=",")


def generate_synthetic_thermal_demand_csv(
    filepath: str, n_hours: int = 8760, base_mw_th: float = 12.0
) -> None:
    """Genera una curva horaria sintética de demanda industrial."""
    hours = np.arange(n_hours)
    hour_of_day = hours % 24
    day_of_week = (hours // 24) % 7  # 0..6

    # Factor día laboral vs fin de semana
    is_weekend = day_of_week >= 5
    day_factor = np.where(is_weekend, 0.4, 1.0)

    # Factor turnos de producción (más bajo de noche)
    shift_factor = np.where((hour_of_day >= 6) & (hour_of_day <= 22), 1.0, 0.7)

    demand = base_mw_th * day_factor * shift_factor

    df = pd.DataFrame({"thermal_demand_mw_th": demand})
    df.to_csv(filepath, index=False, sep=";", decimal=",")
