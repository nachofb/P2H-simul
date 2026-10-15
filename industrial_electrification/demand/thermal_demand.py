"""Módulo común de demanda térmica (ambas alternativas)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.calendar import adapt_hourly_series_to_year
from models.inputs import DemandInputs


def build_thermal_demand(
    inputs: DemandInputs, calendar_df: pd.DataFrame, time_step_hours: float = 1.0
) -> tuple[pd.DataFrame, list[str]]:
    """Construye ``thermal_demand_mw_th[t]`` y ``thermal_demand_mwh_th[t]``.

    Relación: energy_mwh = power_mw * time_step_hours.
    """
    notes: list[str] = []
    n = len(calendar_df)
    year = int(calendar_df["year"].iloc[0])
    if inputs.method == "constant":
        power = np.full(n, inputs.thermal_demand_constant_mw_th, dtype=float)
    elif inputs.method == "weekday_weekend":
        weekend = calendar_df["day_of_week"].isin((6, 7)).to_numpy()
        power = np.where(
            weekend, inputs.weekend_thermal_demand_mw_th, inputs.weekday_thermal_demand_mw_th
        ).astype(float)
    elif inputs.method == "hourly_curve":
        values, adapt_notes = adapt_hourly_series_to_year(
            np.asarray(inputs.curve_values, dtype=float), year, name="Demanda térmica"
        )
        notes += adapt_notes
        if inputs.curve_kind == "power_mw_th":
            power = values
        else:
            power = values / time_step_hours
    else:  # pragma: no cover - protegido por Pydantic
        raise ValueError(f"Método de demanda desconocido: {inputs.method}")

    if len(power) != n:
        raise ValueError(f"Demanda térmica: {len(power)} valores para un calendario de {n} h.")
    if np.isnan(power).any() or (power < 0).any():
        raise ValueError("Demanda térmica: valores nulos o negativos no permitidos.")
    df = pd.DataFrame(
        {
            "thermal_demand_mw_th": power,
            "thermal_demand_mwh_th": power * time_step_hours,
        }
    )
    return df, notes


def annual_thermal_demand_mwh_th(demand_df: pd.DataFrame) -> float:
    return float(demand_df["thermal_demand_mwh_th"].sum())
