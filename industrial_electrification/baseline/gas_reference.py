"""Escenario fósil de referencia (caldera de gas natural).

Convenciones PCS/PCI:
- El precio del gas, los otros costes y el factor de emisión se refieren a
  energía del combustible en PCS (HHV).
- El rendimiento de la caldera se expresa sobre PCI (LHV).
- Consumo PCS = calor útil / (rendimiento_PCI * PCI/PCS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.inputs import GasBaselineInputs


def gas_consumption_mwh_hhv(heat_mwh_th: np.ndarray, gas: GasBaselineInputs) -> np.ndarray:
    return np.asarray(heat_mwh_th, dtype=float) / (gas.fossil_boiler_efficiency_lhv * gas.lhv_hhv_ratio)


def fossil_unit_costs_eur_per_mwh_th(gas: GasBaselineInputs) -> dict[str, float]:
    """Costes variables fósiles por MWh-t útil (coste equivalente del gas)."""
    k = 1.0 / gas.fossil_efficiency_hhv  # MWh-PCS por MWh-t
    fuel = gas.gas_price_eur_per_mwh_hhv * k
    co2 = gas.gas_emission_factor_tco2_per_mwh_hhv * gas.co2_price_eur_per_tco2 * k
    other = gas.gas_other_cost_eur_per_mwh_hhv * k
    return {
        "gas_mwh_hhv_per_mwh_th": k,
        "fuel_eur_per_mwh_th": fuel,
        "co2_eur_per_mwh_th": co2,
        "other_eur_per_mwh_th": other,
        "total_eur_per_mwh_th": fuel + co2 + other,
        "co2_tco2_per_mwh_th": gas.gas_emission_factor_tco2_per_mwh_hhv * k,
    }


def fossil_cost_components(heat_mwh_th: np.ndarray, gas: GasBaselineInputs, prefix: str) -> pd.DataFrame:
    """Consumo, emisiones y costes (combustible, CO2, otros) de un calor fósil dado."""
    gas_mwh = gas_consumption_mwh_hhv(heat_mwh_th, gas)
    fuel = gas_mwh * gas.gas_price_eur_per_mwh_hhv
    co2_t = gas_mwh * gas.gas_emission_factor_tco2_per_mwh_hhv
    co2_cost = co2_t * gas.co2_price_eur_per_tco2
    other = gas_mwh * gas.gas_other_cost_eur_per_mwh_hhv
    return pd.DataFrame(
        {
            f"{prefix}_gas_consumption_mwh_hhv": gas_mwh,
            f"{prefix}_gas_fuel_cost_eur": fuel,
            f"{prefix}_co2_emissions_tco2": co2_t,
            f"{prefix}_co2_cost_eur": co2_cost,
            f"{prefix}_gas_other_cost_eur": other,
            f"{prefix}_total_cost_eur": fuel + co2_cost + other,
        }
    )


def compute_gas_reference(thermal_demand_mwh_th: np.ndarray, gas: GasBaselineInputs) -> pd.DataFrame:
    """Línea base horaria: toda la demanda cubierta con gas."""
    return fossil_cost_components(thermal_demand_mwh_th, gas, prefix="baseline")


def summarize_baseline(baseline_df: pd.DataFrame, annual_demand_mwh_th: float) -> dict[str, float]:
    total = float(baseline_df["baseline_total_cost_eur"].sum())
    return {
        "baseline_gas_consumption_mwh_hhv": float(baseline_df["baseline_gas_consumption_mwh_hhv"].sum()),
        "baseline_gas_fuel_cost_eur": float(baseline_df["baseline_gas_fuel_cost_eur"].sum()),
        "baseline_co2_emissions_tco2": float(baseline_df["baseline_co2_emissions_tco2"].sum()),
        "baseline_co2_cost_eur": float(baseline_df["baseline_co2_cost_eur"].sum()),
        "baseline_gas_other_cost_eur": float(baseline_df["baseline_gas_other_cost_eur"].sum()),
        "annual_baseline_cost_eur": total,
        "annual_baseline_heat_cost_eur_per_mwh_th": total / annual_demand_mwh_th if annual_demand_mwh_th > 0 else float("nan"),
    }
