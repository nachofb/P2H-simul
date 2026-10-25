"""Módulo reutilizable de suministro eléctrico (red + PV onsite).

Cada alternativa usa su propia configuración ``ElectricitySupplyInputs``.

Coste unitario de red (EUR/MWh-e):
    grid_unit_cost[t] = pool[t] + ajustes[t] + peajes[tarifa, periodo(t)]
                        + garantías_origen + otros_costes
Los términos de potencia contratada NO se incluyen (se tratarían por separado).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.calendar import adapt_hourly_series_to_year
from data.loaders import (
    DataBundle,
    adjustment_series_for_year,
    network_tariff_for,
    pool_price_series_for_year,
)
from models.inputs import ElectricitySupplyInputs


def pv_generation_profile(
    supply: ElectricitySupplyInputs, year: int, n_hours: int, time_step_hours: float = 1.0
) -> tuple[np.ndarray, list[str]]:
    """Generación PV horaria (MWh-e por intervalo)."""
    if not supply.has_onsite_pv:
        return np.zeros(n_hours), []
    values, notes = adapt_hourly_series_to_year(
        np.asarray(supply.pv_profile_values, dtype=float), year, name="Perfil PV"
    )
    if supply.pv_profile_kind == "specific_mwh_e_per_mwp":
        gen = values * supply.pv_installed_capacity_mwp
        max_specific = float(values.max()) / time_step_hours
        if max_specific > 1.05:
            notes.append(
                f"AVISO Perfil PV: producción específica máxima {max_specific:.2f} MWh/MWp·h > 1; "
                "¿el archivo contiene energía absoluta?"
            )
    else:
        gen = values
        if supply.pv_installed_capacity_mwp > 0:
            peak = float(values.max()) / time_step_hours
            if peak > 1.05 * supply.pv_installed_capacity_mwp:
                notes.append(
                    f"AVISO Perfil PV: potencia máxima {peak:.2f} MW supera la potencia instalada "
                    f"{supply.pv_installed_capacity_mwp:.2f} MWp."
                )
    if len(gen) != n_hours:
        raise ValueError(f"Perfil PV: {len(gen)} valores para {n_hours} h.")
    return gen, notes


def grid_capacity_profile(
    supply: ElectricitySupplyInputs, tariff_period: np.ndarray
) -> np.ndarray:
    """Capacidad de red disponible (MW). np.inf = sin restricción específica."""
    n = len(tariff_period)
    if not supply.grid_capacity_limit_enabled:
        return np.full(n, np.inf)
    if not supply.grid_capacity_by_period_enabled:
        return np.full(n, float(supply.grid_capacity_mw))
    caps = supply.grid_capacity_by_period_mw()
    return np.array([caps[int(p)] for p in tariff_period], dtype=float)


def build_electricity_supply(
    supply: ElectricitySupplyInputs,
    calendar_df: pd.DataFrame,
    bundle: DataBundle,
    year_index: int = 0,
    time_step_hours: float = 1.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Series horarias de precios eléctricos, PV y capacidad de red de un año."""
    year = int(calendar_df["year"].iloc[0])
    n = len(calendar_df)
    notes: list[str] = []
    if str(supply.pool_price_scenario).strip().upper() == "REAL":
        data_year: int | str = "LTM"
    else:
        data_year = (supply.pool_price_year + year_index) if supply.pool_price_year else year
    pool, pool_notes = pool_price_series_for_year(
        bundle.pool, data_year, supply.pool_price_scenario, target_year=year,
        fill_authorized=supply.pool_missing_data_fill_authorized,
        max_fill_hours=supply.pool_max_fill_hours,
    )
    notes += pool_notes
    pool = pool * supply.pool_price_scaling_factor
    if supply.pool_price_scaling_factor != 1.0:
        notes.append(f"Pool escalado x{supply.pool_price_scaling_factor} (sensibilidad).")

    if supply.adjustment_price_mode == "fixed":
        adjustment = np.full(n, float(supply.adjustment_fixed_eur_per_mwh_e))
    else:
        if bundle.adjustment_values is None:
            raise ValueError("Serie horaria de servicios de ajuste no cargada.")
        adjustment, adj_notes = adjustment_series_for_year(bundle.adjustment_values, year)
        notes += adj_notes

    if supply.is_electrointensive and bundle.network_tariffs_electrointensive is not None:
        tariff_table = bundle.network_tariffs_electrointensive
        notes.append("Aplicados peajes y cargos reducidos para consumidor electrointensivo.")
    else:
        tariff_table = bundle.network_tariffs

    tariff_by_period = network_tariff_for(tariff_table, supply.network_tariff)
    period = calendar_df["tariff_period"].to_numpy()
    network = np.array([tariff_by_period[int(p)] for p in period], dtype=float)
    goo = supply.guarantees_of_origin_cost_eur_per_mwh_e if supply.guarantees_of_origin_enabled else 0.0
    other = supply.grid_other_cost_eur_per_mwh_e

    pv_gen, pv_notes = pv_generation_profile(supply, year, n, time_step_hours)
    notes += pv_notes
    df = pd.DataFrame(
        {
            "pool_price_eur_per_mwh_e": pool,
            "adjustment_price_eur_per_mwh_e": adjustment,
            "network_tariff_eur_per_mwh_e": network,
            "guarantees_of_origin_cost_eur_per_mwh_e": np.full(n, goo),
            "grid_other_cost_eur_per_mwh_e": np.full(n, other),
        }
    )
    df["grid_unit_cost_eur_per_mwh_e"] = (
        df["pool_price_eur_per_mwh_e"]
        + df["adjustment_price_eur_per_mwh_e"]
        + df["network_tariff_eur_per_mwh_e"]
        + df["guarantees_of_origin_cost_eur_per_mwh_e"]
        + df["grid_other_cost_eur_per_mwh_e"]
    )
    df["pv_generation_mwh_e"] = pv_gen
    df["pv_price_eur_per_mwh_e"] = supply.pv_price_eur_per_mwh_e if supply.has_onsite_pv else 0.0
    df["grid_capacity_available_mw"] = grid_capacity_profile(supply, period)
    if df.drop(columns=["grid_capacity_available_mw"]).isna().any().any():
        raise ValueError("Suministro eléctrico: valores nulos en series de precios.")
    return df, notes


def _array_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    """Devuelve (media, P10, P50, P90) de un array numérico, o ceros si está vacío."""
    if len(arr) == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean_val = float(np.mean(arr))
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    return mean_val, float(p10), float(p50), float(p90)


def compute_electricity_price_stats(
    supply_df: pd.DataFrame,
    tariff_period: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Calcula estadísticos (Media, P10, P50, P90) de precios eléctricos para el año completo y por periodos P1-P6.

    Distingue entre:
    - Precio Pool (mercado diario)
    - Componentes adicionales: Peajes y cargos, Servicios de ajuste, Otros costes de red, Garantías de origen
    - Precio Completo de Red (All-in = Pool + Peajes y cargos + Ajustes + Otros costes + GdO)
    """
    periods = np.asarray(tariff_period, dtype=int)
    n_total = len(supply_df)

    slices: list[tuple[str, np.ndarray]] = [
        (f"Anual ({n_total} h)", np.ones(n_total, dtype=bool)),
    ]
    for p in range(1, 7):
        slices.append((f"P{p}", periods == p))

    pool_all = supply_df["pool_price_eur_per_mwh_e"].to_numpy(dtype=float)
    tariff_all = supply_df["network_tariff_eur_per_mwh_e"].to_numpy(dtype=float)
    adj_all = supply_df["adjustment_price_eur_per_mwh_e"].to_numpy(dtype=float)
    other_all = supply_df["grid_other_cost_eur_per_mwh_e"].to_numpy(dtype=float)
    goo_all = (
        supply_df["guarantees_of_origin_cost_eur_per_mwh_e"].to_numpy(dtype=float)
        if "guarantees_of_origin_cost_eur_per_mwh_e" in supply_df.columns
        else (
            supply_df["grid_unit_cost_eur_per_mwh_e"].to_numpy(dtype=float)
            - pool_all
            - tariff_all
            - adj_all
            - other_all
        )
    )
    full_all = supply_df["grid_unit_cost_eur_per_mwh_e"].to_numpy(dtype=float)

    rows = []
    for label, mask in slices:
        pool_mean, pool_p10, pool_p50, pool_p90 = _array_stats(pool_all[mask])
        tar_mean, tar_p10, tar_p50, tar_p90 = _array_stats(tariff_all[mask])
        adj_mean, adj_p10, adj_p50, adj_p90 = _array_stats(adj_all[mask])
        oth_mean, oth_p10, oth_p50, oth_p90 = _array_stats(other_all[mask])
        goo_mean, goo_p10, goo_p50, goo_p90 = _array_stats(goo_all[mask])
        full_mean, full_p10, full_p50, full_p90 = _array_stats(full_all[mask])

        rows.append(
            {
                "period": label,
                "hours": int(mask.sum()),
                # Precio Pool
                "pool_mean_eur_per_mwh_e": pool_mean,
                "pool_p10_eur_per_mwh_e": pool_p10,
                "pool_p50_eur_per_mwh_e": pool_p50,
                "pool_p90_eur_per_mwh_e": pool_p90,
                # Peajes y Cargos
                "network_tariff_mean_eur_per_mwh_e": tar_mean,
                "network_tariff_p10_eur_per_mwh_e": tar_p10,
                "network_tariff_p50_eur_per_mwh_e": tar_p50,
                "network_tariff_p90_eur_per_mwh_e": tar_p90,
                # Servicios de Ajuste
                "adjustment_mean_eur_per_mwh_e": adj_mean,
                "adjustment_p10_eur_per_mwh_e": adj_p10,
                "adjustment_p50_eur_per_mwh_e": adj_p50,
                "adjustment_p90_eur_per_mwh_e": adj_p90,
                # Otros Costes de Red
                "grid_other_cost_mean_eur_per_mwh_e": oth_mean,
                "grid_other_cost_p10_eur_per_mwh_e": oth_p10,
                "grid_other_cost_p50_eur_per_mwh_e": oth_p50,
                "grid_other_cost_p90_eur_per_mwh_e": oth_p90,
                # Garantías de Origen (GdO)
                "guarantees_of_origin_mean_eur_per_mwh_e": goo_mean,
                "guarantees_of_origin_p10_eur_per_mwh_e": goo_p10,
                "guarantees_of_origin_p50_eur_per_mwh_e": goo_p50,
                "guarantees_of_origin_p90_eur_per_mwh_e": goo_p90,
                # Precio Completo de Red (All-in)
                "full_price_mean_eur_per_mwh_e": full_mean,
                "full_price_p10_eur_per_mwh_e": full_p10,
                "full_price_p50_eur_per_mwh_e": full_p50,
                "full_price_p90_eur_per_mwh_e": full_p90,
            }
        )

    return pd.DataFrame(rows)

