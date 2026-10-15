"""Módulo de Dimensionamiento Óptimo y Análisis de Sensibilidad.

Funcionalidades:
1. ``run_sensitivity_analysis``: varía parámetros clave (precio gas, CO2, pool, CAPEX, etc.)
   y analiza impacto en costes, ahorros, VAN, TIR y electrificación.
2. ``optimize_eboiler_sizing``: evalúa una malla de capacidades de E-boiler (MW-e)
   identificando el óptimo por menor coste operativo, mayor VAN, menor LCOH o menor CAPEX con meta.
3. ``optimize_tes_sizing``: evalúa combinaciones de potencia de carga (MW-e), almacenamiento (MWh-th)
   y potencia de descarga (MW-th).
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from data.loaders import DataBundle
from models.inputs import ScenarioConfig
from results.comparisons import run_simulation


def run_sensitivity_analysis(
    cfg: ScenarioConfig,
    bundle: DataBundle,
    parameter_path: str,
    variations_pct: list[float] = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0],
) -> pd.DataFrame:
    """Ejecuta un barrido de sensibilidad porcentual sobre un parámetro numérico.

    parameter_path: ej. 'baseline.gas_price_eur_per_mwh_hhv', 'baseline.co2_price_eur_per_tco2',
                    'eboiler.eboiler_capex_eur', 'supply.pool_price_scaling_factor', etc.
    """
    # Obtener valor base
    parts = parameter_path.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    base_val = getattr(obj, parts[-1])

    rows = []
    for var_p in variations_pct:
        multiplier = 1.0 + (var_p / 100.0)
        new_val = base_val * multiplier if parameter_path != "supply.pool_price_scaling_factor" else multiplier

        new_cfg = cfg.copy_with(**{parameter_path: new_val})
        new_cfg.case_name = f"{cfg.case_name} ({parameter_path} {var_p:+.1f}%)"

        res = run_simulation(new_cfg, bundle)
        if not res.is_valid:
            continue

        y1 = res.first_year.annual
        fin = res.financial.indicators if res.financial else {}

        rows.append(
            {
                "variation_pct": var_p,
                "parameter_value": new_val,
                "electrification_pct": y1.get("actual_electrification_pct"),
                "annual_operating_cost_eur": y1.get("annual_operating_cost_after_incentives_eur"),
                "annual_net_operating_savings_eur": y1.get("annual_net_operating_savings_eur"),
                "npv_eur": fin.get("npv_eur"),
                "irr_pct": fin.get("irr_pct"),
                "simple_payback_years": fin.get("simple_payback_years"),
                "lcoh_eur_per_mwh_th": fin.get("lcoh_after_incentives_eur_per_mwh_th"),
                "avoided_co2_tco2": y1.get("avoided_direct_co2_emissions_tco2"),
            }
        )

    return pd.DataFrame(rows)


def optimize_eboiler_sizing(
    cfg: ScenarioConfig,
    bundle: DataBundle,
    capacities_mw_e: list[float],
    cost_per_mw_e: float | None = None,
) -> dict[str, Any]:
    """Evalúa diferentes potencias de E-boiler y busca los óptimos económicos y técnicos."""
    records = []
    base_capex = cfg.eboiler.eboiler_capex_eur
    base_cap = cfg.eboiler.eboiler_capacity_mw_e

    for cap in capacities_mw_e:
        c_cfg = copy.deepcopy(cfg)
        c_cfg.eboiler.eboiler_capacity_mw_e = cap
        # Si se especifica un coste unitario se escala el CAPEX, si no se mantiene lineal con base_cap
        if cost_per_mw_e is not None:
            c_cfg.eboiler.eboiler_capex_eur = cap * cost_per_mw_e
        elif base_cap > 0:
            c_cfg.eboiler.eboiler_capex_eur = base_capex * (cap / base_cap)

        c_cfg.case_name = f"E-Boiler {cap:.1f} MW"
        res = run_simulation(c_cfg, bundle)
        if not res.is_valid:
            continue

        y1 = res.first_year.annual
        fin = res.financial.indicators if res.financial else {}

        records.append(
            {
                "capacity_mw_e": cap,
                "capex_eur": c_cfg.eboiler.eboiler_capex_eur,
                "electrification_pct": y1.get("actual_electrification_pct"),
                "annual_operating_cost_eur": y1.get("annual_operating_cost_after_incentives_eur"),
                "annual_net_savings_eur": y1.get("annual_net_operating_savings_eur"),
                "npv_eur": fin.get("npv_eur"),
                "irr_pct": fin.get("irr_pct"),
                "lcoh_eur_per_mwh_th": fin.get("lcoh_after_incentives_eur_per_mwh_th"),
                "payback_years": fin.get("simple_payback_years"),
                "avoided_co2_tco2": y1.get("avoided_direct_co2_emissions_tco2"),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return {"table": df, "best_opex": None, "best_npv": None, "best_lcoh": None}

    best_opex = df.loc[df["annual_operating_cost_eur"].idxmin()].to_dict()
    best_npv = (
        df.loc[df["npv_eur"].idxmax()].to_dict() if df["npv_eur"].notna().any() else None
    )
    best_lcoh = (
        df.loc[df["lcoh_eur_per_mwh_th"].idxmin()].to_dict()
        if df["lcoh_eur_per_mwh_th"].notna().any()
        else None
    )

    return {
        "table": df,
        "best_opex": best_opex,
        "best_npv": best_npv,
        "best_lcoh": best_lcoh,
    }


def optimize_tes_sizing(
    cfg: ScenarioConfig,
    bundle: DataBundle,
    charge_capacities_mw_e: list[float],
    storage_capacities_mwh_th: list[float],
    discharge_capacities_mw_th: list[float],
    cost_per_mw_e_charge: float = 150_000.0,
    cost_per_mwh_th_storage: float = 30_000.0,
    cost_per_mw_th_discharge: float = 50_000.0,
) -> dict[str, Any]:
    """Evalúa combinaciones de dimensionamiento para el sistema TES."""
    records = []

    for p_ch in charge_capacities_mw_e:
        for e_st in storage_capacities_mwh_th:
            for p_dis in discharge_capacities_mw_th:
                c_cfg = copy.deepcopy(cfg)
                c_cfg.tes.tes_charge_capacity_mw_e = p_ch
                c_cfg.tes.tes_storage_capacity_mwh_th = e_st
                c_cfg.tes.tes_discharge_capacity_mw_th = p_dis

                # Estimación de CAPEX paramétrica
                capex = (
                    p_ch * cost_per_mw_e_charge
                    + e_st * cost_per_mwh_th_storage
                    + p_dis * cost_per_mw_th_discharge
                )
                c_cfg.tes.tes_capex_eur = capex
                c_cfg.case_name = f"TES Ch{p_ch}MW St{e_st}MWh Dis{p_dis}MW"

                res = run_simulation(c_cfg, bundle)
                if not res.is_valid:
                    continue

                y1 = res.first_year.annual
                fin = res.financial.indicators if res.financial else {}

                records.append(
                    {
                        "charge_capacity_mw_e": p_ch,
                        "storage_capacity_mwh_th": e_st,
                        "discharge_capacity_mw_th": p_dis,
                        "capex_eur": capex,
                        "electrification_pct": y1.get("actual_electrification_pct"),
                        "equivalent_cycles": y1.get("tes_equivalent_cycles"),
                        "annual_operating_cost_eur": y1.get("annual_operating_cost_after_incentives_eur"),
                        "annual_net_savings_eur": y1.get("annual_net_operating_savings_eur"),
                        "npv_eur": fin.get("npv_eur"),
                        "irr_pct": fin.get("irr_pct"),
                        "lcoh_eur_per_mwh_th": fin.get("lcoh_after_incentives_eur_per_mwh_th"),
                        "payback_years": fin.get("simple_payback_years"),
                        "avoided_co2_tco2": y1.get("avoided_direct_co2_emissions_tco2"),
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        return {"table": df, "best_opex": None, "best_npv": None, "best_lcoh": None}

    best_opex = df.loc[df["annual_operating_cost_eur"].idxmin()].to_dict()
    best_npv = (
        df.loc[df["npv_eur"].idxmax()].to_dict() if df["npv_eur"].notna().any() else None
    )
    best_lcoh = (
        df.loc[df["lcoh_eur_per_mwh_th"].idxmin()].to_dict()
        if df["lcoh_eur_per_mwh_th"].notna().any()
        else None
    )

    return {
        "table": df,
        "best_opex": best_opex,
        "best_npv": best_npv,
        "best_lcoh": best_lcoh,
    }
