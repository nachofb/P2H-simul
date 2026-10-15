"""Evaluación económica y financiera de proyectos industriales.

Cálculos incluidos:
- Anualidad de amortización financiera (CAPEX anualizado).
- Flujo de caja del proyecto (unlevered, pre-tax).
- VAN (Valor Actual Neto) y TIR (Tasa Interna de Retorno).
- Payback simple (años y meses) y Payback descontado.
- LCOH plurianual descontado (antes y después de incentivos).
- Coste medio anual del calor (EUR/MWh-th).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from models.inputs import ScenarioConfig
from models.results import FinancialResult, YearResult


def annualized_capex(capex_eur: float, interest_rate_pct: float, amortization_years: int) -> float:
    """Anualidad financiera de amortización del CAPEX (fórmula de anualidad constante)."""
    if capex_eur <= 0 or amortization_years <= 0:
        return 0.0
    r = interest_rate_pct / 100.0
    n = amortization_years
    if r == 0:
        return capex_eur / n
    return capex_eur * r / (1.0 - (1.0 + r) ** (-n))


def compute_irr(cash_flows: list[float]) -> float | None:
    """Calcula la TIR mediante numpy_financial / np.roots / aproximación iterativa."""
    if len(cash_flows) < 2:
        return None
    # Comprobar si hay al menos un cambio de signo
    signs = [1 if cf > 0 else (-1 if cf < 0 else 0) for cf in cash_flows]
    has_pos = any(s > 0 for s in signs)
    has_neg = any(s < 0 for s in signs)
    if not (has_pos and has_neg):
        return None

    # Método de bisección / Newton robusto para IRR
    def npv_rate(r: float) -> float:
        return sum(cf / ((1.0 + r) ** t) for t, cf in enumerate(cash_flows))

    # Búsqueda de rango entre -0.9 y 5.0 (de -90% a 500%)
    low, high = -0.9, 5.0
    f_low, f_high = npv_rate(low), npv_rate(high)
    if f_low * f_high > 0:
        # Explorar con granularidad
        rates = np.linspace(-0.8, 2.0, 100)
        npvs = [npv_rate(r) for r in rates]
        sign_changes = np.where(np.diff(np.sign(npvs)))[0]
        if len(sign_changes) == 0:
            return None
        idx = sign_changes[0]
        low, high = rates[idx], rates[idx + 1]

    # Bisección fina
    for _ in range(100):
        mid = (low + high) / 2.0
        f_mid = npv_rate(mid)
        if abs(f_mid) < 1e-4:
            return mid * 100.0
        if npv_rate(low) * f_mid < 0:
            high = mid
        else:
            low = mid
    return mid * 100.0


def compute_financial_evaluation(
    cfg: ScenarioConfig,
    year_results: list[YearResult],
) -> FinancialResult:
    notes: list[str] = []
    lifetime = cfg.simulation.asset_lifetime_years
    disc_rate = cfg.simulation.discount_rate_pct / 100.0
    capex_initial = cfg.asset_capex_eur
    amort_annual = annualized_capex(
        capex_initial,
        cfg.simulation.interest_rate_pct,
        cfg.simulation.amortization_years,
    )

    if not year_results or not all(y.is_valid for y in year_results):
        notes.append("Resultados de simulación incompletos o infactibles; evaluación financiera no generada.")
        return FinancialResult(cash_flows=pd.DataFrame(), indicators={}, notes=notes)

    # Si se simularon menos años que la vida útil, el último año simulado se proyecta
    n_sim = len(year_results)
    sim_data = [y.annual for y in year_results]

    rows = []
    # Año 0: Inversión inicial
    rows.append(
        {
            "project_year": 0,
            "calendar_year": cfg.simulation.simulation_start_year - 1,
            "capex_eur": -capex_initial,
            "gross_savings_eur": 0.0,
            "opex_incremental_eur": 0.0,
            "innovation_fund_revenue_eur": 0.0,
            "replacement_costs_eur": 0.0,
            "residual_value_eur": 0.0,
            "net_cash_flow_eur": -capex_initial,
            "discount_factor": 1.0,
            "discounted_cash_flow_eur": -capex_initial,
            "thermal_demand_mwh_th": 0.0,
            "electrified_heat_mwh_th": 0.0,
            "project_net_cost_eur": capex_initial,
            "baseline_cost_eur": 0.0,
        }
    )

    # Diccionarios de costes de reposición y capex adicional
    rep_dict = {item.project_year: item.amount_eur for item in cfg.finance.replacement_costs}
    add_capex_dict = {item.project_year: item.amount_eur for item in cfg.finance.additional_capex}

    cum_discounted = -capex_initial
    discounted_payback_years = None
    simple_payback_years = None

    first_year_net_savings = year_results[0].annual["annual_net_operating_savings_eur"]
    if first_year_net_savings > 0 and capex_initial > 0:
        simple_payback_years = capex_initial / first_year_net_savings
    elif capex_initial == 0:
        simple_payback_years = 0.0

    sum_pv_costs_with_if = capex_initial
    sum_pv_costs_no_if = capex_initial
    sum_pv_heat_delivered = 0.0
    sum_pv_total_demand = 0.0
    sum_pv_baseline_costs = 0.0

    # Desgloses de valor presente para LCOH
    pv_capex = capex_initial
    pv_opex_tech = 0.0
    pv_ppa_pv = 0.0
    pv_pool = 0.0
    pv_adjustment = 0.0
    pv_tariffs = 0.0
    pv_gdo = 0.0
    pv_other_elec = 0.0
    pv_fossil_gas = 0.0
    pv_fossil_co2 = 0.0
    pv_fossil_other = 0.0
    pv_if_revenue = 0.0

    pv_base_gas = 0.0
    pv_base_co2 = 0.0
    pv_base_other = 0.0

    for py in range(1, lifetime + 1):
        cal_yr = cfg.simulation.simulation_start_year + (py - 1)
        sim_idx = min(py - 1, n_sim - 1)
        y_ann = sim_data[sim_idx]

        # Comprobar si Innovation Fund sigue activo en este año
        if_active = cfg.innovation_fund.is_active_in_year(cal_yr, cfg.simulation.simulation_start_year)
        if_rev = y_ann["annual_innovation_fund_revenue_eur"] if if_active else 0.0

        gross_sav = y_ann["annual_gross_savings_eur"]
        inc_opex = cfg.incremental_annual_opex_eur
        rep_cost = rep_dict.get(py, 0.0)
        add_capex = add_capex_dict.get(py, 0.0)
        res_val = cfg.finance.residual_value_eur if py == lifetime else 0.0

        operating_savings_year = gross_sav + if_rev - inc_opex
        net_cf = operating_savings_year - rep_cost - add_capex + res_val

        dfactor = 1.0 / ((1.0 + disc_rate) ** py)
        disc_cf = net_cf * dfactor

        prev_cum = cum_discounted
        cum_discounted += disc_cf
        if discounted_payback_years is None and prev_cum < 0 and cum_discounted >= 0:
            fraction = -prev_cum / disc_cf if disc_cf > 0 else 0.0
            discounted_payback_years = (py - 1) + fraction

        # Para LCOH
        # Costes netos del proyecto en el año:
        # Coste operativo + OPEX incremental + Reposición + Capex adicional - IF
        op_cost_before_if = y_ann["annual_operating_cost_before_incentives_eur"]
        cost_with_if = op_cost_before_if + inc_opex + rep_cost + add_capex - if_rev - res_val
        cost_no_if = op_cost_before_if + inc_opex + rep_cost + add_capex - res_val

        sum_pv_costs_with_if += cost_with_if * dfactor
        sum_pv_costs_no_if += cost_no_if * dfactor
        sum_pv_heat_delivered += y_ann["annual_electrified_heat_mwh_th"] * dfactor
        sum_pv_total_demand += y_ann["annual_thermal_demand_mwh_th"] * dfactor
        sum_pv_baseline_costs += y_ann["annual_baseline_cost_eur"] * dfactor

        # Desglose de valor presente para LCOH por componentes del proyecto
        pv_capex += (add_capex + rep_cost - res_val) * dfactor
        pv_opex_tech += inc_opex * dfactor
        pv_ppa_pv += y_ann.get("annual_pv_electricity_cost_eur", 0.0) * dfactor
        pv_pool += y_ann.get("annual_pool_cost_eur", 0.0) * dfactor
        pv_adjustment += y_ann.get("annual_adjustment_cost_eur", 0.0) * dfactor
        pv_tariffs += y_ann.get("annual_network_tariff_cost_eur", 0.0) * dfactor
        pv_gdo += y_ann.get("annual_guarantees_of_origin_cost_eur", 0.0) * dfactor
        pv_other_elec += y_ann.get("annual_other_elec_cost_eur", 0.0) * dfactor
        pv_fossil_gas += y_ann.get("annual_fossil_gas_fuel_cost_eur", 0.0) * dfactor
        pv_fossil_co2 += y_ann.get("annual_fossil_co2_cost_eur", 0.0) * dfactor
        pv_fossil_other += y_ann.get("annual_fossil_gas_other_cost_eur", 0.0) * dfactor
        pv_if_revenue += if_rev * dfactor

        # Desglose de valor presente para LCOH de referencia (Baseline)
        pv_base_gas += y_ann.get("annual_baseline_fuel_cost_eur", 0.0) * dfactor
        pv_base_co2 += y_ann.get("annual_baseline_co2_cost_eur", 0.0) * dfactor
        pv_base_other += y_ann.get("annual_baseline_gas_other_cost_eur", 0.0) * dfactor

        rows.append(
            {
                "project_year": py,
                "calendar_year": cal_yr,
                "capex_eur": -add_capex,
                "gross_savings_eur": gross_sav,
                "opex_incremental_eur": inc_opex,
                "innovation_fund_revenue_eur": if_rev,
                "replacement_costs_eur": rep_cost,
                "residual_value_eur": res_val,
                "net_cash_flow_eur": net_cf,
                "discount_factor": dfactor,
                "discounted_cash_flow_eur": disc_cf,
                "thermal_demand_mwh_th": y_ann["annual_thermal_demand_mwh_th"],
                "electrified_heat_mwh_th": y_ann["annual_electrified_heat_mwh_th"],
                "project_net_cost_eur": cost_with_if,
                "baseline_cost_eur": y_ann["annual_baseline_cost_eur"],
            }
        )

    cf_df = pd.DataFrame(rows)
    cf_df["cumulative_net_cash_flow_eur"] = cf_df["net_cash_flow_eur"].cumsum()
    cf_df["cumulative_discounted_cash_flow_eur"] = cf_df["discounted_cash_flow_eur"].cumsum()

    # VAN (NPV)
    npv_eur = float(cf_df["discounted_cash_flow_eur"].sum())

    # TIR (IRR)
    cfs = cf_df["net_cash_flow_eur"].tolist()
    irr_pct = compute_irr(cfs)

    # LCOH Total y por Demanda Global
    lcoh_with_if = (
        (sum_pv_costs_with_if / sum_pv_total_demand) if sum_pv_total_demand > 0 else None
    )
    lcoh_no_if = (
        (sum_pv_costs_no_if / sum_pv_total_demand) if sum_pv_total_demand > 0 else None
    )
    lcoh_baseline = (
        (sum_pv_baseline_costs / sum_pv_total_demand) if sum_pv_total_demand > 0 else None
    )

    # Desglose LCOH Plurianual Descontado (EUR/MWh-th total de demanda)
    lcoh_breakdown_project = {}
    lcoh_breakdown_baseline = {}
    if sum_pv_total_demand > 0:
        lcoh_breakdown_project = {
            "capex": pv_capex / sum_pv_total_demand,
            "opex_tech": pv_opex_tech / sum_pv_total_demand,
            "ppa_onsite": pv_ppa_pv / sum_pv_total_demand,
            "pool": pv_pool / sum_pv_total_demand,
            "adjustments": pv_adjustment / sum_pv_total_demand,
            "network_tariffs": pv_tariffs / sum_pv_total_demand,
            "guarantees_of_origin": pv_gdo / sum_pv_total_demand,
            "other_grid": pv_other_elec / sum_pv_total_demand,
            "fossil_gas": pv_fossil_gas / sum_pv_total_demand,
            "fossil_co2": pv_fossil_co2 / sum_pv_total_demand,
            "fossil_other": pv_fossil_other / sum_pv_total_demand,
            "innovation_fund": -pv_if_revenue / sum_pv_total_demand,
            "net_lcoh": sum_pv_costs_with_if / sum_pv_total_demand,
        }
        lcoh_breakdown_baseline = {
            "capex": 0.0,
            "opex_tech": 0.0,
            "ppa_onsite": 0.0,
            "pool": 0.0,
            "adjustments": 0.0,
            "network_tariffs": 0.0,
            "guarantees_of_origin": 0.0,
            "other_grid": 0.0,
            "fossil_gas": pv_base_gas / sum_pv_total_demand,
            "fossil_co2": pv_base_co2 / sum_pv_total_demand,
            "fossil_other": pv_base_other / sum_pv_total_demand,
            "innovation_fund": 0.0,
            "net_lcoh": sum_pv_baseline_costs / sum_pv_total_demand,
        }

    # Desglose de Coste Medio del Calor (Año 1) en EUR/MWh-th
    y1 = year_results[0].annual
    y1_demand = y1["annual_thermal_demand_mwh_th"]
    project_heat_cost_year1_eur_per_mwh_th = (
        (y1["annual_operating_cost_after_incentives_eur"] + cfg.incremental_annual_opex_eur + amort_annual)
        / y1_demand
        if y1_demand > 0
        else None
    )
    baseline_heat_cost_year1_eur_per_mwh_th = (
        y1["annual_baseline_cost_eur"] / y1_demand if y1_demand > 0 else None
    )

    heat_cost_breakdown_year1_project = {}
    heat_cost_breakdown_year1_baseline = {}
    if y1_demand > 0:
        heat_cost_breakdown_year1_project = {
            "capex": amort_annual / y1_demand,
            "opex_tech": cfg.incremental_annual_opex_eur / y1_demand,
            "ppa_onsite": y1.get("annual_pv_electricity_cost_eur", 0.0) / y1_demand,
            "pool": y1.get("annual_pool_cost_eur", 0.0) / y1_demand,
            "adjustments": y1.get("annual_adjustment_cost_eur", 0.0) / y1_demand,
            "network_tariffs": y1.get("annual_network_tariff_cost_eur", 0.0) / y1_demand,
            "guarantees_of_origin": y1.get("annual_guarantees_of_origin_cost_eur", 0.0) / y1_demand,
            "other_grid": y1.get("annual_other_elec_cost_eur", 0.0) / y1_demand,
            "fossil_gas": y1.get("annual_fossil_gas_fuel_cost_eur", 0.0) / y1_demand,
            "fossil_co2": y1.get("annual_fossil_co2_cost_eur", 0.0) / y1_demand,
            "fossil_other": y1.get("annual_fossil_gas_other_cost_eur", 0.0) / y1_demand,
            "innovation_fund": -y1.get("annual_innovation_fund_revenue_eur", 0.0) / y1_demand,
            "net_heat_cost": project_heat_cost_year1_eur_per_mwh_th,
        }
        heat_cost_breakdown_year1_baseline = {
            "capex": 0.0,
            "opex_tech": 0.0,
            "ppa_onsite": 0.0,
            "pool": 0.0,
            "adjustments": 0.0,
            "network_tariffs": 0.0,
            "guarantees_of_origin": 0.0,
            "other_grid": 0.0,
            "fossil_gas": y1.get("annual_baseline_fuel_cost_eur", 0.0) / y1_demand,
            "fossil_co2": y1.get("annual_baseline_co2_cost_eur", 0.0) / y1_demand,
            "fossil_other": y1.get("annual_baseline_gas_other_cost_eur", 0.0) / y1_demand,
            "innovation_fund": 0.0,
            "net_heat_cost": baseline_heat_cost_year1_eur_per_mwh_th,
        }

    indicators = {
        "capex_initial_eur": capex_initial,
        "annualized_capex_eur": amort_annual,
        "npv_eur": npv_eur,
        "irr_pct": irr_pct,
        "simple_payback_years": simple_payback_years,
        "simple_payback_months": (simple_payback_years * 12.0) if simple_payback_years else None,
        "discounted_payback_years": discounted_payback_years,
        "lcoh_after_incentives_eur_per_mwh_th": lcoh_with_if,
        "lcoh_before_incentives_eur_per_mwh_th": lcoh_no_if,
        "lcoh_baseline_eur_per_mwh_th": lcoh_baseline,
        "lcoh_breakdown_project": lcoh_breakdown_project,
        "lcoh_breakdown_baseline": lcoh_breakdown_baseline,
        "project_heat_cost_year1_eur_per_mwh_th": project_heat_cost_year1_eur_per_mwh_th,
        "baseline_heat_cost_year1_eur_per_mwh_th": baseline_heat_cost_year1_eur_per_mwh_th,
        "heat_cost_breakdown_year1_project": heat_cost_breakdown_year1_project,
        "heat_cost_breakdown_year1_baseline": heat_cost_breakdown_year1_baseline,
        "asset_lifetime_years": lifetime,
        "discount_rate_pct": cfg.simulation.discount_rate_pct,
    }

    return FinancialResult(cash_flows=cf_df, indicators=indicators, notes=notes)
