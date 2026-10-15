"""Orquestador de simulación y comparación de escenarios.

Funciones clave:
- ``run_simulation(cfg, bundle)``: Ejecuta la simulación completa de un caso.
- ``compare_scenarios(scenarios)``: Compara múltiples resultados de escenarios.
- ``compute_electrification_cost_curve(cfg, bundle)``: Curva de costes incrementales (0% a 100%).
- ``compare_unconstrained_vs_constrained(cfg, bundle)``: Compara libre vs mínima obligatoria.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

import config
from baseline.gas_reference import compute_gas_reference
from data.calendar import build_calendar
from data.loaders import DataBundle
from data.validation import validate_scenario
from demand.thermal_demand import build_thermal_demand
from finance.project_economics import compute_financial_evaluation
from models.inputs import ScenarioConfig
from models.results import ScenarioResult, YearResult
from optimization.common import DispatchInputs, IncentiveTerms
from optimization.eboiler import solve_eboiler
from optimization.tes import solve_tes
from results.eboiler_results import compute_eboiler_year_results
from results.tes_results import compute_tes_year_results
from supply.electricity_supply import build_electricity_supply


def run_single_year(
    cfg: ScenarioConfig,
    bundle: DataBundle,
    year: int,
    year_index: int = 0,
) -> YearResult:
    """Ejecuta la optimización y postproceso de un único año."""
    cal_df = build_calendar(year, time_step_hours=cfg.simulation.time_step_hours)
    demand_df, demand_notes = build_thermal_demand(
        cfg.demand, cal_df, time_step_hours=cfg.simulation.time_step_hours
    )
    baseline_df = compute_gas_reference(demand_df["thermal_demand_mwh_th"].to_numpy(), cfg.baseline)
    supply_df, supply_notes = build_electricity_supply(
        cfg.supply,
        cal_df,
        bundle,
        year_index=year_index,
        time_step_hours=cfg.simulation.time_step_hours,
    )

    all_notes = demand_notes + supply_notes

    # Términos del incentivo Innovation Fund
    inc = cfg.innovation_fund
    active_if = inc.is_active_in_year(year, cfg.simulation.simulation_start_year)
    incentive_terms = IncentiveTerms(
        premium_eur_per_mwh_th=inc.innovation_fund_premium_eur_per_mwh_th if active_if else 0.0,
        pv_heat_eligible=inc.innovation_fund_pv_heat_eligible,
        grid_heat_eligible=inc.innovation_fund_grid_heat_eligible,
        annual_max_eligible_mwh_th=inc.innovation_fund_annual_max_eligible_mwh_th if active_if else None,
        max_revenue_eur=inc.innovation_fund_max_total_amount_eur if active_if else None,
    )

    disp_inputs = DispatchInputs(
        thermal_demand_mwh_th=demand_df["thermal_demand_mwh_th"].to_numpy(),
        grid_unit_cost_eur_per_mwh_e=supply_df["grid_unit_cost_eur_per_mwh_e"].to_numpy(),
        pv_generation_mwh_e=supply_df["pv_generation_mwh_e"].to_numpy(),
        grid_capacity_available_mw=supply_df["grid_capacity_available_mw"].to_numpy(),
        gas=cfg.baseline,
        pv_price_eur_per_mwh_e=cfg.supply.pv_price_eur_per_mwh_e,
        time_step_hours=cfg.simulation.time_step_hours,
        incentive=incentive_terms,
        minimum_electrification_pct=cfg.target.effective_minimum_pct,
        pv_priority_enabled=cfg.supply.pv_priority_enabled,
        pv_priority_cost_tolerance_pct=cfg.supply.pv_priority_cost_tolerance_pct,
        pv_priority_cost_tolerance_eur=cfg.supply.pv_priority_cost_tolerance_eur,
        solver=cfg.solver,
    )

    if cfg.alternative == "eboiler":
        disp_res = solve_eboiler(
            disp_inputs,
            cfg.eboiler,
            compute_max_on_infeasible=cfg.target.compute_max_on_infeasible,
        )
        return compute_eboiler_year_results(
            year=year,
            dispatch=disp_res,
            calendar_df=cal_df,
            demand_df=demand_df,
            baseline_df=baseline_df,
            supply_df=supply_df,
            gas=cfg.baseline,
            params=cfg.eboiler,
            cfg=cfg,
            notes=all_notes,
        )
    else:
        disp_res = solve_tes(
            disp_inputs,
            cfg.tes,
            compute_max_on_infeasible=cfg.target.compute_max_on_infeasible,
        )
        return compute_tes_year_results(
            year=year,
            dispatch=disp_res,
            calendar_df=cal_df,
            demand_df=demand_df,
            baseline_df=baseline_df,
            supply_df=supply_df,
            gas=cfg.baseline,
            params=cfg.tes,
            cfg=cfg,
            notes=all_notes,
        )


def run_simulation(cfg: ScenarioConfig, bundle: DataBundle) -> ScenarioResult:
    """Ejecuta la simulación completa de un caso multianual con evaluación financiera."""
    val = validate_scenario(cfg, bundle)
    if not val.ok:
        return ScenarioResult(
            case_name=cfg.case_name,
            alternative=cfg.alternative,
            config_json=cfg.to_json(),
            years=[],
            errors=val.errors,
            warnings=val.warnings,
            assumptions=val.info,
        )

    year_results: list[YearResult] = []
    years_to_sim = cfg.simulation.simulated_years_list

    for idx, yr in enumerate(years_to_sim):
        y_res = run_single_year(cfg, bundle, year=yr, year_index=idx)
        year_results.append(y_res)
        if not y_res.is_valid:
            break

    fin_res = None
    if all(y.is_valid for y in year_results):
        fin_res = compute_financial_evaluation(cfg, year_results)

    return ScenarioResult(
        case_name=cfg.case_name,
        alternative=cfg.alternative,
        config_json=cfg.to_json(),
        years=year_results,
        financial=fin_res,
        warnings=val.warnings,
        errors=[y.dispatch.solver.message for y in year_results if not y.is_valid],
        assumptions=val.info,
    )


# ---------------------------------------------------------------------------
# Comparación y Curvas de Electrificación
# ---------------------------------------------------------------------------
def compare_unconstrained_vs_constrained(
    cfg: ScenarioConfig, bundle: DataBundle
) -> dict[str, Any]:
    """Compara Escenario A (electrificación libre) vs Escenario B (electrificación obligatoria)."""
    cfg_unconstrained = copy.deepcopy(cfg)
    cfg_unconstrained.target.minimum_electrification_enabled = False
    cfg_unconstrained.case_name = f"{cfg.case_name} (Libre)"

    cfg_constrained = copy.deepcopy(cfg)
    cfg_constrained.target.minimum_electrification_enabled = True
    cfg_constrained.case_name = f"{cfg.case_name} (Mínima {cfg.target.minimum_electrification_pct}%)"

    res_a = run_simulation(cfg_unconstrained, bundle)
    res_b = run_simulation(cfg_constrained, bundle)

    if not res_a.is_valid or not res_b.is_valid:
        return {"status": "error", "result_a": res_a, "result_b": res_b}

    y1_a = res_a.first_year.annual
    y1_b = res_b.first_year.annual

    cost_a = y1_a["annual_operating_cost_after_incentives_eur"]
    cost_b = y1_b["annual_operating_cost_after_incentives_eur"]
    inc_cost = cost_b - cost_a

    elec_heat_a = y1_a["annual_electrified_heat_mwh_th"]
    elec_heat_b = y1_b["annual_electrified_heat_mwh_th"]
    diff_heat = elec_heat_b - elec_heat_a

    inc_cost_per_mwh_th = (inc_cost / diff_heat) if diff_heat > 1e-4 else 0.0

    return {
        "status": "ok",
        "result_a": res_a,
        "result_b": res_b,
        "unconstrained_optimal_cost_eur": cost_a,
        "constrained_optimal_cost_eur": cost_b,
        "minimum_electrification_incremental_cost_eur": inc_cost,
        "electrified_heat_unconstrained_mwh_th": elec_heat_a,
        "electrified_heat_constrained_mwh_th": elec_heat_b,
        "incremental_electrified_heat_mwh_th": diff_heat,
        "incremental_cost_per_mwh_th": inc_cost_per_mwh_th,
    }


def compute_electrification_cost_curve(
    cfg: ScenarioConfig,
    bundle: DataBundle,
    targets_pct: tuple[int, ...] = config.ELECTRIFICATION_CURVE_TARGETS_PCT,
) -> pd.DataFrame:
    """Genera la curva de coste operativo incremental frente al objetivo mínimo de electrificación."""
    # Primer punto: electrificación libre como base
    base_cfg = copy.deepcopy(cfg)
    base_cfg.target.minimum_electrification_enabled = False
    base_res = run_simulation(base_cfg, bundle)

    if not base_res.is_valid:
        return pd.DataFrame()

    base_ann = base_res.first_year.annual
    base_cost = base_ann["annual_operating_cost_after_incentives_eur"]
    base_elec_heat = base_ann["annual_electrified_heat_mwh_th"]

    records = []
    prev_cost = base_cost
    prev_elec_heat = base_elec_heat

    for target in targets_pct:
        target_cfg = copy.deepcopy(cfg)
        target_cfg.target.minimum_electrification_enabled = True
        target_cfg.target.minimum_electrification_pct = float(target)
        target_cfg.case_name = f"Target {target}%"

        res = run_simulation(target_cfg, bundle)
        if not res.is_valid:
            # Detectado punto infactible
            diag = res.first_year.dispatch.infeasibility if res.first_year else None
            max_pct = diag.max_achievable_pct if diag else None
            records.append(
                {
                    "target_pct": target,
                    "is_feasible": False,
                    "actual_electrification_pct": max_pct,
                    "annual_operating_cost_eur": np.nan,
                    "incremental_cost_vs_free_eur": np.nan,
                    "marginal_cost_eur_per_mwh_th": np.nan,
                    "annual_electrified_heat_mwh_th": np.nan,
                    "annual_total_electricity_mwh_e": np.nan,
                    "annual_gas_consumption_mwh_hhv": np.nan,
                    "avoided_direct_co2_emissions_tco2": np.nan,
                }
            )
            continue

        ann = res.first_year.annual
        cost = ann["annual_operating_cost_after_incentives_eur"]
        elec_heat = ann["annual_electrified_heat_mwh_th"]
        inc_cost = cost - base_cost

        # Coste marginal frente al paso anterior
        delta_heat = elec_heat - prev_elec_heat
        delta_cost = cost - prev_cost
        marginal_cost = (delta_cost / delta_heat) if delta_heat > 1e-4 else 0.0

        records.append(
            {
                "target_pct": target,
                "is_feasible": True,
                "actual_electrification_pct": ann["actual_electrification_pct"],
                "annual_operating_cost_eur": cost,
                "incremental_cost_vs_free_eur": inc_cost,
                "marginal_cost_eur_per_mwh_th": marginal_cost,
                "annual_electrified_heat_mwh_th": elec_heat,
                "annual_total_electricity_mwh_e": ann["annual_total_electricity_mwh_e"],
                "annual_gas_consumption_mwh_hhv": ann["annual_gas_consumption_mwh_hhv"],
                "avoided_direct_co2_emissions_tco2": ann["avoided_direct_co2_emissions_tco2"],
            }
        )
        prev_cost = cost
        prev_elec_heat = elec_heat

    return pd.DataFrame(records)


def compare_multiple_scenarios(scenarios: list[ScenarioResult]) -> pd.DataFrame:
    """Crea una tabla comparativa de indicadores clave entre varios casos."""
    rows = []
    for sc in scenarios:
        if not sc.is_valid:
            rows.append({"case_name": sc.case_name, "status": "Infactible/Error"})
            continue
        y1 = sc.first_year.annual
        fin = sc.financial.indicators if sc.financial else {}
        rows.append(
            {
                "case_name": sc.case_name,
                "alternative": sc.alternative,
                "electrification_pct": y1.get("actual_electrification_pct"),
                "annual_baseline_cost_eur": y1.get("annual_baseline_cost_eur"),
                "annual_operating_cost_eur": y1.get("annual_operating_cost_after_incentives_eur"),
                "annual_net_operating_savings_eur": y1.get("annual_net_operating_savings_eur"),
                "capex_initial_eur": fin.get("capex_initial_eur"),
                "npv_eur": fin.get("npv_eur"),
                "irr_pct": fin.get("irr_pct"),
                "simple_payback_years": fin.get("simple_payback_years"),
                "lcoh_after_incentives_eur_per_mwh_th": fin.get("lcoh_after_incentives_eur_per_mwh_th"),
                "avoided_direct_co2_emissions_tco2": y1.get("avoided_direct_co2_emissions_tco2"),
                "direct_co2_reduction_pct": y1.get("direct_co2_reduction_pct"),
            }
        )
    return pd.DataFrame(rows)
