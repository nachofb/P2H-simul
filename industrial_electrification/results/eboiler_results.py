"""Post-procesamiento y cálculo de resultados detallados para E-boiler."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from baseline.gas_reference import fossil_cost_components
from models.inputs import EboilerInputs, GasBaselineInputs, ScenarioConfig
from models.results import DispatchResult, YearResult
from optimization.common import DispatchInputs
from supply.electricity_supply import compute_electricity_price_stats


def compute_eboiler_year_results(
    year: int,
    dispatch: DispatchResult,
    calendar_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    supply_df: pd.DataFrame,
    gas: GasBaselineInputs,
    params: EboilerInputs,
    cfg: ScenarioConfig,
    notes: list[str] | None = None,
) -> YearResult:
    if not dispatch.is_valid:
        return YearResult(
            year=year,
            alternative="eboiler",
            dispatch=dispatch,
            data_notes=notes or [],
        )

    dec = dispatch.decisions
    n = len(dec)
    dt = cfg.simulation.time_step_hours

    # Variables horarias
    pv_in = dec["eboiler_pv_input_mwh_e"].to_numpy()
    grid_in = dec["eboiler_grid_input_mwh_e"].to_numpy()
    elec_tot = pv_in + grid_in
    heat_out = dec["eboiler_heat_output_mwh_th"].to_numpy()
    fossil_heat = dec["fossil_heat_output_mwh_th"].to_numpy()
    eligible_heat = dec["eligible_electrified_heat_mwh_th"].to_numpy()

    pv_gen = supply_df["pv_generation_mwh_e"].to_numpy()
    pv_surplus = np.maximum(0.0, pv_gen - pv_in)

    # Costes fósiles residuales del proyecto
    fossil_df = fossil_cost_components(fossil_heat, gas, prefix="project_fossil")

    # Costes eléctricos
    c_grid = supply_df["grid_unit_cost_eur_per_mwh_e"].to_numpy()
    c_pv = supply_df["pv_price_eur_per_mwh_e"].to_numpy()
    grid_cost = grid_in * c_grid
    pv_cost = pv_in * c_pv

    # Innovation Fund
    premium = (
        cfg.innovation_fund.innovation_fund_premium_eur_per_mwh_th
        if cfg.innovation_fund.is_active_in_year(year, cfg.simulation.simulation_start_year)
        else 0.0
    )
    if_revenue = eligible_heat * premium

    # Desglose de costes eléctricos de red
    pool_cost = grid_in * supply_df["pool_price_eur_per_mwh_e"].to_numpy()
    adj_cost = grid_in * supply_df["adjustment_price_eur_per_mwh_e"].to_numpy()
    tariff_cost = grid_in * supply_df["network_tariff_eur_per_mwh_e"].to_numpy()
    goo_cost = grid_in * supply_df["guarantees_of_origin_cost_eur_per_mwh_e"].to_numpy()
    other_elec_cost = grid_in * supply_df["grid_other_cost_eur_per_mwh_e"].to_numpy()

    # Coste operativo horario y ahorro
    hourly_operating_cost = (
        grid_cost
        + pv_cost
        + fossil_df["project_fossil_total_cost_eur"].to_numpy()
        - if_revenue
    )
    baseline_total = baseline_df["baseline_total_cost_eur"].to_numpy()
    hourly_savings = baseline_total - hourly_operating_cost

    hourly = pd.DataFrame(
        {
            "timestamp": calendar_df["timestamp"],
            "tariff_period": calendar_df["tariff_period"],
            "thermal_demand_mwh_th": demand_df["thermal_demand_mwh_th"],
            "gas_price_eur_per_mwh_hhv": np.full(n, gas.gas_price_eur_per_mwh_hhv),
            "co2_price_eur_per_tco2": np.full(n, gas.co2_price_eur_per_tco2),
            "baseline_total_cost_eur": baseline_total,
            "pool_price_eur_per_mwh_e": supply_df["pool_price_eur_per_mwh_e"],
            "adjustment_price_eur_per_mwh_e": supply_df["adjustment_price_eur_per_mwh_e"],
            "network_tariff_eur_per_mwh_e": supply_df["network_tariff_eur_per_mwh_e"],
            "guarantees_of_origin_cost_eur_per_mwh_e": supply_df["guarantees_of_origin_cost_eur_per_mwh_e"],
            "grid_other_cost_eur_per_mwh_e": supply_df["grid_other_cost_eur_per_mwh_e"],
            "grid_unit_cost_eur_per_mwh_e": c_grid,
            "pv_generation_mwh_e": pv_gen,
            "eboiler_pv_input_mwh_e": pv_in,
            "pv_surplus_mwh_e": pv_surplus,
            "eboiler_grid_input_mwh_e": grid_in,
            "eboiler_total_electricity_mwh_e": elec_tot,
            "eboiler_heat_output_mwh_th": heat_out,
            "fossil_heat_output_mwh_th": fossil_heat,
            "project_gas_consumption_mwh_hhv": fossil_df["project_fossil_gas_consumption_mwh_hhv"],
            "project_gas_fuel_cost_eur": fossil_df["project_fossil_gas_fuel_cost_eur"],
            "project_co2_emissions_tco2": fossil_df["project_fossil_co2_emissions_tco2"],
            "project_co2_cost_eur": fossil_df["project_fossil_co2_cost_eur"],
            "project_gas_other_cost_eur": fossil_df["project_fossil_gas_other_cost_eur"],
            "grid_energy_cost_eur": grid_cost,
            "pv_energy_cost_eur": pv_cost,
            "innovation_fund_premium_eur_per_mwh_th": np.full(n, premium),
            "innovation_fund_revenue_eur": if_revenue,
            "hourly_operating_cost_eur": hourly_operating_cost,
            "hourly_savings_vs_baseline_eur": hourly_savings,
        }
    )

    # Resumen anual
    annual_demand = float(demand_df["thermal_demand_mwh_th"].sum())
    annual_elec_heat = float(heat_out.sum())
    annual_fossil_heat = float(fossil_heat.sum())
    actual_elec_pct = (annual_elec_heat / annual_demand * 100.0) if annual_demand > 0 else 0.0

    capacity_mw = params.eboiler_capacity_mw_e
    full_load_hours = (
        float(elec_tot.sum()) / capacity_mw if capacity_mw > 0 else 0.0
    )
    capacity_factor_pct = (
        (full_load_hours / (n * dt)) * 100.0 if (n * dt) > 0 else 0.0
    )

    baseline_cost = float(baseline_total.sum())
    operating_before_if = float(
        (grid_cost + pv_cost + fossil_df["project_fossil_total_cost_eur"].to_numpy()).sum()
    )
    total_if_rev = float(if_revenue.sum())
    operating_after_if = operating_before_if - total_if_rev
    gross_savings = baseline_cost - operating_before_if
    net_operating_savings = (
        baseline_cost - operating_after_if - cfg.incremental_annual_opex_eur
    )

    # Emisiones directas
    base_co2 = float(baseline_df["baseline_co2_emissions_tco2"].sum())
    proj_co2 = float(fossil_df["project_fossil_co2_emissions_tco2"].sum())
    avoided_co2 = base_co2 - proj_co2
    co2_reduc_pct = (avoided_co2 / base_co2 * 100.0) if base_co2 > 0 else 0.0

    min_target = cfg.target.minimum_electrification_pct if cfg.target.minimum_electrification_enabled else 0.0
    compliance = actual_elec_pct >= (min_target - config.NUMERIC_TOLERANCE)
    margin_pct = actual_elec_pct - min_target

    annual = {
        "annual_thermal_demand_mwh_th": annual_demand,
        "annual_electrified_heat_mwh_th": annual_elec_heat,
        "annual_fossil_heat_mwh_th": annual_fossil_heat,
        "actual_electrification_pct": actual_elec_pct,
        "minimum_electrification_pct": min_target,
        "electrification_compliance": compliance,
        "electrification_margin_pct_points": margin_pct,
        "annual_grid_consumption_mwh_e": float(grid_in.sum()),
        "annual_pv_consumption_mwh_e": float(pv_in.sum()),
        "annual_total_electricity_mwh_e": float(elec_tot.sum()),
        "annual_pv_surplus_mwh_e": float(pv_surplus.sum()),
        "annual_gas_consumption_mwh_hhv": float(fossil_df["project_fossil_gas_consumption_mwh_hhv"].sum()),
        "asset_full_load_hours": full_load_hours,
        "asset_capacity_factor_pct": capacity_factor_pct,
        # Desglose económico
        "annual_baseline_cost_eur": baseline_cost,
        "annual_baseline_fuel_cost_eur": float(baseline_df["baseline_gas_fuel_cost_eur"].sum()),
        "annual_baseline_co2_cost_eur": float(baseline_df["baseline_co2_cost_eur"].sum()),
        "annual_baseline_gas_other_cost_eur": float(baseline_df["baseline_gas_other_cost_eur"].sum()),
        "annual_pool_cost_eur": float(pool_cost.sum()),
        "annual_adjustment_cost_eur": float(adj_cost.sum()),
        "annual_network_tariff_cost_eur": float(tariff_cost.sum()),
        "annual_guarantees_of_origin_cost_eur": float(goo_cost.sum()),
        "annual_other_elec_cost_eur": float(other_elec_cost.sum()),
        "annual_grid_electricity_cost_eur": float(grid_cost.sum()),
        "annual_pv_electricity_cost_eur": float(pv_cost.sum()),
        "annual_total_electricity_cost_eur": float((grid_cost + pv_cost).sum()),
        "annual_captured_electricity_price_eur_per_mwh_e": (
            float((grid_cost + pv_cost).sum()) / float(elec_tot.sum())
            if float(elec_tot.sum()) > 0
            else 0.0
        ),
        "annual_fossil_gas_fuel_cost_eur": float(fossil_df["project_fossil_gas_fuel_cost_eur"].sum()),
        "annual_fossil_co2_cost_eur": float(fossil_df["project_fossil_co2_cost_eur"].sum()),
        "annual_fossil_gas_other_cost_eur": float(fossil_df["project_fossil_gas_other_cost_eur"].sum()),
        "annual_fossil_total_cost_eur": float(fossil_df["project_fossil_total_cost_eur"].sum()),
        "annual_operating_cost_before_incentives_eur": operating_before_if,
        "annual_innovation_fund_revenue_eur": total_if_rev,
        "annual_operating_cost_after_incentives_eur": operating_after_if,
        "annual_gross_savings_eur": gross_savings,
        "annual_net_operating_savings_eur": net_operating_savings,
        "incremental_annual_opex_eur": cfg.incremental_annual_opex_eur,
        # Emisiones
        "baseline_co2_emissions_tco2": base_co2,
        "project_fossil_co2_emissions_tco2": proj_co2,
        "avoided_direct_co2_emissions_tco2": avoided_co2,
        "direct_co2_reduction_pct": co2_reduc_pct,
    }

    # Desglose por periodos tarifarios P1-P6
    by_p = []
    p_series = calendar_df["tariff_period"]
    for p in range(1, 7):
        mask = (p_series == p).to_numpy()
        by_p.append(
            {
                "tariff_period": f"P{p}",
                "hours": int(mask.sum()),
                "grid_mwh_e": float(grid_in[mask].sum()),
                "pv_mwh_e": float(pv_in[mask].sum()),
                "elec_heat_mwh_th": float(heat_out[mask].sum()),
                "fossil_heat_mwh_th": float(fossil_heat[mask].sum()),
                "grid_cost_eur": float(grid_cost[mask].sum()),
            }
        )
    by_period_df = pd.DataFrame(by_p)

    # Desglose mensual
    m_series = calendar_df["month"]
    monthly_rows = []
    for mo in range(1, 13):
        mask = (m_series == mo).to_numpy()
        monthly_rows.append(
            {
                "month": mo,
                "thermal_demand_mwh_th": float(demand_df["thermal_demand_mwh_th"][mask].sum()),
                "electrified_heat_mwh_th": float(heat_out[mask].sum()),
                "fossil_heat_mwh_th": float(fossil_heat[mask].sum()),
                "grid_consumption_mwh_e": float(grid_in[mask].sum()),
                "pv_consumption_mwh_e": float(pv_in[mask].sum()),
                "baseline_cost_eur": float(baseline_total[mask].sum()),
                "project_operating_cost_eur": float(hourly_operating_cost[mask].sum()),
                "net_savings_eur": float(hourly_savings[mask].sum()),
            }
        )
    monthly_df = pd.DataFrame(monthly_rows)

    price_stats_df = compute_electricity_price_stats(supply_df, calendar_df["tariff_period"])

    return YearResult(
        year=year,
        alternative="eboiler",
        dispatch=dispatch,
        hourly=hourly,
        annual=annual,
        by_period=by_period_df,
        price_stats=price_stats_df,
        monthly=monthly_df,
        innovation_fund_active=premium > 0,
        data_notes=notes or [],
    )
