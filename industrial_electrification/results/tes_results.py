"""Post-procesamiento y cálculo de resultados detallados para TES (Almacenamiento Térmico)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from baseline.gas_reference import fossil_cost_components
from models.inputs import GasBaselineInputs, ScenarioConfig, TesInputs
from models.results import DispatchResult, YearResult


def compute_tes_year_results(
    year: int,
    dispatch: DispatchResult,
    calendar_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    supply_df: pd.DataFrame,
    gas: GasBaselineInputs,
    params: TesInputs,
    cfg: ScenarioConfig,
    notes: list[str] | None = None,
) -> YearResult:
    if not dispatch.is_valid:
        return YearResult(
            year=year,
            alternative="tes",
            dispatch=dispatch,
            data_notes=notes or [],
        )

    dec = dispatch.decisions
    n = len(dec)
    dt = cfg.simulation.time_step_hours

    # Variables horarias
    pv_ch = dec["tes_pv_charge_mwh_e"].to_numpy()
    grid_ch = dec["tes_grid_charge_mwh_e"].to_numpy()
    tot_ch = dec["tes_total_charge_mwh_e"].to_numpy()
    ch_heat = dec["tes_charge_heat_mwh_th"].to_numpy()
    dis_storage = dec["tes_discharge_from_storage_mwh_th"].to_numpy()
    heat_deliv = dec["tes_heat_delivered_mwh_th"].to_numpy()
    soc_start = dec["tes_soc_start_mwh_th"].to_numpy()
    soc_end = dec["tes_soc_end_mwh_th"].to_numpy()
    soc_pct = dec["tes_soc_pct"].to_numpy()
    fossil_heat = dec["fossil_heat_output_mwh_th"].to_numpy()
    eligible_heat = dec["eligible_electrified_heat_mwh_th"].to_numpy()

    pv_gen = supply_df["pv_generation_mwh_e"].to_numpy()
    pv_surplus = np.maximum(0.0, pv_gen - pv_ch)

    # Pérdidas
    charge_loss = dec["tes_charge_loss_mwh_e"].to_numpy()
    storage_loss = dec["tes_storage_loss_mwh_th"].to_numpy()
    discharge_loss = dec["tes_discharge_loss_mwh_th"].to_numpy()

    # Costes fósiles residuales
    fossil_df = fossil_cost_components(fossil_heat, gas, prefix="project_fossil")

    # Costes eléctricos de carga
    c_grid = supply_df["grid_unit_cost_eur_per_mwh_e"].to_numpy()
    c_pv = supply_df["pv_price_eur_per_mwh_e"].to_numpy()
    grid_cost = grid_ch * c_grid
    pv_cost = pv_ch * c_pv
    total_charge_cost = grid_cost + pv_cost

    # Innovation Fund
    premium = (
        cfg.innovation_fund.innovation_fund_premium_eur_per_mwh_th
        if cfg.innovation_fund.is_active_in_year(year, cfg.simulation.simulation_start_year)
        else 0.0
    )
    if_revenue = eligible_heat * premium

    # Desglose de costes eléctricos de red
    pool_cost = grid_ch * supply_df["pool_price_eur_per_mwh_e"].to_numpy()
    adj_cost = grid_ch * supply_df["adjustment_price_eur_per_mwh_e"].to_numpy()
    tariff_cost = grid_ch * supply_df["network_tariff_eur_per_mwh_e"].to_numpy()
    goo_cost = grid_ch * supply_df["guarantees_of_origin_cost_eur_per_mwh_e"].to_numpy()
    other_elec_cost = grid_ch * supply_df["grid_other_cost_eur_per_mwh_e"].to_numpy()

    # Coste operativo horario y ahorro
    hourly_operating_cost = (
        total_charge_cost
        + fossil_df["project_fossil_total_cost_eur"].to_numpy()
        - if_revenue
    )
    baseline_total = baseline_df["baseline_total_cost_eur"].to_numpy()
    hourly_savings = baseline_total - hourly_operating_cost

    # Detección de carga y descarga simultáneas
    simultaneous_ops = (tot_ch > 1e-4) & (heat_deliv > 1e-4)

    hourly = pd.DataFrame(
        {
            "timestamp": calendar_df["timestamp"],
            "tariff_period": calendar_df["tariff_period"],
            "thermal_demand_mwh_th": demand_df["thermal_demand_mwh_th"],
            "baseline_total_cost_eur": baseline_total,
            "grid_unit_cost_eur_per_mwh_e": c_grid,
            "pv_generation_mwh_e": pv_gen,
            "tes_pv_charge_mwh_e": pv_ch,
            "pv_surplus_mwh_e": pv_surplus,
            "tes_grid_charge_mwh_e": grid_ch,
            "tes_total_charge_mwh_e": tot_ch,
            "tes_charge_heat_mwh_th": ch_heat,
            "tes_charge_loss_mwh_e": charge_loss,
            "tes_soc_start_mwh_th": soc_start,
            "tes_soc_end_mwh_th": soc_end,
            "tes_soc_pct": soc_pct,
            "tes_storage_loss_mwh_th": storage_loss,
            "tes_discharge_from_storage_mwh_th": dis_storage,
            "tes_discharge_loss_mwh_th": discharge_loss,
            "tes_heat_delivered_mwh_th": heat_deliv,
            "fossil_heat_output_mwh_th": fossil_heat,
            "is_simultaneous_charge_discharge": simultaneous_ops,
            "project_gas_consumption_mwh_hhv": fossil_df["project_fossil_gas_consumption_mwh_hhv"],
            "project_gas_fuel_cost_eur": fossil_df["project_fossil_gas_fuel_cost_eur"],
            "project_co2_emissions_tco2": fossil_df["project_fossil_co2_emissions_tco2"],
            "project_co2_cost_eur": fossil_df["project_fossil_co2_cost_eur"],
            "project_gas_other_cost_eur": fossil_df["project_fossil_gas_other_cost_eur"],
            "tes_charge_cost_eur": total_charge_cost,
            "grid_energy_cost_eur": grid_cost,
            "pv_energy_cost_eur": pv_cost,
            "innovation_fund_premium_eur_per_mwh_th": np.full(n, premium),
            "innovation_fund_revenue_eur": if_revenue,
            "hourly_operating_cost_eur": hourly_operating_cost,
            "hourly_savings_vs_baseline_eur": hourly_savings,
        }
    )

    annual_demand = float(demand_df["thermal_demand_mwh_th"].sum())
    annual_elec_heat = float(heat_deliv.sum())
    annual_fossil_heat = float(fossil_heat.sum())
    actual_elec_pct = (annual_elec_heat / annual_demand * 100.0) if annual_demand > 0 else 0.0

    # Ciclos equivalentes: Energía descargada del almacenamiento / Capacidad nominal de almacenamiento
    cap_storage = params.tes_storage_capacity_mwh_th
    equivalent_cycles = (
        float(dis_storage.sum()) / cap_storage if cap_storage > 0 else 0.0
    )

    # Horas a plena carga del cargador y descargador
    charge_cap = params.tes_charge_capacity_mw_e
    discharge_cap = params.tes_discharge_capacity_mw_th
    charge_flh = float(tot_ch.sum()) / charge_cap if charge_cap > 0 else 0.0
    discharge_flh = float(heat_deliv.sum()) / discharge_cap if discharge_cap > 0 else 0.0

    # Rendimiento global TES = Calor útil entregado / Electricidad total cargada
    tot_ch_annual = float(tot_ch.sum())
    roundtrip_efficiency_pct = (
        (annual_elec_heat / tot_ch_annual * 100.0) if tot_ch_annual > 0 else 0.0
    )

    baseline_cost = float(baseline_total.sum())
    operating_before_if = float(
        (total_charge_cost + fossil_df["project_fossil_total_cost_eur"].to_numpy()).sum()
    )
    total_if_rev = float(if_revenue.sum())
    operating_after_if = operating_before_if - total_if_rev
    gross_savings = baseline_cost - operating_before_if
    net_operating_savings = (
        baseline_cost - operating_after_if - cfg.incremental_annual_opex_eur
    )

    base_co2 = float(baseline_df["baseline_co2_emissions_tco2"].sum())
    proj_co2 = float(fossil_df["project_fossil_co2_emissions_tco2"].sum())
    avoided_co2 = base_co2 - proj_co2
    co2_reduc_pct = (avoided_co2 / base_co2 * 100.0) if base_co2 > 0 else 0.0

    min_target = (
        cfg.target.minimum_electrification_pct
        if cfg.target.minimum_electrification_enabled
        else 0.0
    )
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
        "annual_grid_consumption_mwh_e": float(grid_ch.sum()),
        "annual_pv_consumption_mwh_e": float(pv_ch.sum()),
        "annual_total_electricity_mwh_e": tot_ch_annual,
        "annual_pv_surplus_mwh_e": float(pv_surplus.sum()),
        "annual_gas_consumption_mwh_hhv": float(fossil_df["project_fossil_gas_consumption_mwh_hhv"].sum()),
        # Métricas TES
        "tes_annual_charged_mwh_e": tot_ch_annual,
        "tes_annual_heat_charged_mwh_th": float(ch_heat.sum()),
        "tes_annual_discharged_from_storage_mwh_th": float(dis_storage.sum()),
        "tes_annual_heat_delivered_mwh_th": annual_elec_heat,
        "tes_total_losses_mwh": float(charge_loss.sum() + storage_loss.sum() + discharge_loss.sum()),
        "tes_charge_losses_mwh_e": float(charge_loss.sum()),
        "tes_storage_losses_mwh_th": float(storage_loss.sum()),
        "tes_discharge_losses_mwh_th": float(discharge_loss.sum()),
        "tes_roundtrip_efficiency_pct": roundtrip_efficiency_pct,
        "tes_equivalent_cycles": equivalent_cycles,
        "tes_soc_mean_pct": float(soc_pct.mean()),
        "tes_soc_max_pct": float(soc_pct.max()),
        "tes_soc_min_pct": float(soc_pct.min()),
        "tes_charge_hours": int((tot_ch > 1e-4).sum()),
        "tes_discharge_hours": int((heat_deliv > 1e-4).sum()),
        "tes_simultaneous_hours": int(simultaneous_ops.sum()),
        "asset_full_load_hours_charge": charge_flh,
        "asset_full_load_hours_discharge": discharge_flh,
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
        "annual_total_electricity_cost_eur": float(total_charge_cost.sum()),
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
                "charge_grid_mwh_e": float(grid_ch[mask].sum()),
                "charge_pv_mwh_e": float(pv_ch[mask].sum()),
                "heat_delivered_mwh_th": float(heat_deliv[mask].sum()),
                "fossil_heat_mwh_th": float(fossil_heat[mask].sum()),
                "charge_cost_eur": float(total_charge_cost[mask].sum()),
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
                "electrified_heat_mwh_th": float(heat_deliv[mask].sum()),
                "fossil_heat_mwh_th": float(fossil_heat[mask].sum()),
                "charge_grid_mwh_e": float(grid_ch[mask].sum()),
                "charge_pv_mwh_e": float(pv_ch[mask].sum()),
                "storage_losses_mwh_th": float(storage_loss[mask].sum()),
                "baseline_cost_eur": float(baseline_total[mask].sum()),
                "project_operating_cost_eur": float(hourly_operating_cost[mask].sum()),
                "net_savings_eur": float(hourly_savings[mask].sum()),
            }
        )
    monthly_df = pd.DataFrame(monthly_rows)

    return YearResult(
        year=year,
        alternative="tes",
        dispatch=dispatch,
        hourly=hourly,
        annual=annual,
        by_period=by_period_df,
        monthly=monthly_df,
        innovation_fund_active=premium > 0,
        data_notes=notes or [],
    )
