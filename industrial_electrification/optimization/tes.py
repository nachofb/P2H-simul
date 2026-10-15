"""ALTERNATIVA 2 - Optimizador de despacho Almacenamiento Térmico TES (LP / MILP).

Índices: t in T = {0..N-1}, Δt = time_step_hours.

Variables de decisión (todas >= 0, salvo binarias si MILP):
    tes_pv_charge_mwh_e[t]
    tes_grid_charge_mwh_e[t]
    tes_total_charge_mwh_e[t]
    tes_charge_heat_mwh_th[t]
    tes_discharge_from_storage_mwh_th[t]
    tes_heat_delivered_mwh_th[t]
    tes_soc_start_mwh_th[t]
    tes_soc_end_mwh_th[t]
    fossil_heat_output_mwh_th[t]
    eligible_electrified_heat_mwh_th[t] (si IF activo)
    (Opcional MILP: is_charging[t], is_discharging[t] in {0,1})

Ecuaciones y Restricciones:
1. Balance eléctrico de carga:
   tes_total_charge_mwh_e[t] = tes_pv_charge_mwh_e[t] + tes_grid_charge_mwh_e[t]
2. Conversión térmica de carga:
   tes_charge_heat_mwh_th[t] = tes_total_charge_mwh_e[t] * eta_charge
3. Potencia de carga:
   tes_total_charge_mwh_e[t] <= tes_charge_capacity_mw_e * Δt
4. Disponibilidad PV:
   tes_pv_charge_mwh_e[t] <= pv_generation_mwh_e[t]
5. Capacidad de red:
   tes_grid_charge_mwh_e[t] <= grid_capacity_available_mw[t] * Δt
6. Continuidad SOC:
   tes_soc_start_mwh_th[0] = tes_initial_soc_mwh_th
   tes_soc_start_mwh_th[t] = tes_soc_end_mwh_th[t-1]  (t > 0)
7. Evolución del SOC:
   tes_soc_end_mwh_th[t] = tes_soc_start_mwh_th[t] * (1 - self_discharge_pct/100)
                           + tes_charge_heat_mwh_th[t] - tes_discharge_from_storage_mwh_th[t]
8. Calor útil descargado y entregado:
   tes_heat_delivered_mwh_th[t] = tes_discharge_from_storage_mwh_th[t] * eta_discharge
9. Límites de SOC:
   0 <= tes_soc_end_mwh_th[t] <= tes_storage_capacity_mwh_th
10. Potencia de descarga:
    tes_heat_delivered_mwh_th[t] <= tes_discharge_capacity_mw_th * Δt
11. Balance térmico del proceso:
    tes_heat_delivered_mwh_th[t] + fossil_heat_output_mwh_th[t] = thermal_demand_mwh_th[t]
12. Condición de SOC final:
    según tes_final_soc_mode ('equal_initial', 'fixed', 'minimum', 'free')
13. Electrificación mínima obligatoria:
    sum(tes_heat_delivered_mwh_th[t]) >= (pct_min/100) * sum(thermal_demand_mwh_th[t])
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import config
from models.inputs import TesInputs
from models.results import DispatchResult, InfeasibilityDiagnosis
from optimization.common import (
    DispatchInputs,
    apply_pv_priority,
    binding_count,
    clean,
    solve_model,
    var_values,
)

ALTERNATIVE = "tes"


def build_tes_model(
    inp: DispatchInputs,
    params: TesInputs,
    objective: Literal["min_cost", "max_electrification"] = "min_cost",
    apply_min_target: bool = True,
) -> pyo.ConcreteModel:
    n = inp.n_hours
    dt = inp.time_step_hours
    eta_ch = params.tes_charge_efficiency
    eta_dis = params.tes_discharge_efficiency
    self_dis = params.tes_hourly_self_discharge_pct / 100.0
    soc_retention = 1.0 - self_dis

    demand = inp.thermal_demand_mwh_th
    pv_gen = inp.pv_generation_mwh_e
    grid_cap = inp.grid_capacity_available_mw
    c_grid = inp.grid_unit_cost_eur_per_mwh_e
    c_pv = inp.pv_price_eur_per_mwh_e
    fossil_unit = inp.fossil_cost_eur_per_mwh_th
    c_fossil = fossil_unit["fuel"] + fossil_unit["co2"] + fossil_unit["other"]
    inc = inp.incentive

    m = pyo.ConcreteModel(name="tes_dispatch")
    m.T = pyo.RangeSet(0, n - 1)

    # --- Variables de decisión ---
    m.tes_pv_charge_mwh_e = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.tes_grid_charge_mwh_e = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.tes_total_charge_mwh_e = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.tes_charge_heat_mwh_th = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.tes_discharge_from_storage_mwh_th = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.tes_heat_delivered_mwh_th = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.tes_soc_start_mwh_th = pyo.Var(
        m.T, domain=pyo.NonNegativeReals, bounds=(0, params.tes_storage_capacity_mwh_th)
    )
    m.tes_soc_end_mwh_th = pyo.Var(
        m.T, domain=pyo.NonNegativeReals, bounds=(0, params.tes_storage_capacity_mwh_th)
    )
    m.fossil_heat_output_mwh_th = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Variables binarias si se activa la no simultaneidad MILP
    if params.tes_enforce_no_simultaneous_charge_discharge:
        m.is_charging = pyo.Var(m.T, domain=pyo.Binary)
        m.is_discharging = pyo.Var(m.T, domain=pyo.Binary)
        m.no_simultaneous = pyo.Constraint(
            m.T, rule=lambda m, t: m.is_charging[t] + m.is_discharging[t] <= 1
        )
        m.charge_bin_link = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.tes_total_charge_mwh_e[t]
            <= params.tes_charge_capacity_mw_e * dt * m.is_charging[t],
        )
        m.discharge_bin_link = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.tes_heat_delivered_mwh_th[t]
            <= params.tes_discharge_capacity_mw_th * dt * m.is_discharging[t],
        )

    # --- Restricciones físicas ---
    # 1. Balance eléctrico de carga
    m.charge_balance = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.tes_total_charge_mwh_e[t]
        == m.tes_pv_charge_mwh_e[t] + m.tes_grid_charge_mwh_e[t],
    )

    # 2. Conversión térmica de carga
    m.charge_heat_conv = pyo.Constraint(
        m.T, rule=lambda m, t: m.tes_charge_heat_mwh_th[t] == m.tes_total_charge_mwh_e[t] * eta_ch
    )

    # 3. Potencia de carga
    if not params.tes_enforce_no_simultaneous_charge_discharge:
        m.charge_capacity_limit = pyo.Constraint(
            m.T, rule=lambda m, t: m.tes_total_charge_mwh_e[t] <= params.tes_charge_capacity_mw_e * dt
        )

    # 4. Disponibilidad PV
    m.pv_avail = pyo.Constraint(
        m.T, rule=lambda m, t: m.tes_pv_charge_mwh_e[t] <= float(pv_gen[t])
    )

    # 5. Capacidad de red
    def _grid_rule(m, t):
        if not np.isfinite(grid_cap[t]):
            return pyo.Constraint.Skip
        return m.tes_grid_charge_mwh_e[t] <= float(grid_cap[t]) * dt

    m.grid_cap_limit = pyo.Constraint(m.T, rule=_grid_rule)

    # 6. Continuidad de SOC
    def _soc_start_rule(m, t):
        if t == 0:
            return m.tes_soc_start_mwh_th[t] == params.tes_initial_soc_mwh_th
        return m.tes_soc_start_mwh_th[t] == m.tes_soc_end_mwh_th[t - 1]

    m.soc_continuity = pyo.Constraint(m.T, rule=_soc_start_rule)

    # 7. Evolución del SOC
    m.soc_evolution = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.tes_soc_end_mwh_th[t]
        == m.tes_soc_start_mwh_th[t] * soc_retention
        + m.tes_charge_heat_mwh_th[t]
        - m.tes_discharge_from_storage_mwh_th[t],
    )

    # 8. Conversión de descarga a calor útil
    m.discharge_conv = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.tes_heat_delivered_mwh_th[t]
        == m.tes_discharge_from_storage_mwh_th[t] * eta_dis,
    )

    # 9. Potencia de descarga
    if not params.tes_enforce_no_simultaneous_charge_discharge:
        m.discharge_cap_limit = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.tes_heat_delivered_mwh_th[t] <= params.tes_discharge_capacity_mw_th * dt,
        )

    # 10. Balance térmico
    m.thermal_balance = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.tes_heat_delivered_mwh_th[t] + m.fossil_heat_output_mwh_th[t]
        == float(demand[t]),
    )

    # 11. Condición de SOC final
    if params.tes_final_soc_mode == "equal_initial":
        m.final_soc_cond = pyo.Constraint(
            expr=m.tes_soc_end_mwh_th[n - 1] >= params.tes_initial_soc_mwh_th
        )
    elif params.tes_final_soc_mode == "fixed":
        m.final_soc_cond = pyo.Constraint(
            expr=m.tes_soc_end_mwh_th[n - 1] == params.tes_final_soc_mwh_th
        )
    elif params.tes_final_soc_mode == "minimum":
        m.final_soc_cond = pyo.Constraint(
            expr=m.tes_soc_end_mwh_th[n - 1] >= params.tes_final_soc_mwh_th
        )

    # 12. Electrificación mínima
    m.annual_electrified_heat = pyo.Expression(
        expr=pyo.quicksum(m.tes_heat_delivered_mwh_th[t] for t in m.T)
    )
    if apply_min_target and inp.minimum_electrification_pct is not None:
        required = (inp.minimum_electrification_pct / 100.0) * float(demand.sum())
        m.minimum_electrification = pyo.Constraint(expr=m.annual_electrified_heat >= required)

    # --- Innovation Fund ---
    if inc.needs_eligibility_variable:
        m.eligible_electrified_heat_mwh_th = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        # El calor entregado elegible proviene de la fracción elegible cargada
        m.eligibility = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.eligible_electrified_heat_mwh_th[t] <= m.tes_heat_delivered_mwh_th[t],
        )
        total_eligible = pyo.quicksum(m.eligible_electrified_heat_mwh_th[t] for t in m.T)
        if inc.annual_max_eligible_mwh_th is not None:
            m.if_annual_energy_cap = pyo.Constraint(expr=total_eligible <= inc.annual_max_eligible_mwh_th)
        if inc.max_revenue_eur is not None:
            m.if_revenue_cap = pyo.Constraint(
                expr=inc.premium_eur_per_mwh_th * total_eligible <= inc.max_revenue_eur
            )
        eligible = lambda t: m.eligible_electrified_heat_mwh_th[t]  # noqa: E731
    elif inc.active:
        eligible = lambda t: m.tes_heat_delivered_mwh_th[t]  # noqa: E731
    else:
        eligible = None

    premium = inc.premium_eur_per_mwh_th if inc.active else 0.0
    m.total_pv_use = pyo.Expression(expr=pyo.quicksum(m.tes_pv_charge_mwh_e[t] for t in m.T))

    cost_terms = []
    for t in range(n):
        term = (
            c_pv * m.tes_pv_charge_mwh_e[t]
            + float(c_grid[t]) * m.tes_grid_charge_mwh_e[t]
            + c_fossil * m.fossil_heat_output_mwh_th[t]
        )
        if eligible is not None:
            term = term - premium * eligible(t)
        cost_terms.append(term)
    m.total_variable_cost_eur = pyo.Expression(expr=pyo.quicksum(cost_terms))

    if objective == "min_cost":
        m.objective = pyo.Objective(expr=m.total_variable_cost_eur, sense=pyo.minimize)
    else:
        m.objective = pyo.Objective(expr=m.annual_electrified_heat, sense=pyo.maximize)
    return m


def _extract_tes(m: pyo.ConcreteModel, inp: DispatchInputs, params: TesInputs) -> pd.DataFrame:
    T = list(m.T)
    pv_ch = clean(var_values(m.tes_pv_charge_mwh_e, T))
    grid_ch = clean(var_values(m.tes_grid_charge_mwh_e, T))
    tot_ch = clean(var_values(m.tes_total_charge_mwh_e, T))
    ch_heat = clean(var_values(m.tes_charge_heat_mwh_th, T))
    dis_storage = clean(var_values(m.tes_discharge_from_storage_mwh_th, T))
    heat_deliv = clean(var_values(m.tes_heat_delivered_mwh_th, T))
    soc_start = clean(var_values(m.tes_soc_start_mwh_th, T))
    soc_end = clean(var_values(m.tes_soc_end_mwh_th, T))
    fossil = clean(var_values(m.fossil_heat_output_mwh_th, T))

    if hasattr(m, "eligible_electrified_heat_mwh_th"):
        eligible = clean(var_values(m.eligible_electrified_heat_mwh_th, T))
    elif inp.incentive.active:
        eligible = heat_deliv.copy()
    else:
        eligible = np.zeros(len(T))

    # Pérdidas calculadas
    charge_losses = tot_ch * (1.0 - params.tes_charge_efficiency)
    storage_losses = soc_start * (params.tes_hourly_self_discharge_pct / 100.0)
    discharge_losses = dis_storage * (1.0 - params.tes_discharge_efficiency)

    df = pd.DataFrame(
        {
            "tes_pv_charge_mwh_e": pv_ch,
            "tes_grid_charge_mwh_e": grid_ch,
            "tes_total_charge_mwh_e": tot_ch,
            "tes_charge_heat_mwh_th": ch_heat,
            "tes_discharge_from_storage_mwh_th": dis_storage,
            "tes_heat_delivered_mwh_th": heat_deliv,
            "tes_soc_start_mwh_th": soc_start,
            "tes_soc_end_mwh_th": soc_end,
            "fossil_heat_output_mwh_th": fossil,
            "eligible_electrified_heat_mwh_th": eligible,
            "tes_charge_loss_mwh_e": charge_losses,
            "tes_storage_loss_mwh_th": storage_losses,
            "tes_discharge_loss_mwh_th": discharge_losses,
        }
    )
    if params.tes_storage_capacity_mwh_th > 0:
        df["tes_soc_pct"] = (df["tes_soc_end_mwh_th"] / params.tes_storage_capacity_mwh_th) * 100.0
    else:
        df["tes_soc_pct"] = 0.0

    return df


def maximize_tes_electrification(inp: DispatchInputs, params: TesInputs) -> InfeasibilityDiagnosis:
    m = build_tes_model(inp, params, objective="max_electrification", apply_min_target=False)
    report = solve_model(
        m, inp.solver, is_mip=params.tes_enforce_no_simultaneous_charge_discharge
    )
    requested = inp.minimum_electrification_pct or 0.0
    diag = InfeasibilityDiagnosis(
        requested_minimum_pct=requested,
        max_achievable_pct=None,
        max_achievable_heat_mwh_th=None,
        solver=report,
    )
    if not report.is_optimal:
        diag.limiting_factors.append(f"No se pudo calcular el máximo alcanzable: {report.message}")
        return diag

    dec = _extract_tes(m, inp, params)
    total_demand = inp.annual_thermal_demand_mwh_th
    delivered = dec["tes_heat_delivered_mwh_th"].to_numpy()
    diag.max_achievable_heat_mwh_th = float(delivered.sum())
    diag.max_achievable_pct = (
        100.0 * delivered.sum() / total_demand if total_demand > 0 else 0.0
    )

    dt = inp.time_step_hours
    unmet = inp.thermal_demand_mwh_th - delivered > 1e-6
    dis_bind = binding_count(delivered[unmet], np.full(unmet.sum(), params.tes_discharge_capacity_mw_th * dt))
    ch_bind = binding_count(dec["tes_total_charge_mwh_e"].to_numpy(), np.full(len(delivered), params.tes_charge_capacity_mw_e * dt))
    soc_max = binding_count(dec["tes_soc_end_mwh_th"].to_numpy(), np.full(len(delivered), params.tes_storage_capacity_mwh_th))

    diag.binding_hours = {
        "horas_con_demanda_no_electrificada": int(unmet.sum()),
        "horas_limitadas_por_potencia_descarga": dis_bind,
        "horas_con_carga_a_plena_potencia": ch_bind,
        "horas_con_tes_al_100pct_soc": soc_max,
    }

    if dis_bind:
        diag.limiting_factors.append(
            f"Potencia de descarga ({params.tes_discharge_capacity_mw_th:.2f} MW-th) inferior a la demanda en {dis_bind} h."
        )
    if ch_bind:
        diag.limiting_factors.append(
            f"Potencia de carga ({params.tes_charge_capacity_mw_e:.2f} MW-e) limitante en {ch_bind} h."
        )
    if soc_max:
        diag.limiting_factors.append(
            f"Capacidad de almacenamiento ({params.tes_storage_capacity_mwh_th:.2f} MWh-th) al 100% de llenado en {soc_max} h."
        )
    if not diag.limiting_factors and unmet.any():
        diag.limiting_factors.append("Combinación de capacidades técnicas, pérdidas de almacenamiento y límites de red/PV.")
    return diag


def solve_tes(
    inp: DispatchInputs, params: TesInputs, compute_max_on_infeasible: bool = True
) -> DispatchResult:
    is_mip = params.tes_enforce_no_simultaneous_charge_discharge
    m = build_tes_model(inp, params)
    report = solve_model(m, inp.solver, is_mip=is_mip)

    if not report.is_optimal:
        diag = None
        if report.is_infeasible and inp.minimum_electrification_pct is not None and compute_max_on_infeasible:
            diag = maximize_tes_electrification(inp, params)
        elif report.is_infeasible:
            diag = InfeasibilityDiagnosis(
                requested_minimum_pct=inp.minimum_electrification_pct or 0.0,
                max_achievable_pct=None,
                max_achievable_heat_mwh_th=None,
            )
        return DispatchResult(ALTERNATIVE, report, None, None, infeasibility=diag)

    stage1_cost = float(pyo.value(m.total_variable_cost_eur))
    priority_info = None
    if inp.pv_priority_enabled and inp.pv_generation_mwh_e.sum() > config.NUMERIC_TOLERANCE:
        stage1_elec = float(pyo.value(m.annual_electrified_heat))
        report2, priority_info = apply_pv_priority(
            m,
            m.objective,
            m.total_pv_use,
            m.annual_electrified_heat,
            stage1_cost,
            stage1_elec,
            inp,
            is_mip=is_mip,
        )
        if not report2.is_optimal:
            report.message += f" Etapa de prioridad PV no óptima ({report2.message}); se mantiene la etapa 1."
            m = build_tes_model(inp, params)
            report = solve_model(m, inp.solver, is_mip=is_mip)
        else:
            report.message += " Prioridad PV aplicada (optimización lexicográfica)."

    decisions = _extract_tes(m, inp, params)
    objective = float(pyo.value(m.total_variable_cost_eur))
    return DispatchResult(ALTERNATIVE, report, decisions, objective, pv_priority_stage=priority_info)
