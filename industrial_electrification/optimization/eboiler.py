"""ALTERNATIVA 1 - Optimizador de despacho E-boiler sin almacenamiento (LP).

Índices: t in T = {0..N-1}, Δt = time_step_hours.

Variables de decisión (>= 0, MWh por intervalo):
    eboiler_pv_input_mwh_e[t], eboiler_grid_input_mwh_e[t],
    eboiler_heat_output_mwh_th[t], fossil_heat_output_mwh_th[t]
    eligible_electrified_heat_mwh_th[t]  (solo si hay límites o elegibilidad parcial del IF)

Restricciones:
    heat[t] = (pv_in[t] + grid_in[t]) * eta
    pv_in[t] + grid_in[t] <= P_e * Δt
    pv_in[t] <= pv_generation[t]
    grid_in[t] <= grid_capacity[t] * Δt                 (si hay límite)
    heat[t] + fossil[t] = demand[t]
    Σ heat[t] >= pct_min/100 * Σ demand[t]                (si está activada)
    eligible[t] <= eta * (pv_in[t]*elig_pv + grid_in[t]*elig_grid)
    Σ eligible[t] <= E_max_IF ;  premium * Σ eligible[t] <= importe_restante

Objetivo (min):
    Σ_t [ c_pv*pv_in + c_grid[t]*grid_in + (c_fuel + c_co2 + c_other)*fossil
          - premium_IF*eligible ]
El CAPEX y el OPEX fijo no participan en el despacho.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import config
from models.inputs import EboilerInputs
from models.results import DispatchResult, InfeasibilityDiagnosis
from optimization.common import (
    DispatchInputs,
    apply_pv_priority,
    binding_count,
    clean,
    solve_model,
    var_values,
)

ALTERNATIVE = "eboiler"


def build_eboiler_model(
    inp: DispatchInputs,
    params: EboilerInputs,
    objective: Literal["min_cost", "max_electrification"] = "min_cost",
    apply_min_target: bool = True,
) -> pyo.ConcreteModel:
    n = inp.n_hours
    dt = inp.time_step_hours
    eta = params.eboiler_efficiency
    demand = inp.thermal_demand_mwh_th
    pv_gen = inp.pv_generation_mwh_e
    grid_cap = inp.grid_capacity_available_mw
    c_grid = inp.grid_unit_cost_eur_per_mwh_e
    c_pv = inp.pv_price_eur_per_mwh_e
    fossil_unit = inp.fossil_cost_eur_per_mwh_th
    c_fossil = fossil_unit["fuel"] + fossil_unit["co2"] + fossil_unit["other"]
    inc = inp.incentive

    m = pyo.ConcreteModel(name="eboiler_dispatch")
    m.T = pyo.RangeSet(0, n - 1)

    # --- Variables de decisión ---
    m.eboiler_pv_input_mwh_e = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.eboiler_grid_input_mwh_e = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.eboiler_heat_output_mwh_th = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.fossil_heat_output_mwh_th = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # --- Restricciones ---
    m.conversion = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.eboiler_heat_output_mwh_th[t]
        == (m.eboiler_pv_input_mwh_e[t] + m.eboiler_grid_input_mwh_e[t]) * eta,
    )
    m.eboiler_capacity = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.eboiler_pv_input_mwh_e[t] + m.eboiler_grid_input_mwh_e[t]
        <= params.eboiler_capacity_mw_e * dt,
    )
    m.pv_availability = pyo.Constraint(
        m.T, rule=lambda m, t: m.eboiler_pv_input_mwh_e[t] <= float(pv_gen[t])
    )

    def _grid_rule(m, t):
        if not np.isfinite(grid_cap[t]):
            return pyo.Constraint.Skip
        return m.eboiler_grid_input_mwh_e[t] <= float(grid_cap[t]) * dt

    m.grid_capacity = pyo.Constraint(m.T, rule=_grid_rule)
    m.thermal_balance = pyo.Constraint(
        m.T,
        rule=lambda m, t: m.eboiler_heat_output_mwh_th[t] + m.fossil_heat_output_mwh_th[t]
        == float(demand[t]),
    )
    m.annual_electrified_heat = pyo.Expression(
        expr=pyo.quicksum(m.eboiler_heat_output_mwh_th[t] for t in m.T)
    )
    if apply_min_target and inp.minimum_electrification_pct is not None:
        required = inp.minimum_electrification_pct / 100.0 * float(demand.sum())
        m.minimum_electrification = pyo.Constraint(expr=m.annual_electrified_heat >= required)

    # --- Innovation Fund ---
    if inc.needs_eligibility_variable:
        m.eligible_electrified_heat_mwh_th = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.eligibility = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.eligible_electrified_heat_mwh_th[t]
            <= eta * (
                (m.eboiler_pv_input_mwh_e[t] if inc.pv_heat_eligible else 0)
                + (m.eboiler_grid_input_mwh_e[t] if inc.grid_heat_eligible else 0)
            ),
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
        eligible = lambda t: m.eboiler_heat_output_mwh_th[t]  # noqa: E731
    else:
        eligible = None

    premium = inc.premium_eur_per_mwh_th if inc.active else 0.0
    m.total_pv_use = pyo.Expression(expr=pyo.quicksum(m.eboiler_pv_input_mwh_e[t] for t in m.T))
    cost_terms = []
    for t in range(n):
        term = (
            c_pv * m.eboiler_pv_input_mwh_e[t]
            + float(c_grid[t]) * m.eboiler_grid_input_mwh_e[t]
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


def _extract(m: pyo.ConcreteModel, inp: DispatchInputs) -> pd.DataFrame:
    T = list(m.T)
    pv_in = clean(var_values(m.eboiler_pv_input_mwh_e, T))
    grid_in = clean(var_values(m.eboiler_grid_input_mwh_e, T))
    heat = clean(var_values(m.eboiler_heat_output_mwh_th, T))
    fossil = clean(var_values(m.fossil_heat_output_mwh_th, T))
    if hasattr(m, "eligible_electrified_heat_mwh_th"):
        eligible = clean(var_values(m.eligible_electrified_heat_mwh_th, T))
    elif inp.incentive.active:
        eligible = heat.copy()
    else:
        eligible = np.zeros(len(T))
    return pd.DataFrame(
        {
            "eboiler_pv_input_mwh_e": pv_in,
            "eboiler_grid_input_mwh_e": grid_in,
            "eboiler_heat_output_mwh_th": heat,
            "fossil_heat_output_mwh_th": fossil,
            "eligible_electrified_heat_mwh_th": eligible,
        }
    )


def maximize_eboiler_electrification(inp: DispatchInputs, params: EboilerInputs) -> InfeasibilityDiagnosis:
    """Porcentaje máximo técnicamente alcanzable manteniendo todas las restricciones físicas."""
    m = build_eboiler_model(inp, params, objective="max_electrification", apply_min_target=False)
    report = solve_model(m, inp.solver)
    requested = inp.minimum_electrification_pct or 0.0
    diag = InfeasibilityDiagnosis(
        requested_minimum_pct=requested, max_achievable_pct=None,
        max_achievable_heat_mwh_th=None, solver=report,
    )
    if not report.is_optimal:
        diag.limiting_factors.append(f"No se pudo calcular el máximo alcanzable: {report.message}")
        return diag
    dec = _extract(m, inp)
    total_demand = inp.annual_thermal_demand_mwh_th
    heat = dec["eboiler_heat_output_mwh_th"].to_numpy()
    elec = dec["eboiler_pv_input_mwh_e"].to_numpy() + dec["eboiler_grid_input_mwh_e"].to_numpy()
    diag.max_achievable_heat_mwh_th = float(heat.sum())
    diag.max_achievable_pct = 100.0 * heat.sum() / total_demand if total_demand > 0 else 0.0
    dt = inp.time_step_hours
    unmet = inp.thermal_demand_mwh_th - heat > 1e-6
    cap_bind = binding_count(elec[unmet], np.full(unmet.sum(), params.eboiler_capacity_mw_e * dt))
    grid_lim = inp.grid_capacity_available_mw * dt + inp.pv_generation_mwh_e
    grid_bind = int(np.sum(unmet & np.isfinite(grid_lim) & (elec >= grid_lim - 1e-6)
                           & (grid_lim < params.eboiler_capacity_mw_e * dt - 1e-6)))
    diag.binding_hours = {
        "horas_con_demanda_no_electrificada": int(unmet.sum()),
        "horas_limitadas_por_potencia_eboiler": cap_bind,
        "horas_limitadas_por_capacidad_de_red": grid_bind,
    }
    if cap_bind:
        diag.limiting_factors.append(
            f"Potencia del E-boiler ({params.eboiler_capacity_mw_e:.2f} MW-e x η={params.eboiler_efficiency:.3f}) "
            f"inferior a la demanda en {cap_bind} h."
        )
    if grid_bind:
        diag.limiting_factors.append(f"Capacidad de red disponible limitante en {grid_bind} h.")
    if not diag.limiting_factors and unmet.any():
        diag.limiting_factors.append("Combinación de límites de potencia, red y PV.")
    return diag


def solve_eboiler(
    inp: DispatchInputs, params: EboilerInputs, compute_max_on_infeasible: bool = True
) -> DispatchResult:
    """Resuelve el despacho óptimo E-boiler (con prioridad PV opcional)."""
    m = build_eboiler_model(inp, params)
    report = solve_model(m, inp.solver)
    if not report.is_optimal:
        diag = None
        if report.is_infeasible and inp.minimum_electrification_pct is not None and compute_max_on_infeasible:
            diag = maximize_eboiler_electrification(inp, params)
        elif report.is_infeasible:
            diag = InfeasibilityDiagnosis(
                requested_minimum_pct=inp.minimum_electrification_pct or 0.0,
                max_achievable_pct=None, max_achievable_heat_mwh_th=None,
            )
        return DispatchResult(ALTERNATIVE, report, None, None, infeasibility=diag)

    stage1_cost = float(pyo.value(m.total_variable_cost_eur))
    priority_info = None
    if inp.pv_priority_enabled and inp.pv_generation_mwh_e.sum() > config.NUMERIC_TOLERANCE:
        stage1_elec = float(pyo.value(m.annual_electrified_heat))
        report2, priority_info = apply_pv_priority(
            m, m.objective, m.total_pv_use, m.annual_electrified_heat, stage1_cost, stage1_elec, inp
        )
        if not report2.is_optimal:
            report.message += f" Etapa de prioridad PV no óptima ({report2.message}); se mantiene la etapa 1."
            m = build_eboiler_model(inp, params)
            report = solve_model(m, inp.solver)
        else:
            report.message += " Prioridad PV aplicada (optimización lexicográfica)."
    decisions = _extract(m, inp)
    objective = float(pyo.value(m.total_variable_cost_eur))
    return DispatchResult(ALTERNATIVE, report, decisions, objective, pv_priority_stage=priority_info)
