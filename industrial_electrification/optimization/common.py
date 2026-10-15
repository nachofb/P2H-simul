"""Utilidades comunes de optimización (Pyomo + HiGHS).

Contiene:
- ``DispatchInputs``: parámetros horarios ya procesados que recibe cada
  optimizador (independiente de archivos y de la interfaz).
- ``IncentiveTerms``: condiciones del Innovation Fund aplicables en el año.
- ``solve_model``: resolución con informe de estado, terminación y calidad.
- ``apply_pv_priority``: optimización lexicográfica de prioridad PV.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

import config
from models.inputs import GasBaselineInputs, SolverSettings
from models.results import SolverReport


class OptimizationError(RuntimeError):
    pass


@dataclass
class IncentiveTerms:
    """Condiciones del incentivo en el año optimizado.

    premium_eur_per_mwh_th = puja (EUR/tCO2) x factor de referencia (tCO2/MWh-t).
    """

    premium_eur_per_mwh_th: float = 0.0
    pv_heat_eligible: bool = True
    grid_heat_eligible: bool = True
    annual_max_eligible_mwh_th: Optional[float] = None
    max_revenue_eur: Optional[float] = None  # importe restante disponible en el año

    @property
    def active(self) -> bool:
        return self.premium_eur_per_mwh_th > 0 and (self.pv_heat_eligible or self.grid_heat_eligible)

    @property
    def needs_eligibility_variable(self) -> bool:
        return self.active and (
            not (self.pv_heat_eligible and self.grid_heat_eligible)
            or self.annual_max_eligible_mwh_th is not None
            or self.max_revenue_eur is not None
        )


@dataclass
class DispatchInputs:
    """Parámetros horarios del despacho de un año (todas las series de longitud T)."""

    thermal_demand_mwh_th: np.ndarray
    grid_unit_cost_eur_per_mwh_e: np.ndarray
    pv_generation_mwh_e: np.ndarray
    grid_capacity_available_mw: np.ndarray  # np.inf = sin límite específico
    gas: GasBaselineInputs
    pv_price_eur_per_mwh_e: float = 0.0
    time_step_hours: float = 1.0
    incentive: IncentiveTerms = field(default_factory=IncentiveTerms)
    minimum_electrification_pct: Optional[float] = None
    pv_priority_enabled: bool = False
    pv_priority_cost_tolerance_pct: float = 0.1
    pv_priority_cost_tolerance_eur: float = 0.0
    solver: SolverSettings = field(default_factory=SolverSettings)

    def __post_init__(self) -> None:
        self.thermal_demand_mwh_th = np.asarray(self.thermal_demand_mwh_th, dtype=float)
        self.grid_unit_cost_eur_per_mwh_e = np.asarray(self.grid_unit_cost_eur_per_mwh_e, dtype=float)
        self.pv_generation_mwh_e = np.asarray(self.pv_generation_mwh_e, dtype=float)
        self.grid_capacity_available_mw = np.asarray(self.grid_capacity_available_mw, dtype=float)
        n = len(self.thermal_demand_mwh_th)
        for name in ("grid_unit_cost_eur_per_mwh_e", "pv_generation_mwh_e", "grid_capacity_available_mw"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name}: longitud {len(getattr(self, name))} distinta de la demanda ({n}).")
        if np.isnan(self.thermal_demand_mwh_th).any() or (self.thermal_demand_mwh_th < 0).any():
            raise ValueError("Demanda térmica con valores nulos o negativos.")
        if np.isnan(self.grid_unit_cost_eur_per_mwh_e).any():
            raise ValueError("Coste unitario de red con valores nulos.")
        if np.isnan(self.pv_generation_mwh_e).any() or (self.pv_generation_mwh_e < 0).any():
            raise ValueError("Generación PV con valores nulos o negativos.")
        if np.isnan(self.grid_capacity_available_mw).any() or (self.grid_capacity_available_mw < 0).any():
            raise ValueError("Capacidad de red con valores nulos o negativos.")
        if self.minimum_electrification_pct is not None and not 0 <= self.minimum_electrification_pct <= 100:
            raise ValueError("El porcentaje mínimo de electrificación debe estar entre 0 y 100.")

    @property
    def n_hours(self) -> int:
        return len(self.thermal_demand_mwh_th)

    @property
    def annual_thermal_demand_mwh_th(self) -> float:
        return float(self.thermal_demand_mwh_th.sum())

    @property
    def fossil_cost_eur_per_mwh_th(self) -> dict[str, float]:
        """Coste fósil por MWh-t (combustible, CO2 y otros), convención PCS/PCI."""
        k = 1.0 / self.gas.fossil_efficiency_hhv
        return {
            "fuel": self.gas.gas_price_eur_per_mwh_hhv * k,
            "co2": self.gas.gas_emission_factor_tco2_per_mwh_hhv * self.gas.co2_price_eur_per_tco2 * k,
            "other": self.gas.gas_other_cost_eur_per_mwh_hhv * k,
        }

    def copy_with(self, **changes) -> "DispatchInputs":
        data = dict(self.__dict__)
        data.update(changes)
        return DispatchInputs(**data)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def _get_solver(settings: SolverSettings):
    names = [settings.solver_name] + [s for s in config.SOLVER_FALLBACKS if s != settings.solver_name]
    errors = []
    for name in names:
        try:
            opt = pyo.SolverFactory(name)
            if opt is not None and opt.available(exception_flag=False):
                return name, opt
        except Exception as exc:  # pragma: no cover - depende del entorno
            errors.append(f"{name}: {exc}")
    raise OptimizationError(
        "No hay ningún solver HiGHS disponible. Instale 'highspy' (pip install highspy). "
        + "; ".join(errors)
    )


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def solve_model(model: pyo.ConcreteModel, settings: SolverSettings, is_mip: bool = False) -> SolverReport:
    """Resuelve y carga la solución solo si es óptima."""
    name, opt = _get_solver(settings)
    options = {"time_limit": float(settings.time_limit_seconds)}
    if is_mip:
        options["mip_rel_gap"] = float(settings.mip_rel_gap)
    t0 = time.perf_counter()
    try:
        results = opt.solve(model, load_solutions=False, tee=settings.tee, options=options)
    except TypeError:
        results = opt.solve(model, load_solutions=False, tee=settings.tee)
    except Exception as exc:
        return SolverReport(
            solver_name=name, status="error", termination_condition="exception",
            is_optimal=False, is_infeasible=False, solve_time_s=time.perf_counter() - t0,
            is_mip=is_mip, message=f"Error del solver: {exc}",
        )
    elapsed = time.perf_counter() - t0
    tc = results.solver.termination_condition
    status = results.solver.status
    is_optimal = tc == TerminationCondition.optimal and status in (SolverStatus.ok, SolverStatus.warning)
    is_infeasible = tc in (
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    )
    lb = _safe_float(getattr(results.problem, "lower_bound", None))
    ub = _safe_float(getattr(results.problem, "upper_bound", None))
    objective = None
    gap = None
    if is_optimal:
        model.solutions.load_from(results)
        obj = next(model.component_data_objects(pyo.Objective, active=True))
        objective = float(pyo.value(obj))
        if lb is not None and ub is not None:
            gap = abs(ub - lb) / max(1.0, abs(ub))
    if is_optimal:
        message = "Solución óptima."
    elif is_infeasible:
        message = "Problema INFACTIBLE: no existe despacho que cumpla todas las restricciones."
    elif tc == TerminationCondition.maxTimeLimit:
        message = "Límite de tiempo alcanzado sin solución óptima demostrada. Resultado no válido."
    elif tc == TerminationCondition.unbounded:
        message = "Problema NO ACOTADO: revise precios y límites."
    else:
        message = f"Terminación no óptima: {tc}."
    return SolverReport(
        solver_name=name, status=str(status), termination_condition=str(tc),
        is_optimal=is_optimal, is_infeasible=is_infeasible, objective_value=objective,
        lower_bound=lb, upper_bound=ub, relative_gap=gap, solve_time_s=elapsed,
        is_mip=is_mip, message=message,
    )


def var_values(var, index) -> np.ndarray:
    return np.fromiter((var[t].value if var[t].value is not None else 0.0 for t in index), dtype=float)


def clean(values: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Elimina ruido numérico del solver (|x| < tol -> 0)."""
    v = np.asarray(values, dtype=float).copy()
    v[np.abs(v) < tol] = 0.0
    return v


# ---------------------------------------------------------------------------
# Prioridad PV (lexicográfica)
# ---------------------------------------------------------------------------
def apply_pv_priority(
    model: pyo.ConcreteModel,
    cost_objective: pyo.Objective,
    pv_use_expr,
    electrified_expr,
    stage1_cost: float,
    stage1_electrified: float,
    inputs: DispatchInputs,
    is_mip: bool = False,
) -> tuple[SolverReport, dict]:
    """Etapa 2: maximiza el uso de PV sin empeorar el coste más que la tolerancia.

    Restricciones añadidas:
    - coste <= coste_etapa1 + max(tol_abs, tol_pct * |coste_etapa1|)
    - calor electrificado >= calor electrificado de la etapa 1 (no reduce la
      electrificación ni introduce calor inútil: el balance térmico sigue
      siendo una igualdad).
    """
    tol = max(
        inputs.pv_priority_cost_tolerance_eur,
        inputs.pv_priority_cost_tolerance_pct / 100.0 * abs(stage1_cost),
    )
    model.pv_priority_cost_cap = pyo.Constraint(expr=cost_objective.expr <= stage1_cost + tol)
    model.pv_priority_electrified_floor = pyo.Constraint(
        expr=electrified_expr >= stage1_electrified - config.NUMERIC_TOLERANCE
    )
    cost_objective.deactivate()
    model.pv_priority_objective = pyo.Objective(expr=pv_use_expr, sense=pyo.maximize)
    report = solve_model(model, inputs.solver, is_mip=is_mip)
    info = {
        "stage1_cost_eur": stage1_cost,
        "cost_tolerance_eur": tol,
        "stage2_status": report.termination_condition,
    }
    if report.is_optimal:
        info["stage2_cost_eur"] = float(pyo.value(cost_objective.expr))
        info["stage2_pv_used_mwh_e"] = float(pyo.value(pv_use_expr))
    return report, info


def binding_count(values: np.ndarray, limit: np.ndarray, tol: float = 1e-6) -> int:
    limit = np.broadcast_to(np.asarray(limit, dtype=float), values.shape)
    finite = np.isfinite(limit)
    return int(np.sum(finite & (values >= limit - tol) & (limit > tol)))
