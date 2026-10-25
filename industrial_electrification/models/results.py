"""Estructuras de resultados (variables calculadas).

Se usan dataclasses (no Pydantic) porque contienen DataFrames y arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class SolverReport:
    """Información del solver: estado, condición de terminación y calidad."""

    solver_name: str
    status: str
    termination_condition: str
    is_optimal: bool
    is_infeasible: bool
    objective_value: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    relative_gap: Optional[float] = None
    solve_time_s: Optional[float] = None
    is_mip: bool = False
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class InfeasibilityDiagnosis:
    """Diagnóstico cuando el objetivo mínimo de electrificación no es alcanzable."""

    requested_minimum_pct: float
    max_achievable_pct: Optional[float]
    max_achievable_heat_mwh_th: Optional[float]
    limiting_factors: list[str] = field(default_factory=list)
    binding_hours: dict[str, int] = field(default_factory=dict)
    solver: Optional[SolverReport] = None


@dataclass
class DispatchResult:
    """Resultado de un optimizador de despacho (un año)."""

    alternative: str
    solver: SolverReport
    decisions: Optional[pd.DataFrame]  # variables de decisión horarias
    objective_eur: Optional[float]
    pv_priority_stage: Optional[dict[str, Any]] = None
    infeasibility: Optional[InfeasibilityDiagnosis] = None

    @property
    def is_valid(self) -> bool:
        return self.solver.is_optimal and self.decisions is not None


@dataclass
class YearResult:
    """Resultados de un año simulado."""

    year: int
    alternative: str
    dispatch: DispatchResult
    hourly: Optional[pd.DataFrame] = None
    annual: dict[str, Any] = field(default_factory=dict)
    by_period: Optional[pd.DataFrame] = None
    price_stats: Optional[pd.DataFrame] = None
    monthly: Optional[pd.DataFrame] = None
    innovation_fund_active: bool = False
    data_notes: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.dispatch.is_valid


@dataclass
class FinancialResult:
    cash_flows: pd.DataFrame
    indicators: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Resultado completo de un caso (todos los años + finanzas)."""

    case_name: str
    alternative: str
    config_json: str
    years: list[YearResult]
    financial: Optional[FinancialResult] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors and bool(self.years) and all(y.is_valid for y in self.years)

    @property
    def first_year(self) -> Optional[YearResult]:
        return self.years[0] if self.years else None

    def hourly_all_years(self) -> pd.DataFrame:
        frames = [y.hourly for y in self.years if y.hourly is not None]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def annual_table(self) -> pd.DataFrame:
        rows = [{"year": y.year, **y.annual} for y in self.years if y.annual]
        return pd.DataFrame(rows)
