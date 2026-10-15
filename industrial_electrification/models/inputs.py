"""Modelos Pydantic de parámetros de entrada.

Separación conceptual:
- Este módulo contiene únicamente PARÁMETROS DE ENTRADA (definidos por el
  usuario o por archivos de datos).
- Las VARIABLES DE DECISIÓN se definen en ``optimization/eboiler.py`` y
  ``optimization/tes.py``.
- Las VARIABLES CALCULADAS (resultados) se construyen en ``results/``.

Unidades: ver ``docs/variable_dictionary.md``. Sufijos: ``_mw_e``, ``_mw_th``,
``_mwh_e``, ``_mwh_th``, ``_eur``, ``_eur_per_mwh_e``, ``_eur_per_mwh_hhv``...
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import config

VALID_CURVE_LENGTHS = (8760, 8784)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=False)


def _check_series(values: list[float], name: str, allow_negative: bool = False) -> None:
    if len(values) not in VALID_CURVE_LENGTHS:
        raise ValueError(
            f"{name}: la serie tiene {len(values)} registros; se esperaban 8760 u 8784."
        )
    bad = [i + 1 for i, v in enumerate(values) if v is None or not math.isfinite(v)]
    if bad:
        raise ValueError(f"{name}: valores no numéricos o nulos en registros {bad[:10]}...")
    if not allow_negative:
        neg = [i + 1 for i, v in enumerate(values) if v < 0]
        if neg:
            raise ValueError(f"{name}: valores negativos en registros {neg[:10]}...")


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------
class SimulationSettings(_Base):
    simulation_start_year: int = Field(2027, ge=2000, le=2100, description="Año inicial")
    simulation_years: int = Field(1, ge=1, le=40, description="Años consecutivos optimizados")
    asset_lifetime_years: int = Field(20, ge=1, le=60, description="Vida útil del proyecto")
    amortization_years: int = Field(10, ge=1, le=60, description="Plazo de amortización financiera")
    interest_rate_pct: float = Field(5.0, ge=0, le=50, description="Tipo de interés financiación (%)")
    discount_rate_pct: float = Field(7.0, ge=0, le=50, description="Tasa de descuento (%)")
    time_step_hours: float = Field(config.DEFAULT_TIME_STEP_HOURS, gt=0)

    @field_validator("time_step_hours")
    @classmethod
    def _check_step(cls, v: float) -> float:
        if v not in config.SUPPORTED_TIME_STEP_HOURS:
            raise ValueError(
                f"time_step_hours={v} no soportado en esta versión "
                f"(soportados: {config.SUPPORTED_TIME_STEP_HOURS})."
            )
        return v

    @property
    def simulated_years_list(self) -> list[int]:
        return [self.simulation_start_year + i for i in range(self.simulation_years)]


# ---------------------------------------------------------------------------
# Demanda térmica (módulo común)
# ---------------------------------------------------------------------------
class DemandInputs(_Base):
    method: Literal["hourly_curve", "constant", "weekday_weekend"] = "constant"
    thermal_demand_constant_mw_th: float = Field(10.0, ge=0)
    weekday_thermal_demand_mw_th: float = Field(10.0, ge=0)
    weekend_thermal_demand_mw_th: float = Field(5.0, ge=0)
    curve_values: Optional[list[float]] = None
    curve_kind: Literal["power_mw_th", "energy_mwh_th"] = "power_mw_th"
    curve_source_name: Optional[str] = None

    @model_validator(mode="after")
    def _check_curve(self) -> "DemandInputs":
        if self.method == "hourly_curve":
            if not self.curve_values:
                raise ValueError("Demanda por curva horaria: no se ha cargado la curva.")
            _check_series(self.curve_values, "Curva de demanda térmica")
        return self


# ---------------------------------------------------------------------------
# Escenario fósil de referencia
# ---------------------------------------------------------------------------
class GasBaselineInputs(_Base):
    gas_price_eur_per_mwh_hhv: float = Field(35.0, ge=0)
    co2_price_eur_per_tco2: float = Field(80.0, ge=0)
    gas_other_cost_eur_per_mwh_hhv: float = Field(2.0, ge=0)
    fossil_boiler_efficiency_lhv: float = Field(
        0.92, gt=0, le=1.11, description="Rendimiento sobre PCI (puede superar 1 en calderas de condensación)"
    )
    gas_emission_factor_tco2_per_mwh_hhv: float = Field(
        config.DEFAULT_GAS_EMISSION_FACTOR_TCO2_PER_MWH_HHV, ge=0
    )
    lhv_hhv_ratio: float = Field(config.DEFAULT_LHV_HHV_RATIO, gt=0, le=1)

    @property
    def fossil_efficiency_hhv(self) -> float:
        """Rendimiento referido a PCS = rendimiento PCI x (PCI/PCS)."""
        return self.fossil_boiler_efficiency_lhv * self.lhv_hhv_ratio


# ---------------------------------------------------------------------------
# Suministro eléctrico
# ---------------------------------------------------------------------------
class ElectricitySupplyInputs(_Base):
    # PV onsite
    has_onsite_pv: bool = False
    pv_installed_capacity_mwp: float = Field(0.0, ge=0)
    pv_profile_values: Optional[list[float]] = None
    pv_profile_kind: Literal["absolute_mwh_e", "specific_mwh_e_per_mwp"] = "specific_mwh_e_per_mwp"
    pv_profile_source_name: Optional[str] = None
    pv_price_eur_per_mwh_e: float = Field(40.0)
    pv_priority_enabled: bool = False
    pv_priority_cost_tolerance_pct: float = Field(0.1, ge=0, le=100)
    pv_priority_cost_tolerance_eur: float = Field(0.0, ge=0)

    # Capacidad de red
    grid_capacity_limit_enabled: bool = False
    grid_capacity_by_period_enabled: bool = False
    grid_capacity_mw: float = Field(0.0, ge=0)
    grid_capacity_p1_mw: float = Field(0.0, ge=0)
    grid_capacity_p2_mw: float = Field(0.0, ge=0)
    grid_capacity_p3_mw: float = Field(0.0, ge=0)
    grid_capacity_p4_mw: float = Field(0.0, ge=0)
    grid_capacity_p5_mw: float = Field(0.0, ge=0)
    grid_capacity_p6_mw: float = Field(0.0, ge=0)

    # Precios
    pool_price_scenario: str = config.DEFAULT_POOL_SCENARIO
    pool_price_year: Optional[int] = Field(
        None, description="Año del escenario pool. None = mismo año que el simulado."
    )
    pool_price_scaling_factor: float = Field(
        1.0, ge=0, description="Multiplicador de sensibilidad sobre el pool (1 = sin cambio)"
    )
    pool_missing_data_fill_authorized: bool = False
    pool_max_fill_hours: int = Field(config.DEFAULT_POOL_MAX_FILL_HOURS, ge=0, le=168)
    adjustment_price_mode: Literal["fixed", "hourly_series"] = "hourly_series"
    adjustment_fixed_eur_per_mwh_e: float = 0.0
    network_tariff: Literal["6.1TD", "6.2TD", "6.3TD", "6.4TD"] = "6.1TD"
    is_electrointensive: bool = Field(
        False, description="True = aplicar tarifas reducidas de peajes y cargos para consumidores electrointensivos"
    )
    guarantees_of_origin_enabled: bool = False
    guarantees_of_origin_cost_eur_per_mwh_e: float = Field(0.0, ge=0)
    grid_other_cost_eur_per_mwh_e: float = 0.0

    # Emisiones indirectas (ampliación opcional, no afecta al despacho)
    indirect_emissions_enabled: bool = False
    grid_emission_factor_tco2_per_mwh_e: float = Field(0.0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "ElectricitySupplyInputs":
        if self.has_onsite_pv:
            if not self.pv_profile_values:
                raise ValueError("PV onsite activada: debe cargarse el perfil horario P50.")
            _check_series(self.pv_profile_values, "Perfil PV")
            if self.pv_profile_kind == "specific_mwh_e_per_mwp" and self.pv_installed_capacity_mwp <= 0:
                raise ValueError("Perfil PV específico: indique una potencia instalada > 0 MWp.")
        if self.grid_capacity_limit_enabled and not self.grid_capacity_by_period_enabled:
            if self.grid_capacity_mw <= 0:
                raise ValueError(
                    "Límite de red activado con capacidad 0 MW: el activo no podría importar de red."
                )
        return self

    def grid_capacity_by_period_mw(self) -> dict[int, float]:
        return {
            1: self.grid_capacity_p1_mw, 2: self.grid_capacity_p2_mw, 3: self.grid_capacity_p3_mw,
            4: self.grid_capacity_p4_mw, 5: self.grid_capacity_p5_mw, 6: self.grid_capacity_p6_mw,
        }


# ---------------------------------------------------------------------------
# Electrificación mínima
# ---------------------------------------------------------------------------
class ElectrificationTargetInputs(_Base):
    minimum_electrification_enabled: bool = False
    minimum_electrification_pct: float = Field(0.0, ge=0, le=100)
    compute_max_on_infeasible: bool = True

    @property
    def effective_minimum_pct(self) -> Optional[float]:
        return self.minimum_electrification_pct if self.minimum_electrification_enabled else None


# ---------------------------------------------------------------------------
# Innovation Fund
# ---------------------------------------------------------------------------
class InnovationFundInputs(_Base):
    innovation_fund_enabled: bool = False
    innovation_fund_bid_eur_per_tco2: float = Field(0.0, ge=0)
    innovation_fund_reference_tco2_per_mwh_th: float = Field(
        config.DEFAULT_INNOVATION_FUND_REFERENCE_TCO2_PER_MWH_TH, ge=0
    )
    innovation_fund_duration_years: int = Field(
        config.DEFAULT_INNOVATION_FUND_DURATION_YEARS, ge=1, le=60
    )
    innovation_fund_start_year: Optional[int] = Field(
        None, description="Año de inicio. None = año inicial de la simulación."
    )
    innovation_fund_annual_max_eligible_mwh_th: Optional[float] = Field(None, ge=0)
    innovation_fund_max_total_amount_eur: Optional[float] = Field(None, ge=0)
    innovation_fund_pv_heat_eligible: bool = True
    innovation_fund_grid_heat_eligible: bool = True

    @property
    def innovation_fund_premium_eur_per_mwh_th(self) -> float:
        return self.innovation_fund_bid_eur_per_tco2 * self.innovation_fund_reference_tco2_per_mwh_th

    def is_active_in_year(self, year: int, default_start_year: int) -> bool:
        if not self.innovation_fund_enabled:
            return False
        start = self.innovation_fund_start_year or default_start_year
        return start <= year < start + self.innovation_fund_duration_years


# ---------------------------------------------------------------------------
# Alternativa 1: E-boiler
# ---------------------------------------------------------------------------
class EboilerInputs(_Base):
    eboiler_capacity_mw_e: float = Field(10.0, ge=0)
    eboiler_efficiency: float = Field(0.99, gt=0, le=1)
    eboiler_capex_eur: float = Field(2_500_000.0, ge=0)
    eboiler_annual_opex_eur: float = Field(50_000.0, ge=0)


# ---------------------------------------------------------------------------
# Alternativa 2: TES
# ---------------------------------------------------------------------------
class TesInputs(_Base):
    tes_charge_capacity_mw_e: float = Field(20.0, ge=0)
    tes_storage_capacity_mwh_th: float = Field(200.0, ge=0)
    tes_discharge_capacity_mw_th: float = Field(10.0, ge=0)
    tes_charge_efficiency: float = Field(0.98, gt=0, le=1)
    tes_discharge_efficiency: float = Field(0.95, gt=0, le=1)
    tes_hourly_self_discharge_pct: float = Field(0.05, ge=0, lt=100)
    tes_initial_soc_mwh_th: float = Field(0.0, ge=0)
    tes_final_soc_mode: Literal["equal_initial", "fixed", "minimum", "free"] = "equal_initial"
    tes_final_soc_mwh_th: float = Field(0.0, ge=0)
    tes_enforce_no_simultaneous_charge_discharge: bool = Field(
        False, description="True = formulación MILP con binarias de no simultaneidad"
    )
    tes_capex_eur: float = Field(8_000_000.0, ge=0)
    tes_annual_opex_eur: float = Field(100_000.0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "TesInputs":
        if self.tes_initial_soc_mwh_th > self.tes_storage_capacity_mwh_th + 1e-9:
            raise ValueError("El SOC inicial supera la capacidad de almacenamiento del TES.")
        if self.tes_final_soc_mode in ("fixed", "minimum") and (
            self.tes_final_soc_mwh_th > self.tes_storage_capacity_mwh_th + 1e-9
        ):
            raise ValueError("El SOC final requerido supera la capacidad de almacenamiento del TES.")
        return self

    @property
    def required_final_soc_mwh_th(self) -> Optional[float]:
        if self.tes_final_soc_mode == "equal_initial":
            return self.tes_initial_soc_mwh_th
        if self.tes_final_soc_mode in ("fixed", "minimum"):
            return self.tes_final_soc_mwh_th
        return None


# ---------------------------------------------------------------------------
# Finanzas
# ---------------------------------------------------------------------------
class CashFlowItem(_Base):
    project_year: int = Field(..., ge=1, description="Año del proyecto (1 = primer año de operación)")
    amount_eur: float = Field(..., ge=0)
    description: str = ""


class FinancialInputs(_Base):
    avoided_fossil_fixed_opex_eur: float = Field(
        0.0, ge=0, description="OPEX fijo fósil evitado (reduce el OPEX incremental)"
    )
    replacement_costs: list[CashFlowItem] = Field(default_factory=list)
    additional_capex: list[CashFlowItem] = Field(default_factory=list)
    residual_value_eur: float = Field(0.0, ge=0)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class SolverSettings(_Base):
    solver_name: str = config.DEFAULT_SOLVER_NAME
    time_limit_seconds: float = Field(config.DEFAULT_SOLVER_TIME_LIMIT_S, gt=0)
    mip_rel_gap: float = Field(config.DEFAULT_MIP_REL_GAP, ge=0, le=1)
    tee: bool = False


# ---------------------------------------------------------------------------
# Caso completo
# ---------------------------------------------------------------------------
class ScenarioConfig(_Base):
    case_name: str = "Caso base"
    description: str = ""
    alternative: Literal["eboiler", "tes"] = "eboiler"
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    demand: DemandInputs = Field(default_factory=DemandInputs)
    baseline: GasBaselineInputs = Field(default_factory=GasBaselineInputs)
    supply: ElectricitySupplyInputs = Field(default_factory=ElectricitySupplyInputs)
    target: ElectrificationTargetInputs = Field(default_factory=ElectrificationTargetInputs)
    innovation_fund: InnovationFundInputs = Field(default_factory=InnovationFundInputs)
    eboiler: EboilerInputs = Field(default_factory=EboilerInputs)
    tes: TesInputs = Field(default_factory=TesInputs)
    finance: FinancialInputs = Field(default_factory=FinancialInputs)
    solver: SolverSettings = Field(default_factory=SolverSettings)

    @property
    def asset_capex_eur(self) -> float:
        return self.eboiler.eboiler_capex_eur if self.alternative == "eboiler" else self.tes.tes_capex_eur

    @property
    def asset_annual_opex_eur(self) -> float:
        return (
            self.eboiler.eboiler_annual_opex_eur
            if self.alternative == "eboiler"
            else self.tes.tes_annual_opex_eur
        )

    @property
    def incremental_annual_opex_eur(self) -> float:
        return self.asset_annual_opex_eur - self.finance.avoided_fossil_fixed_opex_eur

    # --- Persistencia --------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "ScenarioConfig":
        return cls.model_validate(json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def copy_with(self, **updates) -> "ScenarioConfig":
        """Duplica el caso aplicando actualizaciones con notación 'seccion.campo'."""
        data = self.model_dump()
        for key, value in updates.items():
            parts = key.split(".")
            target = data
            for p in parts[:-1]:
                target = target[p]
            if parts[-1] not in target:
                raise KeyError(f"Parámetro desconocido: {key}")
            target[parts[-1]] = value
        return ScenarioConfig.model_validate(data)
