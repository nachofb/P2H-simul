"""Validaciones de datos y de coherencia del caso (previas a la optimización).

Distingue ERRORES (bloquean la ejecución) de AVISOS (informativos).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from data.calendar import hours_in_year
from data.loaders import DataBundle, DataFormatError, pool_price_series_for_year
from models.inputs import ScenarioConfig


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "ValidationReport") -> None:
        self.errors += other.errors
        self.warnings += other.warnings
        self.info += other.info


def validate_series(
    values: np.ndarray,
    name: str,
    expected_length: Optional[int] = None,
    allow_negative: bool = False,
    allow_nan: bool = False,
) -> ValidationReport:
    rep = ValidationReport()
    arr = np.asarray(values, dtype=float)
    if expected_length is not None and len(arr) != expected_length:
        rep.errors.append(f"{name}: {len(arr)} valores; se esperaban {expected_length}.")
    if not allow_nan and np.isnan(arr).any():
        rep.errors.append(f"{name}: {int(np.isnan(arr).sum())} valores nulos.")
    if np.isinf(arr).any():
        rep.errors.append(f"{name}: valores infinitos.")
    if not allow_negative and (arr < 0).any():
        rep.errors.append(f"{name}: {int((arr < 0).sum())} valores negativos no permitidos.")
    return rep


def validate_scenario(cfg: ScenarioConfig, bundle: Optional[DataBundle]) -> ValidationReport:
    """Comprueba coherencia del caso y disponibilidad de datos para todos los años."""
    rep = ValidationReport()
    sim = cfg.simulation
    sup = cfg.supply

    if sim.amortization_years > sim.asset_lifetime_years:
        rep.warnings.append("El plazo de amortización supera la vida útil del activo.")
    if sim.simulation_years > sim.asset_lifetime_years:
        rep.warnings.append("Se simulan más años que la vida útil; los años extra no entran en el VAN.")

    # Activo
    if cfg.alternative == "eboiler":
        if cfg.eboiler.eboiler_capacity_mw_e <= 0:
            rep.warnings.append("E-boiler con capacidad 0 MW-e: no habrá electrificación.")
    else:
        t = cfg.tes
        if min(t.tes_charge_capacity_mw_e, t.tes_storage_capacity_mwh_th, t.tes_discharge_capacity_mw_th) <= 0:
            rep.warnings.append("TES con alguna capacidad nula: no habrá electrificación.")
        if t.tes_enforce_no_simultaneous_charge_discharge:
            rep.info.append("TES: formulación MILP (no simultaneidad carga/descarga). Mayor tiempo de cálculo.")

    # Red
    if sup.grid_capacity_limit_enabled and sup.grid_capacity_by_period_enabled:
        caps = sup.grid_capacity_by_period_mw()
        zero = [f"P{p}" for p, v in caps.items() if v <= 0]
        if zero:
            rep.warnings.append(f"Capacidad de red nula en {', '.join(zero)}: sin importación en esos periodos.")
    if not sup.has_onsite_pv and sup.pv_priority_enabled:
        rep.warnings.append("Prioridad PV activada sin planta PV: se ignora.")

    # Objetivo de electrificación
    if cfg.target.minimum_electrification_enabled and cfg.target.minimum_electrification_pct >= 100:
        rep.warnings.append("Objetivo de electrificación del 100 %: puede ser infactible.")

    # Innovation Fund
    fund = cfg.innovation_fund
    if fund.innovation_fund_enabled:
        if fund.innovation_fund_premium_eur_per_mwh_th <= 0:
            rep.warnings.append("Innovation Fund activado con prima nula.")
        if not (fund.innovation_fund_pv_heat_eligible or fund.innovation_fund_grid_heat_eligible):
            rep.warnings.append("Innovation Fund: ninguna fuente es elegible; no habrá ingresos.")
        rep.info.append(
            "Innovation Fund: regla de elegibilidad configurada como hipótesis de simulación, "
            "no como interpretación jurídica de la convocatoria."
        )

    # Datos
    if bundle is None:
        rep.errors.append("No se han cargado los archivos de datos (pool, peajes, ajustes).")
        return rep
    if sup.adjustment_price_mode == "hourly_series" and bundle.adjustment_values is None:
        rep.errors.append("Servicios de ajuste en modo serie horaria pero no se ha cargado el archivo.")
    tariff_table = (
        bundle.network_tariffs_electrointensive
        if sup.is_electrointensive and bundle.network_tariffs_electrointensive is not None
        else bundle.network_tariffs
    )
    if sup.network_tariff.replace(" ", "").upper() not in tariff_table.index:
        rep.errors.append(f"Tarifa {sup.network_tariff} no presente en el archivo de peajes.")
    if sup.is_electrointensive:
        if bundle.network_tariffs_electrointensive is None:
            rep.warnings.append("Consumidor electrointensivo activado pero no se ha cargado el archivo de peajes electrointensivos.")
        else:
            rep.info.append("Peajes y cargos de consumidor electrointensivo activados.")

    for i, year in enumerate(sim.simulated_years_list):
        if str(sup.pool_price_scenario).strip().upper() == "REAL":
            data_year: int | str = "LTM"
        else:
            data_year = (sup.pool_price_year + i) if sup.pool_price_year else year
        try:
            _, notes = pool_price_series_for_year(
                bundle.pool, data_year, sup.pool_price_scenario, target_year=year,
                fill_authorized=sup.pool_missing_data_fill_authorized,
                max_fill_hours=sup.pool_max_fill_hours,
            )
            for n in notes:
                (rep.warnings if n.startswith("AVISO") else rep.info).append(n)
        except DataFormatError as exc:
            rep.errors.append(str(exc))
        n_hours = hours_in_year(year)
        if sup.has_onsite_pv and sup.pv_profile_values and len(sup.pv_profile_values) not in (8760, 8784):
            rep.errors.append(f"Perfil PV con longitud inválida para {year} ({n_hours} h).")
    return rep
