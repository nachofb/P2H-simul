"""Batería exhaustiva de pruebas unitarias y de integración (Pytest).

Verifica:
1. Balances térmicos y de energía.
2. Rendimientos y conversiones PCS/PCI.
3. Límites de red y disponibilidad PV.
4. Restricción de electrificación mínima obligatoria y detección de infactibilidad.
5. Continuidad y límites del SOC en TES.
6. Cálculo de flujos financieros, VAN, TIR, LCOH.
7. Periodos tarifarios y festivos nacionales.
"""
from __future__ import annotations

import numpy as np
import pytest

import config
from baseline.gas_reference import (
    compute_gas_reference,
    fossil_unit_costs_eur_per_mwh_th,
    gas_consumption_mwh_hhv,
)
from data.calendar import build_calendar, calculate_tariff_period, hours_in_year, is_leap_year
from finance.project_economics import annualized_capex, compute_irr
from models.inputs import (
    DemandInputs,
    EboilerInputs,
    GasBaselineInputs,
    ScenarioConfig,
    TesInputs,
)
from optimization.common import DispatchInputs, IncentiveTerms
from optimization.eboiler import solve_eboiler
from optimization.tes import solve_tes


# ---------------------------------------------------------------------------
# 1. Calendario y Periodos Tarifarios
# ---------------------------------------------------------------------------
def test_calendar_leap_and_normal():
    assert hours_in_year(2027) == 8760
    assert hours_in_year(2028) == 8784
    assert not is_leap_year(2027)
    assert is_leap_year(2028)

    cal_2027 = build_calendar(2027)
    assert len(cal_2027) == 8760
    assert "tariff_period" in cal_2027.columns
    assert set(cal_2027["tariff_period"].unique()).issubset({1, 2, 3, 4, 5, 6})


def test_tariff_period_rules():
    # Fin de semana siempre es P6
    assert calculate_tariff_period(month=1, day_of_week=6, hour_of_day=10) == 6
    assert calculate_tariff_period(month=1, day_of_week=7, hour_of_day=10) == 6

    # Festivo nacional siempre es P6
    assert calculate_tariff_period(month=1, day_of_week=1, hour_of_day=10, is_holiday=True) == 6

    # Día laborable en Enero (Alta):
    # 00:00-08:00 -> P6
    assert calculate_tariff_period(month=1, day_of_week=1, hour_of_day=4) == 6
    # 08:00-09:00 -> P2
    assert calculate_tariff_period(month=1, day_of_week=1, hour_of_day=8) == 2
    # 09:00-14:00 -> P1
    assert calculate_tariff_period(month=1, day_of_week=1, hour_of_day=10) == 1
    # 14:00-18:00 -> P2
    assert calculate_tariff_period(month=1, day_of_week=1, hour_of_day=15) == 2
    # 18:00-22:00 -> P1
    assert calculate_tariff_period(month=1, day_of_week=1, hour_of_day=19) == 1


# ---------------------------------------------------------------------------
# 2. Línea Base y Conversiones PCS/PCI
# ---------------------------------------------------------------------------
def test_gas_baseline_conversions():
    gas = GasBaselineInputs(
        gas_price_eur_per_mwh_hhv=30.0,
        co2_price_eur_per_tco2=70.0,
        gas_other_cost_eur_per_mwh_hhv=2.0,
        fossil_boiler_efficiency_lhv=0.90,
        lhv_hhv_ratio=0.90,
        gas_emission_factor_tco2_per_mwh_hhv=0.18,
    )
    # Eficiencia sobre PCS = 0.90 * 0.90 = 0.81
    assert pytest.approx(gas.fossil_efficiency_hhv, 1e-5) == 0.81

    # 81 MWh-th requerirán 100 MWh-HHV
    heat = np.array([81.0])
    cons = gas_consumption_mwh_hhv(heat, gas)
    assert pytest.approx(cons[0], 1e-5) == 100.0

    costs = fossil_unit_costs_eur_per_mwh_th(gas)
    assert costs["gas_mwh_hhv_per_mwh_th"] == pytest.approx(1.0 / 0.81, 1e-4)


# ---------------------------------------------------------------------------
# 3. Optimización E-boiler (24 horas)
# ---------------------------------------------------------------------------
def test_eboiler_optimization_24h():
    n = 24
    demand = np.full(n, 10.0)  # 10 MWh-th/h
    # Precios de red baratos las primeras 12h (20 €/MWh), caros las segundas 12h (100 €/MWh)
    grid_prices = np.array([20.0] * 12 + [100.0] * 12)
    pv = np.zeros(n)
    grid_cap = np.full(n, 20.0)

    gas = GasBaselineInputs(
        gas_price_eur_per_mwh_hhv=50.0,
        co2_price_eur_per_tco2=80.0,
        gas_other_cost_eur_per_mwh_hhv=2.0,
        fossil_boiler_efficiency_lhv=0.90,
        lhv_hhv_ratio=0.90,
    )
    # Coste térmico fósil ~ (50 + 0.1802*80 + 2)/0.81 = 82 €/MWh-th
    # Eboiler η=1.0: con red a 20 € es más barato que gas; con red a 100 € es más caro que gas

    inp = DispatchInputs(
        thermal_demand_mwh_th=demand,
        grid_unit_cost_eur_per_mwh_e=grid_prices,
        pv_generation_mwh_e=pv,
        grid_capacity_available_mw=grid_cap,
        gas=gas,
        minimum_electrification_pct=0.0,
    )

    params = EboilerInputs(
        eboiler_capacity_mw_e=10.0,
        eboiler_efficiency=1.0,
    )

    res = solve_eboiler(inp, params)
    assert res.is_valid
    dec = res.decisions

    # Balance térmico estricto en cada hora
    total_supplied = dec["eboiler_heat_output_mwh_th"] + dec["fossil_heat_output_mwh_th"]
    np.testing.assert_allclose(total_supplied, demand, atol=1e-5)

    # Las primeras 12h debe operar el eboiler a 10 MW
    np.testing.assert_allclose(dec["eboiler_heat_output_mwh_th"][:12], 10.0, atol=1e-4)
    # Las segundas 12h debe operar la caldera de gas
    np.testing.assert_allclose(dec["fossil_heat_output_mwh_th"][12:], 10.0, atol=1e-4)


def test_eboiler_min_electrification_and_infeasibility():
    n = 24
    demand = np.full(n, 10.0)  # Total 240 MWh-th
    grid_prices = np.full(n, 150.0)  # Muy caro, el solver libre elegiría 0%
    gas = GasBaselineInputs(gas_price_eur_per_mwh_hhv=20.0)

    # Caso 1: Obligamos al 50% de electrificación
    inp_50 = DispatchInputs(
        thermal_demand_mwh_th=demand,
        grid_unit_cost_eur_per_mwh_e=grid_prices,
        pv_generation_mwh_e=np.zeros(n),
        grid_capacity_available_mw=np.full(n, 20.0),
        gas=gas,
        minimum_electrification_pct=50.0,
    )
    params = EboilerInputs(eboiler_capacity_mw_e=10.0, eboiler_efficiency=1.0)
    res_50 = solve_eboiler(inp_50, params)
    assert res_50.is_valid
    elec_heat = res_50.decisions["eboiler_heat_output_mwh_th"].sum()
    assert elec_heat >= 120.0 - 1e-4

    # Caso 2: Capacidad E-boiler = 2 MW (máx producción 48 MWh-th = 20%). Pedimos 80% -> Infactible
    params_small = EboilerInputs(eboiler_capacity_mw_e=2.0, eboiler_efficiency=1.0)
    inp_inf = DispatchInputs(
        thermal_demand_mwh_th=demand,
        grid_unit_cost_eur_per_mwh_e=grid_prices,
        pv_generation_mwh_e=np.zeros(n),
        grid_capacity_available_mw=np.full(n, 20.0),
        gas=gas,
        minimum_electrification_pct=80.0,
    )
    res_inf = solve_eboiler(inp_inf, params_small, compute_max_on_infeasible=True)
    assert not res_inf.is_valid
    assert res_inf.solver.is_infeasible
    assert res_inf.infeasibility is not None
    assert pytest.approx(res_inf.infeasibility.max_achievable_pct, 1.0) == 20.0


# ---------------------------------------------------------------------------
# 4. Optimización TES (48 horas: carga de noche, descarga de día)
# ---------------------------------------------------------------------------
def test_tes_optimization_48h():
    n = 48
    # Demanda constante de 5 MWh-th/h
    demand = np.full(n, 5.0)
    # Ciclo de 24h: 8h baratas (10 €/MWh), 16h caras (120 €/MWh) repetido 2 veces
    daily_price = [10.0] * 8 + [120.0] * 16
    grid_prices = np.array(daily_price * 2)

    gas = GasBaselineInputs(gas_price_eur_per_mwh_hhv=40.0)

    inp = DispatchInputs(
        thermal_demand_mwh_th=demand,
        grid_unit_cost_eur_per_mwh_e=grid_prices,
        pv_generation_mwh_e=np.zeros(n),
        grid_capacity_available_mw=np.full(n, 50.0),
        gas=gas,
        minimum_electrification_pct=0.0,
    )

    params = TesInputs(
        tes_charge_capacity_mw_e=20.0,
        tes_storage_capacity_mwh_th=100.0,
        tes_discharge_capacity_mw_th=10.0,
        tes_charge_efficiency=0.95,
        tes_discharge_efficiency=0.95,
        tes_hourly_self_discharge_pct=0.1,
        tes_initial_soc_mwh_th=0.0,
        tes_final_soc_mode="equal_initial",
    )

    res = solve_tes(inp, params)
    assert res.is_valid
    dec = res.decisions

    # 1. Conservación de energía térmica entregada + fósil = demanda
    delivered = dec["tes_heat_delivered_mwh_th"] + dec["fossil_heat_output_mwh_th"]
    np.testing.assert_allclose(delivered, demand, atol=1e-5)

    # 2. SOC dentro de límites [0, 100]
    assert (dec["tes_soc_end_mwh_th"] >= -1e-6).all()
    assert (dec["tes_soc_end_mwh_th"] <= 100.0 + 1e-6).all()

    # 3. Continuidad de SOC
    for t in range(1, n):
        soc_prev = dec["tes_soc_end_mwh_th"].iloc[t - 1]
        soc_start = dec["tes_soc_start_mwh_th"].iloc[t]
        assert pytest.approx(soc_start, 1e-5) == soc_prev

    # 4. El TES debe haber realizado ciclos
    assert dec["tes_total_charge_mwh_e"].sum() > 0
    assert dec["tes_heat_delivered_mwh_th"].sum() > 0


# ---------------------------------------------------------------------------
# 5. Evaluación Financiera
# ---------------------------------------------------------------------------
def test_financial_indicators():
    # Anualidad financiera: 100.000 € a 10 años al 5%
    a = annualized_capex(100_000.0, 5.0, 10)
    assert pytest.approx(a, 1.0) == 12950.46

    # Tasa cero: 100.000 / 10 = 10.000
    a0 = annualized_capex(100_000.0, 0.0, 10)
    assert a0 == 10_000.0

    # TIR de un flujo: -100, 30, 40, 50
    irr = compute_irr([-100.0, 30.0, 40.0, 50.0])
    assert irr is not None
    assert pytest.approx(irr, 0.5) == 8.9  # ~8.9%


def test_electrointensive_and_ltm_real():
    from data.loaders import load_data_bundle
    from models.inputs import ScenarioConfig
    from supply.electricity_supply import build_electricity_supply
    from data.calendar import build_calendar

    bundle = load_data_bundle(use_cache=True)
    assert bundle.network_tariffs_electrointensive is not None
    assert "REAL" in bundle.pool.scenarios
    assert "LTM" in bundle.pool.years

    # 1. Verificar peajes electrointensivos frente a estándar
    cfg = ScenarioConfig()
    cfg.supply.network_tariff = "6.1TD"
    cfg.supply.pool_missing_data_fill_authorized = True
    cal = build_calendar(2027)

    # Estándar
    cfg.supply.is_electrointensive = False
    supply_std, _ = build_electricity_supply(cfg.supply, cal, bundle)

    # Electrointensivo
    cfg.supply.is_electrointensive = True
    supply_electro, notes_electro = build_electricity_supply(cfg.supply, cal, bundle)

    # Peajes electrointensivos deben ser inferiores a los estándar
    assert supply_electro["network_tariff_eur_per_mwh_e"].mean() < supply_std["network_tariff_eur_per_mwh_e"].mean()
    assert any("electrointensivo" in n.lower() for n in notes_electro)

    # 2. Verificar escenario REAL (LTM)
    cfg.supply.pool_price_scenario = "REAL"
    supply_real, notes_real = build_electricity_supply(cfg.supply, cal, bundle)
    assert len(supply_real) == len(cal)
    assert any("LTM" in n for n in notes_real)

