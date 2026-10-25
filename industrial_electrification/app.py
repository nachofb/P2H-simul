"""Interfaz de Usuario en Streamlit - Simulador de Electrificación y Descarbonización Industrial."""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Asegurar path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import streamlit as st

import config
from data.loaders import load_data_bundle, read_hourly_csv
from models.inputs import (
    DemandInputs,
    EboilerInputs,
    ElectricitySupplyInputs,
    ElectrificationTargetInputs,
    FinancialInputs,
    GasBaselineInputs,
    InnovationFundInputs,
    ScenarioConfig,
    SimulationSettings,
    TesInputs,
)
from optimization.sizing import (
    optimize_eboiler_sizing,
    optimize_tes_sizing,
    run_sensitivity_analysis,
)
import importlib
import reporting.dashboards
importlib.reload(reporting.dashboards)
from reporting.dashboards import (
    plot_cash_flow_evolution,
    plot_electricity_dispatch,
    plot_electricity_price_stats,
    plot_electrification_cost_curve,
    plot_lcoh_comparison,
    plot_tes_soc_profile,
    plot_thermal_dispatch,
)
from reporting.exports import export_hourly_results_to_csv, export_scenario_to_excel
from results.comparisons import (
    compare_multiple_scenarios,
    compare_unconstrained_vs_constrained,
    compute_electrification_cost_curve,
    run_simulation,
)
from supply.electricity_supply import compute_electricity_price_stats

# Configuración de página
st.set_page_config(
    page_title="Simulador de Electrificación Industrial",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inicializar sesión
if "bundle" not in st.session_state:
    try:
        st.session_state.bundle = load_data_bundle(use_cache=True)
    except Exception as e:
        st.error(f"Error cargando archivos base: {e}")
        st.session_state.bundle = None

if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = ScenarioConfig()

if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = {}

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Barra lateral: Gestión de Casos y Configuración
with st.sidebar:
    st.title("⚡ Descarbonización P2H")
    st.markdown("Simulador y Optimizador Técnico-Económico de Calor Industrial")

    st.header("1. Alternativa de Proyecto")
    alt = st.radio(
        "Tecnología a evaluar:",
        options=["eboiler", "tes"],
        format_func=lambda x: "Alternativa 1 — E-Boiler" if x == "eboiler" else "Alternativa 2 — TES (Almacenamiento)",
    )
    st.session_state.current_scenario.alternative = alt

    st.subheader("Gestión de Casos")
    case_name = st.text_input("Nombre del caso:", value=st.session_state.current_scenario.case_name)
    st.session_state.current_scenario.case_name = case_name

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("💾 Guardar Caso"):
            st.session_state.saved_scenarios[case_name] = st.session_state.current_scenario.model_dump_json()
            st.success(f"Caso '{case_name}' guardado.")
    with col_s2:
        if st.session_state.saved_scenarios:
            sel_case = st.selectbox("Cargar:", options=list(st.session_state.saved_scenarios.keys()))
            if st.button("📂 Cargar"):
                st.session_state.current_scenario = ScenarioConfig.from_json(st.session_state.saved_scenarios[sel_case])
                st.rerun()

    # Descarga/Carga JSON
    cfg_json = st.session_state.current_scenario.to_json()
    st.download_button(
        "📥 Exportar Caso (JSON)",
        data=cfg_json,
        file_name=f"{case_name}.json",
        mime="application/json",
    )

    uploaded_cfg = st.file_uploader("Subir configuración (JSON):", type=["json"])
    if uploaded_cfg is not None:
        try:
            content = uploaded_cfg.read().decode("utf-8")
            st.session_state.current_scenario = ScenarioConfig.from_json(content)
            st.success("Configuración cargada correctamente.")
        except Exception as exc:
            st.error(f"Error importando JSON: {exc}")

# Cuerpo Principal
st.title("⚡ Simulador de Electrificación y Descarbonización Industrial")
st.markdown(
    """
    Herramienta integral de simulación, optimización matemática (Programación Lineal Pyomo/HiGHS) 
    y análisis financiero para proyectos de electrificación del calor industrial.
    """
)

# Tabs de navegación
tab_inputs, tab_results, tab_sensitivities, tab_sizing, tab_comparison = st.tabs(
    [
        "📝 Configuración de Inputs",
        "📊 Resultados de Simulación",
        "📈 Curva de Electrificación y Sensibilidad",
        "📐 Dimensionamiento Óptimo",
        "⚖️ Comparativa de Casos",
    ]
)

# ===========================================================================
# TAB 1: INPUTS
# ===========================================================================
with tab_inputs:
    cfg = st.session_state.current_scenario

    with st.expander("🕒 1. Caracterización Temporal y Financiera General", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cfg.simulation.simulation_start_year = st.number_input(
                "Año inicial simulación:", min_value=2026, max_value=2060, value=cfg.simulation.simulation_start_year
            )
            cfg.simulation.simulation_years = st.number_input(
                "Años a simular:", min_value=1, max_value=30, value=cfg.simulation.simulation_years
            )
        with col2:
            cfg.simulation.asset_lifetime_years = st.number_input(
                "Vida útil del activo (años):", min_value=1, max_value=50, value=cfg.simulation.asset_lifetime_years
            )
            cfg.simulation.amortization_years = st.number_input(
                "Plazo amortización préstamo (años):", min_value=1, max_value=30, value=cfg.simulation.amortization_years
            )
        with col3:
            cfg.simulation.discount_rate_pct = st.number_input(
                "Tasa de descuento (%):", min_value=0.0, max_value=30.0, value=cfg.simulation.discount_rate_pct, step=0.5
            )
            cfg.simulation.interest_rate_pct = st.number_input(
                "Tipo de interés préstamo (%):", min_value=0.0, max_value=30.0, value=cfg.simulation.interest_rate_pct, step=0.5
            )
        with col4:
            st.info("Paso temporal: 1 hora estándar. Calendario normalizado peninsular con años bisiestos.")

    with st.expander("🔥 2. Demanda Térmica Industrial", expanded=True):
        d_method = st.selectbox(
            "Método de definición de la demanda térmica:",
            options=["constant", "weekday_weekend", "hourly_curve"],
            format_func=lambda x: {
                "constant": "Demanda constante anual",
                "weekday_weekend": "Diferenciada laborable vs fin de semana",
                "hourly_curve": "Importar curva horaria (CSV)",
            }[x],
        )
        cfg.demand.method = d_method

        if d_method == "constant":
            cfg.demand.thermal_demand_constant_mw_th = st.number_input(
                "Potencia térmica continua (MW-th):", min_value=0.1, value=cfg.demand.thermal_demand_constant_mw_th, step=1.0
            )
        elif d_method == "weekday_weekend":
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                cfg.demand.weekday_thermal_demand_mw_th = st.number_input(
                    "Demanda laborable (MW-th):", min_value=0.1, value=cfg.demand.weekday_thermal_demand_mw_th
                )
            with col_d2:
                cfg.demand.weekend_thermal_demand_mw_th = st.number_input(
                    "Demanda fin de semana (MW-th):", min_value=0.0, value=cfg.demand.weekend_thermal_demand_mw_th
                )
        else:
            up_demand = st.file_uploader("Cargar CSV de demanda (8760 u 8784 filas):", type=["csv"])
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                cfg.demand.curve_kind = st.selectbox(
                    "Los datos representan:",
                    options=["power_mw_th", "energy_mwh_th"],
                    format_func=lambda x: "Potencia media (MW-th)" if x == "power_mw_th" else "Energía consumida (MWh-th)",
                )
            if up_demand is not None:
                try:
                    cdata = read_hourly_csv(up_demand, name="Demanda térmica subida")
                    cfg.demand.curve_values = cdata.values.tolist()
                    st.success(f"Curva cargada: {cdata.n_records} registros leídos.")
                except Exception as exc:
                    st.error(f"Error leyendo archivo de demanda: {exc}")

    with st.expander("💨 3. Escenario Fósil de Referencia (Gas Natural)", expanded=False):
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            cfg.baseline.gas_price_eur_per_mwh_hhv = st.number_input(
                "Precio gas natural PCS (EUR/MWh-HHV):", min_value=0.0, value=cfg.baseline.gas_price_eur_per_mwh_hhv, step=1.0
            )
            cfg.baseline.co2_price_eur_per_tco2 = st.number_input(
                "Precio derechos CO2 (EUR/tCO2):", min_value=0.0, value=cfg.baseline.co2_price_eur_per_tco2, step=5.0
            )
        with col_g2:
            cfg.baseline.gas_other_cost_eur_per_mwh_hhv = st.number_input(
                "Otros costes gas PCS (EUR/MWh-HHV):", min_value=0.0, value=cfg.baseline.gas_other_cost_eur_per_mwh_hhv, step=0.5
            )
            cfg.baseline.fossil_boiler_efficiency_lhv = st.number_input(
                "Rendimiento caldera gas PCI:", min_value=0.5, max_value=1.1, value=cfg.baseline.fossil_boiler_efficiency_lhv, step=0.01
            )
        with col_g3:
            cfg.baseline.lhv_hhv_ratio = st.number_input(
                "Relación PCI / PCS gas:", min_value=0.8, max_value=1.0, value=cfg.baseline.lhv_hhv_ratio, step=0.01
            )
            cfg.baseline.gas_emission_factor_tco2_per_mwh_hhv = st.number_input(
                "Factor emisión (tCO2/MWh-HHV):", min_value=0.1, max_value=0.3, value=cfg.baseline.gas_emission_factor_tco2_per_mwh_hhv, step=0.005
            )

    with st.expander("⚡ 4. Suministro Eléctrico (Mercado y PV Onsite)", expanded=False):
        st.markdown("##### Precios del Mercado Eléctrico")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            scenarios = st.session_state.bundle.pool.scenarios if st.session_state.bundle else ["AURORA CENTRAL", "REAL"]
            default_scen = cfg.supply.pool_price_scenario if cfg.supply.pool_price_scenario in scenarios else ("AURORA CENTRAL" if "AURORA CENTRAL" in scenarios else scenarios[0])
            cfg.supply.pool_price_scenario = st.selectbox(
                "Escenario de precios Pool:",
                options=scenarios,
                index=scenarios.index(default_scen),
                format_func=lambda s: f"{s} (Últimos 12 meses)" if s == "REAL" else s,
            )
            if cfg.supply.pool_price_scenario == "REAL":
                st.caption("ℹ️ Curva real de precios de los últimos 12 meses (LTM).")
            cfg.supply.pool_price_scaling_factor = st.number_input(
                "Multiplicador de sensibilidad pool (1.0 = normal):", min_value=0.0, value=cfg.supply.pool_price_scaling_factor, step=0.1
            )
        with col_e2:
            cfg.supply.network_tariff = st.selectbox("Tarifa de acceso:", options=["6.1TD", "6.2TD", "6.3TD", "6.4TD"], index=0)
            cfg.supply.is_electrointensive = st.checkbox(
                "Consumidor electrointensivo",
                value=cfg.supply.is_electrointensive,
                help="Aplica los peajes y cargos reducidos para consumidores electrointensivos (según network_tariffs_electrointensive.xlsx).",
            )
            if cfg.supply.is_electrointensive:
                st.caption("ℹ️ Aplicando peajes y cargos reducidos para consumidor electrointensivo.")
            cfg.supply.adjustment_price_mode = st.radio(
                "Servicios de ajuste:", options=["hourly_series", "fixed"], format_func=lambda x: "Serie horaria archivo" if x == "hourly_series" else "Valor fijo anual"
            )
            if cfg.supply.adjustment_price_mode == "fixed":
                cfg.supply.adjustment_fixed_eur_per_mwh_e = st.number_input("Valor fijo ajustes (EUR/MWh-e):", value=cfg.supply.adjustment_fixed_eur_per_mwh_e)
        with col_e3:
            cfg.supply.grid_other_cost_eur_per_mwh_e = st.number_input("Otros costes red / fee (EUR/MWh-e):", value=cfg.supply.grid_other_cost_eur_per_mwh_e)
            cfg.supply.guarantees_of_origin_enabled = st.checkbox("Garantías de Origen (GdO)", value=cfg.supply.guarantees_of_origin_enabled)
            if cfg.supply.guarantees_of_origin_enabled:
                cfg.supply.guarantees_of_origin_cost_eur_per_mwh_e = st.number_input("Coste GdO (EUR/MWh-e):", value=cfg.supply.guarantees_of_origin_cost_eur_per_mwh_e)

        cfg.supply.pool_missing_data_fill_authorized = st.checkbox(
            "Autorizar completado de horas finales en series de pool (mismo valor del día anterior)",
            value=cfg.supply.pool_missing_data_fill_authorized,
        )

        st.markdown("##### Planta Fotovoltaica Onsite")
        cfg.supply.has_onsite_pv = st.checkbox("¿Dispone de planta PV onsite dedicada?", value=cfg.supply.has_onsite_pv)
        if cfg.supply.has_onsite_pv:
            col_pv1, col_pv2, col_pv3 = st.columns(3)
            with col_pv1:
                cfg.supply.pv_installed_capacity_mwp = st.number_input("Potencia PV (MWp):", min_value=0.1, value=cfg.supply.pv_installed_capacity_mwp or 5.0)
                cfg.supply.pv_price_eur_per_mwh_e = st.number_input("Coste de la energía PV (EUR/MWh-e):", value=cfg.supply.pv_price_eur_per_mwh_e)
            with col_pv2:
                up_pv = st.file_uploader("Curva solar horaria P50 (CSV):", type=["csv"])
                cfg.supply.pv_profile_kind = st.selectbox(
                    "Tipo de curva solar:", options=["specific_mwh_e_per_mwp", "absolute_mwh_e"], format_func=lambda x: "Específica (MWh/MWp)" if x == "specific_mwh_e_per_mwp" else "Energía absoluta (MWh)"
                )
                if up_pv is not None:
                    try:
                        pvdata = read_hourly_csv(up_pv, name="Perfil PV")
                        cfg.supply.pv_profile_values = pvdata.values.tolist()
                        st.success(f"Perfil PV cargado ({pvdata.n_records} registros).")
                    except Exception as e:
                        st.error(f"Error cargando PV: {e}")
                elif not cfg.supply.pv_profile_values:
                    # Cargar por defecto el sintético si existe
                    try:
                        default_pv = root_dir / "examples" / "pv_p50_synthetic.csv"
                        if default_pv.exists():
                            pvdata = read_hourly_csv(default_pv, name="PV por defecto")
                            cfg.supply.pv_profile_values = pvdata.values.tolist()
                    except Exception:
                        pass
            with col_pv3:
                cfg.supply.pv_priority_enabled = st.checkbox("Activar prioridad al uso de PV onsite", value=cfg.supply.pv_priority_enabled)
                if cfg.supply.pv_priority_enabled:
                    cfg.supply.pv_priority_cost_tolerance_pct = st.number_input("Tolerancia de coste en prioridad (%):", value=cfg.supply.pv_priority_cost_tolerance_pct)

        st.markdown("##### Restricción de Capacidad de Red")
        cfg.supply.grid_capacity_limit_enabled = st.checkbox("Limitar potencia máxima disponible de red", value=cfg.supply.grid_capacity_limit_enabled)
        if cfg.supply.grid_capacity_limit_enabled:
            cfg.supply.grid_capacity_by_period_enabled = st.checkbox("Capacidad diferenciada por periodos P1-P6", value=cfg.supply.grid_capacity_by_period_enabled)
            if not cfg.supply.grid_capacity_by_period_enabled:
                cfg.supply.grid_capacity_mw = st.number_input("Potencia disponible de red (MW):", min_value=0.1, value=cfg.supply.grid_capacity_mw or 10.0)
            else:
                p_cols = st.columns(6)
                cfg.supply.grid_capacity_p1_mw = p_cols[0].number_input("P1 (MW):", value=cfg.supply.grid_capacity_p1_mw or 5.0)
                cfg.supply.grid_capacity_p2_mw = p_cols[1].number_input("P2 (MW):", value=cfg.supply.grid_capacity_p2_mw or 5.0)
                cfg.supply.grid_capacity_p3_mw = p_cols[2].number_input("P3 (MW):", value=cfg.supply.grid_capacity_p3_mw or 10.0)
                cfg.supply.grid_capacity_p4_mw = p_cols[3].number_input("P4 (MW):", value=cfg.supply.grid_capacity_p4_mw or 10.0)
                cfg.supply.grid_capacity_p5_mw = p_cols[4].number_input("P5 (MW):", value=cfg.supply.grid_capacity_p5_mw or 15.0)
                cfg.supply.grid_capacity_p6_mw = p_cols[5].number_input("P6 (MW):", value=cfg.supply.grid_capacity_p6_mw or 15.0)

    with st.expander("⚙️ 5. Parámetros Técnicos de la Alternativa Seleccionada", expanded=True):
        if cfg.alternative == "eboiler":
            st.markdown("#### Alternativa 1 — Caldera Eléctrica (E-Boiler)")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                cfg.eboiler.eboiler_capacity_mw_e = st.number_input("Potencia nominal E-boiler (MW-e):", min_value=0.1, value=cfg.eboiler.eboiler_capacity_mw_e, step=1.0)
                cfg.eboiler.eboiler_efficiency = st.number_input("Rendimiento eléctrico E-boiler:", min_value=0.8, max_value=1.0, value=cfg.eboiler.eboiler_efficiency, step=0.01)
            with col_b2:
                cfg.eboiler.eboiler_capex_eur = st.number_input("Inversión inicial CAPEX (EUR):", min_value=0.0, value=cfg.eboiler.eboiler_capex_eur, step=100000.0)
                cfg.eboiler.eboiler_annual_opex_eur = st.number_input("OPEX fijo anual (EUR/año):", min_value=0.0, value=cfg.eboiler.eboiler_annual_opex_eur, step=5000.0)
        else:
            st.markdown("#### Alternativa 2 — Almacenamiento Térmico TES")
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                cfg.tes.tes_charge_capacity_mw_e = st.number_input("Potencia de carga (MW-e):", min_value=0.1, value=cfg.tes.tes_charge_capacity_mw_e, step=1.0)
                cfg.tes.tes_storage_capacity_mwh_th = st.number_input("Capacidad almacenamiento (MWh-th):", min_value=1.0, value=cfg.tes.tes_storage_capacity_mwh_th, step=10.0)
                cfg.tes.tes_discharge_capacity_mw_th = st.number_input("Potencia de descarga (MW-th):", min_value=0.1, value=cfg.tes.tes_discharge_capacity_mw_th, step=1.0)
            with col_t2:
                cfg.tes.tes_charge_efficiency = st.number_input("Rendimiento de carga:", min_value=0.7, max_value=1.0, value=cfg.tes.tes_charge_efficiency, step=0.01)
                cfg.tes.tes_discharge_efficiency = st.number_input("Rendimiento de descarga:", min_value=0.7, max_value=1.0, value=cfg.tes.tes_discharge_efficiency, step=0.01)
                cfg.tes.tes_hourly_self_discharge_pct = st.number_input("Pérdidas de almacenamiento (%/hora):", min_value=0.0, max_value=5.0, value=cfg.tes.tes_hourly_self_discharge_pct, step=0.01)
            with col_t3:
                cfg.tes.tes_capex_eur = st.number_input("Inversión inicial CAPEX (EUR):", min_value=0.0, value=cfg.tes.tes_capex_eur, step=500000.0)
                cfg.tes.tes_annual_opex_eur = st.number_input("OPEX fijo anual (EUR/año):", min_value=0.0, value=cfg.tes.tes_annual_opex_eur, step=10000.0)
                cfg.tes.tes_enforce_no_simultaneous_charge_discharge = st.checkbox("Impedir carga y descarga simultánea (MILP)", value=cfg.tes.tes_enforce_no_simultaneous_charge_discharge)

    with st.expander("🎯 6. Restricción de Electrificación Térmica Mínima Obligatoria", expanded=True):
        cfg.target.minimum_electrification_enabled = st.checkbox(
            "¿Deseas establecer una electrificación térmica mínima obligatoria?",
            value=cfg.target.minimum_electrification_enabled,
        )
        if cfg.target.minimum_electrification_enabled:
            cfg.target.minimum_electrification_pct = st.slider(
                "¿Qué porcentaje mínimo de la demanda térmica anual debe cubrir el activo electrificado? (%)",
                min_value=0.0,
                max_value=100.0,
                value=cfg.target.minimum_electrification_pct,
                step=1.0,
            )
            cfg.target.compute_max_on_infeasible = st.checkbox(
                "En caso de infactibilidad, calcular automáticamente el porcentaje máximo técnicamente alcanzable",
                value=cfg.target.compute_max_on_infeasible,
            )

    with st.expander("💶 7. Incentivos Innovation Fund", expanded=False):
        cfg.innovation_fund.innovation_fund_enabled = st.checkbox(
            "Activar incentivo Innovation Fund (subasta competitiva)", value=cfg.innovation_fund.innovation_fund_enabled
        )
        if cfg.innovation_fund.innovation_fund_enabled:
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                cfg.innovation_fund.innovation_fund_bid_eur_per_tco2 = st.number_input(
                    "Puja del proyecto (EUR/tCO2 evitada):", min_value=0.0, value=cfg.innovation_fund.innovation_fund_bid_eur_per_tco2, step=5.0
                )
            with col_i2:
                cfg.innovation_fund.innovation_fund_reference_tco2_per_mwh_th = st.number_input(
                    "Factor de referencia convocatoria (tCO2/MWh-th):", value=cfg.innovation_fund.innovation_fund_reference_tco2_per_mwh_th
                )
            with col_i3:
                cfg.innovation_fund.innovation_fund_duration_years = st.number_input(
                    "Duración del incentivo (años):", min_value=1, max_value=20, value=cfg.innovation_fund.innovation_fund_duration_years
                )
            prima = cfg.innovation_fund.innovation_fund_premium_eur_per_mwh_th
            st.info(f"Prima resultante equivalente: **{prima:.2f} EUR/MWh-th** de calor útil entregado.")

    st.markdown("---")
    btn_run = st.button("🚀 EJECUTAR SIMULACIÓN Y OPTIMIZACIÓN", type="primary", use_container_width=True)
    if btn_run:
        with st.spinner("Ejecutando optimización matemática con HiGHS y pyomo..."):
            res = run_simulation(cfg, st.session_state.bundle)
            st.session_state.last_result = res
            if res.is_valid:
                st.success("✅ Simulación completada con éxito.")
            else:
                st.error("❌ La simulación resultó infactible o con errores. Revisa los mensajes.")

# ===========================================================================
# TAB 2: RESULTADOS
# ===========================================================================
with tab_results:
    res = st.session_state.last_result
    if res is None:
        st.info("Ejecuta una simulación en la pestaña de configuración para ver los resultados.")
    elif not res.is_valid:
        st.error("### ⚠️ Problema INFACTIBLE")
        st.markdown(
            """
            No existe ningún despacho técnico que pueda satisfacer todas las restricciones impuestas 
            (balance térmico, capacidades máximas, límites de red y porcentaje mínimo de electrificación).
            """
        )
        diag = res.first_year.dispatch.infeasibility if res.first_year else None
        if diag:
            st.warning(f"**Objetivo solicitado:** {diag.requested_minimum_pct:.1f}%")
            if diag.max_achievable_pct is not None:
                st.info(f"**Porcentaje máximo técnicamente alcanzable:** {diag.max_achievable_pct:.2f}% ({diag.max_achievable_heat_mwh_th:,.1f} MWh-th)")
            if diag.limiting_factors:
                st.markdown("**Factores limitantes detectados:**")
                for f in diag.limiting_factors:
                    st.write(f"- {f}")
            if diag.binding_hours:
                st.json(diag.binding_hours)
    else:
        y1 = res.first_year.annual
        fin = res.financial.indicators if res.financial else {}

        st.subheader(f"📊 Dashboard Ejecutivo — {res.case_name} ({res.alternative.upper()})")

        # Tarjetas Métricas Principales
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        m_c1.metric("Electrificación Real", f"{y1['actual_electrification_pct']:.1f}%", delta=f"{y1['electrification_margin_pct_points']:+.1f}% vs mín.")
        m_c2.metric("Ahorro Operativo Neto", f"{y1['annual_net_operating_savings_eur']:,.0f} €/año")
        m_c3.metric("Payback Simple", f"{fin.get('simple_payback_years', 0):.1f} años" if fin.get('simple_payback_years') else "N/A")
        m_c4.metric("VAN Proyecto (NPV)", f"{fin.get('npv_eur', 0):,.0f} €")
        m_c5.metric("Reducción Emisiones CO2", f"{y1['direct_co2_reduction_pct']:.1f}%", f"-{y1['avoided_direct_co2_emissions_tco2']:,.0f} tCO2")

        st.markdown("---")

        # Gráficos horarios interactivos
        st.subheader("Despacho Operativo Horario")
        c_sel1, c_sel2 = st.columns([1, 3])
        with c_sel1:
            day_offset = st.number_input("Día inicial del año a visualizar (1-365):", min_value=1, max_value=360, value=15)
            hours_view = st.slider("Ventana de horas:", min_value=24, max_value=720, value=168, step=24)
            start_hour = (day_offset - 1) * 24

        st.plotly_chart(plot_thermal_dispatch(res.first_year, start_idx=start_hour, num_hours=hours_view), use_container_width=True)
        st.plotly_chart(plot_electricity_dispatch(res.first_year, start_idx=start_hour, num_hours=hours_view), use_container_width=True)

        if res.alternative == "tes":
            st.plotly_chart(plot_tes_soc_profile(res.first_year, start_idx=start_hour, num_hours=hours_view), use_container_width=True)

        st.plotly_chart(plot_cash_flow_evolution(res), use_container_width=True)

        # Tablas de detalle
        st.subheader("Desglose Económico y Energético Anual")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("##### Balance Energético")
            df_energy = pd.DataFrame(
                [
                    {"Concepto": "Demanda térmica total", "Valor": f"{y1['annual_thermal_demand_mwh_th']:,.1f} MWh-th"},
                    {"Concepto": "Calor electrificado útil", "Valor": f"{y1['annual_electrified_heat_mwh_th']:,.1f} MWh-th"},
                    {"Concepto": "Calor residual gas fósil", "Valor": f"{y1['annual_fossil_heat_mwh_th']:,.1f} MWh-th"},
                    {"Concepto": "Consumo electricidad de red", "Valor": f"{y1['annual_grid_consumption_mwh_e']:,.1f} MWh-e"},
                    {"Concepto": "Consumo electricidad PV", "Valor": f"{y1['annual_pv_consumption_mwh_e']:,.1f} MWh-e"},
                    {"Concepto": "Excedentes PV no aprovechados", "Valor": f"{y1['annual_pv_surplus_mwh_e']:,.1f} MWh-e"},
                    {"Concepto": "Consumo de gas PCS", "Valor": f"{y1['annual_gas_consumption_mwh_hhv']:,.1f} MWh-HHV"},
                ]
            )
            st.dataframe(df_energy, hide_index=True, use_container_width=True)

        with col_t2:
            st.markdown("##### Balance Económico")
            elec_cons_total_mwh = y1.get("annual_grid_consumption_mwh_e", 0.0) + y1.get("annual_pv_consumption_mwh_e", 0.0)
            elec_cost_total_eur = y1.get("annual_total_electricity_cost_eur", 0.0)
            captured_price = (
                y1.get("annual_captured_electricity_price_eur_per_mwh_e")
                if "annual_captured_electricity_price_eur_per_mwh_e" in y1
                else ((elec_cost_total_eur / elec_cons_total_mwh) if elec_cons_total_mwh > 0 else 0.0)
            )
            df_costs = pd.DataFrame(
                [
                    {"Partida": "Coste línea base (Gas)", "Importe": f"{y1['annual_baseline_cost_eur']:,.0f} €"},
                    {"Partida": "Gasto electricidad (Red + PV)", "Importe": f"{y1['annual_total_electricity_cost_eur']:,.0f} €"},
                    {"Partida": "Precio eléctrico medio capturado", "Importe": f"{captured_price:.2f} €/MWh-e"},
                    {"Partida": "Gasto gas fósil remanente", "Importe": f"{y1['annual_fossil_total_cost_eur']:,.0f} €"},
                    {"Partida": "Ingresos Innovation Fund", "Importe": f"-{y1['annual_innovation_fund_revenue_eur']:,.0f} €"},
                    {"Partida": "Coste operativo anual proyecto", "Importe": f"{y1['annual_operating_cost_after_incentives_eur']:,.0f} €"},
                    {"Partida": "Ahorro operativo neto", "Importe": f"{y1['annual_net_operating_savings_eur']:,.0f} €"},
                ]
            )
            st.dataframe(df_costs, hide_index=True, use_container_width=True)

        # Desglose Específico y Detallado de LCOH y Coste del Calor
        st.subheader("🔥 Desglose Específico de LCOH y Coste del Calor (EUR/MWh-th)")
        st.markdown(
            """
            Comparativa detallada entre la **Situación de Referencia (Gas)** y la **Solución Futura Electrificada**.
            Incluye la discretización nivelada plurianual (**LCOH**) y la del **Año 1**.
            """
        )

        ind = fin
        lcoh_p = ind.get("lcoh_breakdown_project", {})
        lcoh_b = ind.get("lcoh_breakdown_baseline", {})
        y1_p = ind.get("heat_cost_breakdown_year1_project", {})
        y1_b = ind.get("heat_cost_breakdown_year1_baseline", {})

        st.plotly_chart(
            plot_lcoh_comparison(
                lcoh_b,
                lcoh_p,
                breakdown_proj_year1=y1_p,
                title="Comparativa de LCOH Descontado y Coste Año 1 por Componentes (EUR/MWh-th)",
            ),
            use_container_width=True,
        )

        concepts_display = [
            ("capex", "1. CAPEX e Inversión / Reposiciones"),
            ("opex_tech", "2. OPEX Técnico y Mantenimiento"),
            ("ppa_onsite", "3. Electricidad PPA Onsite (Fotovoltaica)"),
            ("pool", "4. Electricidad de Red — Mercado Diario (Pool)"),
            ("adjustments", "5. Electricidad de Red — Servicios de Ajuste"),
            ("network_tariffs", "6. Electricidad de Red — Peajes y Cargos (6.xTD)"),
            ("guarantees_of_origin", "7. Electricidad de Red — Garantías de Origen (GdO)"),
            ("other_grid", "8. Electricidad de Red — Otros Costes / Fee"),
            ("fossil_gas", "9. Caldera Gas Fósil — Combustible"),
            ("fossil_co2", "10. Caldera Gas Fósil — Derechos de Emisión CO2"),
            ("fossil_other", "11. Caldera Gas Fósil — Otros Costes"),
            ("innovation_fund", "12. Incentivo Innovation Fund"),
        ]

        table_lcoh = []
        for k, label in concepts_display:
            val_base_lcoh = lcoh_b.get(k, 0.0)
            val_proj_lcoh = lcoh_p.get(k, 0.0)
            diff_lcoh = val_proj_lcoh - val_base_lcoh

            val_base_y1 = y1_b.get(k, 0.0)
            val_proj_y1 = y1_p.get(k, 0.0)
            diff_y1 = val_proj_y1 - val_base_y1

            table_lcoh.append(
                {
                    "Componente de Coste": label,
                    "LCOH Referencia": f"{val_base_lcoh:.2f} €",
                    "LCOH Proyecto": f"{val_proj_lcoh:.2f} €",
                    "Diferencia LCOH": f"{diff_lcoh:+.2f} €",
                    "Coste Año 1 Referencia": f"{val_base_y1:.2f} €",
                    "Coste Año 1 Proyecto": f"{val_proj_y1:.2f} €",
                    "Diferencia Año 1": f"{diff_y1:+.2f} €",
                }
            )

        # Totales netos
        tot_base_lcoh = lcoh_b.get("net_lcoh", 0.0)
        tot_proj_lcoh = lcoh_p.get("net_lcoh", 0.0)
        tot_diff_lcoh = tot_proj_lcoh - tot_base_lcoh

        tot_base_y1 = y1_b.get("net_heat_cost", 0.0)
        tot_proj_y1 = y1_p.get("net_heat_cost", 0.0)
        tot_diff_y1 = tot_proj_y1 - tot_base_y1

        table_lcoh.append(
            {
                "Componente de Coste": "TOTAL NETO POR MWh-th",
                "LCOH Referencia": f"{tot_base_lcoh:.2f} €",
                "LCOH Proyecto": f"{tot_proj_lcoh:.2f} €",
                "Diferencia LCOH": f"{tot_diff_lcoh:+.2f} €",
                "Coste Año 1 Referencia": f"{tot_base_y1:.2f} €",
                "Coste Año 1 Proyecto": f"{tot_proj_y1:.2f} €",
                "Diferencia Año 1": f"{tot_diff_y1:+.2f} €",
            }
        )

        st.dataframe(pd.DataFrame(table_lcoh), hide_index=True, use_container_width=True)

        # Estadísticas del Escenario de Precios Eléctricos (Pool vs. Precio Completo de Red)
        st.markdown("---")
        st.subheader("⚡ Estadísticas del Escenario de Precios Eléctricos (EUR/MWh-e)")
        try:
            sim_cfg = ScenarioConfig.from_json(res.config_json) if res.config_json else st.session_state.current_scenario
        except Exception:
            sim_cfg = st.session_state.current_scenario

        sup_cfg = sim_cfg.supply
        electro_tag = " (Electrointensivo)" if sup_cfg.is_electrointensive else " (Estándar)"
        adj_tag = (
            f"Serie horaria"
            if sup_cfg.adjustment_price_mode == "hourly_series"
            else f"Fijo ({sup_cfg.adjustment_fixed_eur_per_mwh_e:.2f} €/MWh-e)"
        )
        goo_val = sup_cfg.guarantees_of_origin_cost_eur_per_mwh_e if sup_cfg.guarantees_of_origin_enabled else 0.0

        st.markdown(
            f"**Escenario Pool:** `{sup_cfg.pool_price_scenario}` ( Factor x{sup_cfg.pool_price_scaling_factor:.2f} ) &nbsp;|&nbsp; "
            f"**Tarifa de Acceso:** `{sup_cfg.network_tariff}{electro_tag}` &nbsp;|&nbsp; "
            f"**Servicios de Ajuste:** `{adj_tag}` &nbsp;|&nbsp; "
            f"**Otros Costes Red:** `{sup_cfg.grid_other_cost_eur_per_mwh_e:.2f} €/MWh-e` &nbsp;|&nbsp; "
            f"**Garantías de Origen (GdO):** `{goo_val:.2f} €/MWh-e`"
        )
        st.caption(
            "El **Precio Completo de Red (All-in)** en cada hora se compone de: "
            "**Precio Pool + Peajes y Cargos (6.xTD) + Servicios de Ajuste + Otros Costes de Red + Garantías de Origen**."
        )

        if len(res.years) > 1:
            sel_yr_idx = st.selectbox(
                "Año simulado para estadísticas de precios eléctricos:",
                options=list(range(len(res.years))),
                format_func=lambda i: f"Año {i + 1} ({res.years[i].year})",
                key="price_stats_year_sel",
            )
            yr_for_prices = res.years[sel_yr_idx]
        else:
            yr_for_prices = res.first_year

        p_stats = getattr(yr_for_prices, "price_stats", None)
        if p_stats is None and yr_for_prices.hourly is not None and "pool_price_eur_per_mwh_e" in yr_for_prices.hourly.columns:
            p_stats = compute_electricity_price_stats(yr_for_prices.hourly, yr_for_prices.hourly["tariff_period"])

        if p_stats is not None and not p_stats.empty:
            ann_row = p_stats.iloc[0]
            n_hrs = int(ann_row["hours"])

            # Tarjetas resumen anual (8760 h)
            st.markdown(f"##### Resumen Anual General ({n_hrs:,} horas)")
            kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8 = st.columns(8)
            kp1.metric("Pool — Media", f"{ann_row['pool_mean_eur_per_mwh_e']:.2f} €")
            kp2.metric("Pool — P10", f"{ann_row['pool_p10_eur_per_mwh_e']:.2f} €")
            kp3.metric("Pool — P50", f"{ann_row['pool_p50_eur_per_mwh_e']:.2f} €")
            kp4.metric("Pool — P90", f"{ann_row['pool_p90_eur_per_mwh_e']:.2f} €")
            kp5.metric(
                "Precio Completo — Media",
                f"{ann_row['full_price_mean_eur_per_mwh_e']:.2f} €",
                delta=f"+{ann_row['full_price_mean_eur_per_mwh_e'] - ann_row['pool_mean_eur_per_mwh_e']:.2f} € vs Pool",
                delta_color="off",
            )
            kp6.metric("Precio Completo — P10", f"{ann_row['full_price_p10_eur_per_mwh_e']:.2f} €")
            kp7.metric("Precio Completo — P50", f"{ann_row['full_price_p50_eur_per_mwh_e']:.2f} €")
            kp8.metric("Precio Completo — P90", f"{ann_row['full_price_p90_eur_per_mwh_e']:.2f} €")

            # Dos tablas comparativas lado a lado: Precio Pool vs Precio Completo
            col_ps1, col_ps2 = st.columns(2)
            with col_ps1:
                st.markdown("##### 1. Precio Pool (Mercado Diario) — EUR/MWh-e")
                df_pool_view = pd.DataFrame(
                    {
                        "Periodo": p_stats["period"],
                        "Horas": p_stats["hours"],
                        "Average (Media)": p_stats["pool_mean_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                        "P10": p_stats["pool_p10_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                        "P50 (Mediana)": p_stats["pool_p50_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                        "P90": p_stats["pool_p90_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    }
                )
                st.dataframe(df_pool_view, hide_index=True, use_container_width=True)

            with col_ps2:
                st.markdown("##### 2. Precio Completo de Red (All-in) — EUR/MWh-e")
                df_full_view = pd.DataFrame(
                    {
                        "Periodo": p_stats["period"],
                        "Horas": p_stats["hours"],
                        "Average (Media)": p_stats["full_price_mean_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                        "P10": p_stats["full_price_p10_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                        "P50 (Mediana)": p_stats["full_price_p50_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                        "P90": p_stats["full_price_p90_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    }
                )
                st.dataframe(df_full_view, hide_index=True, use_container_width=True)

            # Tabla integral con todos los componentes de coste y estadísticos
            st.markdown(
                "##### 3. Desglose Completo por Periodo: Pool + Peajes y Cargos + Servicios de Ajuste + Otros Costes + GdO (EUR/MWh-e)"
            )
            df_breakdown_view = pd.DataFrame(
                {
                    "Periodo": p_stats["period"],
                    "Horas": p_stats["hours"],
                    "Pool Average": p_stats["pool_mean_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Pool P10": p_stats["pool_p10_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Pool P50": p_stats["pool_p50_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Pool P90": p_stats["pool_p90_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Peajes y Cargos": p_stats["network_tariff_mean_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Ajustes Average": p_stats["adjustment_mean_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Ajustes (P10 / P50 / P90)": [
                        f"{r['adjustment_p10_eur_per_mwh_e']:.2f} / {r['adjustment_p50_eur_per_mwh_e']:.2f} / {r['adjustment_p90_eur_per_mwh_e']:.2f} €"
                        for _, r in p_stats.iterrows()
                    ],
                    "Otros Costes Red": p_stats["grid_other_cost_mean_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Garantías Origen (GdO)": p_stats["guarantees_of_origin_mean_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Completo Average": p_stats["full_price_mean_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Completo P10": p_stats["full_price_p10_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Completo P50": p_stats["full_price_p50_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                    "Completo P90": p_stats["full_price_p90_eur_per_mwh_e"].map(lambda v: f"{v:.2f} €"),
                }
            )
            st.dataframe(df_breakdown_view, hide_index=True, use_container_width=True)

            st.plotly_chart(plot_electricity_price_stats(p_stats), use_container_width=True)

        # Descargas Excel y CSV
        st.subheader("📥 Exportación de Resultados")
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            excel_bytes = export_scenario_to_excel(res, st.session_state.current_scenario)
            st.download_button(
                "📊 Descargar Informe Completo en Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"electrificacion_{res.case_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with c_exp2:
            csv_str = export_hourly_results_to_csv(res)
            st.download_button(
                "📄 Descargar Series Horarias en CSV (.csv)",
                data=csv_str,
                file_name=f"despacho_horario_{res.case_name}.csv",
                mime="text/csv",
            )

# ===========================================================================
# TAB 3: CURVA DE COSTE Y SENSIBILIDAD
# ===========================================================================
with tab_sensitivities:
    st.header("📈 Curva de Coste de Electrificación y Análisis de Sensibilidad")

    st.subheader("1. Curva de Coste Incremental vs % Mínimo de Electrificación")
    st.markdown("Calcula el despacho para objetivos mínimos del 0% al 100% (pasos de 10%).")
    if st.button("Generar Curva de Coste de Electrificación"):
        with st.spinner("Optimizando escenarios consecutivos del 0% al 100%..."):
            curve_df = compute_electrification_cost_curve(st.session_state.current_scenario, st.session_state.bundle)
            st.plotly_chart(plot_electrification_cost_curve(curve_df), use_container_width=True)
            st.dataframe(curve_df, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Comparación: Electrificación Libre vs Electrificación Obligatoria")
    if st.button("Comparar Escenario Libre vs Obligatorio"):
        with st.spinner("Calculando ambos casos..."):
            comp = compare_unconstrained_vs_constrained(st.session_state.current_scenario, st.session_state.bundle)
            if comp["status"] == "ok":
                c1, c2, c3 = st.columns(3)
                c1.metric("Coste Operativo Libre", f"{comp['unconstrained_optimal_cost_eur']:,.0f} €")
                c2.metric("Coste Operativo Obligatorio", f"{comp['constrained_optimal_cost_eur']:,.0f} €")
                c3.metric("Sobrecoste por Obligación", f"{comp['minimum_electrification_incremental_cost_eur']:,.0f} €", delta=f"{comp['incremental_cost_per_mwh_th']:.2f} €/MWh-th adic.")
            else:
                st.error("Uno de los escenarios resultó infactible.")

    st.markdown("---")
    st.subheader("3. Análisis de Sensibilidad Unidimensional")
    param_to_vary = st.selectbox(
        "Seleccionar parámetro a sensibilizar (±30%):",
        options=[
            ("baseline.gas_price_eur_per_mwh_hhv", "Precio del gas natural"),
            ("baseline.co2_price_eur_per_tco2", "Precio del CO2"),
            ("supply.pool_price_scaling_factor", "Nivel de precios eléctricos pool"),
            ("eboiler.eboiler_capex_eur" if st.session_state.current_scenario.alternative == "eboiler" else "tes.tes_capex_eur", "CAPEX del activo"),
        ],
        format_func=lambda x: x[1],
    )[0]

    if st.button("Ejecutar Barrido de Sensibilidad"):
        with st.spinner("Simulando variaciones de sensibilidad..."):
            sens_df = run_sensitivity_analysis(st.session_state.current_scenario, st.session_state.bundle, parameter_path=param_to_vary)
            st.dataframe(sens_df, use_container_width=True)
            fig_sens = px.line(
                sens_df,
                x="variation_pct",
                y=["annual_net_operating_savings_eur", "npv_eur"],
                title="Impacto de la variación en Ahorros y VAN",
                markers=True,
            )
            st.plotly_chart(fig_sens, use_container_width=True)

# ===========================================================================
# TAB 4: DIMENSIONAMIENTO ÓPTIMO
# ===========================================================================
with tab_sizing:
    st.header("📐 Dimensionamiento Óptimo de Capacidades")
    cfg = st.session_state.current_scenario

    if cfg.alternative == "eboiler":
        st.subheader("Optimización de Potencia para E-Boiler")
        caps_str = st.text_input("Lista de potencias a evaluar (MW-e separadas por comas):", value="2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0")
        if st.button("Optimizar Dimensionamiento E-Boiler"):
            caps = [float(x.strip()) for x in caps_str.split(",") if x.strip()]
            with st.spinner("Evaluando combinaciones de capacidad..."):
                sizing_res = optimize_eboiler_sizing(cfg, st.session_state.bundle, caps)
                st.dataframe(sizing_res["table"], use_container_width=True)
                if sizing_res["best_npv"]:
                    st.success(f"🏆 **Óptimo de Inversión (Mayor VAN):** {sizing_res['best_npv']['capacity_mw_e']} MW-e (VAN: {sizing_res['best_npv']['npv_eur']:,.0f} €)")
                if sizing_res["best_opex"]:
                    st.info(f"⚡ **Mínimo Coste Operativo:** {sizing_res['best_opex']['capacity_mw_e']} MW-e")
    else:
        st.subheader("Optimización de Capacidades para TES")
        st.markdown("Evalúa la malla tridimensional: Potencia Carga (MW-e) x Almacenamiento (MWh-th) x Potencia Descarga (MW-th).")
        col_s1, col_s2, col_s3 = st.columns(3)
        p_ch_str = col_s1.text_input("Potencias de carga (MW-e):", value="10.0, 20.0")
        e_st_str = col_s2.text_input("Almacenamientos (MWh-th):", value="100.0, 200.0")
        p_dis_str = col_s3.text_input("Potencias descarga (MW-th):", value="5.0, 10.0")

        if st.button("Optimizar Dimensionamiento TES"):
            p_ch = [float(x.strip()) for x in p_ch_str.split(",") if x.strip()]
            e_st = [float(x.strip()) for x in e_st_str.split(",") if x.strip()]
            p_dis = [float(x.strip()) for x in p_dis_str.split(",") if x.strip()]
            with st.spinner("Evaluando malla de capacidades TES..."):
                sizing_res = optimize_tes_sizing(cfg, st.session_state.bundle, p_ch, e_st, p_dis)
                st.dataframe(sizing_res["table"], use_container_width=True)
                if sizing_res["best_npv"]:
                    st.success(f"🏆 **Mayor VAN:** Carga {sizing_res['best_npv']['charge_capacity_mw_e']} MW, Almacenamiento {sizing_res['best_npv']['storage_capacity_mwh_th']} MWh, Descarga {sizing_res['best_npv']['discharge_capacity_mw_th']} MW (VAN: {sizing_res['best_npv']['npv_eur']:,.0f} €)")

# ===========================================================================
# TAB 5: COMPARATIVA DE CASOS
# ===========================================================================
with tab_comparison:
    st.header("⚖️ Comparador Multicaso")
    if len(st.session_state.saved_scenarios) < 2:
        st.info("Guarda al menos 2 casos desde la barra lateral para compararlos aquí.")
    else:
        cases_to_compare = st.multiselect(
            "Seleccionar casos guardados a comparar:",
            options=list(st.session_state.saved_scenarios.keys()),
            default=list(st.session_state.saved_scenarios.keys())[:2],
        )
        if st.button("Comparar Casos Seleccionados"):
            with st.spinner("Ejecutando y comparando casos..."):
                sim_results = []
                for cname in cases_to_compare:
                    c_cfg = ScenarioConfig.from_json(st.session_state.saved_scenarios[cname])
                    sim_results.append(run_simulation(c_cfg, st.session_state.bundle))
                comp_table = compare_multiple_scenarios(sim_results)
                st.dataframe(comp_table, use_container_width=True)
