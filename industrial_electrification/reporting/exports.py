"""Exportación de resultados a Excel multillibro y CSV."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from models.inputs import ScenarioConfig
from models.results import ScenarioResult


def export_hourly_results_to_csv(scenario_res: ScenarioResult) -> str:
    """Exporta los resultados horarios de todos los años a formato CSV."""
    df = scenario_res.hourly_all_years()
    return df.to_csv(index=False, sep=";", decimal=",")


def export_scenario_to_excel(
    scenario_res: ScenarioResult,
    cfg: ScenarioConfig,
    sensitivity_df: pd.DataFrame | None = None,
) -> bytes:
    """Genera un archivo Excel profesional con pestañas detalladas según especificación §22."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # 1. Inputs e Hipótesis
        inputs_dict = cfg.model_dump()
        flat_inputs = []
        for section, values in inputs_dict.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat_inputs.append({"Sección": section, "Parámetro": k, "Valor": str(v)})
            else:
                flat_inputs.append({"Sección": "General", "Parámetro": section, "Valor": str(values)})
        pd.DataFrame(flat_inputs).to_excel(writer, sheet_name="Inputs e Hipótesis", index=False)

        # 2. Resumen Anual
        ann_df = scenario_res.annual_table()
        if not ann_df.empty:
            ann_df.to_excel(writer, sheet_name="Resumen Anual", index=False)

        # 3. Flujos Financieros
        if scenario_res.financial and not scenario_res.financial.cash_flows.empty:
            scenario_res.financial.cash_flows.to_excel(writer, sheet_name="Flujos Financieros", index=False)

        # 4. Desglose LCOH y Coste del Calor
        if scenario_res.financial and "lcoh_breakdown_project" in scenario_res.financial.indicators:
            ind = scenario_res.financial.indicators
            lcoh_p = ind.get("lcoh_breakdown_project", {})
            lcoh_b = ind.get("lcoh_breakdown_baseline", {})
            y1_p = ind.get("heat_cost_breakdown_year1_project", {})
            y1_b = ind.get("heat_cost_breakdown_year1_baseline", {})

            concepts = [
                ("capex", "CAPEX / Inversión Inicial"),
                ("opex_tech", "OPEX Técnico y Mantenimiento"),
                ("ppa_onsite", "PPA Onsite (Electricidad PV)"),
                ("pool", "Mercado Diario (Pool Red)"),
                ("adjustments", "Servicios de Ajuste"),
                ("network_tariffs", "Peajes y Cargos de Red (6.xTD)"),
                ("guarantees_of_origin", "Garantías de Origen (GdO)"),
                ("other_grid", "Otros Costes Eléctricos / Fee"),
                ("fossil_gas", "Gas Natural Fósil (Combustible)"),
                ("fossil_co2", "Derechos de Emisión CO2"),
                ("fossil_other", "Otros Costes Gas Natural"),
                ("innovation_fund", "Incentivo Innovation Fund"),
            ]
            rows_lcoh = []
            for k, label in concepts:
                rows_lcoh.append(
                    {
                        "Concepto": label,
                        "LCOH Referencia (EUR/MWh-th)": lcoh_b.get(k, 0.0),
                        "LCOH Proyecto (EUR/MWh-th)": lcoh_p.get(k, 0.0),
                        "Coste Año 1 Referencia (EUR/MWh-th)": y1_b.get(k, 0.0),
                        "Coste Año 1 Proyecto (EUR/MWh-th)": y1_p.get(k, 0.0),
                    }
                )
            rows_lcoh.append(
                {
                    "Concepto": "TOTAL NETO",
                    "LCOH Referencia (EUR/MWh-th)": lcoh_b.get("net_lcoh", 0.0),
                    "LCOH Proyecto (EUR/MWh-th)": lcoh_p.get("net_lcoh", 0.0),
                    "Coste Año 1 Referencia (EUR/MWh-th)": y1_b.get("net_heat_cost", 0.0),
                    "Coste Año 1 Proyecto (EUR/MWh-th)": y1_p.get("net_heat_cost", 0.0),
                }
            )
            pd.DataFrame(rows_lcoh).to_excel(writer, sheet_name="Desglose LCOH y Coste Calor", index=False)

        # 5. Desglose Periodos Tarifarios (Año 1)
        if scenario_res.first_year and scenario_res.first_year.by_period is not None:
            scenario_res.first_year.by_period.to_excel(writer, sheet_name="Periodos Tarifarios P1-P6", index=False)

        # 6. Desglose Mensual (Año 1)
        if scenario_res.first_year and scenario_res.first_year.monthly is not None:
            scenario_res.first_year.monthly.to_excel(writer, sheet_name="Desglose Mensual", index=False)

        # 7. Despacho Horario (Año 1 o todos si cabe)
        hourly_df = scenario_res.hourly_all_years()
        if not hourly_df.empty:
            # Excel admite 1.048.576 filas
            hourly_df.iloc[:50000].to_excel(writer, sheet_name="Despacho Horario", index=False)

        # 7. Sensibilidad si se ejecutó
        if sensitivity_df is not None and not sensitivity_df.empty:
            sensitivity_df.to_excel(writer, sheet_name="Sensibilidad", index=False)

    return output.getvalue()
