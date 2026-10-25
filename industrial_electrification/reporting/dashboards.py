"""Generación de gráficos interactivos Plotly para visualización de resultados."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.results import ScenarioResult, YearResult


def plot_thermal_dispatch(
    y_res: YearResult, start_idx: int = 0, num_hours: int = 168
) -> go.Figure:
    """Gráfico horario de demanda térmica y cobertura por tecnología."""
    df = y_res.hourly.iloc[start_idx : start_idx + num_hours]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["thermal_demand_mwh_th"],
            name="Demanda Térmica (MWh-th)",
            line=dict(color="black", width=2, dash="dash"),
        )
    )

    if y_res.alternative == "eboiler":
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["eboiler_heat_output_mwh_th"],
                name="E-Boiler (MWh-th)",
                stackgroup="heat",
                line=dict(color="#1f77b4"),
                fill="tonexty",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["tes_heat_delivered_mwh_th"],
                name="Descarga TES Útil (MWh-th)",
                stackgroup="heat",
                line=dict(color="#2ca02c"),
                fill="tonexty",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["fossil_heat_output_mwh_th"],
            name="Caldera Fósil Gas (MWh-th)",
            stackgroup="heat",
            line=dict(color="#d62728"),
            fill="tonexty",
        )
    )

    fig.update_layout(
        title="Despacho de Demanda Térmica por Tecnología",
        xaxis_title="Fecha y Hora",
        yaxis_title="Energía Térmica (MWh-th)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_electricity_dispatch(
    y_res: YearResult, start_idx: int = 0, num_hours: int = 168
) -> go.Figure:
    """Gráfico horario de suministro eléctrico (Red vs PV) y precios."""
    df = y_res.hourly.iloc[start_idx : start_idx + num_hours]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if y_res.alternative == "eboiler":
        pv_col = "eboiler_pv_input_mwh_e"
        grid_col = "eboiler_grid_input_mwh_e"
        label = "E-Boiler"
    else:
        pv_col = "tes_pv_charge_mwh_e"
        grid_col = "tes_grid_charge_mwh_e"
        label = "Carga TES"

    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df[pv_col],
            name=f"{label} desde PV Onsite (MWh-e)",
            marker_color="#ff7f0e",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df[grid_col],
            name=f"{label} desde Red (MWh-e)",
            marker_color="#1f77b4",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["grid_unit_cost_eur_per_mwh_e"],
            name="Coste Unitario Red (EUR/MWh-e)",
            line=dict(color="purple", width=1.5),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Consumo Eléctrico Horario y Precio de Red",
        xaxis_title="Fecha y Hora",
        barmode="stack",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Energía Eléctrica (MWh-e)", secondary_y=False)
    fig.update_yaxes(title_text="Coste Electricidad (EUR/MWh-e)", secondary_y=True)
    return fig


def plot_tes_soc_profile(
    y_res: YearResult, start_idx: int = 0, num_hours: int = 168
) -> go.Figure:
    """Gráfico específico de estado de carga (SOC %) y carga/descarga TES."""
    if y_res.alternative != "tes":
        return go.Figure()

    df = y_res.hourly.iloc[start_idx : start_idx + num_hours]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["tes_total_charge_mwh_e"],
            name="Carga Eléctrica (MWh-e)",
            marker_color="#17becf",
            opacity=0.6,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=-df["tes_discharge_from_storage_mwh_th"],
            name="Descarga Almacenamiento (MWh-th)",
            marker_color="#e377c2",
            opacity=0.6,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["tes_soc_pct"],
            name="SOC Almacenamiento (%)",
            line=dict(color="#2ca02c", width=2.5),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Perfil de Operación TES y Estado de Carga (SOC)",
        xaxis_title="Fecha y Hora",
        template="plotly_white",
        barmode="relative",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Potencia / Energía Flujo (MWh)", secondary_y=False)
    fig.update_yaxes(title_text="SOC (%)", range=[0, 105], secondary_y=True)
    return fig


def plot_cash_flow_evolution(scenario_res: ScenarioResult) -> go.Figure:
    """Evolución del flujo de caja anual y flujo acumulado descontado."""
    if not scenario_res.financial or scenario_res.financial.cash_flows.empty:
        return go.Figure()

    cf = scenario_res.financial.cash_flows

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=cf["project_year"],
            y=cf["net_cash_flow_eur"],
            name="Flujo de Caja Neto (EUR)",
            marker_color=np.where(cf["net_cash_flow_eur"] >= 0, "#2ca02c", "#d62728"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=cf["project_year"],
            y=cf["cumulative_discounted_cash_flow_eur"],
            name="Flujo Descontado Acumulado (EUR)",
            line=dict(color="#1f77b4", width=3),
        ),
        secondary_y=True,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=True)

    fig.update_layout(
        title="Flujo de Caja y Payback Descontado del Proyecto",
        xaxis_title="Año de Proyecto",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Flujo Neto Anual (EUR)", secondary_y=False)
    fig.update_yaxes(title_text="Flujo Descontado Acumulado (EUR)", secondary_y=True)
    return fig


def plot_electrification_cost_curve(curve_df: pd.DataFrame) -> go.Figure:
    """Curva de coste incremental frente a objetivo mínimo de electrificación."""
    if curve_df.empty:
        return go.Figure()

    feasible_df = curve_df[curve_df["is_feasible"]]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=feasible_df["target_pct"],
            y=feasible_df["incremental_cost_vs_free_eur"],
            mode="lines+markers",
            name="Coste Incremental Anual vs Libre (EUR)",
            line=dict(color="#d62728", width=2.5),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=feasible_df["target_pct"],
            y=feasible_df["marginal_cost_eur_per_mwh_th"],
            name="Coste Marginal (EUR/MWh-th adic.)",
            marker_color="#ff7f0e",
            opacity=0.5,
        ),
        secondary_y=True,
    )

    # Identificar límite técnico
    max_achievable = curve_df["actual_electrification_pct"].dropna().max()
    if max_achievable is not None:
        fig.add_vline(
            x=max_achievable,
            line_dash="dot",
            line_color="black",
            annotation_text=f"Máx. Técnico ({max_achievable:.1f}%)",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Curva de Coste de Electrificación Térmica Mínima",
        xaxis_title="Objetivo Mínimo de Electrificación (%)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Coste Incremental Total (EUR/año)", secondary_y=False)
    fig.update_yaxes(title_text="Coste Marginal (EUR/MWh-th)", secondary_y=True)
    return fig


def plot_lcoh_comparison(
    breakdown_base: dict[str, float],
    breakdown_proj: dict[str, float],
    breakdown_proj_year1: dict[str, float] | None = None,
    title: str = "Desglose Comparativo de LCOH / Coste del Calor (EUR/MWh-th)",
) -> go.Figure:
    """Gráfico de barras apiladas comparando los componentes del LCOH de referencia vs proyecto (LCOH y Coste Año 1)."""
    items = [
        ("capex", "CAPEX / Reposiciones", "#1f77b4"),
        ("opex_tech", "OPEX Técnico", "#aec7e8"),
        ("ppa_onsite", "PPA Onsite (PV)", "#ff7f0e"),
        ("pool", "Electricidad Red (Pool)", "#2ca02c"),
        ("adjustments", "Servicios de Ajuste", "#98df8a"),
        ("network_tariffs", "Peajes y Cargos (6.xTD)", "#d62728"),
        ("guarantees_of_origin", "Garantías de Origen (GdO)", "#ff9896"),
        ("other_grid", "Otros Costes Eléctricos / Fee", "#9467bd"),
        ("fossil_gas", "Gas Natural Combustible", "#8c564b"),
        ("fossil_co2", "Derechos de Emisión CO2", "#c49c94"),
        ("fossil_other", "Otros Costes Gas", "#e377c2"),
        ("innovation_fund", "Incentivo Innovation Fund", "#bcbd22"),
    ]

    has_y1 = breakdown_proj_year1 is not None and len(breakdown_proj_year1) > 0
    categories = [
        "Situación de Referencia (Gas)",
        "Proyecto Electrificado (LCOH)",
    ]
    if has_y1:
        categories.append("Proyecto Electrificado (Año 1)")

    fig = go.Figure()

    for key, label, color in items:
        val_base = breakdown_base.get(key, 0.0)
        val_proj = breakdown_proj.get(key, 0.0)
        val_y1 = breakdown_proj_year1.get(key, 0.0) if has_y1 else 0.0

        # Si todos son cero o casi cero no añadir traza innecesaria
        if abs(val_base) < 1e-4 and abs(val_proj) < 1e-4 and abs(val_y1) < 1e-4:
            continue

        y_vals = [val_base, val_proj]
        if has_y1:
            y_vals.append(val_y1)

        fig.add_trace(
            go.Bar(
                name=label,
                x=categories,
                y=y_vals,
                marker_color=color,
            )
        )

    # Añadir puntos con el coste neto total
    net_base = breakdown_base.get("net_lcoh", breakdown_base.get("net_heat_cost", 0.0))
    net_proj = breakdown_proj.get("net_lcoh", breakdown_proj.get("net_heat_cost", 0.0))

    net_points = [net_base, net_proj]
    text_points = [f"{net_base:.2f} €/MWh-th", f"{net_proj:.2f} €/MWh-th"]

    if has_y1:
        net_y1 = breakdown_proj_year1.get("net_heat_cost", breakdown_proj_year1.get("net_lcoh", 0.0))
        net_points.append(net_y1)
        text_points.append(f"{net_y1:.2f} €/MWh-th")

    fig.add_trace(
        go.Scatter(
            x=categories,
            y=net_points,
            mode="text+markers",
            name="Coste Neto",
            text=text_points,
            textposition="top center",
            marker=dict(size=12, color="black", symbol="diamond"),
        )
    )

    fig.update_layout(
        title=title,
        barmode="relative",
        yaxis_title="Coste (EUR/MWh-th)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
    )
    return fig


def plot_electricity_price_stats(price_stats_df: pd.DataFrame) -> go.Figure:
    """Gráfico comparativo de Precio Pool vs Precio Completo de Red (Media, P10, P50, P90) anual y por periodos P1-P6."""
    if price_stats_df is None or price_stats_df.empty:
        return go.Figure()

    periods = price_stats_df["period"].tolist()
    fig = go.Figure()

    # Barras para la Media (Average) con barras de error asimétricas mostrando el rango P10 - P90
    pool_mean = price_stats_df["pool_mean_eur_per_mwh_e"]
    pool_p10 = price_stats_df["pool_p10_eur_per_mwh_e"]
    pool_p50 = price_stats_df["pool_p50_eur_per_mwh_e"]
    pool_p90 = price_stats_df["pool_p90_eur_per_mwh_e"]

    full_mean = price_stats_df["full_price_mean_eur_per_mwh_e"]
    full_p10 = price_stats_df["full_price_p10_eur_per_mwh_e"]
    full_p50 = price_stats_df["full_price_p50_eur_per_mwh_e"]
    full_p90 = price_stats_df["full_price_p90_eur_per_mwh_e"]

    fig.add_trace(
        go.Bar(
            name="Precio Pool — Media (rango P10–P90)",
            x=periods,
            y=pool_mean,
            marker_color="#1f77b4",
            opacity=0.85,
            error_y=dict(
                type="data",
                symmetric=False,
                array=(pool_p90 - pool_mean).tolist(),
                arrayminus=(pool_mean - pool_p10).tolist(),
                color="#0d3b66",
                thickness=1.5,
                width=6,
            ),
            customdata=np.stack([pool_p10, pool_p50, pool_p90], axis=-1),
            hovertemplate=(
                "<b>%{x} — Precio Pool</b><br>"
                "Media: %{y:.2f} €/MWh-e<br>"
                "P10: %{customdata[0]:.2f} €/MWh-e<br>"
                "P50: %{customdata[1]:.2f} €/MWh-e<br>"
                "P90: %{customdata[2]:.2f} €/MWh-e<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            name="Precio Completo Red — Media (rango P10–P90)",
            x=periods,
            y=full_mean,
            marker_color="#2ca02c",
            opacity=0.85,
            error_y=dict(
                type="data",
                symmetric=False,
                array=(full_p90 - full_mean).tolist(),
                arrayminus=(full_mean - full_p10).tolist(),
                color="#145a32",
                thickness=1.5,
                width=6,
            ),
            customdata=np.stack([full_p10, full_p50, full_p90], axis=-1),
            hovertemplate=(
                "<b>%{x} — Precio Completo de Red</b><br>"
                "Media: %{y:.2f} €/MWh-e<br>"
                "P10: %{customdata[0]:.2f} €/MWh-e<br>"
                "P50: %{customdata[1]:.2f} €/MWh-e<br>"
                "P90: %{customdata[2]:.2f} €/MWh-e<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            name="Precio Pool — P50 (Mediana)",
            x=periods,
            y=pool_p50,
            mode="markers",
            marker=dict(color="#ff7f0e", size=9, symbol="diamond"),
            hovertemplate="<b>%{x}</b> — Pool P50: %{y:.2f} €/MWh-e<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            name="Precio Completo Red — P50 (Mediana)",
            x=periods,
            y=full_p50,
            mode="markers",
            marker=dict(color="#d62728", size=9, symbol="diamond"),
            hovertemplate="<b>%{x}</b> — Precio Completo P50: %{y:.2f} €/MWh-e<extra></extra>",
        )
    )

    fig.update_layout(
        title="Distribución de Precios Eléctricos (Media, P50 y Rango P10–P90): Pool vs. Precio Completo de Red",
        xaxis_title="Periodo (Anual 8760h y Periodos Tarifarios P1–P6)",
        yaxis_title="Precio Eléctrico (EUR/MWh-e)",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

