"""Calendario horario normalizado y periodos tarifarios 6.xTD (Península).

Representación de intervalos
----------------------------
- Cada intervalo ``t`` representa la hora [hh:00, hh+1:00) de la hora oficial
  peninsular *normalizada*: todos los días tienen exactamente 24 intervalos.
- No hay cambio de hora: el día del adelanto (marzo) tiene 24 intervalos y el
  del retraso (octubre) también. Así un año normal tiene 8.760 intervalos y un
  bisiesto 8.784.
- ``timestamp`` marca el INICIO del intervalo (naive, sin zona horaria).
- ``hour_of_year`` = 1 ... 8760/8784; ``hour_of_day`` = 0 ... 23.
- ``day_of_week``: lunes = 1 ... domingo = 7.

Conversión de series externas con marcas temporales reales
----------------------------------------------------------
Ver :func:`normalize_timestamped_series`: se convierten a hora local
(Europe/Madrid), la hora duplicada del retraso de octubre se promedia y la hora
inexistente del adelanto de marzo se interpola linealmente; después se
reindexa sobre el calendario normalizado.
"""
from __future__ import annotations

import calendar as _stdlib_calendar
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import config


def is_leap_year(year: int) -> bool:
    return _stdlib_calendar.isleap(year)


def hours_in_year(year: int, time_step_hours: float = 1.0) -> int:
    days = 366 if is_leap_year(year) else 365
    return int(round(days * config.HOURS_PER_DAY / time_step_hours))


# ---------------------------------------------------------------------------
# Festivos
# ---------------------------------------------------------------------------
def national_holidays(
    year: int,
    fixed_mmdd: Iterable[str] = config.NATIONAL_FIXED_HOLIDAYS_MMDD,
    overrides: Optional[dict[int, dict[str, list[str]]]] = None,
) -> set[date]:
    """Festivos nacionales que computan como P6 en el año indicado."""
    overrides = config.HOLIDAY_OVERRIDES_BY_YEAR if overrides is None else overrides
    mmdd = set(fixed_mmdd)
    year_rule = overrides.get(year, {})
    mmdd |= set(year_rule.get("add", []))
    mmdd -= set(year_rule.get("remove", []))
    result = set()
    for item in mmdd:
        month, day = (int(x) for x in item.split("-"))
        result.add(date(year, month, day))
    return result


# ---------------------------------------------------------------------------
# Periodos tarifarios
# ---------------------------------------------------------------------------
def _build_hour_lookup(
    bands: list[tuple[int, int, dict[str, int]]] = config.TARIFF_WORKDAY_HOUR_BANDS,
) -> dict[str, np.ndarray]:
    seasons = set(config.TARIFF_SEASON_BY_MONTH.values())
    lookup = {s: np.zeros(24, dtype=int) for s in seasons}
    covered = np.zeros(24, dtype=int)
    for start, end, mapping in bands:
        for s in seasons:
            lookup[s][start:end] = mapping[s]
        covered[start:end] += 1
    if not np.all(covered == 1):
        raise ValueError("Las bandas horarias tarifarias deben cubrir 0-24 h sin solapes.")
    return lookup


_HOUR_LOOKUP = _build_hour_lookup()


def calculate_tariff_period(
    month: int,
    day_of_week: int,
    hour_of_day: int,
    is_holiday: bool = False,
    season_by_month: dict[int, str] = config.TARIFF_SEASON_BY_MONTH,
    hour_lookup: Optional[dict[str, np.ndarray]] = None,
) -> int:
    """Periodo tarifario P1-P6 (devuelve 1..6) para tarifas 6.xTD peninsulares.

    Parameters
    ----------
    month: 1-12
    day_of_week: 1 (lunes) ... 7 (domingo)
    hour_of_day: 0-23 (inicio del intervalo)
    is_holiday: festivo nacional computable como P6
    """
    if not 1 <= month <= 12 or not 1 <= day_of_week <= 7 or not 0 <= hour_of_day <= 23:
        raise ValueError("Argumentos de calendario fuera de rango.")
    if is_holiday or day_of_week in config.WEEKEND_DAYS:
        return config.NON_WORKING_DAY_PERIOD
    lookup = hour_lookup or _HOUR_LOOKUP
    return int(lookup[season_by_month[month]][hour_of_day])


def build_calendar(
    year: int,
    time_step_hours: float = 1.0,
    holidays: Optional[set[date]] = None,
) -> pd.DataFrame:
    """Calendario horario normalizado de un año con periodo tarifario."""
    if time_step_hours != 1.0:
        raise ValueError("La versión actual solo admite paso horario (1 h).")
    n = hours_in_year(year, time_step_hours)
    timestamps = pd.date_range(start=f"{year}-01-01 00:00", periods=n, freq="h")
    holidays = national_holidays(year) if holidays is None else holidays
    dates = timestamps.date
    holiday_set = set(holidays)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "year": timestamps.year.astype(int),
            "month": timestamps.month.astype(int),
            "day_of_year": timestamps.dayofyear.astype(int),
            "day_of_week": (timestamps.dayofweek + 1).astype(int),
            "hour_of_day": timestamps.hour.astype(int),
            "hour_of_year": np.arange(1, n + 1, dtype=int),
        }
    )
    df["is_holiday"] = np.array([d in holiday_set for d in dates], dtype=bool)
    df["is_weekend"] = df["day_of_week"].isin(config.WEEKEND_DAYS).to_numpy()
    df["tariff_season"] = df["month"].map(config.TARIFF_SEASON_BY_MONTH)
    periods = np.empty(n, dtype=int)
    for season, table in _HOUR_LOOKUP.items():
        mask = (df["tariff_season"] == season).to_numpy()
        periods[mask] = table[df.loc[mask, "hour_of_day"].to_numpy()]
    non_working = df["is_holiday"].to_numpy() | df["is_weekend"].to_numpy()
    periods[non_working] = config.NON_WORKING_DAY_PERIOD
    df["tariff_period"] = periods
    return df


# ---------------------------------------------------------------------------
# Adaptación de series anuales
# ---------------------------------------------------------------------------
def adapt_hourly_series_to_year(
    values: np.ndarray, year: int, name: str = "serie"
) -> tuple[np.ndarray, list[str]]:
    """Adapta una serie anual de 8760/8784 valores al año indicado.

    - 8760 -> bisiesto: se replican las 24 horas del 28 de febrero como 29 de
      febrero y se conserva la secuencia posterior.
    - 8784 -> año normal: se eliminan las 24 posiciones del 29 de febrero.
    """
    values = np.asarray(values, dtype=float)
    notes: list[str] = []
    leap = is_leap_year(year)
    n = len(values)
    i28, i29 = config.FEB_28_FIRST_HOUR_INDEX, config.FEB_29_FIRST_HOUR_INDEX
    if n == 8760 and leap:
        values = np.concatenate([values[:i29], values[i28:i29], values[i29:]])
        notes.append(f"{name}: año {year} bisiesto; 29-feb completado con los valores del 28-feb.")
    elif n == 8784 and not leap:
        values = np.concatenate([values[:i29], values[i29 + 24:]])
        notes.append(f"{name}: año {year} no bisiesto; eliminadas las 24 posiciones del 29-feb.")
    elif n not in (8760, 8784):
        raise ValueError(f"{name}: {n} registros; se esperaban 8760 u 8784.")
    return values, notes


def normalize_timestamped_series(
    series: pd.Series, year: int, tz: str = "Europe/Madrid"
) -> np.ndarray:
    """Convierte una serie con marcas temporales reales al calendario normalizado.

    ``series`` debe tener un índice datetime (con o sin zona horaria). Si el
    índice es naive se interpreta como UTC si contiene horas duplicadas; en
    otro caso, como hora local.
    """
    s = series.copy()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC") if idx.has_duplicates else idx.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    local = idx.tz_convert(tz).tz_localize(None)
    s.index = local
    s = s[~s.index.isna()]
    s = s.groupby(level=0).mean()  # hora duplicada de octubre -> promedio
    target = pd.date_range(f"{year}-01-01", periods=hours_in_year(year), freq="h")
    s = s.reindex(target)
    s = s.interpolate(limit=2, limit_direction="both")  # hora inexistente de marzo
    if s.isna().any():
        raise ValueError(
            f"Serie con marcas temporales: faltan {int(s.isna().sum())} horas tras normalizar."
        )
    return s.to_numpy(dtype=float)
