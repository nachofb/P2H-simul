"""Importación y transformación de archivos de entrada.

Archivos soportados:
- ``pool_price_scenarios.xlsx``: precios pool horarios por año y escenario.
- ``network_tariffs.xlsx``: peajes y cargos de energía por tarifa y periodo.
- ``escenario_servicios_ajustes.csv``: servicios de ajuste horarios.
- CSV genéricos de 8760/8784 valores (demanda, PV).

Reglas comunes:
- Se admite coma decimal y punto/coma como separador de columnas.
- No se asume que los datos empiecen en la primera fila.
- Los valores faltantes NUNCA se sustituyen por cero. Solo se completan
  horas finales si el usuario lo autoriza expresamente
  (``fill_authorized=True``) y se documenta en las notas devueltas.
"""
from __future__ import annotations

import hashlib
import io
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional, Union

import numpy as np
import pandas as pd

import config
from data.calendar import adapt_hourly_series_to_year, hours_in_year, is_leap_year

PathOrBuffer = Union[str, Path, IO[bytes], IO[str], bytes]


class DataFormatError(ValueError):
    """Error de formato o de completitud de un archivo de datos."""


# ---------------------------------------------------------------------------
# Conversión numérica robusta
# ---------------------------------------------------------------------------
_EMPTY_TOKENS = {"", "-", "—", "–", "nan", "NaN", "NAN", "None", "null", "NULL", "n/a", "N/A", "#N/A"}


def parse_number(value) -> float:
    """Convierte un valor (número o texto con coma/punto decimal) a float.

    Devuelve NaN para celdas vacías o no numéricas (nunca 0).
    """
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if s in _EMPTY_TOKENS:
        return math.nan
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")  # 1.234,56
        else:
            s = s.replace(",", "")  # 1,234.56
    elif "," in s:
        if s.count(",") > 1:
            return math.nan
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return math.nan


def to_numeric_array(values) -> np.ndarray:
    return np.array([parse_number(v) for v in values], dtype=float)


def _read_bytes(source: PathOrBuffer) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    data = source.read()
    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except Exception:  # pragma: no cover
            pass
    return data.encode("utf-8") if isinstance(data, str) else data


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DataFormatError("No se pudo decodificar el archivo de texto.")


# ---------------------------------------------------------------------------
# CSV horarios genéricos
# ---------------------------------------------------------------------------
@dataclass
class HourlyCsvData:
    values: np.ndarray
    delimiter: str
    header_rows: list[str]
    column_name: str
    n_records: int
    notes: list[str] = field(default_factory=list)


def _detect_delimiter(lines: list[str]) -> str:
    sample = lines[: min(len(lines), 50)]
    counts = {d: sum(line.count(d) for line in sample) for d in (";", "\t", "|")}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    # Sin ';' ni tabulador: la coma puede ser separador o decimal.
    comma_lines = [line for line in sample if "," in line]
    if comma_lines and all(line.count(",") == 1 for line in comma_lines):
        # Una única coma por línea: si ambas partes son números enteros
        # cortos tipo "9,77" se interpreta como decimal.
        parts_ok = all(
            re.fullmatch(r"\s*-?\d+\s*,\s*\d+\s*", line) is not None for line in comma_lines[1:]
        )
        if parts_ok:
            return "NONE"
    return ","


def read_hourly_csv(
    source: PathOrBuffer,
    value_column: Optional[Union[str, int]] = None,
    name: str = "CSV",
) -> HourlyCsvData:
    """Lee un CSV horario y devuelve la columna numérica de valores.

    - Detecta delimitador (';', tabulador, '|', ',') y coma decimal.
    - Omite filas de encabezado (filas iniciales sin ningún valor numérico).
    - Elige la columna indicada o, por defecto, la última columna numérica que
      no sea un índice correlativo 1..N.
    """
    text = _decode(_read_bytes(source))
    lines = [line for line in text.splitlines() if line.strip() != ""]
    if not lines:
        raise DataFormatError(f"{name}: archivo vacío.")
    delimiter = _detect_delimiter(lines)
    if delimiter == "NONE":
        rows = [[line.strip()] for line in lines]
    else:
        rows = [line.split(delimiter) for line in lines]
    # Eliminar columnas vacías finales (p. ej. por ';' al final de línea).
    rows = [[c.strip().strip('"') for c in r] for r in rows]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    while width > 1 and all(r[-1] == "" for r in rows):
        rows = [r[:-1] for r in rows]
        width -= 1

    header_rows: list[list[str]] = []
    start = 0
    for r in rows:
        if any(not math.isnan(parse_number(c)) for c in r):
            break
        header_rows.append(r)
        start += 1
    body = rows[start:]
    if not body:
        raise DataFormatError(f"{name}: no se han encontrado filas numéricas.")
    header = header_rows[-1] if header_rows else [f"col_{i}" for i in range(width)]

    cols = list(zip(*body))
    parsed = [to_numeric_array(c) for c in cols]
    numeric_idx = [i for i, arr in enumerate(parsed) if np.mean(~np.isnan(arr)) >= 0.95]
    if value_column is not None:
        if isinstance(value_column, int):
            idx = value_column
        else:
            if value_column not in header:
                raise DataFormatError(f"{name}: columna '{value_column}' no encontrada en {header}.")
            idx = header.index(value_column)
    else:
        if not numeric_idx:
            raise DataFormatError(f"{name}: ninguna columna es numérica.")
        candidates = [
            i for i in numeric_idx
            if not np.array_equal(parsed[i], np.arange(1, len(parsed[i]) + 1, dtype=float))
        ] or numeric_idx
        idx = candidates[-1]
    values = parsed[idx]
    nan_rows = np.where(np.isnan(values))[0]
    if len(nan_rows):
        raise DataFormatError(
            f"{name}: {len(nan_rows)} registros no numéricos o vacíos "
            f"(filas de datos {list(nan_rows[:10] + 1)}). No se sustituyen por cero."
        )
    return HourlyCsvData(
        values=values,
        delimiter=delimiter,
        header_rows=[";".join(h) for h in header_rows],
        column_name=header[idx] if idx < len(header) else f"col_{idx}",
        n_records=len(values),
    )


# ---------------------------------------------------------------------------
# Servicios de ajuste
# ---------------------------------------------------------------------------
def read_adjustment_series(source: PathOrBuffer) -> HourlyCsvData:
    data = read_hourly_csv(source, name="Servicios de ajuste")
    if data.n_records not in (8760, 8784):
        raise DataFormatError(
            f"Servicios de ajuste: {data.n_records} registros; se esperaban 8760 (u 8784)."
        )
    return data


def adjustment_series_for_year(values: np.ndarray, year: int) -> tuple[np.ndarray, list[str]]:
    """Adapta la serie de ajustes al año (29-feb = 28-feb en bisiestos)."""
    return adapt_hourly_series_to_year(values, year, name="Servicios de ajuste")


# ---------------------------------------------------------------------------
# Peajes y cargos
# ---------------------------------------------------------------------------
def normalize_tariff_code(raw) -> str:
    return re.sub(r"\s+", "", str(raw)).upper().replace("TD", "TD")


def read_network_tariffs(source: PathOrBuffer, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Devuelve DataFrame índice=tarifa ('6.1TD'), columnas 1..6, EUR/MWh-e."""
    raw_bytes = _read_bytes(source)
    raw = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet_name or 0, header=None, dtype=object)
    header_row = None
    for i in range(min(len(raw), 30)):
        tokens = [str(v).strip().upper() for v in raw.iloc[i].tolist()]
        if sum(1 for t in tokens if re.fullmatch(r"P[1-6]", t)) >= 3:
            header_row = i
            break
    if header_row is None:
        raise DataFormatError("Peajes: no se encuentra la fila de encabezado con P1..P6.")
    tokens = [str(v).strip().upper() for v in raw.iloc[header_row].tolist()]
    period_cols = {int(t[1]): j for j, t in enumerate(tokens) if re.fullmatch(r"P[1-6]", t)}
    name_col = min(period_cols.values()) - 1
    if name_col < 0:
        raise DataFormatError("Peajes: no se encuentra la columna de nombres de tarifa.")
    records = {}
    for i in range(header_row + 1, len(raw)):
        name = raw.iat[i, name_col]
        if name is None or (isinstance(name, float) and math.isnan(name)):
            continue
        code = normalize_tariff_code(name)
        records[code] = {
            p: parse_number(raw.iat[i, j]) * config.NETWORK_TARIFF_UNIT_TO_EUR_PER_MWH_E
            for p, j in period_cols.items()
        }
    df = pd.DataFrame.from_dict(records, orient="index").sort_index(axis=1)
    df.index.name = "tariff"
    return df


def network_tariff_for(df: pd.DataFrame, tariff: str) -> dict[int, float]:
    code = normalize_tariff_code(tariff)
    if code not in df.index:
        raise DataFormatError(f"Peajes: tarifa {tariff} no encontrada. Disponibles: {list(df.index)}")
    row = df.loc[code]
    missing = [p for p in config.TARIFF_PERIODS if p not in row.index or math.isnan(row[p])]
    if missing:
        raise DataFormatError(f"Peajes: la tarifa {tariff} no tiene valor para los periodos {missing}.")
    return {p: float(row[p]) for p in config.TARIFF_PERIODS}


# ---------------------------------------------------------------------------
# Precios pool
# ---------------------------------------------------------------------------
@dataclass
class PoolPriceTable:
    """Tabla de precios pool en bruto (por posición), indexada por (año, escenario)."""

    source_name: str
    series: dict[tuple[int, str], np.ndarray]
    n_positions: int
    detection_notes: list[str] = field(default_factory=list)

    @property
    def years(self) -> list[int]:
        return sorted({y for y, _ in self.series})

    @property
    def scenarios(self) -> list[str]:
        seen: list[str] = []
        for _, s in self.series:
            if s not in seen:
                seen.append(s)
        return seen

    def coverage(self) -> pd.DataFrame:
        """Resumen de completitud por año y escenario."""
        rows = []
        for (y, s), arr in sorted(self.series.items()):
            valid = ~np.isnan(arr)
            first = int(np.argmax(valid)) + 1 if valid.any() else None
            last = int(len(arr) - np.argmax(valid[::-1])) if valid.any() else None
            rows.append(
                {
                    "year": y, "scenario": s, "n_values": int(valid.sum()),
                    "hours_required": hours_in_year(y), "first_position": first,
                    "last_position": last, "negative_values": int((arr[valid] < 0).sum()),
                    "min_eur_per_mwh_e": float(np.nanmin(arr)) if valid.any() else None,
                    "max_eur_per_mwh_e": float(np.nanmax(arr)) if valid.any() else None,
                    "mean_eur_per_mwh_e": float(np.nanmean(arr)) if valid.any() else None,
                }
            )
        return pd.DataFrame(rows)

    def add_scenario(self, year: int, scenario: str, values) -> None:
        """Permite incorporar nuevos escenarios (p. ej. desde la interfaz)."""
        self.series[(int(year), str(scenario).strip().upper())] = np.asarray(values, dtype=float)


def _parse_pool_frame(raw: pd.DataFrame, source_name: str) -> PoolPriceTable:
    notes: list[str] = []
    year_row = None
    for i in range(min(len(raw), 30)):
        vals = raw.iloc[i, 1:].tolist()
        nums = [parse_number(v) for v in vals]
        nonnull = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        years = [n for n in nums if not math.isnan(n) and float(n).is_integer() and 1900 <= n <= 2200]
        if nonnull and len(years) >= 0.8 * len(nonnull):
            year_row = i
            break
    if year_row is None:
        raise DataFormatError("Pool: no se identifica la fila de años.")
    scen_row = None
    for i in range(year_row + 1, min(len(raw), year_row + 10)):
        vals = raw.iloc[i, 1:].tolist()
        texts = [v for v in vals if isinstance(v, str) and math.isnan(parse_number(v))]
        if len(texts) >= 0.5 * max(1, sum(v is not None for v in vals)):
            scen_row = i
            break
    if scen_row is None:
        raise DataFormatError("Pool: no se identifica la fila de escenarios.")
    notes.append(f"Fila de años: {year_row + 1}; fila de escenarios: {scen_row + 1}; datos desde fila {scen_row + 2}.")

    years_raw = [parse_number(v) for v in raw.iloc[year_row, 1:].tolist()]
    # Celdas combinadas: propagar el año hacia la derecha.
    years_ff: list[float] = []
    last = math.nan
    for y in years_raw:
        last = y if not math.isnan(y) else last
        years_ff.append(last)
    scenarios = raw.iloc[scen_row, 1:].tolist()
    data = raw.iloc[scen_row + 1:, 1:]
    series: dict[tuple[int, str], np.ndarray] = {}
    for j, (y, s) in enumerate(zip(years_ff, scenarios)):
        if math.isnan(y) or s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        key = (int(y), str(s).strip().upper())
        if key in series:
            raise DataFormatError(f"Pool: columna duplicada para {key}.")
        series[key] = to_numeric_array(data.iloc[:, j].tolist())
    if not series:
        raise DataFormatError("Pool: no se han encontrado columnas de precios.")
    n_positions = len(data)
    return PoolPriceTable(source_name, series, n_positions, notes)


def _cache_key(raw_bytes: bytes) -> str:
    return hashlib.sha1(raw_bytes).hexdigest()[:16]


def read_pool_price_workbook(
    source: PathOrBuffer, sheet_name: Optional[str] = None, use_cache: bool = True
) -> PoolPriceTable:
    raw_bytes = _read_bytes(source)
    source_name = str(source) if isinstance(source, (str, Path)) else getattr(source, "name", "upload")
    cache_file = config.CACHE_DIR / f"pool_{_cache_key(raw_bytes)}_{sheet_name or 0}.pkl"
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            table: PoolPriceTable = pickle.load(fh)
        table.source_name = source_name
        return table
    raw = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet_name or 0, header=None, dtype=object)
    table = _parse_pool_frame(raw, source_name)
    if use_cache:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as fh:
            pickle.dump(table, fh)
    return table


def pool_price_series_for_year(
    table: PoolPriceTable,
    data_year: int,
    scenario: str,
    target_year: Optional[int] = None,
    fill_authorized: bool = False,
    max_fill_hours: int = config.DEFAULT_POOL_MAX_FILL_HOURS,
) -> tuple[np.ndarray, list[str]]:
    """Extrae y valida la serie pool de ``data_year`` para el calendario de ``target_year``.

    Interpretación posicional (validada con el archivo suministrado): la
    posición 1 es el 1 de enero 00:00-01:00 y las posiciones siguen el
    calendario normalizado (24 h/día). Si un año no bisiesto trae más de 8760
    posiciones se interpreta como calendario bisiesto normalizado y se retiran
    las 24 posiciones del 29 de febrero.
    """
    target_year = target_year or data_year
    key = (int(data_year), str(scenario).strip().upper())
    if key not in table.series:
        raise DataFormatError(
            f"Pool: no existe el escenario '{scenario}' para el año {data_year}. "
            f"Escenarios: {table.scenarios}; años: {table.years[0]}-{table.years[-1]}."
        )
    raw = table.series[key]
    label = f"Pool {key[1]} {data_year}"
    notes: list[str] = []
    valid = ~np.isnan(raw)
    if not valid.any():
        raise DataFormatError(f"{label}: columna sin datos.")
    first = int(np.argmax(valid))
    last = int(len(raw) - np.argmax(valid[::-1]) - 1)
    if first > 0:
        raise DataFormatError(
            f"{label}: la serie empieza en la posición {first + 1}; faltan las {first} primeras horas. "
            "Año incompleto: seleccione otro año del escenario."
        )
    body = raw[: last + 1].copy()
    gaps = np.where(np.isnan(body))[0]
    if len(gaps):
        raise DataFormatError(
            f"{label}: {len(gaps)} valores nulos intermedios (posiciones {list(gaps[:10] + 1)}). "
            "No se sustituyen por cero."
        )
    n = len(body)
    # Calendario bisiesto normalizado en un año no bisiesto -> retirar 29-feb.
    data_leap = is_leap_year(data_year)
    if not data_leap and n > 8760:
        if n > 8784:
            raise DataFormatError(f"{label}: {n} valores (> 8784).")
        i29 = config.FEB_29_FIRST_HOUR_INDEX
        body = np.concatenate([body[:i29], body[i29 + 24:]])
        notes.append(f"{label}: detectado calendario bisiesto normalizado; retiradas posiciones 29-feb.")
        n = len(body)
    expected_data = hours_in_year(data_year)
    if n > expected_data:
        raise DataFormatError(f"{label}: {n} valores; se esperaban {expected_data}.")
    missing = expected_data - n
    if missing > 0:
        if not fill_authorized:
            raise DataFormatError(
                f"{label}: contiene {n} valores y el año requiere {expected_data}; faltan las "
                f"{missing} últimas horas. Autorice expresamente su completado (misma hora del día "
                "anterior) o corrija el archivo."
            )
        if missing > max_fill_hours:
            raise DataFormatError(
                f"{label}: faltan {missing} horas finales (> máximo autorizado {max_fill_hours})."
            )
        filled = list(body)
        for k in range(n, expected_data):
            filled.append(filled[k - config.HOURS_PER_DAY])
        body = np.array(filled, dtype=float)
        notes.append(
            f"AVISO {label}: completadas {missing} horas finales con la misma hora del día anterior "
            "(autorizado por el usuario)."
        )
    if target_year != data_year:
        body, adapt_notes = adapt_hourly_series_to_year(body, target_year, name=label)
        notes.extend(adapt_notes)
        notes.append(f"{label}: utilizado para el año simulado {target_year}.")
    n_neg = int((body < 0).sum())
    if n_neg:
        notes.append(f"{label}: {n_neg} horas con precio negativo (se preservan).")
    return body, notes


# ---------------------------------------------------------------------------
# Paquete de datos
# ---------------------------------------------------------------------------
@dataclass
class DataBundle:
    pool: PoolPriceTable
    network_tariffs: pd.DataFrame
    adjustment_values: Optional[np.ndarray]
    sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def load_data_bundle(
    pool_source: PathOrBuffer = config.DEFAULT_POOL_PRICE_FILE,
    tariff_source: PathOrBuffer = config.DEFAULT_NETWORK_TARIFF_FILE,
    adjustment_source: Optional[PathOrBuffer] = config.DEFAULT_ADJUSTMENT_FILE,
    use_cache: bool = True,
) -> DataBundle:
    pool = read_pool_price_workbook(pool_source, use_cache=use_cache)
    tariffs = read_network_tariffs(tariff_source)
    adjustment = None
    notes = list(pool.detection_notes)
    if adjustment_source is not None:
        adj = read_adjustment_series(adjustment_source)
        adjustment = adj.values
        notes.append(
            f"Servicios de ajuste: {adj.n_records} registros, delimitador '{adj.delimiter}', "
            f"columna '{adj.column_name}'."
        )
    return DataBundle(
        pool=pool,
        network_tariffs=tariffs,
        adjustment_values=adjustment,
        sources={
            "pool": str(pool_source) if isinstance(pool_source, (str, Path)) else "upload",
            "tariffs": str(tariff_source) if isinstance(tariff_source, (str, Path)) else "upload",
            "adjustment": str(adjustment_source) if isinstance(adjustment_source, (str, Path)) else "upload",
        },
        notes=notes,
    )
