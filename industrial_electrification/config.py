"""Configuración global, rutas y supuestos regulatorios del simulador.

Todas las constantes que afectan al cálculo están centralizadas aquí y son
modificables. Ningún módulo de cálculo debe contener constantes "ocultas":
los valores por defecto de los parámetros de usuario se declaran en
``models/inputs.py`` y los supuestos regulatorios (calendario tarifario,
festivos) en este archivo.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input_files"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
CACHE_DIR = PROJECT_ROOT / ".cache"

DEFAULT_POOL_PRICE_FILE = INPUT_DIR / "pool_price_scenarios.xlsx"
DEFAULT_NETWORK_TARIFF_FILE = INPUT_DIR / "network_tariffs.xlsx"
DEFAULT_ELECTROINTENSIVE_TARIFF_FILE = INPUT_DIR / "network_tariffs_electrointensive.xlsx"
DEFAULT_ADJUSTMENT_FILE = INPUT_DIR / "escenario_servicios_ajustes.csv"

# ---------------------------------------------------------------------------
# Resolución temporal
# ---------------------------------------------------------------------------
DEFAULT_TIME_STEP_HOURS = 1.0
# Versión inicial: todas las series de entrada son horarias. Otros pasos
# temporales requerirían remuestrear precios, demanda y PV.
SUPPORTED_TIME_STEP_HOURS = (1.0,)

HOURS_PER_DAY = 24
# Posición (base 0) de la primera hora del 29 de febrero en un calendario
# bisiesto normalizado: 31 días de enero + 28 de febrero = 59 días.
FEB_29_FIRST_HOUR_INDEX = 59 * HOURS_PER_DAY  # 1416
FEB_28_FIRST_HOUR_INDEX = 58 * HOURS_PER_DAY  # 1392

# ---------------------------------------------------------------------------
# Valores iniciales de parámetros físicos (configurables por el usuario)
# ---------------------------------------------------------------------------
DEFAULT_GAS_EMISSION_FACTOR_TCO2_PER_MWH_HHV = 0.1802
DEFAULT_LHV_HHV_RATIO = 0.90
DEFAULT_INNOVATION_FUND_REFERENCE_TCO2_PER_MWH_TH = 0.224
DEFAULT_INNOVATION_FUND_DURATION_YEARS = 5

# ---------------------------------------------------------------------------
# Tarifas de acceso 6.xTD (Península) - Circular CNMC 3/2020
# ---------------------------------------------------------------------------
SUPPORTED_NETWORK_TARIFFS = ("6.1TD", "6.2TD", "6.3TD", "6.4TD")
TARIFF_PERIODS = (1, 2, 3, 4, 5, 6)

# Factor de conversión de las unidades del archivo de peajes a EUR/MWh-e.
# El archivo suministrado ("Energía total") se interpreta en EUR/MWh-e.
# Si se sustituye por un archivo en EUR/kWh-e, fijar 1000.0.
NETWORK_TARIFF_UNIT_TO_EUR_PER_MWH_E = 1.0

TARIFF_SEASON_BY_MONTH: dict[int, str] = {
    1: "high", 2: "high", 3: "medium_high", 4: "low", 5: "low", 6: "medium",
    7: "high", 8: "medium", 9: "medium", 10: "low", 11: "medium_high", 12: "high",
}

SEASON_LABELS_ES = {
    "high": "Alta", "medium_high": "Media-alta", "medium": "Media", "low": "Baja",
}

# Bandas horarias de días laborables: (hora_inicio, hora_fin, {temporada: periodo}).
# hora_fin es exclusiva. Debe cubrir 0-24 sin solapes.
TARIFF_WORKDAY_HOUR_BANDS: list[tuple[int, int, dict[str, int]]] = [
    (0, 8, {"high": 6, "medium_high": 6, "medium": 6, "low": 6}),
    (8, 9, {"high": 2, "medium_high": 3, "medium": 4, "low": 5}),
    (9, 14, {"high": 1, "medium_high": 2, "medium": 3, "low": 4}),
    (14, 18, {"high": 2, "medium_high": 3, "medium": 4, "low": 5}),
    (18, 22, {"high": 1, "medium_high": 2, "medium": 3, "low": 4}),
    (22, 24, {"high": 2, "medium_high": 3, "medium": 4, "low": 5}),
]

# Sábados, domingos y festivos nacionales: P6.
NON_WORKING_DAY_PERIOD = 6
WEEKEND_DAYS = (6, 7)  # Lunes = 1 ... Domingo = 7

# Festivos de ámbito nacional con fecha fija y no sustituibles (Circular
# CNMC 3/2020: se excluyen los festivos sustituibles y los de fecha móvil,
# p. ej. Viernes Santo). Formato MM-DD. NO se incluyen festivos autonómicos
# ni locales.
NATIONAL_FIXED_HOLIDAYS_MMDD: tuple[str, ...] = (
    "01-01",  # Año Nuevo
    "01-06",  # Epifanía del Señor
    "05-01",  # Fiesta del Trabajo
    "08-15",  # Asunción de la Virgen
    "10-12",  # Fiesta Nacional de España
    "11-01",  # Todos los Santos
    "12-06",  # Día de la Constitución
    "12-08",  # Inmaculada Concepción
    "12-25",  # Natividad del Señor
)

# Ajustes por año: {año: {"add": ["MM-DD", ...], "remove": ["MM-DD", ...]}}
HOLIDAY_OVERRIDES_BY_YEAR: dict[int, dict[str, list[str]]] = {}

# ---------------------------------------------------------------------------
# Datos de precios pool
# ---------------------------------------------------------------------------
DEFAULT_POOL_SCENARIO = "AURORA CENTRAL"
# Número máximo por defecto de horas finales que se pueden completar si el
# usuario lo autoriza expresamente (con el valor de la misma hora del día
# anterior). Nunca se completa con ceros.
DEFAULT_POOL_MAX_FILL_HOURS = 24

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
DEFAULT_SOLVER_NAME = "highs"
SOLVER_FALLBACKS = ("highs", "appsi_highs")
DEFAULT_SOLVER_TIME_LIMIT_S = 600.0
DEFAULT_MIP_REL_GAP = 1e-4
# Tolerancia numérica para comprobaciones de balances (MWh / EUR).
NUMERIC_TOLERANCE = 1e-5

# Objetivos de la curva de coste de electrificación (%).
ELECTRIFICATION_CURVE_TARGETS_PCT = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
