# Simulador de Electrificación y Descarbonización Industrial (Power-to-Heat P2H)

Software en Python para **simular, optimizar matemáticamente y evaluar técnica y económicamente proyectos de electrificación del calor industrial**, comparando una instalación convencional de gas natural con:
1. **Alternativa 1 — E-Boiler sin almacenamiento:** Caldera eléctrica que atiende directamente la demanda térmica y caldera de gas de apoyo.
2. **Alternativa 2 — TES (Thermal Energy Storage):** Sistema de almacenamiento térmico cargado con electricidad de la red o planta fotovoltaica (PV) onsite y caldera de gas de apoyo.

---

## 1. Características Técnicas Principales
- **Optimización de Despacho:** Programación lineal exacta (LP / MILP) con **Pyomo** y el solver de alto rendimiento **HiGHS**.
- **Restricción de Electrificación Mínima Obligatoria:** Permite forzar un % mínimo de calor electrificado anual, diagnosticando infactibilidades y calculando el máximo técnicamente alcanzable.
- **Suministro Eléctrico Realista:**
  - Integración nativa con precios horarios del mercado eléctrico español (`pool_price_scenarios.xlsx`).
  - Servicios de ajuste horarios (`escenario_servicios_ajustes.csv`).
  - Tarifas de acceso 6.1TD a 6.4TD (`network_tariffs.xlsx`) y periodos P1–P6 según regulación de la CNMC (Circular 3/2020).
  - Autoconsumo solar PV onsite con lógica lexicográfica de prioridad PV.
- **Incentivos Innovation Fund:** Modelo de subasta competitiva (€/tCO2) sobre calor útil elegible.
- **Evaluación Financiera Completa:** CAPEX anualizado, VAN, TIR, Payback simple y descontado, flujos de caja y LCOH del calor descontado plurianual.
- **Análisis de Sensibilidad y Dimensionamiento Óptimo:** Curvas de coste marginal de electrificación, barridos de sensibilidad paramétrica y optimización multivariable de potencias y almacenamiento.

---

## 2. Instalación y Puesta en Marcha

### Requisitos Previos
- Python 3.12 o superior.

### Pasos de Instalación
```bash
# 1. Clonar o acceder al directorio del proyecto
cd industrial_electrification

# 2. Crear y activar el entorno virtual
python -m venv .venv
# En Windows:
.\.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 3. Ejecución de la Interfaz Web (Streamlit)

Para lanzar el cuadro de mandos interactivo:
```bash
streamlit run app.py
```

La interfaz permite:
1. Seleccionar la alternativa (E-boiler o TES).
2. Configurar la demanda (constante, semanal o mediante curva horaria CSV).
3. Establecer límites de red, capacidad PV y objetivos mínimos de electrificación.
4. Ejecutar la optimización matemática y explorar gráficos de despacho, SOC y flujos de caja.
5. Descargar el informe detallado en Excel multillibro y las series horarias en CSV.

---

## 4. Pruebas Automatizadas (Pytest)

El simulador cuenta con una batería completa de pruebas que aseguran la conservación de energía, la coherencia dimensional, el comportamiento ante infactibilidad y la exactitud matemática de las finanzas:

```bash
# Ejecución con PYTHONPATH en la carpeta del proyecto
pytest tests/test_core.py -v
```

---

## 5. Estructura Modular del Proyecto

```text
industrial_electrification/
├── app.py                      # Interfaz gráfica de usuario en Streamlit
├── config.py                   # Configuración global, calendario y supuestos regulatorios
├── requirements.txt            # Dependencias del proyecto
├── models/
│   ├── inputs.py               # Modelos Pydantic de validación de parámetros de entrada
│   └── results.py              # Estructuras de resultados y diagnósticos del solver
├── data/
│   ├── loaders.py              # Importación robusta de Excel/CSV (coma decimal, detección)
│   ├── calendar.py             # Calendario peninsular, periodos P1-P6 y festivos nacionales
│   ├── validation.py           # Validaciones de coherencia y disponibilidad de datos
│   └── synthetic.py            # Generador de perfiles sintéticos de ejemplo
├── demand/
│   └── thermal_demand.py       # Módulo común de caracterización de la demanda térmica
├── baseline/
│   └── gas_reference.py        # Línea base de caldera de gas natural (PCS/PCI, CO2, costes)
├── supply/
│   └── electricity_supply.py   # Series de costes eléctricos, capacidad de red y PV
├── optimization/
│   ├── common.py               # Envoltorio del solver HiGHS, tolerancia y prioridad PV
│   ├── eboiler.py              # Modelo LP para Alternativa 1 (E-Boiler)
│   ├── tes.py                  # Modelo LP/MILP para Alternativa 2 (Almacenamiento TES)
│   └── sizing.py               # Algoritmos de dimensionamiento óptimo y sensibilidades
├── results/
│   ├── eboiler_results.py      # Postproceso y balances horarios/anuales de E-Boiler
│   ├── tes_results.py          # Postproceso y métricas de almacenamiento TES
│   └── comparisons.py          # Orquestador de simulación y comparativas de escenarios
├── finance/
│   └── project_economics.py    # Evaluación financiera (VAN, TIR, Payback, LCOH)
├── reporting/
│   ├── dashboards.py           # Gráficos interactivos Plotly
│   └── exports.py              # Exportación multillibro a Excel y CSV
└── tests/
    └── test_core.py            # Pruebas unitarias y de integración
```
