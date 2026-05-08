# Análisis de Activos Financieros

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![pandas](https://img.shields.io/badge/pandas-2.2.1-150458)
![yfinance](https://img.shields.io/badge/yfinance-0.2.38-1f77b4)
![License](https://img.shields.io/badge/license-MIT-green)

Proyecto de análisis cuantitativo de activos financieros que descarga precios históricos de mercado, calcula métricas clave de rendimiento y riesgo (retornos, volatilidad, drawdown, Sharpe, VaR), y produce visualizaciones estáticas e interactivas para evaluar y comparar el comportamiento de un portafolio diversificado.

## Objetivos

- Construir un flujo reproducible de descarga, limpieza y persistencia de precios históricos.
- Implementar un módulo de métricas financieras documentado y cubierto por pruebas unitarias.
- Comparar activos de distinta naturaleza (acciones, ETFs, oro, criptomonedas) bajo un marco común.
- Producir visualizaciones estáticas y dashboards interactivos para análisis exploratorio.
- Establecer una base reutilizable de código `src/` para los siguientes proyectos del portafolio.

## Estructura del proyecto

```
financial-asset-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_descarga_datos.ipynb
│   ├── 02_analisis_retornos.ipynb
│   ├── 03_riesgo_volatilidad.ipynb
│   └── 04_visualizaciones.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── visualizer.py
├── reports/
│   └── figures/
├── tests/
│   └── test_metrics.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Activos analizados

Portafolio diversificado entre clases de activos globales y locales colombianos.

| Ticker     | Nombre                              | Tipo                       |
|------------|-------------------------------------|----------------------------|
| AAPL       | Apple Inc.                          | Acción (tecnología)        |
| MSFT       | Microsoft Corp.                     | Acción (tecnología)        |
| SPY        | SPDR S&P 500 ETF Trust              | ETF de índice (EE.UU.)     |
| GLD        | SPDR Gold Shares                    | ETF de oro                 |
| BTC-USD    | Bitcoin                             | Criptomoneda               |
| USDCOP=X   | Tasa de cambio USD / COP            | Divisa                     |
| EURUSD=X   | Tasa de cambio EUR / USD            | Divisa                     |
| ^COLCAP    | Índice COLCAP (Bolsa de Colombia)   | Índice bursátil local      |
| KC=F       | Café Arabica — futuros              | Commodity (exportador)     |

## Instalación

```bash
git clone https://github.com/Jose-Romo16/financial-asset-analysis.git
cd financial-asset-analysis

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Uso rápido

```python
from src.data_loader import descargar_precios, TICKERS_DEFAULT
from src.metrics import resumen_metricas, calcular_retornos
from src.visualizer import graficar_retorno_acumulado

# Usa el portafolio diversificado por defecto (acciones, ETFs, oro, cripto,
# divisas, COLCAP y café). También puedes pasar tu propia lista de tickers.
precios = descargar_precios(
    tickers=TICKERS_DEFAULT,
    inicio="2020-01-01",
    fin="2024-12-31",
    guardar=True,
)

retornos = calcular_retornos(precios, metodo="aritmetico")
resumen = resumen_metricas(retornos, tasa_libre_riesgo=0.05)
print(resumen)

graficar_retorno_acumulado((1 + retornos).cumprod(), guardar=True)
```

## Métricas implementadas

| Métrica                  | Descripción                                                                 |
|--------------------------|------------------------------------------------------------------------------|
| Retorno diario           | Variación porcentual día a día (aritmético o logarítmico).                  |
| Retorno acumulado        | Producto acumulado de (1 + retornos) — crecimiento de 1 unidad invertida.   |
| Volatilidad anualizada   | Desviación estándar de retornos × √252.                                     |
| Drawdown máximo          | Mayor caída desde un máximo histórico hasta el siguiente mínimo.            |
| Ratio de Sharpe          | (Retorno anualizado − tasa libre de riesgo) / volatilidad anualizada.       |
| VaR 95%                  | Pérdida diaria máxima esperada con 95% de confianza (histórico).            |

## Aprendizajes clave

- Diferencia práctica entre retornos aritméticos y logarítmicos al agregar series temporales.
- Importancia de la anualización (`× 252`, `× √252`) para comparar activos de distinta frecuencia.
- Limitaciones del VaR histórico frente a colas gruesas y cambios de régimen.
- Utilidad del drawdown como medida de riesgo psicológico complementaria a la volatilidad.
- Beneficios de diversificar entre clases de activos correlacionados negativamente.

## Próximos pasos

- [ ] Incorporar Conditional VaR (CVaR / Expected Shortfall).
- [ ] Optimización de portafolio mediante frontera eficiente de Markowitz.
- [ ] Backtesting de estrategias simples (medias móviles, rebalanceo periódico).
- [ ] Integración con datos macroeconómicos (tasas, inflación) desde FRED.
- [ ] Despliegue de un dashboard interactivo en Streamlit.

## Autor

**Jose Luis Romo Melo** — Economista | Data Scientist en formación

---

*Proyecto 1 de 8 — Portafolio de Data Science Financiero*
