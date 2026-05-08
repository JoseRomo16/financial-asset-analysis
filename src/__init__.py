"""Paquete `src` del proyecto Financial Asset Analysis.

Expone las funciones públicas de los módulos ``data_loader``, ``metrics`` y
``visualizer`` para facilitar imports directos desde notebooks y scripts.
"""

from src.data_loader import (
    TICKERS_DEFAULT,
    cargar_precios_local,
    descargar_precios,
    resumen_datos,
)
from src.metrics import (
    DIAS_HABILES,
    calcular_drawdown,
    calcular_max_drawdown,
    calcular_retorno_acumulado,
    calcular_retorno_anualizado,
    calcular_retorno_total,
    calcular_retornos,
    calcular_sharpe,
    calcular_var,
    calcular_volatilidad,
    calcular_volatilidad_total,
    resumen_metricas,
)
from src.visualizer import (
    PALETA,
    graficar_correlacion,
    graficar_distribucion_retornos,
    graficar_drawdown,
    graficar_precios,
    graficar_retorno_acumulado,
    graficar_retorno_interactivo,
    graficar_volatilidad_interactiva,
)

__all__ = [
    # data_loader
    "TICKERS_DEFAULT",
    "descargar_precios",
    "cargar_precios_local",
    "resumen_datos",
    # metrics
    "DIAS_HABILES",
    "calcular_retornos",
    "calcular_retorno_acumulado",
    "calcular_retorno_total",
    "calcular_retorno_anualizado",
    "calcular_volatilidad",
    "calcular_volatilidad_total",
    "calcular_drawdown",
    "calcular_max_drawdown",
    "calcular_sharpe",
    "calcular_var",
    "resumen_metricas",
    # visualizer
    "PALETA",
    "graficar_precios",
    "graficar_retorno_acumulado",
    "graficar_drawdown",
    "graficar_correlacion",
    "graficar_distribucion_retornos",
    "graficar_retorno_interactivo",
    "graficar_volatilidad_interactiva",
]
