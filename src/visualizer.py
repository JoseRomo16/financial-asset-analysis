"""Módulo de visualización estática (matplotlib + seaborn) e interactiva (plotly).

Todas las funciones aceptan el flag ``guardar`` para persistir la figura en
``reports/figures/`` (PNG para gráficos estáticos, HTML para interactivos).
La paleta corporativa del proyecto se centraliza en :data:`PALETA`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# Paleta de cinco colores: azul, verde, rojo, ámbar, violeta.
PALETA: List[str] = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]

# Carpeta de salida (relativa al proyecto, no al notebook que invoque).
RUTA_PROYECTO: Path = Path(__file__).resolve().parent.parent
RUTA_FIGURAS: Path = RUTA_PROYECTO / "reports" / "figures"

# Configuración global de estilo.
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

DataFrameLike = Union[pd.DataFrame, pd.Series]


def _asegurar_dataframe(datos: DataFrameLike) -> pd.DataFrame:
    """Normaliza Series a DataFrame de una columna."""
    if isinstance(datos, pd.Series):
        nombre = datos.name if datos.name is not None else "activo"
        return datos.to_frame(name=str(nombre))
    return datos


def _ruta_salida(nombre_archivo: str) -> Path:
    """Construye la ruta de salida garantizando que la carpeta exista."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    return RUTA_FIGURAS / nombre_archivo


def _colores_para(columnas: List[str]) -> List[str]:
    """Devuelve una lista de colores reciclando la paleta corporativa."""
    return [PALETA[i % len(PALETA)] for i in range(len(columnas))]


# ---------------------------------------------------------------------------
# Visualizaciones estáticas (matplotlib + seaborn)
# ---------------------------------------------------------------------------


def graficar_precios(df: DataFrameLike, guardar: bool = False) -> plt.Figure:
    """Grafica precios normalizados a base 100 para comparar trayectorias."""
    df = _asegurar_dataframe(df).dropna(how="all")
    base100 = df.divide(df.iloc[0]).multiply(100)

    fig, ax = plt.subplots(figsize=(12, 6))
    for color, columna in zip(_colores_para(list(base100.columns)), base100.columns):
        ax.plot(base100.index, base100[columna], label=columna, color=color, linewidth=1.6)

    ax.set_title("Precios normalizados (base 100)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (base 100)")
    ax.legend(loc="upper left")
    fig.tight_layout()

    if guardar:
        fig.savefig(_ruta_salida("precios_normalizados.png"), bbox_inches="tight")
    return fig


def graficar_retorno_acumulado(
    ret_acum: DataFrameLike, guardar: bool = False
) -> plt.Figure:
    """Grafica el retorno acumulado expresado en porcentaje."""
    ret_acum = _asegurar_dataframe(ret_acum)
    pct = (ret_acum - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for color, columna in zip(_colores_para(list(pct.columns)), pct.columns):
        ax.plot(pct.index, pct[columna], label=columna, color=color, linewidth=1.6)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Retorno acumulado (%)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Retorno acumulado (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()

    if guardar:
        fig.savefig(_ruta_salida("retorno_acumulado.png"), bbox_inches="tight")
    return fig


def graficar_drawdown(drawdown: DataFrameLike, guardar: bool = False) -> plt.Figure:
    """Grafica drawdown rellenando el área bajo la curva (siempre ≤ 0)."""
    drawdown = _asegurar_dataframe(drawdown) * 100
    columnas = list(drawdown.columns)
    colores = _colores_para(columnas)

    fig, axes = plt.subplots(
        nrows=len(columnas), ncols=1,
        figsize=(12, 2.5 * len(columnas)),
        sharex=True,
    )
    if len(columnas) == 1:
        axes = [axes]

    for ax, columna, color in zip(axes, columnas, colores):
        serie = drawdown[columna]
        ax.fill_between(serie.index, serie.values, 0, color=color, alpha=0.35)
        ax.plot(serie.index, serie.values, color=color, linewidth=1.2)
        ax.set_title(f"Drawdown — {columna}")
        ax.set_ylabel("Drawdown (%)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Fecha")
    fig.tight_layout()

    if guardar:
        fig.savefig(_ruta_salida("drawdown.png"), bbox_inches="tight")
    return fig


def graficar_correlacion(
    retornos: DataFrameLike, guardar: bool = False
) -> plt.Figure:
    """Heatmap triangular inferior de la matriz de correlaciones de retornos."""
    retornos = _asegurar_dataframe(retornos)
    matriz = retornos.corr()
    mascara = np.triu(np.ones_like(matriz, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matriz,
        mask=mascara,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlación"},
        ax=ax,
    )
    ax.set_title("Matriz de correlación de retornos diarios")
    fig.tight_layout()

    if guardar:
        fig.savefig(_ruta_salida("correlacion.png"), bbox_inches="tight")
    return fig


def graficar_distribucion_retornos(
    retornos: DataFrameLike, guardar: bool = False
) -> plt.Figure:
    """Histograma + KDE de retornos diarios para cada activo."""
    retornos = _asegurar_dataframe(retornos)
    columnas = list(retornos.columns)
    colores = _colores_para(columnas)

    n = len(columnas)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    # Aplanamos para iterar uniformemente sin importar la forma de la grilla.
    ejes = np.atleast_1d(axes).flatten()

    for ax, columna, color in zip(ejes, columnas, colores):
        sns.histplot(
            retornos[columna].dropna(),
            kde=True,
            color=color,
            stat="density",
            ax=ax,
        )
        ax.set_title(f"Distribución — {columna}")
        ax.set_xlabel("Retorno diario")
        ax.set_ylabel("Densidad")

    # Apagamos ejes sobrantes cuando la grilla no cuadra exactamente.
    for ax in ejes[len(columnas):]:
        ax.set_visible(False)

    fig.tight_layout()

    if guardar:
        fig.savefig(_ruta_salida("distribucion_retornos.png"), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Visualizaciones interactivas (plotly)
# ---------------------------------------------------------------------------


def graficar_retorno_interactivo(
    ret_acum: DataFrameLike, guardar: bool = False
) -> go.Figure:
    """Retorno acumulado interactivo con range selector y rangeslider."""
    ret_acum = _asegurar_dataframe(ret_acum)
    pct = (ret_acum - 1) * 100
    colores = _colores_para(list(pct.columns))

    fig = go.Figure()
    for color, columna in zip(colores, pct.columns):
        fig.add_trace(
            go.Scatter(
                x=pct.index,
                y=pct[columna],
                mode="lines",
                name=columna,
                line=dict(color=color, width=1.8),
            )
        )

    fig.update_layout(
        title="Retorno acumulado (%) — interactivo",
        xaxis_title="Fecha",
        yaxis_title="Retorno acumulado (%)",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1A", step="year", stepmode="backward"),
                    dict(count=3, label="3A", step="year", stepmode="backward"),
                    dict(step="all", label="Todo"),
                ]
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )

    if guardar:
        fig.write_html(_ruta_salida("retorno_acumulado_interactivo.html"))
    return fig


def graficar_volatilidad_interactiva(
    vol: DataFrameLike, guardar: bool = False
) -> go.Figure:
    """Volatilidad rodante interactiva (anualizada en %)."""
    vol = _asegurar_dataframe(vol) * 100
    colores = _colores_para(list(vol.columns))

    fig = go.Figure()
    for color, columna in zip(colores, vol.columns):
        fig.add_trace(
            go.Scatter(
                x=vol.index,
                y=vol[columna],
                mode="lines",
                name=columna,
                line=dict(color=color, width=1.8),
            )
        )

    fig.update_layout(
        title="Volatilidad rodante anualizada (%)",
        xaxis_title="Fecha",
        yaxis_title="Volatilidad anualizada (%)",
        template="plotly_white",
        hovermode="x unified",
    )

    if guardar:
        fig.write_html(_ruta_salida("volatilidad_interactiva.html"))
    return fig
