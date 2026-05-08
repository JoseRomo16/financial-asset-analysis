"""Módulo de descarga y carga de precios históricos de activos financieros.

Este módulo encapsula la interacción con `yfinance` y la persistencia local
de los datos crudos y procesados, manteniendo todas las rutas relativas a
la raíz del proyecto mediante `pathlib.Path`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import yfinance as yf

# Raíz del proyecto: dos niveles arriba de este archivo (src/data_loader.py).
RUTA_PROYECTO: Path = Path(__file__).resolve().parent.parent
RUTA_DATA_RAW: Path = RUTA_PROYECTO / "data" / "raw"
RUTA_DATA_PROCESSED: Path = RUTA_PROYECTO / "data" / "processed"

# Tickers por defecto del portafolio diversificado.
# Combina renta variable global, oro, cripto, divisas, índice local
# colombiano (COLCAP) y café como commodity exportador.
TICKERS_DEFAULT: List[str] = [
    "AAPL",       # Apple — acción tech (NASDAQ)
    "MSFT",       # Microsoft — acción tech (NASDAQ)
    "SPY",        # SPDR S&P 500 — ETF de mercado amplio EE.UU.
    "GLD",        # SPDR Gold Shares — ETF de oro físico
    "BTC-USD",    # Bitcoin — criptomoneda
    "USDCOP=X",   # Tasa de cambio USD/COP
    "EURUSD=X",   # Tasa de cambio EUR/USD
    "^COLCAP",    # Índice COLCAP — bolsa de Colombia
    "KC=F",       # Café Arabica — futuros (commodity)
]


def _normalizar_panel_precios(
    datos: pd.DataFrame, tickers: Sequence[str]
) -> pd.DataFrame:
    """Convierte la respuesta de yfinance a un DataFrame plano de precios.

    yfinance devuelve estructuras distintas según se solicite uno o varios
    tickers. Esta función uniformiza la salida a un DataFrame con índice
    temporal y una columna por ticker (precio de cierre ajustado).

    Parameters
    ----------
    datos : pd.DataFrame
        Salida cruda de ``yf.download``.
    tickers : Sequence[str]
        Lista de tickers solicitados.

    Returns
    -------
    pd.DataFrame
        DataFrame con índice de fechas y columnas = tickers.
    """
    # Caso multi-ticker: yfinance devuelve columnas con MultiIndex.
    if isinstance(datos.columns, pd.MultiIndex):
        # Preferimos 'Adj Close' si está disponible; si no, 'Close'.
        nivel_superior = datos.columns.get_level_values(0).unique().tolist()
        campo_precio = "Adj Close" if "Adj Close" in nivel_superior else "Close"
        precios = datos[campo_precio].copy()
    else:
        # Caso de un solo ticker: columnas planas (Open, High, Low, Close, ...).
        campo_precio = "Adj Close" if "Adj Close" in datos.columns else "Close"
        precios = datos[[campo_precio]].copy()
        precios.columns = [tickers[0]]

    # Aseguramos el orden de columnas según los tickers solicitados.
    columnas_existentes = [t for t in tickers if t in precios.columns]
    precios = precios[columnas_existentes]
    precios.index = pd.to_datetime(precios.index)
    precios.index.name = "fecha"
    return precios


def descargar_precios(
    tickers: Optional[Iterable[str]] = None,
    inicio: str = "2020-01-01",
    fin: Optional[str] = None,
    guardar: bool = True,
) -> pd.DataFrame:
    """Descarga precios históricos diarios desde Yahoo Finance.

    Parameters
    ----------
    tickers : Iterable[str], opcional
        Lista de símbolos a descargar. Si es ``None`` se usa
        :data:`TICKERS_DEFAULT`.
    inicio : str
        Fecha de inicio en formato 'YYYY-MM-DD'.
    fin : str, opcional
        Fecha final en formato 'YYYY-MM-DD'. Si es ``None`` se usa la fecha
        actual del sistema.
    guardar : bool
        Si es ``True`` persiste el panel crudo en ``data/raw/precios.csv`` y
        la versión con forward-fill en ``data/processed/precios.csv``.

    Returns
    -------
    pd.DataFrame
        Panel de precios con índice de fechas y una columna por ticker.
        Los valores faltantes se rellenan con forward fill.
    """
    if tickers is None:
        tickers = TICKERS_DEFAULT
    tickers = list(tickers)

    # Descarga: progreso silenciado para no contaminar la salida en notebooks.
    datos = yf.download(
        tickers=tickers,
        start=inicio,
        end=fin,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )

    if datos is None or datos.empty:
        raise ValueError(
            "yfinance no devolvió datos. Verifica los tickers, fechas o "
            "conexión a internet."
        )

    precios = _normalizar_panel_precios(datos, tickers)

    if guardar:
        RUTA_DATA_RAW.mkdir(parents=True, exist_ok=True)
        RUTA_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        precios.to_csv(RUTA_DATA_RAW / "precios.csv")

    # Forward fill para rellenar días no operados (cripto vs. acciones).
    precios = precios.ffill().dropna(how="all")

    if guardar:
        precios.to_csv(RUTA_DATA_PROCESSED / "precios.csv")

    return precios


def cargar_precios_local(ruta: Optional[Path] = None) -> pd.DataFrame:
    """Carga un panel de precios previamente descargado a CSV.

    Parameters
    ----------
    ruta : pathlib.Path, opcional
        Ruta al archivo CSV. Si es ``None`` se asume
        ``data/processed/precios.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame con índice de fechas y columnas por ticker.
    """
    if ruta is None:
        ruta = RUTA_DATA_PROCESSED / "precios.csv"
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de precios en: {ruta}. "
            "Ejecuta primero `descargar_precios(...)`."
        )

    precios = pd.read_csv(ruta, index_col=0, parse_dates=True)
    precios.index.name = "fecha"
    return precios


def resumen_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Construye un resumen tabular del panel de precios.

    Incluye: número de observaciones, fecha inicial, fecha final, valores
    nulos, precio mínimo, precio máximo y precio medio por ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Panel de precios con columnas por ticker.

    Returns
    -------
    pd.DataFrame
        Resumen con una fila por ticker.
    """
    # Soporte para el caso de un solo ticker (Series → DataFrame).
    if isinstance(df, pd.Series):
        df = df.to_frame()

    resumen = pd.DataFrame(
        {
            "observaciones": df.count(),
            "fecha_inicio": df.apply(lambda s: s.first_valid_index()),
            "fecha_fin": df.apply(lambda s: s.last_valid_index()),
            "nulos": df.isna().sum(),
            "precio_min": df.min(),
            "precio_max": df.max(),
            "precio_medio": df.mean(),
        }
    )
    resumen.index.name = "ticker"
    return resumen
