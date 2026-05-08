"""Módulo de métricas financieras de rendimiento y riesgo.

Implementa cálculos estándar sobre paneles de precios diarios: retornos
(aritméticos y logarítmicos), volatilidad, drawdown, Sharpe y VaR
histórico. Todas las funciones aceptan tanto un panel multi-activo como
un único ticker (Series o DataFrame de una columna).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

# Constante estándar de días hábiles bursátiles al año.
DIAS_HABILES: int = 252

DataFrameLike = Union[pd.DataFrame, pd.Series]


def _asegurar_dataframe(datos: DataFrameLike) -> pd.DataFrame:
    """Convierte una Series a DataFrame de una columna sin alterar DataFrames.

    yfinance devuelve estructuras distintas para uno y varios tickers; esta
    función uniformiza la entrada para que el resto de las métricas pueda
    asumir un DataFrame.
    """
    if isinstance(datos, pd.Series):
        nombre = datos.name if datos.name is not None else "activo"
        return datos.to_frame(name=str(nombre))
    return datos


def calcular_retornos(df: DataFrameLike, metodo: str = "aritmetico") -> pd.DataFrame:
    """Calcula retornos diarios sobre un panel de precios.

    Parameters
    ----------
    df : pd.DataFrame o pd.Series
        Precios con índice temporal y una columna por activo.
    metodo : {'aritmetico', 'logaritmico'}
        - ``'aritmetico'``: r_t = P_t / P_{t-1} − 1.
        - ``'logaritmico'``: r_t = ln(P_t / P_{t-1}).

    Returns
    -------
    pd.DataFrame
        Retornos diarios. La primera fila se elimina (no hay r_0 definido),
        por lo que la longitud es ``len(df) - 1``.

    Raises
    ------
    ValueError
        Si ``metodo`` no es uno de los valores permitidos.
    """
    df = _asegurar_dataframe(df)

    if metodo == "aritmetico":
        retornos = df.pct_change()
    elif metodo == "logaritmico":
        retornos = np.log(df / df.shift(1))
    else:
        raise ValueError(
            f"Método '{metodo}' no soportado. Usa 'aritmetico' o 'logaritmico'."
        )

    return retornos.dropna(how="all")


def calcular_retorno_acumulado(retornos: DataFrameLike) -> pd.DataFrame:
    """Construye la serie de retornos acumulados (crecimiento de 1 unidad).

    Equivale a ``(1 + r).cumprod()``: una unidad invertida al inicio del
    periodo se transforma en este valor al cierre de cada día.
    """
    retornos = _asegurar_dataframe(retornos)
    return (1 + retornos).cumprod()


def calcular_retorno_total(retornos: DataFrameLike) -> pd.Series:
    """Retorno total compuesto del periodo, expresado como decimal.

    Returns
    -------
    pd.Series
        Una entrada por activo: ``∏(1 + r_t) − 1``.
    """
    retornos = _asegurar_dataframe(retornos)
    return (1 + retornos).prod() - 1


def calcular_retorno_anualizado(retornos: DataFrameLike) -> pd.Series:
    """Retorno geométrico anualizado asumiendo 252 días hábiles.

    Fórmula: ``(1 + r_total) ** (252 / n_observaciones) − 1``.
    """
    retornos = _asegurar_dataframe(retornos)
    n_obs = retornos.count()
    retorno_total = (1 + retornos).prod()
    # ``np.power`` devuelve ndarray; reconstruimos la Series con su índice.
    valores = np.power(retorno_total.values, DIAS_HABILES / n_obs.values) - 1
    return pd.Series(valores, index=retorno_total.index)


def calcular_volatilidad(
    retornos: DataFrameLike, ventana: int = 21, anualizar: bool = True
) -> pd.DataFrame:
    """Volatilidad rodante (desviación estándar móvil de retornos).

    Parameters
    ----------
    retornos : DataFrameLike
        Retornos diarios.
    ventana : int
        Tamaño de la ventana móvil en días (por defecto 21 ≈ 1 mes bursátil).
    anualizar : bool
        Si es ``True`` multiplica por ``√252`` para anualizar.
    """
    retornos = _asegurar_dataframe(retornos)
    vol = retornos.rolling(window=ventana).std()
    if anualizar:
        vol = vol * np.sqrt(DIAS_HABILES)
    return vol


def calcular_volatilidad_total(retornos: DataFrameLike) -> pd.Series:
    """Volatilidad anualizada del periodo completo.

    Equivale a ``std(retornos) * √252``.
    """
    retornos = _asegurar_dataframe(retornos)
    return retornos.std() * np.sqrt(DIAS_HABILES)


def calcular_drawdown(retornos: DataFrameLike) -> pd.DataFrame:
    """Drawdown diario respecto al máximo histórico acumulado.

    Drawdown = (V_t / max_{s ≤ t} V_s) − 1, donde V_t es el valor del
    portafolio (retorno acumulado). Es siempre ≤ 0.
    """
    retornos = _asegurar_dataframe(retornos)
    valor = (1 + retornos).cumprod()
    maximo_acumulado = valor.cummax()
    return valor / maximo_acumulado - 1


def calcular_max_drawdown(retornos: DataFrameLike) -> pd.Series:
    """Máximo drawdown del periodo (valor más negativo de la serie).

    Returns
    -------
    pd.Series
        Una entrada por activo, siempre ≤ 0.
    """
    return calcular_drawdown(retornos).min()


def calcular_sharpe(
    retornos: DataFrameLike, tasa_libre_riesgo: float = 0.05
) -> pd.Series:
    """Ratio de Sharpe anualizado.

    Fórmula: ``(retorno_anualizado − tasa_libre_riesgo) / volatilidad_anualizada``.

    Parameters
    ----------
    retornos : DataFrameLike
        Retornos diarios.
    tasa_libre_riesgo : float
        Tasa libre de riesgo anual expresada como decimal (0.05 = 5%).
    """
    retorno_anual = calcular_retorno_anualizado(retornos)
    vol_anual = calcular_volatilidad_total(retornos)
    # Evitamos división por cero devolviendo NaN explícito.
    return (retorno_anual - tasa_libre_riesgo) / vol_anual.replace(0, np.nan)


def calcular_var(
    retornos: DataFrameLike, confianza: float = 0.95
) -> pd.Series:
    """Value at Risk histórico diario, expresado como pérdida positiva.

    Para un nivel de confianza ``c``, el VaR es el percentil ``(1 − c)`` de
    la distribución de retornos, devuelto como número positivo (pérdida).

    Parameters
    ----------
    retornos : DataFrameLike
        Retornos diarios.
    confianza : float
        Nivel de confianza en el rango (0, 1). Por defecto 0.95.

    Returns
    -------
    pd.Series
        VaR positivo por activo. Un VaR de 0.025 significa que con 95% de
        confianza la pérdida diaria no excederá el 2.5%.
    """
    retornos = _asegurar_dataframe(retornos)
    percentil = retornos.quantile(1 - confianza)
    # Convertimos a pérdida positiva: VaR ≥ 0 cuando hay riesgo de caída.
    return -percentil


def resumen_metricas(
    retornos: DataFrameLike, tasa_libre_riesgo: float = 0.05
) -> pd.DataFrame:
    """Tabla resumen con todas las métricas clave por activo.

    Columnas: ``retorno_total``, ``retorno_anualizado``,
    ``volatilidad_anualizada``, ``max_drawdown``, ``sharpe``, ``var_95``.
    """
    retornos = _asegurar_dataframe(retornos)
    resumen = pd.DataFrame(
        {
            "retorno_total": calcular_retorno_total(retornos),
            "retorno_anualizado": calcular_retorno_anualizado(retornos),
            "volatilidad_anualizada": calcular_volatilidad_total(retornos),
            "max_drawdown": calcular_max_drawdown(retornos),
            "sharpe": calcular_sharpe(retornos, tasa_libre_riesgo),
            "var_95": calcular_var(retornos, confianza=0.95),
        }
    )
    resumen.index.name = "ticker"
    return resumen
