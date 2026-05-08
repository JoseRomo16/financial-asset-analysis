"""Pruebas unitarias para el módulo `src.metrics`.

Usa una fixture pequeña con dos activos controlados:
- Activo "A": serie creciente (100 → 120) con un retroceso intermedio.
- Activo "B": serie decreciente (200 → 180) con un repunte intermedio.

Las trayectorias incluyen al menos un retorno negativo en cada activo, lo
que asegura que el cuantil al 5% sea negativo y, por tanto, el VaR positivo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Permite ejecutar `pytest tests/test_metrics.py` desde la raíz del proyecto
# sin necesidad de instalar el paquete (añade la raíz al path).
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUTA_PROYECTO))

from src.metrics import (  # noqa: E402  (import after sys.path manipulation)
    calcular_drawdown,
    calcular_max_drawdown,
    calcular_retorno_total,
    calcular_retornos,
    calcular_sharpe,
    calcular_var,
    calcular_volatilidad_total,
)


@pytest.fixture
def precios() -> pd.DataFrame:
    """Panel de 5 puntos: A sube 100→120, B baja 200→180."""
    fechas = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "A": [100.0, 110.0, 105.0, 115.0, 120.0],
            "B": [200.0, 195.0, 200.0, 190.0, 180.0],
        },
        index=fechas,
    )


# ---------------------------------------------------------------------------
# Validación de forma y valores
# ---------------------------------------------------------------------------


def test_retornos_tiene_una_fila_menos(precios: pd.DataFrame) -> None:
    retornos = calcular_retornos(precios)
    assert len(retornos) == len(precios) - 1
    assert list(retornos.columns) == list(precios.columns)


def test_primer_retorno_aritmetico_correcto(precios: pd.DataFrame) -> None:
    retornos = calcular_retornos(precios, metodo="aritmetico")
    # (110 - 100) / 100 = 0.10
    assert retornos["A"].iloc[0] == pytest.approx(0.10)
    # (195 - 200) / 200 = -0.025
    assert retornos["B"].iloc[0] == pytest.approx(-0.025)


# ---------------------------------------------------------------------------
# Retornos totales: signo según trayectoria
# ---------------------------------------------------------------------------


def test_retorno_total_positivo_para_activo_creciente(precios: pd.DataFrame) -> None:
    retornos = calcular_retornos(precios)
    total = calcular_retorno_total(retornos)
    assert total["A"] > 0
    assert total["A"] == pytest.approx(0.20)


def test_retorno_total_negativo_para_activo_decreciente(precios: pd.DataFrame) -> None:
    retornos = calcular_retornos(precios)
    total = calcular_retorno_total(retornos)
    assert total["B"] < 0
    assert total["B"] == pytest.approx(-0.10)


# ---------------------------------------------------------------------------
# Volatilidad y drawdown: signos esperados
# ---------------------------------------------------------------------------


def test_volatilidad_siempre_positiva(precios: pd.DataFrame) -> None:
    retornos = calcular_retornos(precios)
    vol = calcular_volatilidad_total(retornos)
    assert (vol > 0).all()


def test_drawdown_siempre_menor_o_igual_a_cero(precios: pd.DataFrame) -> None:
    retornos = calcular_retornos(precios)
    dd = calcular_drawdown(retornos)
    # La primera fila puede ser 0 (máximo inicial); ninguna fila debe ser > 0.
    assert (dd <= 1e-12).all().all()


def test_max_drawdown_siempre_menor_o_igual_a_cero(precios: pd.DataFrame) -> None:
    retornos = calcular_retornos(precios)
    max_dd = calcular_max_drawdown(retornos)
    assert (max_dd <= 0).all()


# ---------------------------------------------------------------------------
# VaR: signo de pérdida
# ---------------------------------------------------------------------------


def test_var_siempre_positivo(precios: pd.DataFrame) -> None:
    retornos = calcular_retornos(precios)
    var = calcular_var(retornos, confianza=0.95)
    # VaR se reporta como pérdida positiva (o exactamente cero como caso límite).
    assert (var >= 0).all()


# ---------------------------------------------------------------------------
# Comparación entre métodos de retorno
# ---------------------------------------------------------------------------


def test_retornos_log_difieren_de_aritmeticos(precios: pd.DataFrame) -> None:
    aritm = calcular_retornos(precios, metodo="aritmetico")
    logar = calcular_retornos(precios, metodo="logaritmico")
    # Misma forma, pero valores distintos (excepto en el caso trivial r = 0).
    assert aritm.shape == logar.shape
    diferencias = (aritm - logar).abs()
    assert (diferencias > 1e-9).all().all()


def test_metodo_invalido_lanza_value_error(precios: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        calcular_retornos(precios, metodo="exponencial")


# ---------------------------------------------------------------------------
# Sharpe: finitud
# ---------------------------------------------------------------------------


def test_sharpe_es_finito(precios: pd.DataFrame) -> None:
    retornos = calcular_retornos(precios)
    sharpe = calcular_sharpe(retornos, tasa_libre_riesgo=0.05)
    assert np.isfinite(sharpe.values).all()
