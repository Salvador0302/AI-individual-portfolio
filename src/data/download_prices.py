"""Descarga precios históricos con yfinance para todos los tickers del proyecto.

Este módulo utiliza la lista de `TICKERS` definida en `src.config` y
descarga precios diarios ajustados de los últimos 5 años, devolviendo
un DataFrame con índice de fecha y columnas multi-índice `(ticker, campo)`.
Los datos combinados se guardan en `data/raw/prices.parquet`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.config import RAW_DATA_DIR, TICKERS


def download_prices(tickers: list[str] | None = None, years: int = 5) -> pd.DataFrame:
    """Descarga precios ajustados diarios para los tickers indicados.

    Parámetros
    ----------
    tickers: lista de símbolos a descargar. Si es None se usa `config.TICKERS`.
    years: horizonte histórico en años (por defecto 5).

    Devuelve
    --------
    DataFrame con índice de fecha y columnas multi-índice (ticker, campo).
    """

    if tickers is None:
        tickers = TICKERS

    end = datetime.today().date()
    start = end - timedelta(days=365 * years)

    # yfinance permite descargar varios tickers en un solo llamado
    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )

    # Si sólo hay un ticker, yfinance devuelve un DataFrame simple; lo normalizamos
    if isinstance(data.columns, pd.MultiIndex):
        # ya viene como (ticker, campo)
        df = data
    else:
        # un solo ticker, creamos multi-índice
        ticker = tickers[0]
        df = pd.concat({ticker: data}, axis=1)

    df.index.name = "date"
    return df


def save_prices_parquet(df: pd.DataFrame, filename: str = "prices.parquet") -> str:
    """Guarda el DataFrame de precios en `data/raw` como parquet."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / filename
    df.to_parquet(path)
    return str(path)


if __name__ == "__main__":
    prices_df = download_prices()
    output_path = save_prices_parquet(prices_df)
    print(f"Datos de precios guardados en: {output_path}")
