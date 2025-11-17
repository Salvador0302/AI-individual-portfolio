"""Construcción de features cuantitativos para precios y sentimiento.

Este módulo carga:

- ``data/raw/prices.parquet``: precios diarios por ticker (MultiIndex columnas).
- ``data/processed/news_sentiment.parquet``: sentimiento por titular y ticker.

Y genera un DataFrame de features por fecha y ticker con, entre otros:

- Retornos diarios y logarítmicos.
- Medias móviles a 5 y 20 días.
- Volatilidad móvil a 20 días.
- Indicador agregado de sentimiento (media de los últimos N días).

El resultado se guarda en ``data/processed/features.parquet``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


PRICES_FILE = "prices.parquet"
NEWS_SENTIMENT_FILE = "news_sentiment.parquet"
FEATURES_FILE = "features.parquet"


def _load_prices() -> pd.DataFrame:
    """Carga el parquet de precios y lo normaliza a un formato largo.

    Se espera un índice de fecha y columnas MultiIndex (ticker, campo).
    Devuelve un DataFrame con columnas: date, ticker, open, high, low,
    close, adj_close, volume (las que existan en los datos).
    """

    path = RAW_DATA_DIR / PRICES_FILE
    df = pd.read_parquet(path)

    # df tiene columnas MultiIndex: (campo) o (ticker, campo)
    if isinstance(df.columns, pd.MultiIndex):
        # Reorganizamos a formato largo
        df_long = (
            df.stack(0)  # nivel de ticker
            .rename_axis(["date", "ticker"])  # nombres de índices
            .reset_index()
        )
    else:
        # Caso raro: una sola serie sin MultiIndex, usamos ticker genérico
        df_long = df.copy()
        df_long["ticker"] = "UNKNOWN"
        df_long = df_long.reset_index().rename(columns={"index": "date"})

    # Normalizamos nombres de columnas
    rename_map = {}
    for col in df_long.columns:
        if isinstance(col, str):
            if col.lower() == "adj close":
                rename_map[col] = "adj_close"
        # otras columnas se mantienen igual

    df_long = df_long.rename(columns=rename_map)

    return df_long


def _load_news_sentiment() -> pd.DataFrame:
    """Carga el parquet de sentimiento y agrega por fecha y ticker.

    Se asume que el parquet contiene al menos: ticker, sentiment_label,
    sentiment_score y opcionalmente una columna de fecha o timestamp. En
    caso de no disponer de fecha, se trata todo como un único día y se
    usa la media por ticker.
    """

    path = PROCESSED_DATA_DIR / NEWS_SENTIMENT_FILE
    df = pd.read_parquet(path)

    # Intentamos inferir una columna de fecha si existe
    date_col = None
    for candidate in ["date", "published_at", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df["date"] = df[date_col].dt.date
    else:
        # Si no hay fecha, asignamos una fecha dummy (por ejemplo, hoy)
        df["date"] = pd.Timestamp.today().date()

    # Agregamos por fecha y ticker: media del sentiment_score
    df_agg = (
        df.groupby(["date", "ticker"], as_index=False)["sentiment_score"].mean()
        .rename(columns={"sentiment_score": "sentiment_daily_mean"})
    )

    return df_agg


def _add_price_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    """Añade retornos, log-retornos y stats de ventana sobre precios."""

    df = df_prices.sort_values(["ticker", "date"]).copy()

    if "adj_close" in df.columns:
        price_col = "adj_close"
    elif "close" in df.columns:
        price_col = "close"
    else:
        raise ValueError("No se encontró columna de precio ('adj_close' o 'close').")

    # Retornos
    df["return_1d"] = df.groupby("ticker")[price_col].pct_change(1)
    df["return_5d"] = df.groupby("ticker")[price_col].pct_change(5)

    # Retorno logarítmico
    df["log_return_1d"] = np.log1p(df["return_1d"].fillna(0))

    # Medias móviles
    df["ma_5"] = df.groupby("ticker")[price_col].transform(lambda s: s.rolling(5).mean())
    df["ma_20"] = df.groupby("ticker")[price_col].transform(lambda s: s.rolling(20).mean())

    # Volatilidad móvil 20 días (desviación estándar de los retornos diarios)
    df["vol_20"] = (
        df.groupby("ticker")["return_1d"]
        .transform(lambda s: s.rolling(20).std())
    )

    return df


def _add_sentiment_features(df: pd.DataFrame, df_sent: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Combina precios con sentimiento y añade indicador agregado.

    `window` controla cuántos días incluir en la media móvil del
    sentimiento (por defecto 5).
    """

    # Convertimos date a datetime (sin hora) para merge consistente
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.date

    merged = df.merge(df_sent, on=["date", "ticker"], how="left")

    # Media móvil del sentimiento por ticker
    merged = merged.sort_values(["ticker", "date"]).copy()
    merged["sentiment_rolling_mean"] = (
        merged.groupby("ticker")["sentiment_daily_mean"]
        .transform(lambda s: s.rolling(window).mean())
    )

    return merged


def build_features() -> Tuple[pd.DataFrame, str]:
    """Construye el conjunto de features y lo guarda en parquet.

    Devuelve el DataFrame resultante y la ruta del archivo parquet.
    """

    prices_long = _load_prices()
    prices_with_features = _add_price_features(prices_long)

    news_sentiment = _load_news_sentiment()
    features_df = _add_sentiment_features(prices_with_features, news_sentiment)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / FEATURES_FILE
    features_df.to_parquet(out_path)

    return features_df, str(out_path)


if __name__ == "__main__":
    df_features, path = build_features()
    print(f"Features construidas y guardadas en: {path}")

