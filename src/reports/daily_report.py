"""Generación de informe diario en Markdown para el portafolio individual.

Este script combina:

- Precios recientes (desde ``features.parquet``).
- Sentimiento medio de noticias por ticker.
- Señales del modelo ML (BUY / HOLD / SELL).
- Un breve resumen en lenguaje natural por ticker, usando
  ``summarize_ticker_news``.

El resultado es un archivo Markdown en ``models/reports/daily_report_<fecha>.md``
que resume la situación diaria del portafolio.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from src.config import MODELS_REPORTS_DIR, PROCESSED_DATA_DIR
from src.models.ml_signals import predict_latest_signals
from src.news.summarize_gemini import summarize_texts as summarize_ticker_news


def _load_features_for_date(date: str) -> pd.DataFrame:
    """Carga features y filtra filas de la fecha indicada (YYYY-MM-DD)."""

    path = PROCESSED_DATA_DIR / "features.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[df["date"] == date].copy()


def _aggregate_sentiment(df_day: pd.DataFrame) -> pd.DataFrame:
    """Devuelve sentimiento medio por ticker para la fecha dada."""

    cols = [c for c in df_day.columns if c.startswith("sentiment_")]
    if not cols:
        return pd.DataFrame(columns=["ticker", "sentiment_mean"])

    agg = (
        df_day.groupby("ticker")[cols]
        .mean()
        .mean(axis=1)
        .reset_index(name="sentiment_mean")
    )
    return agg


def _build_ticker_table(date: str) -> pd.DataFrame:
    """Construye una tabla con métricas por ticker para la fecha dada."""

    df_day = _load_features_for_date(date)
    if df_day.empty:
        return pd.DataFrame()

    # Precio último y variación diaria
    day_prices = df_day.copy()
    price_col = "adj_close" if "adj_close" in day_prices.columns else "close"

    price_info = day_prices[["ticker", price_col, "return_1d"]].rename(
        columns={price_col: "last_price", "return_1d": "daily_return"}
    )

    # Sentimiento medio
    sent_info = _aggregate_sentiment(df_day)

    # Señales ML (usa la fecha más reciente disponible en features)
    signals = predict_latest_signals()
    signals = signals[["ticker", "signal", "prob_up"]]

    # Merge de todo
    merged = (
        price_info.merge(sent_info, on="ticker", how="left")
        .merge(signals, on="ticker", how="left")
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    return merged


def _build_ticker_summaries(tickers: List[str]) -> dict[str, str]:
    """Crea resúmenes en lenguaje natural por ticker.

    Por simplicidad, usamos el propio nombre del ticker como texto base,
    pero en un flujo completo se deberían pasar los titulares/noticias
    recientes de cada ticker a ``summarize_ticker_news``.
    """

    texts = [f"Noticias recientes y contexto para {t}" for t in tickers]
    summaries = summarize_ticker_news(texts)
    return dict(zip(tickers, summaries))


def generate_daily_report(date: str | None = None) -> Path:
    """Genera el informe diario para la fecha indicada (o hoy por defecto)."""

    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")

    table = _build_ticker_table(date)

    MODELS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODELS_REPORTS_DIR / f"daily_report_{date}.md"

    lines: list[str] = []
    lines.append(f"# Informe diario del portafolio – {date}\n")

    if table.empty:
        lines.append("No hay datos disponibles para esta fecha.\n")
    else:
        # Tabla general
        table_to_show = table[[
            "ticker",
            "last_price",
            "daily_return",
            "sentiment_mean",
            "signal",
            "prob_up",
        ]].copy()
        lines.append("## Resumen cuantitativo por ticker\n")
        lines.append(table_to_show.to_markdown(index=False))
        lines.append("\n")

        # Resúmenes en lenguaje natural
        lines.append("## Resúmenes en lenguaje natural\n")
        summaries = _build_ticker_summaries(table["ticker"].tolist())
        for ticker in table["ticker"].tolist():
            lines.append(f"### {ticker}\n")
            lines.append(summaries.get(ticker, "(Sin resumen disponible)"))
            lines.append("\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    path = generate_daily_report()
    print(f"Informe generado en: {path}")

