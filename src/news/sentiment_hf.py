"""Módulo de análisis de sentimiento para noticias financieras.

Usa modelos de Hugging Face especializados en texto financiero
(`ProsusAI/finbert` o `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`).

Dado una lista de noticias (por ejemplo, cargadas del JSON producido por
`scrape_news.py`), calcula el sentimiento de cada titular y devuelve un
DataFrame con columnas: ``ticker``, ``headline``, ``sentiment_label``,
``sentiment_score``. Además, guarda los resultados en
``data/processed/news_sentiment.parquet``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.config import HF_API_TOKEN, PROCESSED_DATA_DIR, RAW_DATA_DIR


DEFAULT_MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"


def _get_device() -> int:
    """Devuelve el índice de dispositivo para Transformers (GPU si hay, si no CPU).

    Transformers espera ``device=-1`` para CPU y ``device>=0`` para GPU.
    """

    if torch.cuda.is_available():
        return 0
    return -1


def _build_pipeline(model_name: str = DEFAULT_MODEL_NAME):
    """Construye un pipeline de sentimiento para noticias financieras."""

    if HF_API_TOKEN:
        os.environ["HF_API_TOKEN"] = HF_API_TOKEN

    device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    clf = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return clf


def score_headlines(news_items: List[Dict]) -> pd.DataFrame:
    """Aplica el modelo de sentimiento a una lista de noticias.

    Parámetros
    ----------
    news_items: lista de diccionarios, típicamente con al menos:
        ``ticker``, ``headline``.

    Devuelve
    --------
    DataFrame con columnas
        ``ticker``, ``headline``, ``sentiment_label``, ``sentiment_score``.
    """

    if not news_items:
        return pd.DataFrame(
            columns=["ticker", "headline", "sentiment_label", "sentiment_score"]
        )

    clf = _build_pipeline()

    headlines = [str(item.get("headline", "")) for item in news_items]
    tickers = [item.get("ticker", "") for item in news_items]

    # Procesar por lotes automáticamente (Transformers lo maneja internamente)
    results = clf(headlines)

    df = pd.DataFrame(
        {
            "ticker": tickers,
            "headline": headlines,
            "sentiment_label": [r["label"] for r in results],
            "sentiment_score": [float(r["score"]) for r in results],
        }
    )

    return df


def save_sentiment(df: pd.DataFrame, filename: str = "news_sentiment.parquet") -> str:
    """Guarda el DataFrame de sentimiento en ``data/processed`` como parquet."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_parquet(path)
    return str(path)


def load_news_json(path: Path) -> List[Dict]:
    """Carga noticias desde un archivo JSON como lista de diccionarios."""

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El JSON de noticias debe contener una lista de objetos")
    return data


if __name__ == "__main__":
    # Ejemplo de uso: busca el fichero de noticias del día más reciente
    raw_dir = RAW_DATA_DIR
    json_files = sorted(raw_dir.glob("news_*.json"))
    if not json_files:
        raise SystemExit("No se encontraron archivos news_*.json en data/raw.")

    latest_file = json_files[-1]
    print(f"Cargando noticias desde: {latest_file}")

    news_items = load_news_json(latest_file)
    df_scores = score_headlines(news_items)
    out_path = save_sentiment(df_scores)
    print(f"Resultados de sentimiento guardados en: {out_path}")

