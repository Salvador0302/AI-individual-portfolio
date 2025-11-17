"""Utilidades para resumir noticias usando la API de Gemini.

En un entorno real, aquí se implementaría la llamada HTTP a la API
de Gemini. Para este proyecto académico devolvemos resúmenes simples
basados en el texto de entrada, de forma determinista y sin acceso a
servicios externos (útil para desarrollo offline).
"""

from __future__ import annotations

import os
from typing import List

import pandas as pd

from src.config import DATA_PROCESSED_DIR


def summarize_texts(texts: List[str]) -> List[str]:
    """Devuelve un "pseudo-resumen" simple para cada texto.

    Este comportamiento es offline y determinista, pensado como
    sustituto cuando no se tiene acceso a Gemini. Si más adelante
    quieres integrar la API real, puedes reemplazar aquí la lógica.
    """

    summaries: List[str] = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            summaries.append("Sin noticias relevantes recientes.")
            continue

        snippet = t[:240].replace("\n", " ")
        summaries.append(snippet + ("..." if len(t) > 240 else ""))
    return summaries


def add_summaries(input_csv: str, text_column: str = "content", output_csv: str = "news_with_summaries.csv") -> str:
    """Añade una columna de resúmenes a un CSV de noticias procesadas."""

    path = input_csv if os.path.isabs(input_csv) else os.path.join(DATA_PROCESSED_DIR, input_csv)
    df = pd.read_csv(path)

    df["summary"] = summarize_texts(df[text_column].fillna("").astype(str).tolist())

    out_path = os.path.join(DATA_PROCESSED_DIR, output_csv)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    out = add_summaries("news_with_sentiment.csv")
    print(f"Noticias con resúmenes guardadas en: {out}")
