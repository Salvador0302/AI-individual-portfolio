"""Scraping de noticias financieras para los tickers del proyecto.

IMPORTANTE: antes de usar este script con un sitio concreto, debes
revisar y respetar SIEMPRE:

- El archivo robots.txt del sitio web.
- Los términos y condiciones de uso (Terms of Service).

Este ejemplo está pensado como plantilla educativa. Asegúrate de que el
scraping está permitido y de limitar la frecuencia de peticiones.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR, TICKERS


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_news_for_ticker(ticker: str, limit: int = 20) -> List[Dict[str, str]]:
    """Obtiene noticias para un ticker mediante scraping básico.

    Esta función es un ejemplo de scraping educativo. Adáptala al sitio
    concreto que quieras usar (estructura HTML, parámetros de búsqueda,
    etc.) y respeta siempre robots.txt y los términos de uso.

    Devuelve una lista de diccionarios con claves:
    ``{'ticker', 'headline', 'url', 'published_at', 'source'}``.
    """

    # Ejemplo: búsqueda en DuckDuckGo de noticias recientes sobre el ticker.
    # Sustituye esta URL por la de la web que quieras usar, siempre que
    # esté permitido por sus términos.
    query = f"{ticker} stock news"
    url = "https://duckduckgo.com/html/"
    params = {"q": query}

    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    results: List[Dict[str, str]] = []

    # Este selector CSS es solo ilustrativo; deberías ajustarlo al HTML real
    for result in soup.select(".result__body")[:limit]:
        title_el = result.select_one(".result__a")
        if not title_el:
            continue

        headline = title_el.get_text(strip=True)
        url_target = title_el.get("href", "")

        # DuckDuckGo no proporciona fecha ni fuente de forma directa en este HTML
        # aquí se ponen valores aproximados/placeholder
        published_at = datetime.utcnow().isoformat()
        source = "duckduckgo_search"

        results.append(
            {
                "ticker": ticker,
                "headline": headline,
                "url": url_target,
                "published_at": published_at,
                "source": source,
            }
        )

    # Respetar buenas prácticas: pequeña pausa entre peticiones
    time.sleep(2)

    return results


def scrape_all_tickers(limit_per_ticker: int = 20) -> List[Dict[str, str]]:
    """Scrapea noticias para todos los tickers definidos en `config.TICKERS`."""

    all_news: List[Dict[str, str]] = []
    for ticker in TICKERS:
        news = fetch_news_for_ticker(ticker, limit=limit_per_ticker)
        all_news.extend(news)
    return all_news


def save_news_json(records: List[Dict[str, str]]) -> str:
    """Guarda todas las noticias en `data/raw/news_<fecha>.json`."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().strftime("%Y%m%d")
    path = RAW_DATA_DIR / f"news_{today}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return str(path)


if __name__ == "__main__":
    news_records = scrape_all_tickers(limit_per_ticker=20)
    output_path = save_news_json(news_records)
    print(f"Noticias guardadas en: {output_path}")
