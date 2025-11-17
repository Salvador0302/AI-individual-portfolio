"""Une precios históricos y noticias procesadas en un único dataset."""

import os
import pandas as pd

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR


def merge_prices_and_news(prices_file: str, news_file: str, output_file: str = "merged_prices_news.csv") -> str:
    prices_path = prices_file if os.path.isabs(prices_file) else os.path.join(DATA_RAW_DIR, prices_file)
    news_path = news_file if os.path.isabs(news_file) else os.path.join(DATA_PROCESSED_DIR, news_file)

    prices = pd.read_csv(prices_path, parse_dates=["date"])
    news = pd.read_csv(news_path, parse_dates=["date"])

    merged = pd.merge_asof(
        prices.sort_values("date"),
        news.sort_values("date"),
        on="date",
        direction="backward",
    )

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED_DIR, output_file)
    merged.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    output = merge_prices_and_news("prices_AAPL.csv", "news_sentiment_AAPL.csv")
    print(f"Dataset combinado guardado en: {output}")
