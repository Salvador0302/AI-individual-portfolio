"""Configuración central del proyecto de portafolio fintech."""

from pathlib import Path

from dotenv import load_dotenv
import os


# --- Tickers del juego de bolsa ---
TICKERS: list[str] = [
    "LMND",
    "SOFI",
    "SQ",
    "PYPL",
    "NU",
    "OPFI",
    "STNE",
    "ALLY",
]


# --- Rutas base del proyecto ---
BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
TRAINED_MODELS_DIR: Path = MODELS_DIR / "trained"
REPORTS_MODELS_DIR: Path = MODELS_DIR / "reports"


# --- Carga de variables de entorno ---
ENV_PATH: Path = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


# Claves de APIs y otros secretos
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
ALPHA_VANTAGE_API_KEY: str | None = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY: str | None = os.getenv("NEWS_API_KEY")
HF_API_TOKEN: str | None = os.getenv("HF_API_TOKEN")


__all__ = [
    "TICKERS",
    "BASE_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "TRAINED_MODELS_DIR",
    "REPORTS_MODELS_DIR",
    "GEMINI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "NEWS_API_KEY",
    "HF_API_TOKEN",
]

