"""Módulo para modelos deep learning de señales.

Para este proyecto el foco principal está en el modelo ML clásico
(`RandomForest`), pero se deja este archivo preparado para que puedas
experimentar con arquitecturas de deep learning (LSTM, Transformers...) 
sobre ``features.parquet`` si lo deseas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR


def load_features() -> pd.DataFrame:
    """Carga ``features.parquet`` como punto de partida para modelos DL."""

    path = PROCESSED_DATA_DIR / "features.parquet"
    df = pd.read_parquet(path)
    return df


def train_dl_model() -> None:
    """Punto de entrada para entrenar un modelo deep learning.

    Implementa aquí tu arquitectura de red neuronal si decides extender
    el proyecto (por ejemplo usando PyTorch o Keras). Esta función solo
    documenta el lugar donde debería ir ese código.
    """

    df = load_features()
    print(
        "Se han cargado", len(df),
        "filas de features. Implementa aquí tu modelo DL si lo deseas.",
    )


if __name__ == "__main__":
    train_dl_model()
