"""Modelos ML para generar señales de trading a partir de features.

Flujo principal:

- Carga ``data/processed/features.parquet``.
- Genera una variable objetivo binaria: ``1`` si el retorno del día
  siguiente es positivo, ``0`` en otro caso.
- Separa train/test respetando el orden temporal (80%/20%).
- Entrena un ``RandomForestClassifier`` (o similar) y calcula métricas
  como accuracy, F1 y ROC-AUC sobre el test.
- Guarda el modelo en ``models/trained/ml_signal_model.pkl``.
- Expone una función ``predict_latest_signals()`` que devuelve, para la
  fecha más reciente, una tabla con ``ticker`` y recomendación
  ``BUY`` / ``HOLD`` / ``SELL`` según la probabilidad de subida.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import TRAINED_MODELS_DIR, PROCESSED_DATA_DIR


MODEL_FILENAME = "ml_signal_model.pkl"


@dataclass
class ModelMetrics:
    accuracy: float
    f1: float
    roc_auc: float | None


def _load_features() -> pd.DataFrame:
    """Carga el parquet de features y ordena por fecha y ticker."""

    path = PROCESSED_DATA_DIR / "features.parquet"
    df = pd.read_parquet(path)

    # Aseguramos tipos y orden
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def _create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Crea la variable objetivo binaria basada en el retorno del día siguiente.

    Para cada ticker, target = 1 si ``return_1d`` del día siguiente es
    > 0, 0 en otro caso. La última observación de cada ticker se descarta
    al no tener "día siguiente" conocido.
    """

    if "return_1d" not in df.columns:
        raise ValueError("El DataFrame de features debe contener la columna 'return_1d'.")

    df = df.sort_values(["ticker", "date"]).copy()

    # shift -1 para mirar el retorno del día siguiente dentro de cada ticker
    df["next_return_1d"] = df.groupby("ticker")["return_1d"].shift(-1)
    df = df.dropna(subset=["next_return_1d"])  # elimina último día de cada ticker

    df["target"] = (df["next_return_1d"] > 0).astype(int)
    return df


def _train_test_split_time(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en train/test respetando el orden temporal.

    Se utiliza el índice temporal global: primeros `train_ratio` de las
    filas para train, el resto para test.
    """

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    df_train = df.iloc[:n_train].copy()
    df_test = df.iloc[n_train:].copy()
    return df_train, df_test


def _select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Selecciona columnas numéricas como features y devuelve X, y."""

    if "target" not in df.columns:
        raise ValueError("El DataFrame debe contener una columna 'target'.")

    y = df["target"].astype(int)

    # Excluimos columnas no numéricas o identificadores
    drop_cols = {"target", "ticker", "next_return_1d"}
    if "date" in df.columns:
        drop_cols.add("date")

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].select_dtypes(include=["number"]).copy()

    return X, y


def train_ml_model() -> Tuple[RandomForestClassifier, ModelMetrics]:
    """Entrena el modelo ML de señales y devuelve modelo y métricas."""

    df = _load_features()
    df = _create_target(df)

    df_train, df_test = _train_test_split_time(df, train_ratio=0.8)

    X_train, y_train = _select_features(df_train)
    X_test, y_test = _select_features(df_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    try:
        roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc = None

    metrics = ModelMetrics(accuracy=acc, f1=f1, roc_auc=roc)

    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = TRAINED_MODELS_DIR / MODEL_FILENAME
    joblib.dump(model, model_path)

    print("Métricas en test:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-score: {f1:.4f}")
    if roc is not None:
        print(f"  ROC-AUC:  {roc:.4f}")
    else:
        print("  ROC-AUC: no disponible (solo una clase en test)")

    print(f"Modelo guardado en: {model_path}")

    return model, metrics


def _load_trained_model() -> RandomForestClassifier:
    """Carga el modelo entrenado desde disco."""

    model_path = TRAINED_MODELS_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en {model_path}. Ejecuta train_ml_model() primero."
        )
    return joblib.load(model_path)


def predict_latest_signals(
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
) -> pd.DataFrame:
    """Devuelve señales BUY / HOLD / SELL para la fecha más reciente.

    La recomendación se basa en la probabilidad predicha de retorno
    positivo (clase 1):

    - ``prob >= buy_threshold``  → ``BUY``
    - ``prob <= sell_threshold`` → ``SELL``
    - en otro caso              → ``HOLD``
    """

    df = _load_features()
    df = _create_target(df)

    latest_date = df["date"].max()
    df_latest = df[df["date"] == latest_date].copy()

    if df_latest.empty:
        raise ValueError("No hay datos para la fecha más reciente.")

    model = _load_trained_model()

    # Para predecir sobre el último día, seleccionamos las mismas columnas de features
    X_latest, _ = _select_features(df_latest)

    probs = model.predict_proba(X_latest)[:, 1]

    recommendations: List[str] = []
    for p in probs:
        if p >= buy_threshold:
            recommendations.append("BUY")
        elif p <= sell_threshold:
            recommendations.append("SELL")
        else:
            recommendations.append("HOLD")

    result = pd.DataFrame(
        {
            "date": df_latest["date"].values,
            "ticker": df_latest["ticker"].values,
            "prob_up": probs,
            "signal": recommendations,
        }
    ).sort_values(["ticker"]).reset_index(drop=True)

    return result


if __name__ == "__main__":
    model, metrics = train_ml_model()
    latest_signals = predict_latest_signals()
    print("Señales para la fecha más reciente:")
    print(latest_signals)

