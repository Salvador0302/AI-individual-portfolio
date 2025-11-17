"""Dashboard de Streamlit para el portafolio fintech/insurtech.

Esta app permite:
- Cargar datos de precios y features para los 8 tickers del juego.
- Ajustar pesos del portafolio con sliders (suma 100%).
- Visualizar la evolución del portafolio simulado.
- Ver métricas de riesgo (volatilidad, Sharpe ratio, drawdown máximo).
- Mostrar un heatmap de correlaciones entre activos.
- Incluir la última información de sentimiento y resumen por ticker.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import TICKERS, PROCESSED_DATA_DIR
from src.models.ml_signals import predict_latest_signals


st.set_page_config(page_title="Dashboard Portafolio Fintech", layout="wide")


@st.cache_data
def load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "features.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_sentiment() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "news_sentiment.parquet"
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["ticker", "sentiment_score"])
    return df


def compute_portfolio_series(df_features: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Calcula la serie de valor del portafolio a partir de retornos por ticker."""

    df = df_features.copy()
    df = df[df["ticker"].isin(weights.keys())]

    # Filtramos columnas relevantes
    df = df[["date", "ticker", "return_1d"]].dropna()

    # Pivot a formato ancho: filas -> fechas, columnas -> tickers
    pivot = df.pivot(index="date", columns="ticker", values="return_1d").sort_index()

    # Alineamos pesos al pivot
    w = np.array([weights.get(t, 0.0) for t in pivot.columns])
    w = w / w.sum() if w.sum() > 0 else w

    # Retorno diario del portafolio
    port_ret = pivot.fillna(0).dot(w)

    # Serie de valor inicial 100
    value = (1 + port_ret).cumprod() * 100
    out = pd.DataFrame({"date": value.index, "portfolio_value": value.values})
    return out


def compute_risk_metrics(portfolio_df: pd.DataFrame) -> Dict[str, float]:
    """Calcula volatilidad anualizada, Sharpe ratio y drawdown máximo."""

    df = portfolio_df.sort_values("date").copy()
    df["ret"] = df["portfolio_value"].pct_change().fillna(0)

    mean_ret = df["ret"].mean()
    vol = df["ret"].std()

    ann_factor = np.sqrt(252)
    ann_vol = vol * ann_factor
    ann_ret = mean_ret * 252

    rf = 0.0
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

    cum_max = df["portfolio_value"].cummax()
    drawdown = df["portfolio_value"] / cum_max - 1
    max_dd = drawdown.min()

    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def compute_correlation_matrix(df_features: pd.DataFrame) -> pd.DataFrame:
    """Calcula matriz de correlación de retornos entre tickers."""

    df = df_features[["date", "ticker", "return_1d"]].dropna().copy()
    pivot = df.pivot(index="date", columns="ticker", values="return_1d")
    corr = pivot.corr()
    return corr


def build_sentiment_table(df_features: pd.DataFrame, df_sent: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla con sentimiento medio reciente y señal ML por ticker."""

    # Sentimiento medio por ticker
    if not df_sent.empty:
        sent = df_sent.groupby("ticker")["sentiment_score"].mean().reset_index()
        sent = sent.rename(columns={"sentiment_score": "sentiment_mean"})
    else:
        sent = pd.DataFrame({"ticker": TICKERS, "sentiment_mean": np.nan})

    # Señales ML (último día)
    try:
        signals = predict_latest_signals()
    except Exception:
        signals = pd.DataFrame(columns=["ticker", "signal", "prob_up"])

    # Último precio
    latest_date = df_features["date"].max()
    latest = df_features[df_features["date"] == latest_date].copy()
    price_col = "adj_close" if "adj_close" in latest.columns else "close"
    prices = latest[["ticker", price_col]].rename(columns={price_col: "last_price"})

    # Merge
    merged = (
        prices.merge(sent, on="ticker", how="left")
        .merge(signals, on="ticker", how="left")
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    return merged


def main() -> None:
    st.title("📊 Dashboard Portafolio Fintech & Insurtech")
    st.markdown(
        """Este panel resume el comportamiento del portafolio individual 
        y grupal sobre los 8 tickers del juego, combinando precios, 
        riesgo, correlaciones y sentimiento de noticias."""
    )

    df_features = load_features()
    df_sent = load_sentiment()

    st.sidebar.header("Parámetros del portafolio")
    st.sidebar.markdown("Asigna pesos a cada ticker (suma 100%).")

    default_weight = 100 / len(TICKERS)
    weights: Dict[str, float] = {}
    total = 0.0
    for ticker in TICKERS:
        w = st.sidebar.slider(f"Peso {ticker} (%)", min_value=0.0, max_value=100.0, value=float(default_weight), step=1.0)
        weights[ticker] = w / 100.0
        total += w

    st.sidebar.markdown(f"**Suma de pesos:** {total:.1f}%")
    if abs(total - 100.0) > 1e-6:
        st.sidebar.error("La suma de pesos debe ser 100%.")

    # Layout principal
    tab1, tab2, tab3, tab4 = st.tabs([
        "Evolución del portafolio",
        "Riesgo y métricas",
        "Correlaciones",
        "Noticias y sentimiento",
    ])

    with tab1:
        st.subheader("Evolución del valor del portafolio")
        port_df = compute_portfolio_series(df_features, weights)
        fig = px.line(port_df, x="date", y="portfolio_value", title="Valor del portafolio (base 100)")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Métricas de riesgo")
        metrics = compute_risk_metrics(port_df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rentab. anualizada", f"{metrics['annual_return']*100:.2f}%")
        col2.metric("Volatilidad anualizada", f"{metrics['annual_vol']*100:.2f}%")
        col3.metric("Sharpe ratio", f"{metrics['sharpe']:.2f}")
        col4.metric("Drawdown máximo", f"{metrics['max_drawdown']*100:.2f}%")

    with tab3:
        st.subheader("Heatmap de correlaciones entre activos")
        corr = compute_correlation_matrix(df_features)
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
        fig_corr.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        st.subheader("Último resumen de noticias y sentimiento por ticker")
        sentiment_table = build_sentiment_table(df_features, df_sent)
        st.dataframe(sentiment_table, use_container_width=True)


if __name__ == "__main__":
    main()
