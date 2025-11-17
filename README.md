## Portfolio Individual – Estrategia Agresiva Fintech & Insurtech

Repositorio para el portafolio individual del juego de bolsa de MarketWatch
con USD 10,000, centrado en 8 empresas fintech/insurtech:
`LMND`, `SOFI`, `SQ`, `PYPL`, `NU`, `OPFI`, `STNE`, `ALLY`.

El objetivo es construir una estrategia agresiva pero defendible a nivel
académico, apoyada en datos de mercado, noticias, análisis de sentimiento,
modelos de Machine Learning y un dashboard interactivo.

### Flujo principal

1. **Descarga de datos de mercado**  
	`src/data/download_prices.py` descarga precios diarios de los 8 tickers
	y guarda `data/raw/prices.parquet`.

2. **Scraping de noticias y sentimiento**  
	- `src/news/scrape_news.py` recopila noticias y guarda
	  `data/raw/news_YYYYMMDD.json`.
	- `src/news/sentiment_hf.py` aplica un modelo de Hugging Face para
	  etiquetar titulares y guarda `data/processed/news_sentiment.parquet`.

3. **Construcción de features**  
	`src/features/build_features.py` combina precios y sentimiento para
	generar `data/processed/features.parquet` con retornos, medias móviles,
	volatilidad y agregados de sentimiento.

4. **Modelo de señales ML**  
	`src/models/ml_signals.py` entrena un `RandomForestClassifier` para
	predecir si el retorno del siguiente día será positivo, guarda el
	modelo en `models/trained/ml_signal_model.pkl` y expone
	`predict_latest_signals()` (BUY / HOLD / SELL por ticker).

5. **Informe diario e IA generativa**  
	- `src/reports/daily_report.py` genera un informe Markdown diario con
	  precios, sentimiento y señales por ticker.
	- `src/news/summarize_gemini.py` produce resúmenes en lenguaje natural
	  (stub offline, fácilmente sustituible por la API real de Gemini).

6. **Dashboard interactivo**  
	`src/dashboard/streamlit_app.py` ofrece una app Streamlit para:
	- Ajustar pesos del portafolio con sliders.
	- Ver la evolución del portafolio simulado.
	- Consultar volatilidad, Sharpe ratio y drawdown máximo.
	- Visualizar un heatmap de correlaciones.
	- Revisar sentimiento y señales ML por ticker.

### Puesta en marcha (resumen)

```powershell
cd "c:\Users\SALVADOR\Desktop\escritorio\AI-individual-portfolio"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 1) Precios
python -m src.data.download_prices

# 2) Noticias y sentimiento
python -m src.news.scrape_news
python -m src.news.sentiment_hf

# 3) Features y modelo
python -m src.features.build_features
python -m src.models.ml_signals

# 4) Informe diario
python -m src.reports.daily_report

# 5) Dashboard Streamlit
streamlit run src/dashboard/streamlit_app.py
```

Configura tus claves (si usas APIs externas) en `.env` a partir de
`.env.example`. Para fines académicos, el código está preparado para
funcionar también en modo offline (sin Gemini real).