# Solana Memecoin — ML Analysis Platform

An end-to-end machine-learning research platform that analyzes the price behavior of
**30,000+ Solana memecoins** in their first hours of trading, in order to discover
behavioral archetypes and predict short-horizon price movements (pumps).

> ⚠️ **Research project / work in progress.** The goal is to reach trading-ready
> predictive performance. The pattern discovery, feature engineering and models here
> are the approaches explored toward that goal — the emphasis is on rigorous, leakage-safe
> time-series methodology, not on a finished trading bot.

---

## Highlights

- **Scale:** a [Polars](https://pola.rs/)-based pipeline that processes **30,519 tokens**
  end-to-end in ~45 seconds, with caching.
- **Behavioral archetypes:** unsupervised discovery of **9 archetypes** (e.g. *Quick Pump & Death*,
  *Phoenix Attempt*, *Survivor Pump*) via ACF features + K-means + t-SNE, across three
  time resolutions — Sprint (0–399 min), Standard (400–1199 min), Marathon (1200+ min).
- **Death detection:** an algorithm that flags "dead" tokens (flatlined price) with
  1e-12 numerical precision, so models only train on the live portion of each series.
- **Leakage-safe ML:** strict per-token temporal splitting, per-token scaling fitted only
  on the training window, and rolling/log features designed to be look-ahead free.
- **Multiple model families:** LightGBM & XGBoost (directional + multi-class), a unified
  and an advanced hybrid attention-based **LSTM**, plus logistic-regression / linear baselines;
  experiments with transformer forecasting and an RL agent.
- **Feature engineering:** up to **81 features** on a 10-minute observation window
  (cumulative returns, rolling volatility, momentum, ACF-based and technical indicators).
- **Interactive dashboards:** Streamlit apps for data-quality inspection, feature analysis
  and archetype exploration.
- **Tested:** 200+ passing tests, including dedicated mathematical-validation tests for the
  leakage and scaling logic.

## Repository structure

```
data_analysis/          # EDA + interactive data-quality dashboard
data_cleaning/          # category-aware cleaning (normal / extreme / dead tokens)
feature_engineering/    # feature builders + Streamlit feature app
time_series/            # ACF analysis, archetype clustering, forecasting
quant_analysis/         # quantitative analysis & visualization apps
ML/                     # models (LightGBM, XGBoost, LSTM, baselines, tuning)
solana_memecoin_bot_new/# newer bot/analysis module (clustering, RL, transformer)
utils/                  # shared utilities (scaling, splitting, metrics, IO)
streamlit_utils/        # shared Streamlit components
run_pipeline.py         # end-to-end pipeline entry point
PROJECT_COMPREHENSIVE_SUMMARY.md  # detailed write-up of approaches & results
PROJECT_STRUCTURE.md    # data layout & module map
```

## Data

- **30,519** Solana memecoins, minute-by-minute price data over roughly the first 24h.
- Categories: ~3k normal, ~4k extreme (99.9% dumps / 1M%+ pumps — legitimate signal, not noise),
  ~25k tokens that trade then flatline ("dead"), plus data-quality edge cases.
- Data lives under `data/` (`raw/ → processed/ → cleaned/ → features/ → with_archetypes/`)
  and is not committed (see `.gitignore`).

## Methodology notes (what makes it leakage-safe)

- **Temporal splitting** per token — never random splits (which would leak the future).
- **Per-token scaling** (RobustScaler / winsorization) fitted on the training window only.
- **Death-aware processing** — features and labels use only pre-death data.
- **Multi-resolution** modeling so Sprint / Standard / Marathon behaviors don't get mixed
  into a single unstable model.

See [`PROJECT_COMPREHENSIVE_SUMMARY.md`](./PROJECT_COMPREHENSIVE_SUMMARY.md) for the full
account of what worked, what didn't, and current metrics.

## Tech stack

Python · Polars · pandas · NumPy · scikit-learn · LightGBM · XGBoost · PyTorch · TensorFlow/Keras ·
statsmodels · Optuna · Streamlit · Plotly · Matplotlib/Seaborn · pytest · Black

## Getting started

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. run the end-to-end pipeline (expects data under data/)
python run_pipeline.py

# 3. launch a dashboard (examples)
streamlit run data_analysis/data_quality.py
streamlit run feature_engineering/app.py

# 4. run the tests
pytest
```

## Status & roadmap

Current models discover clear behavioral structure (e.g. **78.7%** of high-volatility
"marathon" tokens pump >50% after minute 5) but early-classification F1 is still below the
trading-ready target. Next steps: integrate volume / liquidity / buy-sell data, stronger
early-window features, ensemble methods, and a real-time inference path.

---

*Personal research project. Not financial advice — memecoin trading is extremely risky.*
