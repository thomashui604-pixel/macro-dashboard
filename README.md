# Global Macro Dashboard

A professional buy-side macro analyst morning briefing tool built with Streamlit and Plotly.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)

## Features

- **Rates & Curve** — Live UST yield curve with historical overlays, curve spreads with recession shading, real rates & breakevens, SOFR/EFFR
- **Policy Path** — Fed Funds futures implied rate strip, FOMC dot plot vs market pricing, SOFR futures
- **Cross-Asset** — 35+ instruments across equities, bonds, FX, commodities, vol with sparklines and color-coded returns
- **VIX & Vol** — Term structure, contango/backwardation detection, realized vs implied vol, risk premium
- **Macro Data** — Inflation, labor market, activity, and financial conditions with NBER recession shading

## Setup

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
FRED_API_KEY = "your_key_here"
```

Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html

## Run

```bash
streamlit run app.py
```

## Data Sources

- **FRED** — Rates, macro releases, financial conditions
- **Yahoo Finance** — Equities, bonds, FX, commodities, volatility
- **US Treasury** — Yield curve reference data
