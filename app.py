"""
Global Macro Dashboard — Morning Briefing Tool
================================================
A professional buy-side macro analyst dashboard built with Streamlit.
Prioritizes signal density over decoration.

Run: streamlit run app.py
Secrets: .streamlit/secrets.toml → FRED_API_KEY = "your_key_here"
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import requests
import json
import time
import shutil
import math
import calendar
from collections import defaultdict
from pathlib import Path
import warnings

from regime_model import (
    REGIME_LOOKUP,
    REGIME_COLORS,
    GROWTH_KEYS,
    INFLATION_KEYS,
    LIQUIDITY_KEYS,
    GROWTH_LABELS,
    INFLATION_LABELS,
    LIQUIDITY_LABELS,
    INVERT,
    FRED_IDS,
    DEFAULT_START as REGIME_DEFAULT_START,
    compute_regime,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0e1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2333;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --blue: #58a6ff;
    --cyan: #39d2c0;
}

.stApp { font-family: 'DM Sans', sans-serif; }

div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
}
div[data-testid="stMetric"] label { font-size: 0.72rem !important; color: var(--text-secondary) !important; font-family: 'JetBrains Mono', monospace !important; text-transform: uppercase; letter-spacing: 0.05em; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.25rem !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 600; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important; }

div[data-testid="stTabs"] button { font-family: 'JetBrains Mono', monospace !important; font-weight: 500; letter-spacing: 0.02em; font-size: 0.82rem; }

.kpi-strip {
    display: flex; gap: 8px; flex-wrap: wrap; padding: 8px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
}
.kpi-item {
    background: #1c2333; border: 1px solid #30363d; border-radius: 5px;
    padding: 6px 12px; display: inline-flex; align-items: center; gap: 6px;
}
.kpi-label { color: #8b949e; }
.kpi-value { color: #e6edf3; font-weight: 600; }
.kpi-green { color: #3fb950; }
.kpi-red { color: #f85149; }

section[data-testid="stSidebar"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; }

div.stAlert { border-radius: 6px; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# PLOTLY LAYOUT DEFAULTS
# ─────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(family="JetBrains Mono, monospace", size=11, color="#e6edf3"),
    margin=dict(l=50, r=30, t=60, b=100), # Increased bottom padding for legends
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.35, # Moved lower to clear X-axis
        xanchor="center",
        x=0.5,
        font=dict(size=10),
        bgcolor="rgba(0,0,0,0)"
    ),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", title=dict(standoff=20)),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", title=dict(standoff=15)),
    hovermode="x unified",
)

GREEN = "#3fb950"
RED = "#f85149"
BLUE = "#58a6ff"
CYAN = "#39d2c0"
YELLOW = "#d29922"
MUTED = "#484f58"
COLORS = ["#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff", "#39d2c0", "#f0883e", "#ff7b72"]

def make_layout(title="", height=420, **kwargs):
    layout = {**PLOTLY_LAYOUT, "title": dict(text=title, font=dict(size=13), x=0.01), "height": height}
    layout.update(kwargs)
    return go.Layout(**layout)

# ─────────────────────────────────────────────────────────────────────
# DATA FETCHING HELPERS
# ─────────────────────────────────────────────────────────────────────
def get_fred_key():
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        return None

# ── Disk cache for FRED data (survives reboots) ──────────────────────
CACHE_DIR = Path("data_cache")

_FRED_TTL = {
    "DFF": 4*3600, "SOFR": 4*3600,
    "USREC": 24*3600, "A191RL1Q225SBEA": 24*3600, "GDP": 24*3600,
    "DGS": 4*3600, "DFII": 4*3600, "T5Y": 4*3600, "T10Y": 4*3600,
    "CPIAUCSL": 12*3600, "CPILFESL": 12*3600, "PCEPI": 12*3600,
    "PCEPILFE": 12*3600, "PAYEMS": 12*3600, "UNRATE": 12*3600,
    "JOLTS": 12*3600, "ICSA": 12*3600, "INDPRO": 12*3600,
}
_DEFAULT_TTL = 12 * 3600

def _fred_ttl(series_id):
    if series_id in _FRED_TTL:
        return _FRED_TTL[series_id]
    for prefix, ttl in _FRED_TTL.items():
        if series_id.startswith(prefix):
            return ttl
    return _DEFAULT_TTL

def _cache_paths(series_id, start):
    slug = f"{series_id}__{start or 'none'}".replace("/", "-").replace(":", "-")
    return CACHE_DIR / f"{slug}.parquet", CACHE_DIR / f"{slug}.json"

def _disk_load(series_id, start):
    pq, meta = _cache_paths(series_id, start)
    if not pq.exists() or not meta.exists():
        return None
    try:
        m = json.loads(meta.read_text())
        if time.time() - m["fetched_at"] > _fred_ttl(series_id):
            return None
        return pd.read_parquet(pq)["value"]
    except Exception:
        return None

def _disk_save(series, series_id, start):
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        pq, meta = _cache_paths(series_id, start)
        series.rename("value").to_frame().to_parquet(pq)
        meta.write_text(json.dumps({"series_id": series_id, "fetched_at": time.time()}))
    except Exception:
        pass

def _fetch_fred_raw(series_id, start=None, end=None):
    """Direct HTTP fetch from FRED — no caching."""
    key = get_fred_key()
    if not key:
        return pd.Series(dtype=float)
    try:
        params = {"series_id": series_id, "api_key": key,
                  "file_type": "json", "sort_order": "asc"}
        if start:
            params["observation_start"] = start.strftime("%Y-%m-%d") if isinstance(start, datetime) else start
        if end:
            params["observation_end"] = end.strftime("%Y-%m-%d") if isinstance(end, datetime) else end
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params=params, timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date")["value"].dropna()
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fred_series(series_id, start=None, end=None):
    """Fetch a FRED series: disk cache → FRED API fallback."""
    cached = _disk_load(series_id, start)
    if cached is not None:
        return cached
    series = _fetch_fred_raw(series_id, start, end)
    if len(series) > 0:
        _disk_save(series, series_id, start)
    return series

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fred_multi(series_ids, start=None, end=None):
    """Fetch multiple FRED series into a DataFrame."""
    result = {}
    for sid in series_ids:
        s = fetch_fred_series(sid, start, end)
        if len(s) > 0:
            result[sid] = s
    if result:
        return pd.DataFrame(result)
    return pd.DataFrame()

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_recession_dates():
    """Fetch NBER recession indicator from FRED."""
    s = fetch_fred_series("USREC", start="1960-01-01")
    if len(s) == 0:
        return []
    # Find recession start/end pairs
    recessions = []
    in_recession = False
    start = None
    for date_val, val in s.items():
        if val == 1 and not in_recession:
            start = date_val
            in_recession = True
        elif val == 0 and in_recession:
            recessions.append((start, date_val))
            in_recession = False
    if in_recession:
        recessions.append((start, s.index[-1]))
    return recessions

def add_recession_shading(fig, recessions, x_min=None, x_max=None):
    """Add NBER recession shading to a plotly figure, filtered to data range."""
    for rec_start, rec_end in recessions:
        # Skip recessions entirely outside the data range
        if x_min is not None and rec_end < x_min:
            continue
        if x_max is not None and rec_start > x_max:
            continue
        # Clip to data range
        draw_start = max(rec_start, x_min) if x_min else rec_start
        draw_end = min(rec_end, x_max) if x_max else rec_end
        fig.add_vrect(
            x0=draw_start, x1=draw_end,
            fillcolor="rgba(255,255,255,0.04)", line_width=0,
            layer="below"
        )

def safe_pct_change(current, previous):
    if previous is None or previous == 0 or pd.isna(previous) or pd.isna(current):
        return None
    return (current / previous - 1) * 100

def color_for_change(val):
    if val is None or pd.isna(val):
        return "#8b949e"
    return GREEN if val >= 0 else RED

def fmt_pct(val):
    if val is None or pd.isna(val):
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"

def lookback_date(lookback_str):
    """Convert lookback string to start date (returns a tz-naive pd.Timestamp)."""
    now = pd.Timestamp.now()
    mapping = {
        "1M": timedelta(days=30),
        "3M": timedelta(days=91),
        "6M": timedelta(days=182),
        "1Y": timedelta(days=365),
        "2Y": timedelta(days=730),
        "3Y": timedelta(days=1095),
        "5Y": timedelta(days=1825),
        "10Y": timedelta(days=3650),
        "Max": timedelta(days=20000),
    }
    result = now - mapping.get(lookback_str, timedelta(days=365))
    # Return tz-naive so pandas will coerce safely against any index
    return result.tz_localize(None) if result.tzinfo else result


# ─────────────────────────────────────────────────────────────────────
# AI SUMMARY (Gemini Flash)
# ─────────────────────────────────────────────────────────────────────
def get_gemini_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

def ai_summary(tab_key, context_str):
    """Show AI summary widget with manual refresh button."""
    ss_key = f"ai_summary_{tab_key}"
    col_btn, col_txt = st.columns([1, 6])
    with col_btn:
        generate = st.button("🤖 AI Summary", key=f"btn_{tab_key}")
    if generate:
        api_key = get_gemini_key()
        if not api_key:
            st.warning("Add GEMINI_API_KEY to Streamlit secrets.")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = (
                "You are a senior macro strategist. Based on the data below, "
                "write a 2-3 sentence market summary. Be specific with numbers. "
                "No disclaimers or caveats.\n\n" + context_str
            )
            resp = model.generate_content(prompt)
            st.session_state[ss_key] = resp.text
        except Exception as e:
            st.session_state[ss_key] = f"Summary unavailable: {e}"
    if ss_key in st.session_state:
        with col_txt:
            st.markdown(
                f'<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; '
                f'padding:10px 14px; font-size:0.82rem; color:#e6edf3; line-height:1.5;">'
                f'{st.session_state[ss_key]}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    global_lookback = st.selectbox("Default Lookback", ["1Y", "2Y", "5Y", "10Y"], index=0)
    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        st.rerun()
    st.divider()
    st.caption("Data: FRED · yfinance · Treasury")
    st.caption(f"As of {datetime.now().strftime('%Y-%m-%d %H:%M')} ET")
    fred_key = get_fred_key()
    if not fred_key:
        st.warning("⚠️ No FRED_API_KEY in secrets. FRED data will be unavailable.")


def _yf_closes(tickers, **kwargs):
    """yf.download → 'Close' DataFrame, robust to MultiIndex / single-ticker shape.

    Returns an empty DataFrame on failure or when no Close data is available.
    """
    try:
        tlist = [tickers] if isinstance(tickers, str) else list(tickers)
        data = yf.download(tlist, auto_adjust=True, progress=False, threads=True, **kwargs)
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            return data["Close"] if "Close" in data.columns.get_level_values(0) else pd.DataFrame()
        if "Close" in data.columns:
            return pd.DataFrame({tlist[0]: data["Close"]})
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# KPI STRIP (compact header)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kpi_data():
    """Fetch the KPI summary data."""
    kpis = {}
    # yfinance quick data
    tickers_dict = {"SPX": "^GSPC", "DXY": "DX-Y.NYB", "Gold": "GC=F", "VIX": "^VIX"}
    tickers_list = list(tickers_dict.values())
    closes = _yf_closes(tickers_list, period="5d")
    for name, ticker in tickers_dict.items():
        if ticker in closes.columns:
            s = closes[ticker].dropna()
            if len(s) >= 2:
                kpis[name] = {"price": s.iloc[-1], "chg": safe_pct_change(s.iloc[-1], s.iloc[-2])}
            elif len(s) == 1:
                kpis[name] = {"price": s.iloc[-1], "chg": None}
    # FRED
    effr = fetch_fred_series("DFF")
    if len(effr) > 0:
        kpis["EFFR"] = {"price": effr.iloc[-1], "chg": None}
    t10y = fetch_fred_series("DGS10")
    if len(t10y) > 0:
        kpis["10Y"] = {"price": t10y.iloc[-1], "chg": None}
    t2y = fetch_fred_series("DGS2")
    if len(t2y) > 0 and len(t10y) > 0:
        kpis["2s10s"] = {"price": t10y.iloc[-1] - t2y.iloc[-1], "chg": None}
    return kpis

def render_kpi_strip(kpis):
    items = []
    order = ["EFFR", "10Y", "2s10s", "SPX", "DXY", "Gold", "VIX"]
    for name in order:
        if name not in kpis:
            continue
        d = kpis[name]
        price = d["price"]
        chg = d["chg"]
        if name in ["EFFR", "10Y", "2s10s"]:
            price_str = f"{price:.2f}%"
        elif name == "VIX":
            price_str = f"{price:.1f}"
        else:
            price_str = f"{price:,.1f}"
        chg_html = ""
        if chg is not None:
            color_class = "kpi-green" if chg >= 0 else "kpi-red"
            sign = "+" if chg >= 0 else ""
            chg_html = f' <span class="{color_class}">{sign}{chg:.2f}%</span>'
        items.append(f'<span class="kpi-item"><span class="kpi-label">{name}</span><span class="kpi-value">{price_str}</span>{chg_html}</span>')
    return '<div class="kpi-strip">' + ''.join(items) + '</div>'

@st.fragment
def _kpi_fragment():
    st.markdown(render_kpi_strip(fetch_kpi_data()), unsafe_allow_html=True)

_kpi_fragment()

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 Rates & Curve",
    "🏛 Policy Path",
    "🌍 Cross-Asset",
    "📊 VIX & Vol",
    "📋 Macro Data",
    "🏗 Sector Analysis",
    "📊 Relative Strength",
    "🧭 Macro Regime",
    "🧪 Regime Backtest",
])

# ═════════════════════════════════════════════════════════════════════
# TAB 1 — RATES & CURVE
# ═════════════════════════════════════════════════════════════════════
with tab1:
    YIELD_SERIES = {
        "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1",
        "2Y": "DGS2", "3Y": "DGS3", "5Y": "DGS5", "7Y": "DGS7",
        "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
    }
    TENOR_ORDER = list(YIELD_SERIES.keys())
    TENOR_X = list(range(len(TENOR_ORDER)))

    @st.cache_data(ttl=21600, show_spinner=False)
    def get_yield_curve_data():
        df = fetch_fred_multi(list(YIELD_SERIES.values()), start="2023-01-01")
        df.columns = [k for k, v in YIELD_SERIES.items() if v in df.columns][:len(df.columns)]
        return df

    yield_df = get_yield_curve_data()

    if yield_df.empty:
        st.warning("⚠️ Yield curve data unavailable. Check FRED API key.")
    else:
        # ── AI Summary ──
        if not yield_df.empty:
            _row = yield_df.iloc[-1]
            _prev = yield_df.iloc[-6] if len(yield_df) > 5 else _row
            _ctx = "US Treasury Yields (current vs 1 week ago):\n"
            for t in ["2Y", "5Y", "10Y", "30Y"]:
                if t in _row.index:
                    chg = _row[t] - _prev.get(t, _row[t])
                    _ctx += f"  {t}: {_row[t]:.3f}% ({chg:+.0f}bp)\n"
            if "2Y" in _row.index and "10Y" in _row.index:
                _ctx += f"  2s10s spread: {(_row['10Y'] - _row['2Y'])*100:.0f}bp\n"
            ai_summary("rates", _ctx)

        # ── Yield Curve Chart ──
        st.markdown("#### US Treasury Par Yield Curve")
        fig_curve = go.Figure()

        def get_curve_at_date(df, target_date, label, color, dash="solid", width=2):
            # Find nearest available date
            available = df.index[df.index <= target_date]
            if len(available) == 0:
                return None
            row = df.loc[available[-1]]
            vals = [row.get(t, np.nan) for t in TENOR_ORDER]
            return go.Scatter(
                x=TENOR_ORDER, y=vals, name=f"{label} ({available[-1].strftime('%m/%d')})",
                line=dict(color=color, dash=dash, width=width),
                hovertemplate="%{x}: %{y:.3f}%<extra></extra>"
            )

        today = yield_df.index[-1]
        overlays = [
            (today, "Today", BLUE, "solid", 3),
            (today - timedelta(days=7), "1W Ago", MUTED, "dot", 1.5),
            (today - timedelta(days=30), "1M Ago", YELLOW, "dash", 1.5),
            (today - timedelta(days=91), "3M Ago", CYAN, "dash", 1.5),
            (today - timedelta(days=365), "1Y Ago", RED, "dash", 1.5),
        ]

        for target, label, color, dash, width in overlays:
            trace = get_curve_at_date(yield_df, target, label, color, dash, width)
            if trace:
                fig_curve.add_trace(trace)

        # Annotate 3M10Y spread
        try:
            today_row = yield_df.iloc[-1]
            spread_3m10y = today_row.get("10Y", np.nan) - today_row.get("3M", np.nan)
            if not pd.isna(spread_3m10y):
                inv_text = " ⚠️ INVERTED" if spread_3m10y < 0 else ""
                fig_curve.add_annotation(
                    x="5Y", y=max(today_row.dropna()) + 0.15,
                    text=f"3M10Y: {spread_3m10y:+.0f}bp{inv_text}",
                    showarrow=False, font=dict(size=12, color=RED if spread_3m10y < 0 else GREEN, family="JetBrains Mono"),
                    bgcolor="#1c2333", bordercolor="#30363d", borderwidth=1, borderpad=4,
                )
        except Exception:
            pass

        fig_curve.update_layout(make_layout("", height=380))
        fig_curve.update_layout(yaxis_title="Yield (%)", xaxis_title="Tenor")
        st.plotly_chart(fig_curve, use_container_width=True)

        # ── Curve Spreads & Regime ──
        st.markdown("#### Curve Spreads & Regime")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            spread_lb = st.selectbox("Chart Lookback", ["1Y", "2Y", "5Y", "10Y", "Max"], index=["1Y", "2Y", "5Y", "10Y", "Max"].index(global_lookback if global_lookback in ["1Y", "2Y", "5Y", "10Y"] else "1Y"), key="spread_lb")
        with col2:
            curve_pair = st.selectbox("Curve Pair", ["2s10s", "2s30s", "5s30s", "3M10Y"], index=0, key="curve_pair")
        with col3:
            calc_timeframe = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], index=0, key="calc_tf")
        with col4:
            lookback_period = st.number_input("Change Lookback", min_value=1, value=50, step=1, key="lb_period")

        # Determine how much prior history we need to fetch to calculate the shifted properties
        # before truncating to the actual view window.
        buffer_days = lookback_period * 2  # Daily buffer
        if calc_timeframe == "Weekly":
            buffer_days = lookback_period * 8
        elif calc_timeframe == "Monthly":
            buffer_days = lookback_period * 32
            
        chart_start_date = lookback_date(spread_lb)
        if spread_lb == "Max":
            fetch_start_date = None
        else:
            fetch_start_date = chart_start_date - timedelta(days=buffer_days)

        fetch_start_str = fetch_start_date.strftime("%Y-%m-%d") if fetch_start_date else None
        pair_map = {"2s10s": ("DGS10", "DGS2"), "2s30s": ("DGS30", "DGS2"), "5s30s": ("DGS30", "DGS5"), "3M10Y": ("DGS10", "DGS3MO")}
        long_sym, short_sym = pair_map[curve_pair]
        
        spread_series = fetch_fred_multi([long_sym, short_sym], start=fetch_start_str)

        if not spread_series.empty and long_sym in spread_series and short_sym in spread_series:
            df_curve = spread_series.copy()
            # Forward fill to prevent gaps when one FRED series is updated before another
            df_curve = df_curve.ffill()
            
            if calc_timeframe == "Weekly":
                df_curve = df_curve.resample("W-FRI").last().dropna()
            elif calc_timeframe == "Monthly":
                df_curve = df_curve.resample("ME").last().dropna()
            else:
                df_curve = df_curve.dropna()

            df_curve["long_yield"] = df_curve[long_sym]
            df_curve["short_yield"] = df_curve[short_sym]
            df_curve["yield_spread"] = (df_curve["long_yield"] - df_curve["short_yield"]) * 100

            df_curve["yield_spread_prev"] = df_curve["yield_spread"].shift(lookback_period)
            df_curve["long_yield_prev"] = df_curve["long_yield"].shift(lookback_period)
            df_curve["short_yield_prev"] = df_curve["short_yield"].shift(lookback_period)

            df_curve["spread_change"] = df_curve["yield_spread"] - df_curve["yield_spread_prev"]
            df_curve["long_yield_change"] = df_curve["long_yield"] - df_curve["long_yield_prev"]
            df_curve["short_yield_change"] = df_curve["short_yield"] - df_curve["short_yield_prev"]

            # Trim to the actual requested view window to strip the buffer
            if spread_lb != "Max":
                df_curve = df_curve[df_curve.index >= chart_start_date]

            sc = df_curve["spread_change"]
            lc = df_curve["long_yield_change"]
            shc = df_curve["short_yield_change"]
            conds = [
                sc.isna() | lc.isna() | shc.isna(),
                (lc > 0) & (shc < 0),
                (lc < 0) & (shc > 0),
                (sc > 0) & (lc > 0),
                (sc > 0) & (lc < 0),
                (sc < 0) & (lc > 0),
                (sc < 0) & (lc < 0),
            ]
            color_choices = ["rgba(128,128,128,0.5)", "#f97316", "#eab308", "#ef4444", "#22c55e", "#a855f7", "#3b82f6"]
            text_choices = ["Neutral", "Steepener Twist", "Flattener Twist", "Bear Steepener", "Bull Steepener", "Bear Flattener", "Bull Flattener"]
            df_curve["regime_color"] = np.select(conds, color_choices, default="rgba(128,128,128,0.5)")
            df_curve["regime_text"] = np.select(conds, text_choices, default="Neutral")

            fig_spreads = go.Figure()

            # Regime bars
            fig_spreads.add_trace(go.Bar(
                x=df_curve.index, y=df_curve["yield_spread"],
                name="Regime", marker_color=df_curve["regime_color"],
                opacity=0.8,
                hovertemplate="Spread: %{y:.0f}bp<br>Regime: %{customdata}<extra></extra>",
                customdata=df_curve["regime_text"]
            ))

            # Spread line
            fig_spreads.add_trace(go.Scatter(
                x=df_curve.index, y=df_curve["yield_spread"],
                name="Yield Spread", line=dict(color="rgba(255,255,255,0.7)", width=2),
                mode="lines", hoverinfo="skip"
            ))

            # Zero line
            fig_spreads.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=1)
            
            # Recession shading
            recessions = fetch_recession_dates()
            add_recession_shading(fig_spreads, recessions, x_min=df_curve.index.min(), x_max=df_curve.index.max())

            # Annotate current value
            last_val = df_curve["yield_spread"].iloc[-1]
            last_regime = df_curve["regime_text"].iloc[-1]
            last_color = df_curve["regime_color"].iloc[-1]
            if not pd.isna(last_val):
                fig_spreads.add_annotation(
                    x=df_curve.index[-1], y=last_val,
                    text=f" {last_val:.0f}bp ({last_regime})", showarrow=False,
                    xanchor="left", font=dict(size=11, color=last_color),
                )

            fig_spreads.update_layout(make_layout("", height=350))
            fig_spreads.update_layout(yaxis_title="Basis Points", barmode='relative')
            
            if calc_timeframe == "Daily":
                missing_bdays = pd.bdate_range(start=df_curve.index.min(), end=df_curve.index.max()).difference(df_curve.index)
                fig_spreads.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]), # Weekends
                        dict(values=missing_bdays.strftime("%Y-%m-%d").tolist()) # Dynamic holidays
                    ]
                )
                
            st.plotly_chart(fig_spreads, use_container_width=True)

            # Legend
            st.markdown(
                '<div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; font-size:0.8rem; margin-top: -15px;">'
                '<span style="color:#ef4444;">■ Bear Steepener</span>'
                '<span style="color:#22c55e;">■ Bull Steepener</span>'
                '<span style="color:#a855f7;">■ Bear Flattener</span>'
                '<span style="color:#3b82f6;">■ Bull Flattener</span>'
                '<span style="color:#f97316;">■ Steepener Twist</span>'
                '<span style="color:#eab308;">■ Flattener Twist</span>'
                '</div>', unsafe_allow_html=True
            )

        # ── Real Rates & Breakevens ──
        st.markdown("#### Real Rates & Breakevens")
        real_start = lookback_date(global_lookback).strftime("%Y-%m-%d")
        real_series = fetch_fred_multi(["DFII5", "DFII10", "T5YIFR", "T10YIE"], start=real_start)
        if not real_series.empty:
            fig_real = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=("Real Yields (TIPS)", "Breakeven Inflation"),
            )
            if "DFII5" in real_series:
                fig_real.add_trace(go.Scatter(x=real_series.index, y=real_series["DFII5"], name="5Y Real", line=dict(color=BLUE, width=1.5)), row=1, col=1)
            if "DFII10" in real_series:
                fig_real.add_trace(go.Scatter(x=real_series.index, y=real_series["DFII10"], name="10Y Real", line=dict(color=CYAN, width=1.5)), row=1, col=1)
            if "T10YIE" in real_series:
                fig_real.add_trace(go.Scatter(x=real_series.index, y=real_series["T10YIE"], name="10Y BEI", line=dict(color=YELLOW, width=1.5)), row=2, col=1)
            if "T5YIFR" in real_series:
                fig_real.add_trace(go.Scatter(x=real_series.index, y=real_series["T5YIFR"], name="5Y5Y Fwd BEI", line=dict(color=RED, width=1.5, dash="dot")), row=2, col=1)
            # Zero line on real yields panel
            fig_real.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=1, row=1, col=1)
            layout = make_layout("", height=460)
            fig_real.update_layout(layout)
            fig_real.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            fig_real.update_yaxes(title_text="Yield (%)", row=1, col=1, gridcolor="#21262d")
            fig_real.update_yaxes(title_text="Rate (%)", row=2, col=1, gridcolor="#21262d")
            st.plotly_chart(fig_real, use_container_width=True)

            # ── 5Y5Y Forward Real Rate (r-star proxy) ──
            st.markdown("#### Real Forward Dynamics (5Y5Y)")
            st.caption("The 5-year real rate, 5 years forward. Often used as a market-based proxy for 'r-star' (the neutral real rate). Implied from TIPS yields: (10 × 10Y Real - 5 × 5Y Real) / 5.")
            
            if "DFII5" in real_series and "DFII10" in real_series:
                # 5Y5Y Forward Real Rate = (10 * 10Y_Real - 5 * 5Y_Real) / 5
                # Simplified: 2 * 10Y_Real - 5Y_Real
                fwd_real = (2 * real_series["DFII10"] - real_series["DFII5"]).dropna()
                
                if not fwd_real.empty:
                    fig_fwd = go.Figure()
                    fig_fwd.add_trace(go.Scatter(
                        x=fwd_real.index, y=fwd_real.values, name="5Y5Y Fwd Real",
                        line=dict(color=CYAN, width=2), fill="tozeroy",
                        fillcolor="rgba(57,210,192,0.05)",
                    ))
                    y_pad = (fwd_real.max() - fwd_real.min()) * 0.1
                    fig_fwd.update_layout(make_layout("", height=320))
                    fig_fwd.update_layout(
                        yaxis_title="Implied Real Rate (%)",
                        yaxis=dict(range=[fwd_real.min() - y_pad, fwd_real.max() + y_pad]),
                    )
                    st.plotly_chart(fig_fwd, use_container_width=True)
            else:
                st.info("Insufficient TIPS data to compute 5Y5Y Real Forward.")
        else:
            st.info("Real rates data unavailable.")

        # ── SOFR / Policy Rate ──
        st.markdown("#### SOFR & Fed Funds Rate")
        policy_start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        policy_series = fetch_fred_multi(["DFF", "SOFR"], start=policy_start)
        if not policy_series.empty:
            fig_policy = go.Figure()
            if "DFF" in policy_series:
                fig_policy.add_trace(go.Scatter(x=policy_series.index, y=policy_series["DFF"], name="Eff. Fed Funds", line=dict(color=BLUE, width=2)))
            if "SOFR" in policy_series:
                fig_policy.add_trace(go.Scatter(x=policy_series.index, y=policy_series["SOFR"], name="SOFR", line=dict(color=CYAN, width=1.5, dash="dot")))
            fig_policy.update_layout(make_layout("", height=320))
            fig_policy.update_layout(yaxis_title="Rate (%)")
            st.plotly_chart(fig_policy, use_container_width=True)
        else:
            st.info("Policy rate data unavailable.")


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — POLICY PATH
# ═════════════════════════════════════════════════════════════════════
with tab2:
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_ff_futures():
        """Fetch Fed Funds Futures strip from yfinance."""
        months = {"F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
                  "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec"}
        month_codes =["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
        now = datetime.now()

        tickers = []
        labels =[]
        # FIX: Increased from 18 to 24 months to ensure we have the "next month" 
        # contract for CME math on late-year meetings.
        for i in range(24):
            m_idx = (now.month - 1 + i) % 12
            year = now.year + (now.month - 1 + i) // 12
            code = month_codes[m_idx]
            yr = str(year)[-2:]
            ticker = f"ZQ{code}{yr}.CBT"
            label = f"{months[code]} {year}"
            tickers.append(ticker)
            labels.append(label)

        results =[]
        closes = _yf_closes(tickers, period="5d")
        for ticker, label in zip(tickers, labels):
            if ticker in closes.columns:
                s = closes[ticker].dropna()
                if not s.empty:
                    price = s.iloc[-1]
                    implied_rate = 100 - price
                    results.append({"contract": label, "ticker": ticker, "price": price, "implied_rate": implied_rate})
        return pd.DataFrame(results)

    ff_df = fetch_ff_futures()
    _effr_tab2 = fetch_fred_series("DFF")

    # ── AI Summary ──
    _policy_ctx = ""
    if len(_effr_tab2) > 0:
        _policy_ctx += f"Current EFFR: {_effr_tab2.iloc[-1]:.2f}%\n"
    if not ff_df.empty:
        _policy_ctx += "Fed Funds Futures strip (implied rates):\n"
        for _, r in ff_df.head(6).iterrows():
            _policy_ctx += f"  {r['contract']}: {r['implied_rate']:.3f}%\n"
    if _policy_ctx:
        ai_summary("policy", _policy_ctx)

    # ── FedWatch-Style Per-Meeting Probabilities ──
    st.markdown("#### FOMC Meeting Probabilities — FedWatch Style")
    st.caption("Interpolated from monthly FF futures (CME FedWatch methodology). "
               "**P(Cut)** is the cumulative probability that the target rate is net lower "
               "than today by that meeting, built from the marginal 25bp step tree.")

    def get_dynamic_fomc_dates(num_upcoming=12):
        """Dynamically generates future FOMC dates so the app never breaks."""
        today = date.today()
        # Known standard schedule
        known_dates =[
            # 2026
            date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6), date(2026, 6, 17),
            date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 16),
            # 2027
            date(2027, 1, 27), date(2027, 3, 17), date(2027, 5, 5), date(2027, 6, 16),
            date(2027, 7, 28), date(2027, 9, 22), date(2027, 10, 27), date(2027, 12, 15),
        ]
        future_dates = [d for d in known_dates if d > today]
        
        # Algorithmic fallback if we run out of known dates (rolls forward infinitely)
        standard_months =[1, 3, 5, 6, 7, 9, 10, 12]
        last_year = future_dates[-1].year if future_dates else today.year
        while len(future_dates) < num_upcoming:
            last_year += 1
            for m in standard_months:
                approx_date = date(last_year, m, 16) # ~Mid-month approximation
                if approx_date > today:
                    future_dates.append(approx_date)
                    
        return future_dates[:num_upcoming]

    FOMC_DATES = get_dynamic_fomc_dates()

    def compute_fedwatch(contracts_dict, current_effr, fomc_dates):
        """
        Bloomberg WIRP-style FedWatch.
        Maps EFFR to Target Midpoint and builds a step-function probability tree.
        """
        # EFFR is usually ~7-8bps below the Target Upper Bound (e.g. 5.33 in a 5.25-5.50 range)
        # We assume target midpoint is ~4-5bps above current EFFR.
        EFFR_TARGET_SPREAD = 0.045 
        current_target_mid = current_effr + EFFR_TARGET_SPREAD

        month_key = lambda d: f"{d.strftime('%b')} {d.year}"
        results =[]
        prev_post_rate = None
        prev_mtg = None
        cum_prob_dist = {0: 1.0}

        for mtg in fomc_dates:
            mk = month_key(mtg)
            if mk not in contracts_dict:
                prev_post_rate = None
                prev_mtg = mtg
                continue

            month_rate = contracts_dict[mk]
            days_in_month = calendar.monthrange(mtg.year, mtg.month)[1]
            d_b = mtg.day
            d_a = days_in_month - d_b

            # 1. Establish Pre-Meeting Rate
            use_chain = (prev_post_rate is not None and prev_mtg is not None 
                         and prev_mtg.year == mtg.year and prev_mtg.month == mtg.month)
            if use_chain:
                pre_rate = prev_post_rate
            else:
                prior_month = mtg.month - 1 if mtg.month > 1 else 12
                prior_year = mtg.year if mtg.month > 1 else mtg.year - 1
                prior_key = month_key(date(prior_year, prior_month, 1))
                
                if prior_key in contracts_dict:
                    pre_rate = contracts_dict[prior_key]
                elif prev_post_rate is not None:
                    pre_rate = prev_post_rate
                else:
                    pre_rate = current_effr

            # 2. Extract Implied Post-Meeting Rate
            next_month = mtg.month + 1 if mtg.month < 12 else 1
            next_year = mtg.year if mtg.month < 12 else mtg.year + 1
            next_key = month_key(date(next_year, next_month, 1))
            
            next_mtgs =[m for m in fomc_dates if m.year == next_year and m.month == next_month]
            has_next_month_mtg = len(next_mtgs) > 0
            next_mtg_day = next_mtgs[0].day if has_next_month_mtg else 31

            if not has_next_month_mtg and next_key in contracts_dict:
                post_rate = contracts_dict[next_key]
            elif d_a <= 10 and next_key in contracts_dict and next_mtg_day >= 15:
                post_rate = contracts_dict[next_key]
            elif d_a <= 10 and next_key not in contracts_dict:
                post_rate = month_rate
            else:
                if d_a == 0: d_a = 1
                post_rate = (month_rate * days_in_month - pre_rate * d_b) / d_a

            # Sanity clamp
            if post_rate < 0 or abs(post_rate - month_rate) > 0.75:
                post_rate = contracts_dict.get(next_key, month_rate)

            # 3. Step-Function Probabilities (25bp nodes)
            # We calculate the delta from the original current target midpoint
            delta_from_start = (post_rate + EFFR_TARGET_SPREAD - current_target_mid) * 100
            m_moves = delta_from_start / 25.0
            
            # Marginal calculation (from previous meeting rate)
            delta_marginal = (post_rate - pre_rate) * 100
            m_marginal = delta_marginal / 25.0
            
            lower_step = math.floor(m_marginal)
            upper_step = lower_step + 1
            prob_upper = m_marginal - lower_step
            prob_lower = 1.0 - prob_upper
            
            marginal_probs = {lower_step * 25: prob_lower, upper_step * 25: prob_upper}

            # 4. Advance Probability Tree
            new_cum_dist = defaultdict(float)
            for cum_bp, cum_prob in cum_prob_dist.items():
                for marg_bp, marg_prob in marginal_probs.items():
                    new_cum_dist[cum_bp + marg_bp] += cum_prob * marg_prob
            cum_prob_dist = dict(new_cum_dist)

            # 5. Extract Step Probabilities for UI (e.g. Prob of -25, -50)
            # Bloomberg WIRP shows the probability of specific "buckets"
            p_25 = sum(p for bp, p in marginal_probs.items() if abs(bp) == 25)
            p_50 = sum(p for bp, p in marginal_probs.items() if abs(bp) == 50)
            
            expected_cum_bp = sum(bp * p for bp, p in cum_prob_dist.items())

            results.append({
                "meeting": mtg.strftime("%b %d, %Y"),
                "implied_rate": post_rate + EFFR_TARGET_SPREAD,
                "delta_bp": delta_marginal,
                "p_25": p_25,
                "p_50": p_50,
                "cum_bp": expected_cum_bp,
                "cum_moves": expected_cum_bp / 25.0,
            })
            prev_post_rate = post_rate

        return results

    if not ff_df.empty:
        contracts_dict = dict(zip(ff_df["contract"], ff_df["implied_rate"]))
        effr_for_fw = _effr_tab2.iloc[-1] if len(_effr_tab2) > 0 else None

        if effr_for_fw is not None:
            fw_results = compute_fedwatch(contracts_dict, effr_for_fw, FOMC_DATES)

            if fw_results:
                # ── Probability table (Bloomberg WIRP Style) ──
                grid_cols = "120px 80px 85px 85px 75px 75px 75px"
                tbl_hdr = f'<div style="display:grid; grid-template-columns:{grid_cols}; gap:6px; padding:4px 8px; font-size:0.65rem; color:#8b949e; font-family:JetBrains Mono,monospace; border-bottom:1px solid #30363d; text-transform:uppercase;">'
                tbl_hdr += '<span>Meeting</span><span style="text-align:right">#Hikes/Cuts</span><span style="text-align:right">Imp. Rate</span><span style="text-align:right">Imp. Δ</span><span style="text-align:right">Prob 0</span><span style="text-align:right">Prob 25</span><span style="text-align:right">Prob 50</span></div>'
                st.markdown(tbl_hdr, unsafe_allow_html=True)

                for r in fw_results:
                    cm = r["cum_moves"]
                    if abs(cm) < 0.05:
                        cum_label, cum_color = "—", MUTED
                    elif cm < 0:
                        cum_label, cum_color = f"{abs(cm):.1f} cuts", GREEN
                    else:
                        cum_label, cum_color = f"{cm:.1f} hikes", RED

                    p0_val = 1.0 - (r["p_25"] + r["p_50"])
                    
                    row = f'<div style="display:grid; grid-template-columns:{grid_cols}; gap:6px; padding:4px 8px; font-size:0.75rem; font-family:JetBrains Mono,monospace; color:#e6edf3; border-bottom:1px solid #161b22;">'
                    row += f'<span style="color:#8b949e;">{r["meeting"]}</span>'
                    row += f'<span style="text-align:right; color:{cum_color}; font-weight:600;">{cum_label}</span>'
                    row += f'<span style="text-align:right; font-weight:600;">{r["implied_rate"]:.3f}%</span>'
                    row += f'<span style="text-align:right; color:{BLUE};">{r["delta_bp"]:+.1f}</span>'
                    row += f'<span style="text-align:right; color:{MUTED};">{p0_val*100:.0f}%</span>'
                    row += f'<span style="text-align:right; color:{GREEN if r["delta_bp"] < 0 else RED};">{r["p_25"]*100:.0f}%</span>'
                    row += f'<span style="text-align:right; color:{GREEN if r["delta_bp"] < 0 else RED}; font-weight:700;">{r["p_50"]*100:.0f}%</span>'
                    row += '</div>'
                    st.markdown(row, unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

                # ── FedWatch-style chart ──
                fw_df = pd.DataFrame(fw_results)
                fig_fw = go.Figure()

                bar_colors =[GREEN if v < -0.05 else RED if v > 0.05 else MUTED for v in fw_df["cum_moves"]]
                bar_hover = []
                for m, v in zip(fw_df["meeting"], fw_df["cum_moves"]):
                    if abs(v) < 0.05:
                        bar_hover.append(f"{m}<br>No change priced")
                    elif v < 0:
                        bar_hover.append(f"{m}<br>{abs(v):.1f} cuts priced")
                    else:
                        bar_hover.append(f"{m}<br>{v:.1f} hikes priced")
                fig_fw.add_trace(go.Bar(
                    x=fw_df["meeting"], y=fw_df["cum_moves"],
                    name="# Hikes/Cuts", marker_color=bar_colors, opacity=0.75,
                    yaxis="y2", hovertext=bar_hover, hovertemplate="%{hovertext}<extra></extra>",
                ))

                fig_fw.add_trace(go.Scatter(
                    x=fw_df["meeting"], y=fw_df["post_rate"],
                    name="Implied Policy Rate (%)",
                    line=dict(color=BLUE, width=3), mode="lines+markers",
                    marker=dict(size=7, color=BLUE),
                    hovertemplate="%{x}<br>Implied rate: %{y:.3f}%<extra></extra>",
                ))

                fig_fw.add_trace(go.Scatter(
                    x=["Current"], y=[effr_for_fw],
                    mode="markers+text", marker=dict(size=10, color="white", line=dict(color=BLUE, width=2)),
                    text=[f"{effr_for_fw:.2f}%"], textposition="top center", textfont=dict(color="white", size=11),
                    showlegend=False, hovertemplate="Current EFFR: %{y:.3f}%<extra></extra>",
                ))

                fig_fw.add_hline(y=effr_for_fw, line_dash="solid", line_color="rgba(255,255,255,0.25)", line_width=1)
                fig_fw.update_layout(make_layout("Implied Overnight Rate & Number of Cuts Priced In", height=450))
                fig_fw.update_layout(
                    xaxis=dict(tickangle=-45, categoryorder="array", categoryarray=["Current"] + fw_df["meeting"].tolist(), gridcolor="#21262d"),
                    yaxis=dict(title="Implied Policy Rate (%)", side="left", gridcolor="#21262d"),
                    yaxis2=dict(title="# Hikes/Cuts", overlaying="y", side="right", showgrid=False, zeroline=True, zerolinecolor="rgba(255,255,255,0.15)", dtick=1, rangemode="tozero"),
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, font=dict(size=10)), bargap=0.3,
                )
                st.plotly_chart(fig_fw, use_container_width=True)
            else:
                st.info("No upcoming FOMC meetings found in contract range.")
        else:
            st.info("EFFR data unavailable — cannot compute meeting probabilities.")
    else:
        st.info("FF Futures data unavailable — cannot compute meeting probabilities.")

    st.divider()
    st.markdown("#### FOMC SEP Median Dots vs Market Pricing")
    st.caption("SEP Medians from Dec 2024 projections — update quarterly")

    sep_data = {"2025": 3.875, "2026": 3.375, "2027": 3.125, "Longer Run": 3.00}
    market_implied = {}
    if not ff_df.empty:
        for yr in["2025", "2026", "2027"]:
            yr_contracts = ff_df[ff_df["contract"].str.endswith(yr)]
            if not yr_contracts.empty:
                market_implied[yr] = yr_contracts.iloc[-1]["implied_rate"]

    if sep_data:
        fig_dots = go.Figure()
        years = list(sep_data.keys())
        fig_dots.add_trace(go.Bar(
            x=years, y=[sep_data[y] for y in years],
            name="SEP Median", marker_color=YELLOW, opacity=0.8,
            text=[f"{sep_data[y]:.3f}%" for y in years], textposition="outside",
        ))
        if market_implied:
            mkt_years = list(market_implied.keys())
            fig_dots.add_trace(go.Bar(
                x=mkt_years, y=[market_implied[y] for y in mkt_years],
                name="Market Implied", marker_color=BLUE, opacity=0.8,
                text=[f"{market_implied[y]:.3f}%" for y in mkt_years], textposition="outside",
            ))
        fig_dots.update_layout(make_layout("", height=350, barmode="group"))
        fig_dots.update_layout(yaxis_title="Rate (%)")
        st.plotly_chart(fig_dots, use_container_width=True)

    st.markdown("#### SOFR Futures Strip (SR1)")
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_sofr_futures():
        month_codes =["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
        months_map = {"F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
                      "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec"}
        now = datetime.now()
        tickers = []
        labels = []
        for i in range(8):
            m_idx = (now.month - 1 + i) % 12
            year = now.year + (now.month - 1 + i) // 12
            code = month_codes[m_idx]
            yr = str(year)[-2:]
            ticker = f"SR1{code}{yr}.CME"
            label = f"{months_map[code]} {year}"
            tickers.append(ticker)
            labels.append(label)

        results =[]
        closes = _yf_closes(tickers, period="5d")
        for ticker, label in zip(tickers, labels):
            if ticker in closes.columns:
                s = closes[ticker].dropna()
                if not s.empty:
                    results.append({"contract": label, "implied_rate": 100 - s.iloc[-1]})
        return pd.DataFrame(results)

    sofr_strip = fetch_sofr_futures()
    if not sofr_strip.empty:
        fig_sofr = go.Figure()
        fig_sofr.add_trace(go.Scatter(
            x=sofr_strip["contract"], y=sofr_strip["implied_rate"],
            mode="lines+markers+text",
            line=dict(color=CYAN, width=2),
            marker=dict(color=CYAN, size=8),
            text=[f"{r:.3f}%" for r in sofr_strip["implied_rate"]], textposition="top center",
        ))
        fig_sofr.update_layout(make_layout("", height=320))
        fig_sofr.update_layout(yaxis_title="Implied Rate (%)", xaxis_tickangle=-45)
        st.plotly_chart(fig_sofr, use_container_width=True)
    else:
        st.info("SOFR futures data unavailable via yfinance.")

# ═════════════════════════════════════════════════════════════════════
# TAB 3 — CROSS-ASSET DASHBOARD
# ═════════════════════════════════════════════════════════════════════
with tab3:
    ASSET_GROUPS = {
        "Equity Indices": {
            "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Russell 2000": "^RUT", "DJIA": "^DJI",
            "DAX": "^GDAXI", "Nikkei": "^N225", "Hang Seng": "^HSI", "Euro Stoxx 50": "^STOXX50E",
            "MSCI EM (EEM)": "EEM",
        },
        "Rates / Bond ETFs": {
            "SHY (1-3Y)": "SHY", "IEF (7-10Y)": "IEF", "TLT (20Y+)": "TLT",
            "HYG (HY Credit)": "HYG", "LQD (IG Credit)": "LQD",
            "EMB (EM Sov)": "EMB", "TIP (TIPS)": "TIP",
        },
        "FX": {
            "DXY": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X",
            "GBP/USD": "GBPUSD=X", "USD/CNY": "CNY=X", "AUD/USD": "AUDUSD=X",
        },
        "Commodities": {
            "Gold": "GC=F", "Silver": "SI=F", "WTI Crude": "CL=F",
            "Brent Crude": "BZ=F", "Copper": "HG=F", "Natural Gas": "NG=F",
        },
        "Vol / Risk": {
            "VIX": "^VIX", "VIX3M": "^VIX3M", "VIX6M": "^VIX6M", "MOVE": "^MOVE",
        },
    }

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_cross_asset_data(_ticker_tuple):
        all_tickers = list(_ticker_tuple)
        # Fetch 1 year of data for all calcs
        return _yf_closes(all_tickers, period="1y", interval="1d")

    all_tickers = []
    for group in ASSET_GROUPS.values():
        all_tickers.extend(group.values())
    cross_data = fetch_cross_asset_data(tuple(all_tickers))
    if not cross_data.empty and cross_data.index.tz is not None:
        cross_data.index = cross_data.index.tz_localize(None)

    # ── AI Summary ──
    if not cross_data.empty:
        _ca_ctx = "Cross-asset snapshot (1D change):\n"
        _key_assets = {"S&P 500": "^GSPC", "Nasdaq": "^NDX", "VIX": "^VIX",
                       "Gold": "GC=F", "WTI Crude": "CL=F", "DXY": "DX-Y.NYB",
                       "TLT (20Y+)": "TLT", "HYG (HY)": "HYG"}
        for name, tk in _key_assets.items():
            if tk in cross_data.columns:
                s = cross_data[tk].dropna()
                if len(s) >= 2:
                    prc = s.iloc[-1]
                    chg = (s.iloc[-1] / s.iloc[-2] - 1) * 100
                    _ca_ctx += f"  {name}: {prc:.2f} ({chg:+.2f}%)\n"
        ai_summary("cross_asset", _ca_ctx)

    def calc_asset_stats(df, ticker):
        """Calculate price, changes, sparkline for a ticker."""
        if df.empty or ticker not in df.columns:
            return None
        s = df[ticker].dropna()
        if len(s) < 2:
            return None
        current = s.iloc[-1]
        stats = {"price": current}
        # 1D change
        if len(s) >= 2:
            stats["1D"] = safe_pct_change(s.iloc[-1], s.iloc[-2])
        # 1W
        if len(s) >= 6:
            stats["1W"] = safe_pct_change(current, s.iloc[-6] if len(s) >= 6 else s.iloc[0])
        # 1M
        idx_1m = s.index[s.index <= s.index[-1] - timedelta(days=28)]
        if len(idx_1m) > 0:
            stats["1M"] = safe_pct_change(current, s.loc[idx_1m[-1]])
        # 3M
        idx_3m = s.index[s.index <= s.index[-1] - timedelta(days=88)]
        if len(idx_3m) > 0:
            stats["3M"] = safe_pct_change(current, s.loc[idx_3m[-1]])
        # YTD
        year_start = pd.Timestamp(s.index[-1].year, 1, 1)
        idx_ytd = s.index[s.index >= year_start]
        if len(idx_ytd) > 0:
            stats["YTD"] = safe_pct_change(current, s.loc[idx_ytd[0]])
        # Sparkline data (last 30 days)
        stats["spark"] = s.tail(30).values.tolist()
        return stats

    def render_sparkline_svg(data, width=80, height=24):
        if not data or len(data) < 2:
            return ""
        mn, mx = min(data), max(data)
        rng = mx - mn if mx != mn else 1
        points = []
        for i, v in enumerate(data):
            x = i / (len(data) - 1) * width
            y = height - ((v - mn) / rng) * height
            points.append(f"{x:.1f},{y:.1f}")
        color = GREEN if data[-1] >= data[0] else RED
        polyline = " ".join(points)
        return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"><polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'

    for group_name, tickers in ASSET_GROUPS.items():
        st.markdown(f"##### {group_name}")
        # Compute stats for all assets in group
        asset_stats = []
        for name, ticker in tickers.items():
            stats = calc_asset_stats(cross_data, ticker)
            if stats:
                stats["name"] = name
                stats["ticker"] = ticker
                asset_stats.append(stats)
            else:
                asset_stats.append({"name": name, "ticker": ticker, "price": None})

        # Sort by 1D performance
        asset_stats.sort(key=lambda x: x.get("1D", 0) or 0, reverse=True)

        if not asset_stats:
            st.info(f"No data for {group_name}")
            continue

        # Header row
        hdr = '<div style="display: grid; grid-template-columns: 80px 60px 90px repeat(5, 70px); gap: 4px; padding: 2px 6px; font-size: 0.68rem; color: #8b949e; font-family: JetBrains Mono, monospace; border-bottom: 1px solid #21262d; text-transform: uppercase;">'
        hdr += '<span>Asset</span><span></span><span style="text-align:right">Price</span>'
        for period in ["1D", "1W", "1M", "3M", "YTD"]:
            hdr += f'<span style="text-align:right">{period}</span>'
        hdr += '</div>'
        st.markdown(hdr, unsafe_allow_html=True)

        for a in asset_stats:
            if a["price"] is None:
                row = f'<div style="display: grid; grid-template-columns: 80px 60px 90px repeat(5, 70px); gap: 4px; padding: 3px 6px; font-size: 0.76rem; font-family: JetBrains Mono, monospace; color: #484f58;"><span>{a["name"]}</span><span></span><span style="text-align:right">N/A</span></div>'
                st.markdown(row, unsafe_allow_html=True)
                continue

            price = a["price"]
            # Format price
            if price > 1000:
                p_str = f"{price:,.0f}"
            elif price > 10:
                p_str = f"{price:,.2f}"
            else:
                p_str = f"{price:,.4f}"

            spark_svg = render_sparkline_svg(a.get("spark", []))
            row = f'<div style="display: grid; grid-template-columns: 80px 60px 90px repeat(5, 70px); gap: 4px; padding: 3px 6px; font-size: 0.76rem; font-family: JetBrains Mono, monospace; color: #e6edf3; border-bottom: 1px solid #161b22;">'
            row += f'<span style="color: #8b949e; font-size: 0.7rem;">{a["name"]}</span>'
            row += f'<span>{spark_svg}</span>'
            row += f'<span style="text-align:right; font-weight: 600;">{p_str}</span>'
            for period in ["1D", "1W", "1M", "3M", "YTD"]:
                val = a.get(period)
                color = color_for_change(val)
                row += f'<span style="text-align:right; color: {color};">{fmt_pct(val)}</span>'
            row += '</div>'
            st.markdown(row, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 16px'></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 4 — VIX & VOL
# ═════════════════════════════════════════════════════════════════════
with tab4:
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_vix_data():
        tickers = ["^VIX", "^VIX3M", "^VIX6M", "^MOVE"]
        results = {}
        closes = _yf_closes(tickers, period="3y")
        for t in tickers:
            if t in closes.columns:
                s = closes[t].dropna()
                if not s.empty:
                    results[t] = s
        return results

    vix_data = fetch_vix_data()

    # ── AI Summary ──
    _vol_ctx = "Volatility snapshot:\n"
    for label, tk in [("VIX", "^VIX"), ("VIX3M", "^VIX3M"), ("VIX6M", "^VIX6M"), ("MOVE", "^MOVE")]:
        if tk in vix_data and len(vix_data[tk]) > 0:
            cur = vix_data[tk].iloc[-1]
            prev = vix_data[tk].iloc[-6] if len(vix_data[tk]) > 5 else cur
            _vol_ctx += f"  {label}: {cur:.1f} (1W chg: {cur - prev:+.1f})\n"
    if "^VIX" in vix_data and "^VIX3M" in vix_data and len(vix_data["^VIX"]) > 0 and len(vix_data["^VIX3M"]) > 0:
        ratio = vix_data["^VIX3M"].iloc[-1] / vix_data["^VIX"].iloc[-1]
        _vol_ctx += f"  VIX3M/VIX ratio: {ratio:.2f} ({'contango' if ratio > 1 else 'backwardation'})\n"
    ai_summary("vol", _vol_ctx)

    # ── VIX Term Structure ──
    st.markdown("#### VIX Term Structure")

    # Term structure bar chart
    vix_current = {}
    for label, ticker in [("VIX", "^VIX"), ("VIX3M", "^VIX3M"), ("VIX6M", "^VIX6M")]:
        if ticker in vix_data and len(vix_data[ticker]) > 0:
            vix_current[label] = vix_data[ticker].iloc[-1]

    if vix_current:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_ts = go.Figure()
            names = list(vix_current.keys())
            vals = [vix_current[n] for n in names]
            colors_bar = []
            for i, v in enumerate(vals):
                if i == 0 and len(vals) > 1 and v > vals[1]:
                    colors_bar.append(RED)  # backwardation
                else:
                    colors_bar.append(BLUE)
            fig_ts.add_trace(go.Bar(
                x=names, y=vals, marker_color=colors_bar,
                text=[f"{v:.1f}" for v in vals], textposition="outside",
            ))
            fig_ts.update_layout(make_layout("", height=300))
            st.plotly_chart(fig_ts, use_container_width=True)

        with col2:
            st.markdown("**Term Structure Metrics**")
            if "VIX" in vix_current and "VIX3M" in vix_current:
                ratio = vix_current["VIX3M"] / vix_current["VIX"]
                state = "Contango (risk-on)" if ratio > 1 else "**Backwardation (risk-off)**"
                st.metric("VIX3M/VIX Ratio", f"{ratio:.3f}")
                st.markdown(f"Structure: {state}")
            for name, val in vix_current.items():
                st.metric(name, f"{val:.2f}")

    # ── VIX Time Series ──
    st.markdown("#### VIX Time Series")
    vol_lb_col, _ = st.columns([1, 4])
    with vol_lb_col:
        vix_lb = st.selectbox("Lookback", ["3M", "1Y", "3Y", "Max"], index=1, key="vix_lb")

    if "^VIX" in vix_data:
        vix_series = vix_data["^VIX"]
        vix_series.index = vix_series.index.tz_localize(None) if vix_series.index.tz else vix_series.index
        start = lookback_date(vix_lb)
        vix_plot = vix_series[vix_series.index >= start]

        if len(vix_plot) > 0:
            fig_vix = go.Figure()
            fig_vix.add_trace(go.Scatter(
                x=vix_plot.index, y=vix_plot.values, name="VIX",
                line=dict(color=BLUE, width=1.5), fill="tozeroy",
                fillcolor="rgba(88,166,255,0.08)",
            ))
            # 20-day MA
            ma20 = vix_plot.rolling(20).mean()
            fig_vix.add_trace(go.Scatter(
                x=ma20.index, y=ma20.values, name="20d MA",
                line=dict(color=YELLOW, width=1, dash="dot"),
            ))
            fig_vix.update_layout(make_layout("", height=350))

            # Percentile ranks
            current_vix = vix_series.iloc[-1]
            pct_cols = st.columns(3)
            for col, (period, days) in zip(pct_cols, [("1Y", 252), ("3Y", 756), ("5Y", 1260)]):
                lookback_data = vix_series.tail(days)
                if len(lookback_data) > 0:
                    pct = (lookback_data < current_vix).sum() / len(lookback_data) * 100
                    col.metric(f"Percentile ({period})", f"{pct:.0f}th")
            st.plotly_chart(fig_vix, use_container_width=True)

    # ── Realized vs Implied Vol ──
    st.markdown("#### SPX Realized vs Implied Volatility")

    @st.cache_data(ttl=3600, show_spinner=False)
    def calc_vol_premium():
        try:
            closes = _yf_closes(["^GSPC", "^VIX"], period="2y")
            if closes.empty or "^GSPC" not in closes.columns or "^VIX" not in closes.columns:
                return pd.DataFrame()

            spx = closes["^GSPC"].dropna()
            vix_hist = closes["^VIX"].dropna()
            
            if spx.empty or vix_hist.empty:
                return pd.DataFrame()

            # Strip timezones so indexes align (SPX=New_York, VIX=Chicago)
            spx.index = spx.index.tz_localize(None) if spx.index.tz else spx.index
            vix_hist.index = vix_hist.index.tz_localize(None) if vix_hist.index.tz else vix_hist.index
            # Realized vol: 20-day annualized
            returns = spx.pct_change()
            realized = returns.rolling(20).std() * np.sqrt(252) * 100
            # Align dates
            df = pd.DataFrame({"Realized Vol (20d)": realized, "VIX (Implied)": vix_hist})
            df = df.dropna()
            df["Vol Risk Premium"] = df["VIX (Implied)"] - df["Realized Vol (20d)"]
            return df
        except Exception:
            return pd.DataFrame()

    vol_df = calc_vol_premium()
    if not vol_df.empty:
        vol_df.index = vol_df.index.tz_localize(None) if vol_df.index.tz else vol_df.index
        # Filter to lookback
        vol_start = lookback_date(global_lookback)
        vol_plot = vol_df[vol_df.index >= vol_start]
        if len(vol_plot) > 0:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vol_plot.index, y=vol_plot["VIX (Implied)"], name="VIX (Implied)",
                line=dict(color=BLUE, width=1.5),
            ))
            fig_vol.add_trace(go.Scatter(
                x=vol_plot.index, y=vol_plot["Realized Vol (20d)"], name="20d Realized",
                line=dict(color=CYAN, width=1.5),
            ))
            fig_vol.add_trace(go.Scatter(
                x=vol_plot.index, y=vol_plot["Vol Risk Premium"], name="Vol Risk Premium",
                line=dict(color=YELLOW, width=1, dash="dash"), fill="tozeroy",
                fillcolor="rgba(210,153,34,0.06)",
            ))
            fig_vol.add_hline(y=0, line_dash="dot", line_color=MUTED)
            fig_vol.update_layout(make_layout("", height=350))
            fig_vol.update_layout(yaxis_title="Volatility (%)")
            # Current premium
            curr_premium = vol_plot["Vol Risk Premium"].iloc[-1]
            st.markdown(f"**Current vol risk premium:** {curr_premium:.1f}pts (implied {'>' if curr_premium > 0 else '<'} realized)")
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("Volatility premium data unavailable.")

    # ── MOVE Index (Rates Vol) ──
    st.divider()
    st.markdown("#### MOVE Index — Bond Market Volatility")

    if "^MOVE" in vix_data:
        move_series = vix_data["^MOVE"]
        move_series.index = move_series.index.tz_localize(None) if move_series.index.tz else move_series.index

        # MOVE metrics + time series side by side
        move_m_col, move_ts_col = st.columns([1, 3])

        with move_m_col:
            current_move = move_series.iloc[-1]
            st.metric("MOVE", f"{current_move:.1f}")
            # Percentile ranks
            for period, days in [("1Y", 252), ("3Y", 756), ("5Y", 1260)]:
                lb_data = move_series.tail(days)
                if len(lb_data) > 0:
                    pct = (lb_data < current_move).sum() / len(lb_data) * 100
                    st.metric(f"Pctl ({period})", f"{pct:.0f}th")

        with move_ts_col:
            move_lb_col2, _ = st.columns([1, 3])
            with move_lb_col2:
                move_lb = st.selectbox("Lookback", ["3M", "1Y", "3Y", "Max"], index=1, key="move_lb")
            move_start = lookback_date(move_lb)
            move_plot = move_series[move_series.index >= move_start]
            if len(move_plot) > 0:
                fig_move = go.Figure()
                fig_move.add_trace(go.Scatter(
                    x=move_plot.index, y=move_plot.values, name="MOVE",
                    line=dict(color="#f0883e", width=1.5), fill="tozeroy",
                    fillcolor="rgba(240,136,62,0.08)",
                ))
                ma20_move = move_plot.rolling(20).mean()
                fig_move.add_trace(go.Scatter(
                    x=ma20_move.index, y=ma20_move.values, name="20d MA",
                    line=dict(color=YELLOW, width=1, dash="dot"),
                ))
                fig_move.update_layout(make_layout("", height=320))
                fig_move.update_layout(yaxis_title="MOVE Index")
                st.plotly_chart(fig_move, use_container_width=True)

        # VIX vs MOVE comparison (dual axis)
        st.markdown("#### VIX vs MOVE — Equity Vol vs Rates Vol")
        if "^VIX" in vix_data:
            vix_comp = vix_data["^VIX"].copy()
            vix_comp.index = vix_comp.index.tz_localize(None) if vix_comp.index.tz else vix_comp.index
            comp_start = lookback_date(move_lb)
            vix_c = vix_comp[vix_comp.index >= comp_start]
            move_c = move_series[move_series.index >= comp_start]

            if len(vix_c) > 0 and len(move_c) > 0:
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=vix_c.index, y=vix_c.values, name="VIX",
                    line=dict(color=BLUE, width=1.5),
                ))
                fig_comp.add_trace(go.Scatter(
                    x=move_c.index, y=move_c.values, name="MOVE",
                    line=dict(color="#f0883e", width=1.5), yaxis="y2",
                ))
                fig_comp.update_layout(make_layout("", height=350))
                fig_comp.update_layout(
                    yaxis=dict(title="VIX", gridcolor="#21262d"),
                    yaxis2=dict(title="MOVE", overlaying="y", side="right", gridcolor="#21262d", showgrid=False),
                )
                # Current ratio
                try:
                    ratio_vm = move_series.iloc[-1] / vix_comp.iloc[-1]
                    st.markdown(f"**MOVE/VIX ratio:** {ratio_vm:.1f}× — {'rates vol elevated vs equity vol' if ratio_vm > 4 else 'normal range'}")
                except Exception:
                    pass
                st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("MOVE Index data unavailable.")


# ═════════════════════════════════════════════════════════════════════
# TAB 5 — MACRO DATA
# ═════════════════════════════════════════════════════════════════════
with tab5:
    macro_lb_col, _ = st.columns([1, 4])
    with macro_lb_col:
        macro_lb = st.selectbox("Lookback", ["1Y", "2Y", "5Y", "10Y"], index=["1Y", "2Y", "5Y", "10Y"].index(global_lookback), key="macro_lb")
    macro_start = lookback_date(macro_lb).strftime("%Y-%m-%d")
    recessions = fetch_recession_dates()

    # ── AI Summary ──
    _macro_ctx = "Latest US macro data:\n"
    _macro_series = [("CPI YoY", "CPIAUCSL"), ("Core CPI YoY", "CPILFESL"),
                     ("Unemployment", "UNRATE"), ("EFFR", "DFF")]
    for label, sid in _macro_series:
        s = fetch_fred_series(sid)
        if len(s) > 12:
            cur = s.iloc[-1]
            if sid in ["CPIAUCSL", "CPILFESL"]:
                yoy = (cur / s.iloc[-13] - 1) * 100 if s.iloc[-13] != 0 else 0
                _macro_ctx += f"  {label}: {yoy:.1f}%\n"
            else:
                _macro_ctx += f"  {label}: {cur:.2f}%\n"
    ai_summary("macro", _macro_ctx)

    def make_macro_chart(series_dict, title, height=350, yoy_compute=None, mom_diff=None, ylabel="", chart_type="line", min_years=None):
        """Helper to build macro time series charts."""
        # Allow individual charts to override the global lookback floor (e.g. quarterly GDP
        # data is too sparse with a 1Y window).
        effective_start_dt = lookback_date(macro_lb)
        if min_years is not None:
            min_start_dt = pd.Timestamp.now().normalize() - timedelta(days=int(min_years * 365))
            if min_start_dt < effective_start_dt:
                effective_start_dt = min_start_dt
        effective_start_str = effective_start_dt.strftime("%Y-%m-%d")

        # For YoY/MoM transforms we need extra history, so fetch 2 extra years
        needs_extra = bool(yoy_compute or mom_diff)
        if needs_extra:
            extra_start = (effective_start_dt - timedelta(days=730)).strftime("%Y-%m-%d")
        else:
            extra_start = effective_start_str
        data = fetch_fred_multi(list(series_dict.values()), start=extra_start)
        if data.empty:
            st.info(f"{title}: data unavailable")
            return

        plot_start = pd.Timestamp(effective_start_str)
        fig = go.Figure()
        all_dates = []
        color_idx = 0
        for label, sid in series_dict.items():
            if sid not in data.columns:
                continue
            s = data[sid].dropna()
            if len(s) == 0:
                continue

            # Apply transformations
            if yoy_compute and sid in yoy_compute:
                s = s.pct_change(12) * 100  # YoY %
                s = s.dropna()

            if mom_diff and sid in mom_diff:
                s = s.diff()
                s = s.dropna()

            # Trim to the actual lookback window AFTER transformation
            s = s[s.index >= plot_start]

            if len(s) == 0:
                continue

            all_dates.extend(s.index.tolist())
            color = COLORS[color_idx % len(COLORS)]
            if chart_type == "stacked_bar":
                fig.add_trace(go.Bar(
                    x=s.index, y=s.values, name=label,
                    marker_color=color,
                    hovertemplate=f"{label}: " + "%{y:.2f}<extra></extra>",
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=label,
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"{label}: " + "%{y:.2f}<extra></extra>",
                ))

                # Annotate latest value (line charts only — stacked bars are illegible with overlapping annotations)
                last_date = s.index[-1]
                last_val = s.iloc[-1]
                fig.add_annotation(
                    x=last_date, y=last_val,
                    text=f" {last_val:.1f} ({last_date.strftime('%b %y')})",
                    showarrow=False, xanchor="left",
                    font=dict(size=9, color=color),
                )
            color_idx += 1

        if all_dates:
            add_recession_shading(fig, recessions, x_min=min(all_dates), x_max=max(all_dates))
        fig.update_layout(make_layout(title, height=height))
        if chart_type == "stacked_bar":
            fig.update_layout(barmode="stack")
        if ylabel:
            fig.update_layout(yaxis_title=ylabel)
        st.plotly_chart(fig, use_container_width=True)

    # ── Inflation ──
    st.markdown("#### Inflation")
    infl_col1, infl_col2 = st.columns(2)
    with infl_col1:
        make_macro_chart(
            {"CPI YoY": "CPIAUCSL", "Core CPI YoY": "CPILFESL"},
            "CPI & Core CPI (YoY %)", yoy_compute=["CPIAUCSL", "CPILFESL"], ylabel="%"
        )
    with infl_col2:
        make_macro_chart(
            {"PCE YoY": "PCEPI", "Core PCE YoY": "PCEPILFE"},
            "PCE & Core PCE (YoY %)", yoy_compute=["PCEPI", "PCEPILFE"], ylabel="%"
        )
    st.caption("5-year forward inflation expectation. Implied from the difference between nominal and inflation-indexed Treasury yields (FRED: T5YIFR). Often used to measure the anchoring of long-term inflation expectations.")
    make_macro_chart(
        {"5Y5Y Forward Inflation": "T5YIFR"},
        "5Y5Y Forward Inflation Expectation", ylabel="%"
    )

    # ── CPI Component Breakdown ──
    st.markdown("#### CPI Component Breakdown")
    make_macro_chart(
        {
            "Food": "CUSR0000SAF1",
            "Energy": "CPIENGSL",
            "Shelter": "CUSR0000SAH1",
            "Core Goods": "CUSR0000SACL1E",
            "Services": "CUSR0000SAS",
        },
        "CPI Major Components (YoY %)",
        height=380,
        yoy_compute=["CUSR0000SAF1", "CPIENGSL", "CUSR0000SAH1", "CUSR0000SACL1E", "CUSR0000SAS"],
        ylabel="%",
        chart_type="stacked_bar",
    )
    cpi_col1, cpi_col2 = st.columns(2)
    with cpi_col1:
        make_macro_chart(
            {
                "Shelter": "CUSR0000SAH1",
                "Owners' Equiv. Rent": "CUSR0000SEHA",
                "Rent of Primary Res.": "CUSR0000SAS11",
            },
            "Shelter Components (YoY %)",
            yoy_compute=["CUSR0000SAH1", "CUSR0000SEHA", "CUSR0000SAS11"],
            ylabel="%",
        )
    with cpi_col2:
        make_macro_chart(
            {
                "Used Cars & Trucks": "CUSR0000SETA02",
                "Medical Care": "CPIMEDSL",
                "Apparel": "CPIAPPSL",
            },
            "Selected CPI Components (YoY %)",
            yoy_compute=["CUSR0000SETA02", "CPIMEDSL", "CPIAPPSL"],
            ylabel="%",
        )

    # ── Labor Market ──
    st.markdown("#### Labor Market")
    labor_col1, labor_col2 = st.columns(2)
    with labor_col1:
        make_macro_chart({"Unemployment Rate": "UNRATE"}, "Unemployment Rate (%)", ylabel="%")
    with labor_col2:
        make_macro_chart(
            {"NFP (MoM Change, K)": "PAYEMS"},
            "Nonfarm Payrolls (MoM Change)", mom_diff=["PAYEMS"], ylabel="Thousands"
        )
    labor_col3, labor_col4 = st.columns(2)
    with labor_col3:
        make_macro_chart({"JOLTS Openings": "JTSJOL"}, "JOLTS Job Openings", ylabel="Thousands")
    with labor_col4:
        make_macro_chart({"Initial Claims": "ICSA"}, "Initial Jobless Claims", ylabel="Thousands")
    with st.columns(1)[0]:
        make_macro_chart(
            {"Avg Hourly Earnings": "CES0500000003"},
            "Avg Hourly Earnings (YoY %)", yoy_compute=["CES0500000003"], ylabel="%"
        )

    # ── Activity ──
    st.markdown("#### Activity")
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        make_macro_chart(
            {"Real GDP (QoQ% Ann.)": "A191RL1Q225SBEA"},
            "Real GDP (QoQ % Annualized)", ylabel="%", min_years=10
        )
    with act_col2:
        make_macro_chart(
            {"Nominal GDP (YoY%)": "GDP"},
            "Nominal GDP (YoY %)", yoy_compute=["GDP"], ylabel="%", min_years=10
        )
    
    act_col3, act_col4 = st.columns(2)
    with act_col3:
        make_macro_chart({"Industrial Production": "INDPRO"}, "Industrial Production Index", ylabel="Index")
    with act_col4:
        make_macro_chart({"Mfg Production": "IPMAN"}, "Industrial Production: Manufacturing", ylabel="Index")
    
    with st.columns(1)[0]:
        make_macro_chart({"CFNAI": "CFNAI"}, "Chicago Fed National Activity Index", ylabel="Index")

    # ── Consumer ──
    st.markdown("#### Consumer")
    cons_col1, cons_col2 = st.columns(2)
    with cons_col1:
        make_macro_chart(
            {"Retail Sales": "RSAFS"},
            "Retail Sales (YoY %)", yoy_compute=["RSAFS"], ylabel="%"
        )
    with cons_col2:
        make_macro_chart({"Michigan Sentiment": "UMCSENT"}, "U. of Michigan Consumer Sentiment", ylabel="Index")

    # ── Housing ──
    st.markdown("#### Housing")
    hous_col1, hous_col2 = st.columns(2)
    with hous_col1:
        make_macro_chart({"Building Permits": "PERMIT"}, "Building Permits", ylabel="Thousands")
    with hous_col2:
        make_macro_chart({"Housing Starts": "HOUST"}, "Housing Starts", ylabel="Thousands")

    # ── Financial Conditions ──
    st.markdown("#### Financial Conditions")
    fin_col1, fin_col2 = st.columns(2)
    with fin_col1:
        make_macro_chart(
            {"NFCI": "NFCI", "Fin. Stress (STLFSI)": "STLFSI4"},
            "Financial Conditions & Stress", ylabel="Index"
        )
    with fin_col2:
        make_macro_chart(
            {"HY OAS": "BAMLH0A0HYM2", "IG OAS": "BAMLC0A0CM"},
            "Credit Spreads (OAS)", ylabel="Basis Points"
        )


# ═════════════════════════════════════════════════════════════════════
# TAB 6 — SECTOR ANALYSIS
# ═════════════════════════════════════════════════════════════════════
with tab6:
    # ── Constants ──
    SECTOR_ETFS = {
        "Technology":       "XLK",
        "Health Care":      "XLV",
        "Financials":       "XLF",
        "Cons. Discr.":     "XLY",
        "Comm. Services":   "XLC",
        "Industrials":      "XLI",
        "Cons. Staples":    "XLP",
        "Energy":           "XLE",
        "Real Estate":      "XLRE",
        "Utilities":        "XLU",
        "Materials":        "XLB",
    }

    # Approximate S&P 500 GICS sector weights (~Q1 2025)
    SECTOR_WEIGHTS = {
        "XLK": 0.295, "XLV": 0.115, "XLF": 0.135, "XLY": 0.100,
        "XLC": 0.090, "XLI": 0.085, "XLP": 0.060, "XLE": 0.040,
        "XLRE": 0.025, "XLU": 0.030, "XLB": 0.025,
    }

    SECTOR_COLORS = [
        "#58a6ff", "#3fb950", "#f85149", "#d29922",
        "#bc8cff", "#39d2c0", "#f0883e", "#ff7b72",
        "#a5d6ff", "#7ee787", "#ffa657",
    ]

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_sector_data():
        tickers = list(SECTOR_ETFS.values()) + ["^GSPC"]
        closes = _yf_closes(tickers, period="2y", interval="1d").dropna(how="all")
        if not closes.empty and closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)
        return closes

    sector_closes = fetch_sector_data()

    if sector_closes.empty:
        st.warning("⚠️ Sector ETF data unavailable.")
    else:
        # ── AI Summary ──
        _sec_ctx = "S&P 500 sector performance (1M returns):\n"
        for sname, ticker in SECTOR_ETFS.items():
            if ticker in sector_closes.columns:
                s = sector_closes[ticker].dropna()
                if len(s) > 21:
                    chg = (s.iloc[-1] / s.iloc[-22] - 1) * 100
                    _sec_ctx += f"  {sname}: {chg:+.1f}%\n"
        if "^GSPC" in sector_closes.columns:
            sp = sector_closes["^GSPC"].dropna()
            if len(sp) > 21:
                _sec_ctx += f"  S&P 500: {(sp.iloc[-1]/sp.iloc[-22]-1)*100:+.1f}%\n"
        ai_summary("sectors", _sec_ctx)

        # ── Sector Returns Table ──
        st.markdown("#### Sector Returns")
        now_date = sector_closes.index[-1]
        period_starts = {
            "1D":  now_date - timedelta(days=1),
            "1W":  now_date - timedelta(days=7),
            "1M":  now_date - timedelta(days=30),
            "3M":  now_date - timedelta(days=91),
            "YTD": pd.Timestamp(now_date.year, 1, 1),
        }
        period_raw = {p: {} for p in period_starts}
        for pname, pstart in period_starts.items():
            window = sector_closes[sector_closes.index >= pstart]
            if window.empty:
                continue
            base_row = window.iloc[0]
            for sname, ticker in SECTOR_ETFS.items():
                if ticker in window.columns:
                    b = base_row.get(ticker, float("nan"))
                    c = window[ticker].iloc[-1]
                    if b and b != 0 and not pd.isna(b) and not pd.isna(c):
                        period_raw[pname][sname] = (c / b - 1) * 100

        tbl_rows = []
        for sname, ticker in SECTOR_ETFS.items():
            row = {"Sector": sname}
            for pname in ["1D", "1W", "1M", "3M", "YTD"]:
                val = period_raw[pname].get(sname)
                if val is not None:
                    sign = "+" if val >= 0 else ""
                    row[pname] = f"{sign}{val:.2f}%"
                else:
                    row[pname] = "—"
            tbl_rows.append(row)
        def _sort_key(r):
            v = r.get("1D", "—")
            try:
                return float(v.replace("%", "").replace("+", ""))
            except Exception:
                return -999
        tbl_rows.sort(key=_sort_key, reverse=True)
        tbl_df = pd.DataFrame(tbl_rows).set_index("Sector")
        st.dataframe(tbl_df, use_container_width=True)

        # ── Cumulative Contribution Area Chart ──
        st.markdown("#### Sector Contribution to S&P 500")
        bar_col, range_col, _ = st.columns([1, 1, 3])
        with bar_col:
            bar_window = st.selectbox(
                "Bar Period", ["1D", "1W", "2W", "1M", "3M"],
                index=1, key="sector_bar_window",
            )
        with range_col:
            contrib_range = st.selectbox(
                "Range", ["1M", "3M", "6M", "YTD", "1Y", "2Y"],
                index=3, key="sector_contrib_range",
            )
        resample_rule = {"1D": "D", "1W": "W-FRI", "2W": "2W-FRI", "1M": "ME", "3M": "QE"}[bar_window]
        if contrib_range == "YTD":
            contrib_start = pd.Timestamp(datetime.now().year, 1, 1)
        else:
            contrib_start = lookback_date(contrib_range)

        trimmed = sector_closes[sector_closes.index >= contrib_start].copy()
        if not trimmed.empty and "^GSPC" in trimmed.columns:
            # Cumulative S&P 500 return line
            spx_base = trimmed["^GSPC"].iloc[0]
            spx_cum = (trimmed["^GSPC"] / spx_base - 1) * 100

            # Resample to period boundaries for bar contributions
            period_closes = trimmed.resample(resample_rule).last().dropna(how="all")

            fig_contrib = go.Figure()

            # Stacked bars: cumulative sector contribution at each period
            for i, (sname, ticker) in enumerate(SECTOR_ETFS.items()):
                if ticker in period_closes.columns:
                    sec_base = period_closes[ticker].iloc[0]
                    if sec_base and sec_base != 0 and not pd.isna(sec_base):
                        cum_contrib = (period_closes[ticker] / sec_base - 1) * 100 * SECTOR_WEIGHTS.get(ticker, 0)
                        fig_contrib.add_trace(go.Bar(
                            x=period_closes.index[1:], y=cum_contrib.iloc[1:],
                            name=sname,
                            marker_color=SECTOR_COLORS[i],
                            opacity=0.85,
                            hovertemplate=f"{sname}: %{{y:.2f}}%<extra></extra>",
                        ))

            # S&P 500 cumulative return line at period boundaries
            spx_period = period_closes["^GSPC"].dropna()
            spx_cum_period = (spx_period / spx_base - 1) * 100
            fig_contrib.add_trace(go.Scatter(
                x=spx_cum_period.index[1:], y=spx_cum_period.iloc[1:],
                name="S&P 500",
                mode="lines+markers",
                line=dict(color="#e6edf3", width=2.5),
                marker=dict(size=4, color="#e6edf3"),
                hovertemplate="S&P 500: %{y:.2f}%<extra></extra>",
            ))

            fig_contrib.update_layout(
                make_layout(f"S&P 500 Return & {bar_window} Sector Attribution", height=460),
                barmode="relative",
                yaxis=dict(title="Return (%)", gridcolor="#21262d"),
            )
            # Only apply rangebreaks for 1D bars to remove gaps. 
            # For 1W+ periods, rangebreaks can cause visual disjoints on holiday weeks.
            if bar_window == "1D":
                missing_bdays = pd.bdate_range(start=period_closes.index.min(), end=period_closes.index.max()).difference(period_closes.index)
                fig_contrib.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]), # Weekends
                        dict(values=missing_bdays.strftime("%Y-%m-%d").tolist()) # Dynamic holidays
                    ],
                    gridcolor="#21262d"
                )
            else:
                fig_contrib.update_xaxes(gridcolor="#21262d")
            st.plotly_chart(fig_contrib, use_container_width=True)

        # ── Relative Rotation Graph ──
        st.markdown("#### Relative Rotation Graph")
        rrg_ctl_col, _ = st.columns([1, 4])
        with rrg_ctl_col:
            rrg_tail = st.selectbox(
                "Tail Length (weeks)", [2, 3, 4, 6], index=1,
                key="rrg_tail",
            )

        RRG_WINDOW = 10  # rolling window in weeks

        # Weekly resample
        weekly = sector_closes.resample("W-FRI").last().dropna(how="all")
        benchmark_weekly = weekly.get("^GSPC")

        rrg_data = {}
        if benchmark_weekly is not None and len(benchmark_weekly) > RRG_WINDOW * 2:
            for sname, ticker in SECTOR_ETFS.items():
                if ticker not in weekly.columns:
                    continue
                sec_w = weekly[ticker].dropna()
                bench_w = benchmark_weekly.reindex(sec_w.index).dropna()
                sec_w = sec_w.reindex(bench_w.index)

                rs = sec_w / bench_w
                rs_sma = rs.rolling(window=RRG_WINDOW, min_periods=RRG_WINDOW).mean()
                rs_ratio = (rs / rs_sma) * 100

                rs_ratio_sma = rs_ratio.rolling(window=RRG_WINDOW, min_periods=RRG_WINDOW).mean()
                rs_momentum = (rs_ratio / rs_ratio_sma) * 100

                valid = pd.DataFrame({
                    "rs_ratio": rs_ratio,
                    "rs_momentum": rs_momentum,
                }).dropna()

                if len(valid) >= 2:
                    tail_data = valid.tail(rrg_tail + 1)
                    rrg_data[sname] = {
                        "ticker": ticker,
                        "rs_ratio": tail_data["rs_ratio"].tolist(),
                        "rs_momentum": tail_data["rs_momentum"].tolist(),
                        "dates": [d.strftime("%b %d") for d in tail_data.index],
                    }

        if rrg_data:
            all_rs = [v for d in rrg_data.values() for v in d["rs_ratio"]]
            all_mom = [v for d in rrg_data.values() for v in d["rs_momentum"]]
            # Center axes symmetrically around (100, 100), fit to data spread
            x_half = max(abs(max(all_rs) - 100.0), abs(min(all_rs) - 100.0), 0.5) * 1.15
            y_half = max(abs(max(all_mom) - 100.0), abs(min(all_mom) - 100.0), 0.5) * 1.15
            x_min, x_max = 100.0 - x_half, 100.0 + x_half
            y_min, y_max = 100.0 - y_half, 100.0 + y_half

            fig_rrg = go.Figure()

            # Quadrant background shading
            quad_fills = [
                (100, x_max, 100, y_max, "rgba(63,185,80,0.06)"),    # Leading
                (100, x_max, y_min, 100, "rgba(248,81,73,0.06)"),     # Weakening
                (x_min, 100, y_min, 100, "rgba(72,79,88,0.08)"),      # Lagging
                (x_min, 100, 100, y_max, "rgba(88,166,255,0.06)"),    # Improving
            ]
            for x0, x1, y0, y1, color in quad_fills:
                fig_rrg.add_shape(
                    type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                    fillcolor=color, line_width=0, layer="below",
                )

            # Crosshair lines
            fig_rrg.add_hline(y=100, line_dash="dot", line_color=MUTED, line_width=1)
            fig_rrg.add_vline(x=100, line_dash="dot", line_color=MUTED, line_width=1)

            # Quadrant labels — pinned to plot corners via paper coords
            quad_labels = [
                (0.98, 0.98, "Leading",   GREEN, "right", "top"),
                (0.98, 0.02, "Weakening", RED,   "right", "bottom"),
                (0.02, 0.02, "Lagging",   MUTED, "left",  "bottom"),
                (0.02, 0.98, "Improving", BLUE,  "left",  "top"),
            ]
            for qx, qy, qtxt, qcolor, qanchor, qyanchor in quad_labels:
                fig_rrg.add_annotation(
                    x=qx, y=qy, text=qtxt, showarrow=False,
                    xref="paper", yref="paper",
                    font=dict(size=11, color=qcolor), opacity=0.6,
                    xanchor=qanchor, yanchor=qyanchor,
                )

            # Sector tails and current positions
            for i, (sname, dct) in enumerate(rrg_data.items()):
                rs_r = dct["rs_ratio"]
                rs_m = dct["rs_momentum"]
                color = SECTOR_COLORS[i]
                hex_c = color.lstrip("#")
                cr, cg, cb = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
                n = len(rs_r)
                # Fade opacity along the tail (oldest → newest)
                marker_colors = [f"rgba({cr},{cg},{cb},{0.15 + 0.6 * j / max(n - 2, 1)})" for j in range(n - 1)]
                marker_colors.append(color)
                # Tail line
                fig_rrg.add_trace(go.Scatter(
                    x=rs_r, y=rs_m,
                    mode="lines+markers",
                    name=sname,
                    line=dict(color=f"rgba({cr},{cg},{cb},0.4)", width=1),
                    marker=dict(
                        size=[3] * (n - 1) + [9],
                        color=marker_colors,
                        symbol=["circle"] * (n - 1) + ["diamond"],
                        line=dict(width=0),
                    ),
                    customdata=dct["dates"],
                    hovertemplate=(
                        f"{sname}<br>"
                        "Date: %{customdata}<br>"
                        "RS-Ratio: %{x:.2f}<br>"
                        "RS-Mom: %{y:.2f}<extra></extra>"
                    ),
                ))
                # Label at latest point
                fig_rrg.add_annotation(
                    x=rs_r[-1], y=rs_m[-1],
                    text=f" {sname}",
                    showarrow=False, xanchor="left",
                    font=dict(size=9, color=color),
                )

            fig_rrg.update_layout(
                make_layout("Relative Rotation Graph — S&P 500 Sectors", height=580),
                xaxis_title="RS-Ratio →",
                yaxis_title="RS-Momentum →",
                hovermode="closest",
                xaxis=dict(range=[x_min, x_max], gridcolor="#21262d", zerolinecolor="#30363d"),
                yaxis=dict(range=[y_min, y_max], gridcolor="#21262d", zerolinecolor="#30363d"),
            )
            st.plotly_chart(fig_rrg, use_container_width=True)
        else:
            st.info("Insufficient history for RRG calculation (need ~20+ weeks).")


# ═════════════════════════════════════════════════════════════════════
# TAB 7 — RELATIVE STRENGTH ANALYSIS
# ═════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("### 📊 Macro Relative Strength Dashboard")
    st.caption("Global Equity Neutral — ACWI Denominator. Strips out global equity beta to surface which themes and regions are attracting capital relative to the world average.")

    # ── Universe loader (enriched schema: see universe_rules.py) ──
    from universe_rules import (
        AssetClass, THEMES, apply_rules, enrich_metadata, enrich_one,
        classify_new_ticker,
    )
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    UNIVERSE_CSV = Path("universe.csv")

    def load_universe_full():
        """Raw enriched CSV (all rows, all columns) — for admin UI."""
        if not UNIVERSE_CSV.exists():
            return pd.DataFrame()
        return pd.read_csv(UNIVERSE_CSV)

    def load_universe():
        """Filtered + renamed for downstream RS code.

        Returns only `included == True` rows, normalized to the legacy column
        names (Symbol/Description/Theme) so the RS analytics code is unchanged.
        """
        df = load_universe_full()
        if df.empty:
            return pd.DataFrame(columns=["Symbol", "Description", "Theme"])
        df = df[df["included"] == True].copy()
        df = df.rename(columns={"symbol": "Symbol", "name": "Description", "theme": "Theme"})
        return df

    # ── Admin: Universe Editor (st.data_editor) ──
    STALE_DAYS = 7

    def _stale_symbols(df: pd.DataFrame) -> list[str]:
        """Symbols with missing or older-than-STALE_DAYS last_refreshed."""
        if "last_refreshed" not in df.columns:
            return df["symbol"].tolist()
        cutoff = (_dt.now(_tz.utc) - _td(days=STALE_DAYS)).date().isoformat()
        mask = df["last_refreshed"].isna() | (df["last_refreshed"].astype(str) < cutoff)
        return df.loc[mask, "symbol"].tolist()

    with st.expander("🛠️ Universe Management (Admin)"):
        full = load_universe_full()
        if full.empty:
            st.warning("universe.csv not found.")
        else:
            n_in = int(full["included"].sum())
            n_ex = len(full) - n_in
            n_stale = len(_stale_symbols(full))
            st.caption(f"{len(full)} ETFs · {n_in} included · {n_ex} excluded · "
                       f"{n_stale} with stale metadata (>{STALE_DAYS}d). "
                       "Edit `theme`, `included`, `canonical_for`, or `override` below. "
                       "Set `override=True` to preserve a manual `included` choice across rule re-runs.")

            # ── Quick add ──
            qa1, qa2 = st.columns([1, 4])
            with qa1:
                new_sym = st.text_input("Add ticker", placeholder="e.g. SMH", key="qa_symbol").strip().upper()
            with qa2:
                st.caption("Paste a ticker → fetches metadata, proposes asset_class + theme, applies rules. ~2s.")
                if new_sym and st.button(f"➕ Add {new_sym}", key="qa_add"):
                    if new_sym in set(full["symbol"].astype(str).str.upper()):
                        st.warning(f"{new_sym} already in universe.")
                    else:
                        with st.spinner(f"Fetching {new_sym}…"):
                            md = enrich_one(new_sym)
                        if md.aum is None and md.long_name is None:
                            st.error(f"Could not fetch metadata for {new_sym}. Check the ticker.")
                        else:
                            name, ac, theme = classify_new_ticker(new_sym, md)
                            new_row = {
                                "symbol": new_sym, "name": name,
                                "asset_class": ac, "theme": theme,
                                "included": True, "exclude_reason": None,
                                "canonical_for": None,
                                "aum": md.aum, "avg_volume_30d": md.avg_volume_30d,
                                "inception_date": md.inception_date,
                                "expense_ratio": md.expense_ratio,
                                "last_refreshed": md.last_refreshed,
                                "is_leveraged": md.is_leveraged,
                                "is_single_stock": md.is_single_stock,
                                "override": False,
                            }
                            full2 = pd.concat([full, pd.DataFrame([new_row])], ignore_index=True)
                            full2 = apply_rules(full2, preserve_overrides=True)
                            full2.to_csv(UNIVERSE_CSV, index=False)
                            st.cache_data.clear()
                            row = full2[full2["symbol"] == new_sym].iloc[0]
                            verdict = "✅ included" if row["included"] else f"❌ excluded ({row['exclude_reason']})"
                            st.success(f"Added {new_sym} — {name} → {ac} / {theme} — {verdict}")
                            st.rerun()

            show_excluded = st.checkbox("Show excluded rows", value=False)
            view = full if show_excluded else full[full["included"] == True]

            edited = st.data_editor(
                view,
                use_container_width=True, height=420, hide_index=True,
                num_rows="dynamic",
                column_config={
                    "symbol":         st.column_config.TextColumn(disabled=True),
                    "name":           st.column_config.TextColumn(disabled=True),
                    "asset_class":    st.column_config.SelectboxColumn(options=[ac.value for ac in AssetClass]),
                    "theme":          st.column_config.SelectboxColumn(options=THEMES),
                    "included":       st.column_config.CheckboxColumn(),
                    "exclude_reason": st.column_config.TextColumn(disabled=True),
                    "canonical_for":  st.column_config.TextColumn(help="Set to canonical sibling ticker to deduplicate."),
                    "override":       st.column_config.CheckboxColumn(help="If true, manual `included` is preserved across rule re-runs."),
                    "aum":            st.column_config.NumberColumn(format="$%d", disabled=True),
                    "avg_volume_30d": st.column_config.NumberColumn(format="$%d", disabled=True),
                    "inception_date": st.column_config.TextColumn(disabled=True),
                    "expense_ratio":  st.column_config.NumberColumn(format="%.4f", disabled=True),
                    "last_refreshed": st.column_config.TextColumn(disabled=True),
                    "is_leveraged":   st.column_config.CheckboxColumn(disabled=True),
                    "is_single_stock":st.column_config.CheckboxColumn(disabled=True),
                },
                key="universe_editor",
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("💾 Save changes"):
                    edited_syms = set(edited["symbol"].astype(str))
                    view_syms = set(view["symbol"].astype(str))
                    deleted_syms = view_syms - edited_syms

                    # Splice edited rows back into the full frame
                    merged = full[~full["symbol"].isin(deleted_syms)].copy()
                    merged = merged.set_index("symbol")
                    merged.update(edited.set_index("symbol"))
                    merged = merged.reset_index()

                    merged = apply_rules(merged, preserve_overrides=True)
                    merged.to_csv(UNIVERSE_CSV, index=False)
                    st.cache_data.clear()
                    msg = "Saved."
                    if deleted_syms:
                        msg += f" Deleted: {', '.join(sorted(deleted_syms))}"
                    st.success(msg)
                    st.rerun()
            with c2:
                stale = _stale_symbols(full)
                btn_label = f"🔄 Refresh stale ({len(stale)})" if stale else "🔄 Refresh stale (0)"
                if st.button(btn_label, disabled=not stale):
                    with st.spinner(f"Fetching yfinance metadata for {len(stale)} stale symbols…"):
                        meta = enrich_metadata(stale)
                        full2 = full.set_index("symbol")
                        full2.update(meta)
                        full2 = full2.reset_index()
                        full2 = apply_rules(full2, preserve_overrides=True)
                        full2.to_csv(UNIVERSE_CSV, index=False)
                        st.cache_data.clear()
                    st.success(f"Refreshed {len(stale)} symbols.")
                    st.rerun()
            with c3:
                if st.button("⚙️ Re-apply rules"):
                    full2 = apply_rules(full, preserve_overrides=True)
                    full2.to_csv(UNIVERSE_CSV, index=False)
                    st.cache_data.clear()
                    st.success("Rules re-applied.")
                    st.rerun()

    # ── Constants ──
    RS_BASE = "ACWI"

    @st.cache_data(ttl=3600, show_spinner="Calculating Relative Strength...")
    def get_rs_analytics(rs_rank_lookback=126):
        universe = load_universe()
        if universe.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        tickers = universe["Symbol"].tolist()
        if RS_BASE not in tickers: tickers.append(RS_BASE)
        
        # Batch download 2Y for 252d rolling window
        prices = _yf_closes(tickers, period="2y").dropna(axis=1, how="all").ffill().bfill()
        base_price = prices[RS_BASE]
        
        # 1. RS Ratio & Trend
        rs_ratio = prices.drop(columns=[RS_BASE], errors="ignore").div(base_price, axis=0)
        rs_sma20 = rs_ratio.rolling(20).mean()

        # 2. Z-scored returns of the RS ratio (Option 1 + 2)
        # Use RS ratio returns (geometric relative perf) then Z-score vs 252d distribution
        rs_ret_5d  = rs_ratio.pct_change(5)
        rs_ret_20d = rs_ratio.pct_change(20)

        def zscore_latest(series_df, window=252):
            mu  = series_df.rolling(window).mean()
            sig = series_df.rolling(window).std()
            return ((series_df - mu) / sig).iloc[-1]

        z5d  = zscore_latest(rs_ret_5d)
        z20d = zscore_latest(rs_ret_20d)

        # 3. RS Rank — percentile of RS ratio *performance* over the lookback window
        # Indexes each ETF to a common baseline so absolute price level is irrelevant
        lookback_start = max(0, len(rs_ratio) - rs_rank_lookback)
        rs_perf = rs_ratio.iloc[-1] / rs_ratio.iloc[lookback_start] - 1
        rs_rank = rs_perf.rank(pct=True).mul(98).add(1).round(0).astype(int)

        # 4. RS Trend — RS ratio above/below SMA20 with SMA sloping
        sma_slope = rs_sma20.iloc[-1] - rs_sma20.iloc[-5] if len(rs_sma20) >= 5 else pd.Series(0, index=rs_sma20.columns)
        rs_trend = np.where(
            (rs_ratio.iloc[-1] > rs_sma20.iloc[-1]) & (sma_slope > 0), "↑ Rising",
            np.where(
                (rs_ratio.iloc[-1] < rs_sma20.iloc[-1]) & (sma_slope < 0), "↓ Falling",
                "→ Neutral"
            )
        )

        snap = pd.DataFrame({
            "Symbol":  rs_ratio.columns,
            "RS_Rank": rs_rank.values,
            "Z5D":     z5d.values,
            "Z20D":    z20d.values,
            "RS_Trend": rs_trend,
            "RS_Ratio": rs_ratio.iloc[-1].values,
            "RS_SMA20": rs_sma20.iloc[-1].values,
        })
        snap = snap.merge(universe[["Symbol", "Description", "Theme"]], on="Symbol", how="inner")
        return snap.dropna(subset=["RS_Rank", "Z5D", "Z20D"]), rs_ratio, rs_sma20

    _lookback_options = {"3 Months (63d)": 63, "6 Months (126d)": 126, "12 Months (252d)": 252, "24 Months (504d)": 504}
    _lookback_label = st.selectbox("RS Rank lookback", list(_lookback_options.keys()), index=0, help="Window used to measure RS ratio performance for the percentile rank. Shorter = more responsive to recent shifts; longer = structural leaders.")
    rs_rank_lookback = _lookback_options[_lookback_label]

    rs_snap, rs_ratio_df, rs_sma_df = get_rs_analytics(rs_rank_lookback)

    if rs_snap.empty:
        st.warning("⚠️ Relative Strength data unavailable.")
    else:
        # ── AI Summary ──
        _rs_ctx = "Relative Strength (RS) leaders (by RS Rank vs ACWI):\n"
        for _, r in rs_snap.nlargest(5, "RS_Rank").iterrows():
            _rs_ctx += f"  {r['Symbol']}: RS Rank {r['RS_Rank']}, Z5D {r['Z5D']:+.2f}, Z20D {r['Z20D']:+.2f}\n"
        ai_summary("rs", _rs_ctx)

        rs_tabs = st.tabs(["🔥 Leaders & Laggards", "🔄 Rotation & Rebounds", "🗺️ Theme Breakdown", "📈 Sparklines"])
        cols_to_show = ["Symbol", "Description", "Theme", "RS_Rank", "Z5D", "Z20D", "RS_Trend"]
        fmt = {"RS_Rank": "{:.0f}", "Z5D": "{:+.2f}", "Z20D": "{:+.2f}"}

        with rs_tabs[0]:
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("##### 🟢 Top 20 Leaders")
                top20 = rs_snap.nlargest(20, "RS_Rank")[cols_to_show].reset_index(drop=True)
                st.dataframe(top20.style.format(fmt).background_gradient(subset=["Z5D", "Z20D"], cmap="RdYlGn"), use_container_width=True, hide_index=True)

            with col_r:
                st.markdown("##### 🔴 Bottom 20 Laggards")
                bot20 = rs_snap.nsmallest(20, "RS_Rank")[cols_to_show].reset_index(drop=True)
                st.dataframe(bot20.style.format(fmt).background_gradient(subset=["Z5D", "Z20D"], cmap="RdYlGn"), use_container_width=True, hide_index=True)

        with rs_tabs[1]:
            st.markdown("#### Relative Rotation Analysis")
            st.caption("Axes are Z-scores of RS ratio returns vs 252d distribution — comparable across all ETF types.")
            fig_rot = px.scatter(
                rs_snap, x="Z5D", y="Z20D", color="Theme",
                hover_name="Symbol", hover_data=["Description", "RS_Rank", "RS_Trend"],
                labels={"Z5D": "5D RS Z-Score", "Z20D": "20D RS Z-Score"},
                height=600
            )
            fig_rot.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig_rot.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
            for txt, xp, yp in [("Sustained Leaders", 1.5, 1.8), ("Turning Up", 1.5, -1.8), ("Laggards", -1.5, -1.8), ("Fading", -1.5, 1.8)]:
                fig_rot.add_annotation(x=xp, y=yp, text=f"<b>{txt}</b>", showarrow=False, font=dict(size=12, color="grey"))
            fig_rot.update_layout(make_layout(""))
            st.plotly_chart(fig_rot, use_container_width=True)

            col_trn, col_brk = st.columns(2)
            with col_trn:
                st.markdown("##### 🔍 Turning Up (Rebounding)")
                st.caption("Z20D < 0 (lagging medium-term) AND Z5D > 0.5 (short-term thrust beginning)")
                turning = rs_snap[(rs_snap['Z20D'] < 0) & (rs_snap['Z5D'] > 0.5)].sort_values('Z5D', ascending=False).head(15)
                st.dataframe(turning[["Symbol", "Theme", "RS_Rank", "Z5D", "Z20D", "RS_Trend"]].style.format(fmt), use_container_width=True)

            with col_brk:
                st.markdown("##### ⚠️ Fading (Breaking Down)")
                st.caption("Z20D > 0 (leading medium-term) AND Z5D < -0.5 (short-term deterioration)")
                breaking = rs_snap[(rs_snap['Z20D'] > 0) & (rs_snap['Z5D'] < -0.5)].sort_values('Z5D', ascending=True).head(15)
                st.dataframe(breaking[["Symbol", "Theme", "RS_Rank", "Z5D", "Z20D", "RS_Trend"]].style.format(fmt), use_container_width=True)

        with rs_tabs[2]:
            st.markdown("#### Theme Breakdown")
            theme_agg = rs_snap.groupby("Theme").agg(
                Avg_Rank=("RS_Rank", "mean"),
                Avg_Z5D=("Z5D", "mean"),
                Avg_Z20D=("Z20D", "mean"),
                N=("Symbol", "count")
            ).sort_values("Avg_Z20D", ascending=False)

            fig_theme = px.bar(
                theme_agg.reset_index(), x="Avg_Z20D", y="Theme", orientation="h",
                color="Avg_Z20D", color_continuous_scale="RdYlGn", range_color=[-2, 2],
                text="N",
                labels={"Avg_Z20D": "20D RS Momentum (z-score)", "Theme": ""},
                height=520
            )
            fig_theme.update_traces(textposition="outside", texttemplate="%{text} ETFs")
            fig_theme.update_layout(make_layout(""), yaxis=dict(autorange="reversed"),
                                    coloraxis_showscale=False)
            st.plotly_chart(fig_theme, use_container_width=True)

            display_agg = theme_agg.rename(columns={
                "Avg_Rank": "Avg RS Rank",
                "Avg_Z5D":  "5D Momentum (z)",
                "Avg_Z20D": "20D Momentum (z)",
                "N":        "# ETFs",
            })
            st.dataframe(
                display_agg.style
                    .format({"Avg RS Rank": "{:.0f}", "5D Momentum (z)": "{:+.2f}",
                             "20D Momentum (z)": "{:+.2f}", "# ETFs": "{:.0f}"})
                    .background_gradient(subset=["5D Momentum (z)", "20D Momentum (z)"], cmap="RdYlGn"),
                use_container_width=True
            )

        with rs_tabs[3]:
            st.markdown("#### RS Ratio vs SMA(20) Sparklines (Trailing 60d)")
            spark_view = st.selectbox("View", ["Top 20 RS Rank", "Bottom 20 RS Rank", "Turning Up"])

            if spark_view == "Top 20 RS Rank":
                tkrs_to_plot = rs_snap.nlargest(20, "RS_Rank")["Symbol"].tolist()
            elif spark_view == "Bottom 20 RS Rank":
                tkrs_to_plot = rs_snap.nsmallest(20, "RS_Rank")["Symbol"].tolist()
            else:
                tkrs_to_plot = rs_snap[(rs_snap['Z20D'] < 0) & (rs_snap['Z5D'] > 0.5)].sort_values('Z5D', ascending=False).head(20)["Symbol"].tolist()

            if tkrs_to_plot:
                cols = 5
                rows = -(-len(tkrs_to_plot) // cols)
                fig_sparks = make_subplots(rows=rows, cols=cols, subplot_titles=tkrs_to_plot, horizontal_spacing=0.04, vertical_spacing=0.08)
                
                for i, tkr in enumerate(tkrs_to_plot):
                    r, c = divmod(i, cols)
                    if tkr in rs_ratio_df.columns:
                        ratio_line = rs_ratio_df[tkr].tail(60)
                        sma_line = rs_sma_df[tkr].tail(60)
                        fig_sparks.add_trace(go.Scatter(x=ratio_line.index, y=ratio_line.values, mode="lines", line=dict(color=BLUE, width=1.5), showlegend=False), row=r+1, col=c+1)
                        fig_sparks.add_trace(go.Scatter(x=sma_line.index, y=sma_line.values, mode="lines", line=dict(color=YELLOW, width=1, dash="dash"), showlegend=False), row=r+1, col=c+1)
                
                fig_sparks.update_xaxes(showticklabels=False, showgrid=False)
                fig_sparks.update_yaxes(showticklabels=False, showgrid=False)
                fig_sparks.update_layout(make_layout("", height=140 * rows))
                st.plotly_chart(fig_sparks, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════
# TAB 8 — MACRO REGIME (Growth × Inflation × Liquidity)
# ═════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown("### 🧭 Macro Regime — Growth × Inflation × Liquidity")
    st.caption("Three-dimensional macro framework: composite z-scores → regime classification → asset class implications.")

    # ── Regime computation (logic lives in regime_model.py) ─────────
    @st.cache_data(ttl=24 * 3600, show_spinner="Loading macro regime data...")
    def _regime_compute():
        raw = {}
        for sid in FRED_IDS:
            s = fetch_fred_series(sid, start=REGIME_DEFAULT_START)
            if s is not None and len(s) > 0:
                raw[sid] = s
        return compute_regime(raw)


    data = _regime_compute()
    combined = data["combined"]

    if combined.empty or combined["Regime"].dropna().empty:
        st.warning("Macro regime data unavailable. Check FRED API key and try refreshing.")
    else:
        valid = combined.dropna(subset=["Regime"])
        latest = valid.iloc[-1]
        latest_date = valid.index[-1]
        regime = latest["Regime"]
        color = REGIME_COLORS.get(regime, "#8b949e")

        # ── Section 1 — Current Regime Dashboard ─────────────────────
        st.markdown("#### Current Regime")
        st.markdown(
            f'<div style="background:linear-gradient(135deg,{color}22,{color}08);'
            f'border:1px solid {color};border-radius:8px;padding:18px 22px;'
            f'margin-bottom:14px;">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
            f'color:#8b949e;text-transform:uppercase;letter-spacing:0.08em;">'
            f'Regime · {latest_date.strftime("%b %Y")}</div>'
            f'<div style="font-family:DM Sans,sans-serif;font-size:1.9rem;'
            f'font-weight:700;color:{color};margin-top:4px;">{regime}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        for col, dim, score_key, class_key in [
            (c1, "Growth",    "Growth",    "Growth_Class"),
            (c2, "Inflation", "Inflation", "Inflation_Class"),
            (c3, "Liquidity", "Liquidity", "Liquidity_Class"),
        ]:
            score = latest[score_key]
            cls = latest[class_key]
            score_color = GREEN if score > 0.5 else (RED if score < -0.5 else YELLOW)
            with col:
                st.markdown(
                    f'<div style="background:#1c2333;border:1px solid #30363d;'
                    f'border-radius:6px;padding:12px 16px;">'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
                    f'color:#8b949e;text-transform:uppercase;letter-spacing:0.06em;">{dim} Score</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:1.6rem;'
                    f'font-weight:600;color:{score_color};">{score:+.2f}</div>'
                    f'<div style="font-family:DM Sans,sans-serif;font-size:0.9rem;'
                    f'color:#e6edf3;margin-top:2px;">{cls}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            f'<div style="margin-top:14px;padding:10px 14px;background:#161b22;'
            f'border-left:3px solid {color};border-radius:4px;font-size:0.92rem;'
            f'color:#e6edf3;line-height:1.6;">'
            f'The current regime is <b style="color:{color};">{regime}</b>. '
            f'Growth is <b>{latest["Growth_Class"]}</b> ({latest["Growth"]:+.2f}σ), '
            f'inflation is <b>{latest["Inflation_Class"]}</b> ({latest["Inflation"]:+.2f}σ), '
            f'and financial conditions are <b>{latest["Liquidity_Class"]}</b> '
            f'({latest["Liquidity"]:+.2f}σ).'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Section 2 — Regime Components Detail ─────────────────────
        st.markdown("#### Regime Components")

        def _component_section(title, dim_score, keys, labels, feats, zscores):
            with st.expander(title, expanded=False):
                rows = []
                for k in keys:
                    if k not in feats:
                        rows.append({"Series": labels.get(k, k), "Latest": np.nan,
                                     "12M Δ": np.nan, "Z-score": np.nan})
                        continue
                    s = feats[k].dropna()
                    if len(s) == 0:
                        rows.append({"Series": labels.get(k, k), "Latest": np.nan,
                                     "12M Δ": np.nan, "Z-score": np.nan})
                        continue
                    latest_v = s.iloc[-1]
                    prior = s.iloc[-13] if len(s) >= 13 else np.nan
                    chg = (latest_v - prior) if not pd.isna(prior) else np.nan
                    z = zscores.get(k)
                    z_v = z.dropna().iloc[-1] if (z is not None and z.dropna().size > 0) else np.nan
                    rows.append({
                        "Series": labels.get(k, k),
                        "Latest": round(latest_v, 3),
                        "12M Δ": round(chg, 3) if not pd.isna(chg) else np.nan,
                        "Z-score": round(z_v, 2) if not pd.isna(z_v) else np.nan,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dim_score.index, y=dim_score.values, mode="lines",
                    name=f"{title.split('—')[0].strip()} Score",
                    line=dict(color=BLUE, width=2),
                ))
                fig.add_hline(y=0.5, line=dict(color=GREEN, width=1, dash="dash"),
                              annotation_text="+0.5", annotation_position="top right")
                fig.add_hline(y=-0.5, line=dict(color=RED, width=1, dash="dash"),
                              annotation_text="-0.5", annotation_position="bottom right")
                fig.add_hline(y=0, line=dict(color=MUTED, width=1))
                fig.update_layout(make_layout(f"{title} Composite Z-score", height=320))
                st.plotly_chart(fig, use_container_width=True)

        _component_section(
            "Growth Components",
            combined["Growth"],
            GROWTH_KEYS, GROWTH_LABELS, data["feats"], data["zscores"],
        )
        _component_section(
            "Inflation Components",
            combined["Inflation"],
            INFLATION_KEYS, INFLATION_LABELS, data["feats"], data["zscores"],
        )
        _component_section(
            "Liquidity Components",
            combined["Liquidity"],
            LIQUIDITY_KEYS, LIQUIDITY_LABELS, data["feats"], data["zscores"],
        )

        st.divider()

        # ── Section 3 — Regime History Timeline ──────────────────────
        st.markdown("#### Regime History Timeline")

        valid_idx = valid.index
        hist_min = valid_idx.min().to_pydatetime().date()
        hist_max = valid_idx.max().to_pydatetime().date()
        default_start = max(hist_min, (valid_idx.max() - pd.DateOffset(years=15)).to_pydatetime().date())

        date_range = st.date_input(
            "Date range",
            value=(default_start, hist_max),
            min_value=hist_min,
            max_value=hist_max,
            key="regime_date_range",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            ds, de = date_range
        else:
            ds, de = default_start, hist_max
        ds_ts = pd.Timestamp(ds)
        de_ts = pd.Timestamp(de)
        view = valid.loc[(valid.index >= ds_ts) & (valid.index <= de_ts)]

        if view.empty:
            st.info("No regime data in the selected range.")
        else:
            # Build segments of consecutive same-regime months
            segs = []
            cur_regime = None
            cur_start = None
            prev_date = None
            for d, r in view["Regime"].items():
                if cur_regime is None:
                    cur_regime, cur_start, prev_date = r, d, d
                    continue
                if r != cur_regime:
                    segs.append((cur_regime, cur_start, prev_date))
                    cur_regime, cur_start = r, d
                prev_date = d
            segs.append((cur_regime, cur_start, prev_date))

            fig = go.Figure()
            seen = set()
            for r, s_dt, e_dt in segs:
                end_show = e_dt + pd.offsets.MonthEnd(1)
                color = REGIME_COLORS.get(r, "#8b949e")
                show_legend = r not in seen
                seen.add(r)
                fig.add_trace(go.Scatter(
                    x=[s_dt, end_show, end_show, s_dt, s_dt],
                    y=[0, 0, 1, 1, 0],
                    fill="toself",
                    fillcolor=color,
                    line=dict(width=0),
                    mode="lines",
                    name=r,
                    legendgroup=r,
                    showlegend=show_legend,
                    hovertemplate=f"<b>{r}</b><br>%{{x|%b %Y}}<extra></extra>",
                ))
            fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1])
            fig.update_layout(make_layout("Regime by Month", height=260,
                                          legend=dict(orientation="h", y=-0.25,
                                                      xanchor="center", x=0.5,
                                                      font=dict(size=9))))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Section 4 — Regime Transition Matrix ─────────────────────
        st.markdown("#### Regime Transitions & Persistence")

        regimes_seq = valid["Regime"].tolist()
        regime_dates = valid.index.tolist()

        # Build runs (regime, start, end_inclusive, length_months)
        runs = []
        if regimes_seq:
            cur = regimes_seq[0]
            cur_s = regime_dates[0]
            cur_len = 1
            for i in range(1, len(regimes_seq)):
                if regimes_seq[i] == cur:
                    cur_len += 1
                else:
                    runs.append((cur, cur_s, regime_dates[i - 1], cur_len))
                    cur = regimes_seq[i]
                    cur_s = regime_dates[i]
                    cur_len = 1
            runs.append((cur, cur_s, regime_dates[-1], cur_len))

        # Average duration per regime
        dur_by = defaultdict(list)
        for r, _, _, ln in runs:
            dur_by[r].append(ln)
        dur_rows = [{"Regime": r,
                     "Avg Duration (mo)": round(np.mean(v), 1),
                     "Occurrences": len(v),
                     "Total Months": int(np.sum(v))}
                    for r, v in dur_by.items()]
        dur_df = pd.DataFrame(dur_rows).sort_values("Total Months", ascending=False)

        # Transition matrix between distinct successive runs
        from_to = defaultdict(lambda: defaultdict(int))
        for i in range(len(runs) - 1):
            from_to[runs[i][0]][runs[i + 1][0]] += 1

        regimes_order = sorted({r for r, _, _, _ in runs})
        if from_to and regimes_order:
            mat = np.zeros((len(regimes_order), len(regimes_order)), dtype=float)
            for i, r_from in enumerate(regimes_order):
                row_total = sum(from_to[r_from].values())
                if row_total == 0:
                    continue
                for j, r_to in enumerate(regimes_order):
                    mat[i, j] = from_to[r_from][r_to] / row_total * 100

            colA, colB = st.columns([3, 2])
            with colA:
                fig = go.Figure(data=go.Heatmap(
                    z=mat,
                    x=regimes_order,
                    y=regimes_order,
                    colorscale="Blues",
                    colorbar=dict(title="% →"),
                    hovertemplate="From <b>%{y}</b><br>To <b>%{x}</b><br>%{z:.0f}%<extra></extra>",
                ))
                fig.update_layout(make_layout("Transition Probability (% from row → column)",
                                              height=460,
                                              margin=dict(l=140, r=30, t=60, b=140)))
                fig.update_xaxes(tickangle=-40)
                st.plotly_chart(fig, use_container_width=True)
            with colB:
                st.markdown("**Average regime duration**")
                st.dataframe(dur_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough regime history for transition statistics.")

# ═════════════════════════════════════════════════════════════════════
# TAB 9 — REGIME BACKTEST
# ═════════════════════════════════════════════════════════════════════
with tab9:
    st.markdown("### 🧪 Regime Backtest")
    st.caption("Historical average forward returns for major asset classes, grouped by macro regime. This provides visual validation that the model captures real economic shifts.")
    
    @st.cache_data(ttl=24 * 3600, show_spinner="Computing regime backtest...")
    def _compute_backtest():
        # Re-fetch the regime data
        regime_data = _regime_compute()
        combined = regime_data.get("combined")
        if combined is None or combined.empty or "Regime" not in combined.columns:
            return None, None
            
        labels = combined[["Regime"]].copy()
        
        # Collapse into coarse buckets for better sample size
        GOLDILOCKS = {"Goldilocks / Late Cycle", "Goldilocks / Early Cycle", "Disinflationary Boom", "Soft Landing / Recovery"}
        EXPANSION = {"Mid-Cycle Expansion", "Steady Expansion", "Late Cycle", "Reflation", "Reflation / Recovery"}
        OVERHEATING = {"Overheating", "Late Cycle Slowdown Risk", "Late Cycle Disinflation", "Mild Tightening"}
        SLOWDOWN = {"Slowdown", "Disinflationary Slowdown", "Policy Easing", "Policy Easing / Trough", "Transition", "Transition (Inflation)", "Transition (Disinflation)"}
        STAGFLATION = {"Stagflation", "Stagflation Risk", "Stagflation w/ Easing"}
        RECESSION = {"Recession", "Recession (Recovery Setup)", "Hard Landing Risk"}

        def bucket(r):
            if pd.isna(r): return None
            if r in GOLDILOCKS:    return "Goldilocks"
            if r in EXPANSION:     return "Expansion"
            if r in OVERHEATING:   return "Overheating"
            if r in SLOWDOWN:      return "Slowdown"
            if r in STAGFLATION:   return "Stagflation"
            if r in RECESSION:     return "Recession"
            return "Other"
            
        labels["Coarse_Regime"] = labels["Regime"].apply(bucket)
        
        # Shift index to end of month so it aligns with forward prices better
        labels.index = labels.index + pd.offsets.MonthEnd(0)
        
        # Fetch asset prices: SPY (Eq), AGG (Bonds), GLD (Gold), UUP (Dollar)
        assets = {"SPY": "SPY", "AGG": "AGG", "GLD": "GLD", "DXY": "DX-Y.NYB"}
        prices = _yf_closes(list(assets.values()), period="max")
        if prices.empty:
            return None, None
            
        prices = prices.rename(columns={v: k for k, v in assets.items()})
        prices = prices.resample("ME").last().dropna(how="all")
        
        # Join labels with prices
        df = labels.join(prices, how="inner")
        
        # Calculate forward returns
        horizons = {"1M": 1, "3M": 3, "6M": 6}
        results = []
        
        for asset in assets.keys():
            if asset not in df.columns:
                continue
            for h_label, h_months in horizons.items():
                # Forward return
                df[f"{asset}_{h_label}"] = df[asset].shift(-h_months) / df[asset] - 1.0
                
                # Group by coarse regime
                grp = df.groupby("Coarse_Regime")[f"{asset}_{h_label}"]
                means = grp.mean() * 100 # percentage
                counts = grp.count()
                
                for regime in means.index:
                    if counts[regime] > 3: # Need at least a few data points
                        results.append({
                            "Regime": regime,
                            "Asset": asset,
                            "Horizon": h_label,
                            "Avg_Return": means[regime],
                            "N": counts[regime]
                        })
                        
        return pd.DataFrame(results), df

    bt_results, bt_df = _compute_backtest()
    
    if bt_results is None or bt_results.empty:
        st.warning("Backtest data is currently unavailable. Please check data sources.")
    else:
        st.markdown("#### Average Forward Returns by Macro Regime")
        
        # Controls
        bt_col1, bt_col2 = st.columns([1, 3])
        with bt_col1:
            sel_asset = st.selectbox("Asset", ["SPY", "AGG", "GLD", "DXY"], index=0, key="bt_asset")
            sel_horizon = st.selectbox("Forward Horizon", ["1M", "3M", "6M"], index=1, key="bt_horizon")
            
        # Filter data
        filtered = bt_results[(bt_results["Asset"] == sel_asset) & (bt_results["Horizon"] == sel_horizon)]
        
        if filtered.empty:
            st.info(f"Not enough data for {sel_asset} over {sel_horizon}.")
        else:
            # Sort logically by cycle
            regime_order = ["Goldilocks", "Expansion", "Overheating", "Stagflation", "Slowdown", "Recession"]
            filtered["Regime_Cat"] = pd.Categorical(filtered["Regime"], categories=regime_order, ordered=True)
            filtered = filtered.sort_values("Regime_Cat")
            
            fig_bt = go.Figure()
            colors = [GREEN if val >= 0 else RED for val in filtered["Avg_Return"]]
            
            fig_bt.add_trace(go.Bar(
                x=filtered["Regime"],
                y=filtered["Avg_Return"],
                text=[f"{v:+.1f}%<br>(n={n})" for v, n in zip(filtered["Avg_Return"], filtered["N"])],
                textposition="outside",
                marker_color=colors,
                hovertemplate="Regime: %{x}<br>Avg Return: %{y:.2f}%<br>Occurrences: %{customdata}<extra></extra>",
                customdata=filtered["N"]
            ))
            
            fig_bt.add_hline(y=0, line_dash="solid", line_color=MUTED, line_width=1)
            
            y_max = filtered["Avg_Return"].abs().max() * 1.3
            if pd.isna(y_max) or y_max == 0: y_max = 5
            
            fig_bt.update_layout(
                make_layout(f"Average {sel_horizon} Forward Return for {sel_asset} by Regime Bucket", height=400),
                yaxis_title="Average Return (%)",
                yaxis=dict(range=[-y_max, y_max])
            )
            st.plotly_chart(fig_bt, use_container_width=True)
            
            with st.expander("Show Data Table"):
                # Pivot table to show all horizons for the selected asset
                asset_all = bt_results[bt_results["Asset"] == sel_asset].copy()
                pivot = asset_all.pivot_table(index="Regime", columns="Horizon", values=["Avg_Return", "N"])
                # Flatten multi-index columns
                pivot.columns = [f"{col[1]} {col[0].replace('Avg_Return', 'Return (%)').replace('N', 'Count')}" for col in pivot.columns]
                
                # Order columns
                cols = []
                for h in ["1M", "3M", "6M"]:
                    cols.extend([c for c in pivot.columns if h in c])
                pivot = pivot[cols]
                
                # Format
                format_dict = {c: "{:+.2f}%" for c in pivot.columns if "Return" in c}
                format_dict.update({c: "{:.0f}" for c in pivot.columns if "Count" in c})
                
                st.dataframe(pivot.style.format(format_dict).background_gradient(subset=[c for c in pivot.columns if "Return" in c], cmap="RdYlGn"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Data: FRED (Federal Reserve Economic Data) · Yahoo Finance · US Treasury | Not investment advice. Data may be delayed.")
