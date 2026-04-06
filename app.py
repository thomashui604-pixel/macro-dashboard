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
from datetime import datetime, timedelta
import requests
import json
import time
import shutil
from pathlib import Path
import warnings

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
    margin=dict(l=50, r=30, t=50, b=60),
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="right", x=1, font=dict(size=10)),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
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
    "USREC": 24*3600, "A191RL1Q225SBEA": 24*3600,
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

@st.cache_data(ttl=60, show_spinner=False)
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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yf(tickers, period="1y", interval="1d"):
    """Fetch yfinance data for one or more tickers."""
    try:
        if isinstance(tickers, str):
            tickers = [tickers]
        data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False, threads=True)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            # Return Close prices
            if "Close" in data.columns.get_level_values(0):
                return data["Close"]
        else:
            if "Close" in data.columns:
                return data[["Close"]]
            return data
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yf_single(ticker, period="1y", interval="1d"):
    """Fetch yfinance data for a single ticker, returning a DataFrame."""
    try:
        t = yf.Ticker(ticker)
        data = t.history(period=period, interval=interval, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        return data
    except Exception:
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
    for date, val in s.items():
        if val == 1 and not in_recession:
            start = date
            in_recession = True
        elif val == 0 and in_recession:
            recessions.append((start, date))
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

def fmt_num(val, decimals=2):
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:,.{decimals}f}"

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


# ─────────────────────────────────────────────────────────────────────
# KPI STRIP (compact header)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kpi_data():
    """Fetch the KPI summary data."""
    kpis = {}
    # yfinance quick data
    tickers = {"SPX": "^GSPC", "DXY": "DX-Y.NYB", "Gold": "GC=F", "VIX": "^VIX"}
    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d", auto_adjust=True)
            if len(hist) >= 2:
                kpis[name] = {"price": hist["Close"].iloc[-1], "chg": safe_pct_change(hist["Close"].iloc[-1], hist["Close"].iloc[-2])}
            elif len(hist) == 1:
                kpis[name] = {"price": hist["Close"].iloc[-1], "chg": None}
        except Exception:
            pass
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

kpi_data = fetch_kpi_data()

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

st.markdown(render_kpi_strip(kpi_data), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Rates & Curve",
    "🏛 Policy Path",
    "🌍 Cross-Asset",
    "📊 VIX & Vol",
    "📋 Macro Data",
    "🏗 Sector Analysis",
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

        # ── Curve Spreads ──
        st.markdown("#### Curve Spreads")
        lb_col, _ = st.columns([1, 4])
        with lb_col:
            spread_lb = st.selectbox("Lookback", ["1Y", "2Y", "5Y", "10Y"], index=["1Y", "2Y", "5Y", "10Y"].index(global_lookback), key="spread_lb")

        spread_start = lookback_date(spread_lb).strftime("%Y-%m-%d")
        spread_series = fetch_fred_multi(["DGS2", "DGS10", "DGS30", "DGS5", "DGS3MO"], start=spread_start)
        if not spread_series.empty:
            spreads = pd.DataFrame(index=spread_series.index)
            if "DGS2" in spread_series and "DGS10" in spread_series:
                spreads["2s10s"] = (spread_series["DGS10"] - spread_series["DGS2"]) * 100
            if "DGS2" in spread_series and "DGS30" in spread_series:
                spreads["2s30s"] = (spread_series["DGS30"] - spread_series["DGS2"]) * 100
            if "DGS5" in spread_series and "DGS30" in spread_series:
                spreads["5s30s"] = (spread_series["DGS30"] - spread_series["DGS5"]) * 100
            if "DGS3MO" in spread_series and "DGS10" in spread_series:
                spreads["3M10Y"] = (spread_series["DGS10"] - spread_series["DGS3MO"]) * 100
            spreads = spreads.dropna(how="all")

            if not spreads.empty:
                fig_spreads = go.Figure()
                colors = {"2s10s": BLUE, "2s30s": CYAN, "5s30s": YELLOW, "3M10Y": RED}
                for col in spreads.columns:
                    fig_spreads.add_trace(go.Scatter(
                        x=spreads.index, y=spreads[col], name=col,
                        line=dict(color=colors.get(col, MUTED), width=1.5),
                        hovertemplate=f"{col}: " + "%{y:.0f}bp<extra></extra>"
                    ))
                # Zero line
                fig_spreads.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=1)
                # Recession shading
                recessions = fetch_recession_dates()
                add_recession_shading(fig_spreads, recessions, x_min=spreads.index.min(), x_max=spreads.index.max())
                # Annotate current values
                for col in spreads.columns:
                    last_val = spreads[col].dropna().iloc[-1] if len(spreads[col].dropna()) > 0 else None
                    if last_val is not None:
                        fig_spreads.add_annotation(
                            x=spreads.index[-1], y=last_val,
                            text=f" {last_val:.0f}bp", showarrow=False,
                            xanchor="left", font=dict(size=10, color=colors.get(col, MUTED)),
                        )
                fig_spreads.update_layout(make_layout("", height=350))
                fig_spreads.update_layout(yaxis_title="Basis Points")
                st.plotly_chart(fig_spreads, use_container_width=True)

        # ── Real Rates & Breakevens ──
        st.markdown("#### Real Rates & Breakevens")
        real_start = lookback_date(global_lookback).strftime("%Y-%m-%d")
        real_series = fetch_fred_multi(["DFII5", "DFII10", "T5YIFR", "T10YIE"], start=real_start)
        if not real_series.empty:
            fig_real = go.Figure()
            # Real rates on y1
            if "DFII5" in real_series:
                fig_real.add_trace(go.Scatter(x=real_series.index, y=real_series["DFII5"], name="5Y TIPS Real", line=dict(color=BLUE, width=1.5)))
            if "DFII10" in real_series:
                fig_real.add_trace(go.Scatter(x=real_series.index, y=real_series["DFII10"], name="10Y TIPS Real", line=dict(color=CYAN, width=1.5)))
            # Breakevens on y2
            if "T5YIFR" in real_series:
                fig_real.add_trace(go.Scatter(x=real_series.index, y=real_series["T5YIFR"], name="5Y5Y Fwd BEI", line=dict(color=YELLOW, dash="dash", width=1.5), yaxis="y2"))
            if "T10YIE" in real_series:
                fig_real.add_trace(go.Scatter(x=real_series.index, y=real_series["T10YIE"], name="10Y BEI", line=dict(color=RED, dash="dash", width=1.5), yaxis="y2"))
            layout = make_layout("", height=350)
            fig_real.update_layout(layout)
            fig_real.update_layout(
                yaxis=dict(title="Real Yield (%)", gridcolor="#21262d"),
                yaxis2=dict(title="Breakeven (%)", overlaying="y", side="right", gridcolor="#21262d", showgrid=False),
            )
            st.plotly_chart(fig_real, use_container_width=True)
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
        # Construct tickers for upcoming contracts
        months = {"F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
                  "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec"}
        month_codes = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
        now = datetime.now()

        tickers = []
        labels = []
        # Generate 18 contracts ahead
        for i in range(18):
            m_idx = (now.month - 1 + i) % 12
            year = now.year + (now.month - 1 + i) // 12
            code = month_codes[m_idx]
            yr = str(year)[-2:]
            ticker = f"ZQ{code}{yr}.CBT"
            label = f"{months[code]} {year}"
            tickers.append(ticker)
            labels.append(label)

        results = []
        for ticker, label in zip(tickers, labels):
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d", auto_adjust=True)
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
                    implied_rate = 100 - price
                    results.append({"contract": label, "ticker": ticker, "price": price, "implied_rate": implied_rate})
            except Exception:
                continue
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
    st.caption("Interpolated from monthly FF futures using the CME FedWatch methodology: isolates the implied rate change at each meeting using day-weighting within the contract month.")

    # Scheduled FOMC announcement dates
    FOMC_DATES = [
        # 2026
        (2026, 1, 28), (2026, 3, 18), (2026, 5, 6), (2026, 6, 17),
        (2026, 7, 29), (2026, 9, 16), (2026, 10, 28), (2026, 12, 16),
        # 2027
        (2027, 1, 27), (2027, 3, 17), (2027, 5, 5), (2027, 6, 16),
        (2027, 7, 28), (2027, 9, 22), (2027, 10, 27), (2027, 12, 15),
    ]

    def compute_fedwatch(contracts_dict, current_effr, fomc_dates_raw):
        """
        CME FedWatch interpolation — chained post-rates.
        contracts_dict: {"Mar 2026": 3.64, ...} implied rates from FF futures
        Returns list of dicts with per-meeting implied rates and probabilities.
        """
        import calendar
        from datetime import date

        today = date.today()
        fomc_dates = [date(y, m, d) for y, m, d in fomc_dates_raw if date(y, m, d) > today]

        month_key = lambda d: f"{d.strftime('%b')} {d.year}"
        results = []
        prev_post_rate = None  # chain from prior meeting
        prev_mtg = None

        for mtg in fomc_dates:
            mk = month_key(mtg)
            if mk not in contracts_dict:
                prev_post_rate = None  # break chain if month missing
                prev_mtg = mtg
                continue

            month_rate = contracts_dict[mk]
            days_in_month = calendar.monthrange(mtg.year, mtg.month)[1]
            effective_day = mtg.day + 1  # rate change effective day after announcement
            days_before = effective_day - 1
            days_after = days_in_month - days_before

            # Pre-meeting rate: chain from prior meeting only if it was in the
            # same or immediately prior month; otherwise use the prior month's
            # contract to avoid amplifying interpolation errors across gaps.
            use_chain = False
            if prev_post_rate is not None and prev_mtg is not None:
                months_gap = (mtg.year - prev_mtg.year) * 12 + (mtg.month - prev_mtg.month)
                if months_gap <= 1:
                    use_chain = True

            if use_chain:
                pre_rate = prev_post_rate
            else:
                prior_month = mtg.month - 1 if mtg.month > 1 else 12
                prior_year = mtg.year if mtg.month > 1 else mtg.year - 1
                prior_key = month_key(date(prior_year, prior_month, 1))
                if prior_key in contracts_dict:
                    pre_rate = contracts_dict[prior_key]
                else:
                    pre_rate = current_effr

            # If meeting is very late in month (<=2 days after), the month's
            # contract is dominated by pre-meeting rate — use next month instead
            if days_after <= 2:
                next_month = mtg.month + 1 if mtg.month < 12 else 1
                next_year = mtg.year if mtg.month < 12 else mtg.year + 1
                next_key = month_key(date(next_year, next_month, 1))
                post_rate = contracts_dict.get(next_key, month_rate)
            else:
                post_rate = (month_rate * days_in_month - pre_rate * days_before) / days_after

            prev_post_rate = post_rate
            prev_mtg = mtg
            delta_bp = (post_rate - pre_rate) * 100

            # Probability of 25bp move vs hold
            prob_25bp = min(abs(delta_bp) / 25, 1.0)
            prob_hold = 1.0 - prob_25bp
            move_type = "cut" if delta_bp < -1 else "hike" if delta_bp > 1 else "hold"

            # Cumulative change from current EFFR
            cum_bp = (post_rate - current_effr) * 100
            cum_moves = cum_bp / 25

            results.append({
                "meeting": mtg.strftime("%b %d, %Y"),
                "pre_rate": pre_rate,
                "post_rate": post_rate,
                "delta_bp": delta_bp,
                "prob_hold": prob_hold,
                "prob_25bp": prob_25bp,
                "move_type": move_type,
                "cum_bp": cum_bp,
                "cum_moves": cum_moves,
            })

        return results

    if not ff_df.empty:
        # Build contracts dict from ff_df
        contracts_dict = dict(zip(ff_df["contract"], ff_df["implied_rate"]))
        effr_for_fw = _effr_tab2.iloc[-1] if len(_effr_tab2) > 0 else None

        if effr_for_fw is not None:
            fw_results = compute_fedwatch(contracts_dict, effr_for_fw, FOMC_DATES)

            if fw_results:
                # ── Probability table (Bloomberg style) ──
                grid_cols = "120px 110px 110px 110px 110px"
                tbl_hdr = f'<div style="display:grid; grid-template-columns:{grid_cols}; gap:6px; padding:4px 8px; font-size:0.68rem; color:#8b949e; font-family:JetBrains Mono,monospace; border-bottom:1px solid #30363d; text-transform:uppercase;">'
                tbl_hdr += '<span>Meeting Date</span><span style="text-align:right">#Hikes/Cuts</span><span style="text-align:right">%Hike/Cut</span><span style="text-align:right">Δ Impl Rate</span><span style="text-align:right">Implied Rate</span></div>'
                st.markdown(tbl_hdr, unsafe_allow_html=True)

                for r in fw_results:
                    # #Hikes/Cuts — cumulative 25bp moves from EFFR
                    cm = r["cum_moves"]
                    if abs(cm) < 0.05:
                        cum_label = "—"
                        cum_color = MUTED
                    elif cm < 0:
                        cum_label = f"{abs(cm):.1f} cuts"
                        cum_color = GREEN
                    else:
                        cum_label = f"{cm:.1f} hikes"
                        cum_color = RED

                    # %Hike/Cut — probability of a move at this meeting
                    move_label = f'{r["prob_25bp"]*100:.0f}% {r["move_type"]}'
                    move_color = GREEN if r["move_type"] == "cut" else RED if r["move_type"] == "hike" else MUTED

                    # Delta implied rate
                    delta_color = GREEN if r["delta_bp"] < -1 else RED if r["delta_bp"] > 1 else MUTED

                    row = f'<div style="display:grid; grid-template-columns:{grid_cols}; gap:6px; padding:4px 8px; font-size:0.78rem; font-family:JetBrains Mono,monospace; color:#e6edf3; border-bottom:1px solid #161b22;">'
                    row += f'<span style="color:#8b949e;">{r["meeting"]}</span>'
                    row += f'<span style="text-align:right; color:{cum_color}; font-weight:600;">{cum_label}</span>'
                    row += f'<span style="text-align:right; color:{move_color}; font-weight:600;">{move_label}</span>'
                    row += f'<span style="text-align:right; color:{delta_color};">{r["delta_bp"]:+.1f}</span>'
                    row += f'<span style="text-align:right; font-weight:600;">{r["post_rate"]:.3f}%</span>'
                    row += '</div>'
                    st.markdown(row, unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

                # ── FedWatch-style chart: Implied Rate + Cumulative Cuts ──
                fw_df = pd.DataFrame(fw_results)

                fig_fw = go.Figure()

                # Bars: cumulative cuts/hikes (right axis)
                bar_colors = [GREEN if v < -0.05 else RED if v > 0.05 else MUTED for v in fw_df["cum_moves"]]
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
                    name="# Hikes/Cuts",
                    marker_color=bar_colors, opacity=0.75,
                    yaxis="y2",
                    hovertext=bar_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                ))

                # Line: implied post-meeting policy rate (left axis)
                fig_fw.add_trace(go.Scatter(
                    x=fw_df["meeting"], y=fw_df["post_rate"],
                    name="Implied Policy Rate (%)",
                    line=dict(color=BLUE, width=3),
                    mode="lines+markers",
                    marker=dict(size=7, color=BLUE),
                    hovertemplate="%{x}<br>Implied rate: %{y:.3f}%<extra></extra>",
                ))

                # Current EFFR anchor
                fig_fw.add_trace(go.Scatter(
                    x=["Current"], y=[effr_for_fw],
                    mode="markers+text",
                    marker=dict(size=10, color="white", line=dict(color=BLUE, width=2)),
                    text=[f"{effr_for_fw:.2f}%"], textposition="top center",
                    textfont=dict(color="white", size=11),
                    showlegend=False,
                    hovertemplate="Current EFFR: %{y:.3f}%<extra></extra>",
                ))

                # Reference line at current EFFR
                fig_fw.add_hline(y=effr_for_fw, line_dash="solid", line_color="rgba(255,255,255,0.25)", line_width=1)

                fig_fw.update_layout(make_layout("Implied Overnight Rate & Number of Cuts Priced In", height=450))
                fig_fw.update_layout(
                    xaxis=dict(
                        tickangle=-45,
                        categoryorder="array",
                        categoryarray=["Current"] + fw_df["meeting"].tolist(),
                        gridcolor="#21262d",
                    ),
                    yaxis=dict(title="Implied Policy Rate (%)", side="left", gridcolor="#21262d"),
                    yaxis2=dict(
                        title="# Hikes/Cuts",
                        overlaying="y", side="right", showgrid=False,
                        zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
                        dtick=1,
                        rangemode="tozero",
                    ),
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, font=dict(size=10)),
                    bargap=0.3,
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

    # Hardcoded SEP median dots (Dec 2024 SEP)
    sep_data = {
        "2025": 3.875,
        "2026": 3.375,
        "2027": 3.125,
        "Longer Run": 3.00,
    }

    market_implied = {}
    if not ff_df.empty:
        # Use the last available contract in each year as proxy for year-end rate
        for yr in ["2025", "2026", "2027"]:
            yr_contracts = ff_df[ff_df["contract"].str.endswith(yr)]
            if not yr_contracts.empty:
                # Last contract in the year (latest month available)
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

    # ── SOFR Futures Strip ──
    st.markdown("#### SOFR Futures Strip (SR1)")

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_sofr_futures():
        month_codes = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
        months_map = {"F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
                      "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec"}
        now = datetime.now()
        results = []
        for i in range(8):
            m_idx = (now.month - 1 + i) % 12
            year = now.year + (now.month - 1 + i) // 12
            code = month_codes[m_idx]
            yr = str(year)[-2:]
            ticker = f"SR1{code}{yr}.CME"
            label = f"{months_map[code]} {year}"
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d", auto_adjust=True)
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
                    results.append({"contract": label, "implied_rate": 100 - price})
            except Exception:
                continue
        return pd.DataFrame(results)

    sofr_strip = fetch_sofr_futures()
    if not sofr_strip.empty:
        fig_sofr = go.Figure()
        fig_sofr.add_trace(go.Bar(
            x=sofr_strip["contract"], y=sofr_strip["implied_rate"],
            marker_color=CYAN, text=[f"{r:.3f}%" for r in sofr_strip["implied_rate"]],
            textposition="outside",
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
        try:
            data = yf.download(all_tickers, period="1y", interval="1d", auto_adjust=True, progress=False, threads=True)
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"] if "Close" in data.columns.get_level_values(0) else pd.DataFrame()
            else:
                closes = data[["Close"]] if "Close" in data.columns else pd.DataFrame()
            return closes
        except Exception:
            return pd.DataFrame()

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
        for t in tickers:
            try:
                d = yf.Ticker(t)
                hist = d.history(period="3y", auto_adjust=True)
                if not hist.empty:
                    results[t] = hist["Close"]
            except Exception:
                pass
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
            spx = yf.Ticker("^GSPC").history(period="2y", auto_adjust=True)
            vix_hist = yf.Ticker("^VIX").history(period="2y", auto_adjust=True)
            if spx.empty or vix_hist.empty:
                return pd.DataFrame()
            # Strip timezones so indexes align (SPX=New_York, VIX=Chicago)
            spx.index = spx.index.tz_localize(None) if spx.index.tz else spx.index
            vix_hist.index = vix_hist.index.tz_localize(None) if vix_hist.index.tz else vix_hist.index
            # Realized vol: 20-day annualized
            returns = spx["Close"].pct_change()
            realized = returns.rolling(20).std() * np.sqrt(252) * 100
            # Align dates
            df = pd.DataFrame({"Realized Vol (20d)": realized, "VIX (Implied)": vix_hist["Close"]})
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

    def make_macro_chart(series_dict, title, height=350, yoy_compute=None, mom_diff=None, ylabel=""):
        """Helper to build macro time series charts."""
        # For YoY/MoM transforms we need extra history, so fetch 2 extra years
        needs_extra = bool(yoy_compute or mom_diff)
        if needs_extra:
            extra_start = (lookback_date(macro_lb) - timedelta(days=730)).strftime("%Y-%m-%d")
        else:
            extra_start = macro_start
        data = fetch_fred_multi(list(series_dict.values()), start=extra_start)
        if data.empty:
            st.info(f"{title}: data unavailable")
            return

        plot_start = pd.Timestamp(macro_start)
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
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, name=label,
                line=dict(color=COLORS[color_idx % len(COLORS)], width=1.5),
                hovertemplate=f"{label}: " + "%{y:.2f}<extra></extra>",
            ))

            # Annotate latest value
            last_date = s.index[-1]
            last_val = s.iloc[-1]
            fig.add_annotation(
                x=last_date, y=last_val,
                text=f" {last_val:.1f} ({last_date.strftime('%b %y')})",
                showarrow=False, xanchor="left",
                font=dict(size=9, color=COLORS[color_idx % len(COLORS)]),
            )
            color_idx += 1

        if all_dates:
            add_recession_shading(fig, recessions, x_min=min(all_dates), x_max=max(all_dates))
        fig.update_layout(make_layout(title, height=height))
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
    labor_col5, labor_col6 = st.columns(2)
    with labor_col5:
        make_macro_chart(
            {"Avg Hourly Earnings": "CES0500000003"},
            "Avg Hourly Earnings (YoY %)", yoy_compute=["CES0500000003"], ylabel="%"
        )
    with labor_col6:
        pass

    # ── Activity ──
    st.markdown("#### Activity")
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        make_macro_chart(
            {"Real GDP (QoQ% Ann.)": "A191RL1Q225SBEA"},
            "Real GDP (QoQ % Annualized)", ylabel="%"
        )
    with act_col2:
        make_macro_chart({"Industrial Production": "INDPRO"}, "Industrial Production Index", ylabel="Index")
    act_col3, act_col4 = st.columns(2)
    with act_col3:
        make_macro_chart({"Mfg Production": "IPMAN"}, "Industrial Production: Manufacturing", ylabel="Index")
    with act_col4:
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
        import yfinance as yf_local
        tickers = list(SECTOR_ETFS.values()) + ["^GSPC"]
        try:
            data = yf_local.download(
                tickers, period="2y", interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"]
            else:
                closes = data[["Close"]]
            closes = closes.dropna(how="all")
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            return closes
        except Exception:
            return pd.DataFrame()

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
            for pname in ["1W", "1M", "3M", "YTD"]:
                val = period_raw[pname].get(sname)
                if val is not None:
                    sign = "+" if val >= 0 else ""
                    row[pname] = f"{sign}{val:.1f}%"
                else:
                    row[pname] = "—"
            tbl_rows.append(row)
        def _sort_key(r):
            v = r.get("YTD", "—")
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
                "Bar Period", ["1W", "2W", "1M", "3M"],
                index=0, key="sector_bar_window",
            )
        with range_col:
            contrib_range = st.selectbox(
                "Range", ["3M", "6M", "YTD", "1Y", "2Y"],
                index=2, key="sector_contrib_range",
            )
        resample_rule = {"1W": "W-FRI", "2W": "2W-FRI", "1M": "ME", "3M": "QE"}[bar_window]
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


# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Data: FRED (Federal Reserve Economic Data) · Yahoo Finance · US Treasury | Not investment advice. Data may be delayed.")
