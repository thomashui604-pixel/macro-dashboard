"""
Macro regime model — shared logic.

Pure pandas/numpy; no Streamlit dependency. Imported by both the dashboard
(`app.py`) and the validation package (`validation/`) so the empirical tests
operate on the *deployed* model rather than a copy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Regime lookup (3 × 3 × 3 = 27 combinations) ──────────────────────
REGIME_LOOKUP = {
    ("Expansion",  "Inflationary",   "Loose"):   "Goldilocks / Late Cycle",
    ("Expansion",  "Inflationary",   "Neutral"): "Late Cycle",
    ("Expansion",  "Inflationary",   "Tight"):   "Overheating",
    ("Expansion",  "Neutral",        "Loose"):   "Mid-Cycle Expansion",
    ("Expansion",  "Neutral",        "Neutral"): "Steady Expansion",
    ("Expansion",  "Neutral",        "Tight"):   "Late Cycle Slowdown Risk",
    ("Expansion",  "Disinflationary","Loose"):   "Goldilocks / Early Cycle",
    ("Expansion",  "Disinflationary","Neutral"): "Disinflationary Boom",
    ("Expansion",  "Disinflationary","Tight"):   "Late Cycle Disinflation",
    ("Neutral",    "Inflationary",   "Loose"):   "Reflation",
    ("Neutral",    "Inflationary",   "Neutral"): "Transition (Inflation)",
    ("Neutral",    "Inflationary",   "Tight"):   "Stagflation Risk",
    ("Neutral",    "Neutral",        "Loose"):   "Reflation / Recovery",
    ("Neutral",    "Neutral",        "Neutral"): "Transition",
    ("Neutral",    "Neutral",        "Tight"):   "Mild Tightening",
    ("Neutral",    "Disinflationary","Loose"):   "Soft Landing / Recovery",
    ("Neutral",    "Disinflationary","Neutral"): "Transition (Disinflation)",
    ("Neutral",    "Disinflationary","Tight"):   "Disinflationary Slowdown",
    ("Contraction","Inflationary",   "Loose"):   "Stagflation w/ Easing",
    ("Contraction","Inflationary",   "Neutral"): "Stagflation",
    ("Contraction","Inflationary",   "Tight"):   "Stagflation",
    ("Contraction","Neutral",        "Loose"):   "Policy Easing",
    ("Contraction","Neutral",        "Neutral"): "Slowdown",
    ("Contraction","Neutral",        "Tight"):   "Hard Landing Risk",
    ("Contraction","Disinflationary","Loose"):   "Policy Easing / Trough",
    ("Contraction","Disinflationary","Neutral"): "Recession (Recovery Setup)",
    ("Contraction","Disinflationary","Tight"):   "Recession",
}

REGIME_COLORS = {
    "Goldilocks / Late Cycle":    "#3fb950",
    "Goldilocks / Early Cycle":   "#3fb950",
    "Late Cycle":                 "#7CB342",
    "Mid-Cycle Expansion":        "#56d364",
    "Steady Expansion":           "#3fb950",
    "Disinflationary Boom":       "#56d364",
    "Late Cycle Slowdown Risk":   "#d29922",
    "Late Cycle Disinflation":    "#d29922",
    "Overheating":                "#f0883e",
    "Reflation":                  "#39d2c0",
    "Reflation / Recovery":       "#39d2c0",
    "Transition (Inflation)":     "#8b949e",
    "Transition":                 "#8b949e",
    "Transition (Disinflation)":  "#8b949e",
    "Mild Tightening":            "#a39922",
    "Stagflation Risk":           "#f85149",
    "Stagflation":                "#f85149",
    "Stagflation w/ Easing":      "#f85149",
    "Soft Landing / Recovery":    "#58a6ff",
    "Disinflationary Slowdown":   "#bc8cff",
    "Policy Easing":              "#58a6ff",
    "Policy Easing / Trough":     "#58a6ff",
    "Slowdown":                   "#bc8cff",
    "Hard Landing Risk":          "#a31515",
    "Recession (Recovery Setup)": "#7c4dff",
    "Recession":                  "#7c1e1e",
}

GROWTH_KEYS    = ["MANEMP_YoY", "ICSA", "RSXFS_YoY", "INDPRO_YoY", "USSLIND"]
INFLATION_KEYS = ["CPIAUCSL_YoY", "CPIAUCSL_3M_SAAR", "PCEPILFE_YoY", "T5YIFR", "PPIFID_YoY"]
LIQUIDITY_KEYS = ["FEDFUNDS_DEV", "WALCL_YoY", "REAL_M2_YoY", "HY_OAS", "T10Y2Y_ADJ"]

GROWTH_LABELS = {
    "MANEMP_YoY":   "Manufacturing Employment YoY (proxy for ISM PMI)",
    "ICSA":         "Initial Jobless Claims (inverted)",
    "RSXFS_YoY":    "Real Retail Sales YoY",
    "INDPRO_YoY":   "Industrial Production YoY",
    "USSLIND":      "Leading Index (USSLIND)",
}
INFLATION_LABELS = {
    "CPIAUCSL_YoY":     "CPI YoY",
    "CPIAUCSL_3M_SAAR": "CPI 3M Annualized (SAAR)",
    "PCEPILFE_YoY":     "Core PCE YoY",
    "T5YIFR":           "5Y5Y Fwd Breakeven",
    "PPIFID_YoY":       "PPI Final Demand YoY",
}
LIQUIDITY_LABELS = {
    "FEDFUNDS_DEV":  "Fed Funds vs 12M Trend (inverted)",
    "WALCL_YoY":     "Fed Balance Sheet YoY",
    "REAL_M2_YoY":   "Real M2 YoY (M2 / CPI)",
    "HY_OAS":        "HY OAS (inverted)",
    "T10Y2Y_ADJ":    "2s10s Slope (Adjusted for 10Y Trend)",
}

# Series whose z-score should be flipped so that +z always means "expansionary
# / inflationary / loose", aligning sign with the pillar interpretation.
INVERT = {"ICSA", "FEDFUNDS_DEV", "HY_OAS"}

# FRED series IDs the model needs.
FRED_IDS = [
    "MANEMP", "ICSA", "RSXFS", "INDPRO", "USSLIND",
    "CPIAUCSL", "PCEPILFE", "T5YIFR", "PPIFID",
    "FEDFUNDS", "WALCL", "M2SL", "BAMLH0A0HYM2", "T10Y2Y", "DGS10",
]

DEFAULT_START = "1985-01-01"


def rolling_z(s: pd.Series, window: int = 36, min_periods: int = 24) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - m) / sd.replace(0, np.nan)


def classify(score, labels_pos_neutral_neg, threshold: float = 0.5):
    if score is None or pd.isna(score):
        return None
    if score > threshold:
        return labels_pos_neutral_neg[0]
    if score < -threshold:
        return labels_pos_neutral_neg[2]
    return labels_pos_neutral_neg[1]


def build_features(monthly: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Transform raw monthly FRED series into the model's 15 features."""
    feats: dict[str, pd.Series] = {}
    if "MANEMP" in monthly:
        feats["MANEMP_YoY"] = monthly["MANEMP"].pct_change(12) * 100
    if "ICSA" in monthly:
        feats["ICSA"] = monthly["ICSA"]
    if "RSXFS" in monthly:
        feats["RSXFS_YoY"] = monthly["RSXFS"].pct_change(12) * 100
    if "INDPRO" in monthly:
        feats["INDPRO_YoY"] = monthly["INDPRO"].pct_change(12) * 100
    if "USSLIND" in monthly:
        # USSLIND is Philadelphia Fed's predicted 6-month growth rate of the
        # Coincident Index — already a growth rate, not a raw index level.
        feats["USSLIND"] = monthly["USSLIND"]

    if "CPIAUCSL" in monthly:
        feats["CPIAUCSL_YoY"] = monthly["CPIAUCSL"].pct_change(12) * 100
        # 3M annualized rate catches turning points ~9 months faster than YoY.
        feats["CPIAUCSL_3M_SAAR"] = (
            (monthly["CPIAUCSL"] / monthly["CPIAUCSL"].shift(3)) ** 4 - 1
        ) * 100
    if "PCEPILFE" in monthly:
        feats["PCEPILFE_YoY"] = monthly["PCEPILFE"].pct_change(12) * 100
    if "T5YIFR" in monthly:
        feats["T5YIFR"] = monthly["T5YIFR"]
    if "PPIFID" in monthly:
        feats["PPIFID_YoY"] = monthly["PPIFID"].pct_change(12) * 100

    if "FEDFUNDS" in monthly:
        ff = monthly["FEDFUNDS"]
        ff_trend = ff.rolling(12, min_periods=6).mean()
        feats["FEDFUNDS_DEV"] = ff - ff_trend
    if "WALCL" in monthly:
        feats["WALCL_YoY"] = monthly["WALCL"].pct_change(12) * 100
    if "M2SL" in monthly and "CPIAUCSL" in monthly:
        real_m2 = monthly["M2SL"] / monthly["CPIAUCSL"]
        feats["REAL_M2_YoY"] = real_m2.pct_change(12) * 100
    if "BAMLH0A0HYM2" in monthly:
        feats["HY_OAS"] = monthly["BAMLH0A0HYM2"]

    if "T10Y2Y" in monthly and "DGS10" in monthly:
        # Regime-aware yield curve score:
        # - Bull steepener (rates falling): T10Y2Y_ADJ rises (loosening).
        # - Bear flattener (rates rising):  T10Y2Y_ADJ falls (tightening).
        # Subtracting 2× the 10Y YoY change encodes the four curve regimes
        # into one continuous financial-conditions z-score.
        feats["T10Y2Y_ADJ"] = monthly["T10Y2Y"] - (2.0 * monthly["DGS10"].diff(12))

    return feats


def compute_regime(
    raw: dict[str, pd.Series],
    z_window: int = 36,
    z_min_periods: int = 24,
    threshold: float = 0.5,
) -> dict:
    """Compute the full macro regime panel from raw FRED series.

    Parameters
    ----------
    raw : dict[str, pd.Series]
        Raw FRED series keyed by FRED ID. Must contain at least the IDs in
        ``FRED_IDS`` for the model to populate every pillar.
    z_window, z_min_periods : int
        Rolling z-score window (months) and minimum periods.
    threshold : float
        Pillar classification cutoff in z-score units.

    Returns
    -------
    dict with keys: raw, monthly, feats, zscores, signed,
    growth_df, inflation_df, liquidity_df, combined.
    """
    monthly = {sid: s.resample("ME").last() for sid, s in raw.items()}
    feats = build_features(monthly)

    zscores = {n: rolling_z(s, z_window, z_min_periods) for n, s in feats.items()}
    signed = {n: (zscores[n] * (-1 if n in INVERT else 1)) for n in zscores}

    def _avg(keys):
        cols = {k: signed[k] for k in keys if k in signed}
        if not cols:
            return pd.Series(dtype=float), pd.DataFrame()
        df = pd.DataFrame(cols)
        return df.mean(axis=1, skipna=True), df

    g_score, g_df = _avg(GROWTH_KEYS)
    i_score, i_df = _avg(INFLATION_KEYS)
    l_score, l_df = _avg(LIQUIDITY_KEYS)

    combined = pd.DataFrame({
        "Growth":    g_score,
        "Inflation": i_score,
        "Liquidity": l_score,
    }).dropna(how="all")

    combined["Growth_Class"] = combined["Growth"].apply(
        lambda x: classify(x, ("Expansion", "Neutral", "Contraction"), threshold))
    combined["Inflation_Class"] = combined["Inflation"].apply(
        lambda x: classify(x, ("Inflationary", "Neutral", "Disinflationary"), threshold))
    combined["Liquidity_Class"] = combined["Liquidity"].apply(
        lambda x: classify(x, ("Loose", "Neutral", "Tight"), threshold))
    combined["Regime"] = combined.apply(
        lambda r: REGIME_LOOKUP.get(
            (r["Growth_Class"], r["Inflation_Class"], r["Liquidity_Class"])),
        axis=1,
    )

    return {
        "raw": raw, "monthly": monthly, "feats": feats,
        "zscores": zscores, "signed": signed,
        "growth_df": g_df, "inflation_df": i_df, "liquidity_df": l_df,
        "combined": combined,
    }
