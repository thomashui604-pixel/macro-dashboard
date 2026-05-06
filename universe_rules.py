"""
Universe schema, classification, metadata enrichment, and inclusion rules
for the Relative Strength dashboard.

The RS dashboard is scoped to equity-only. Non-equity ETFs (commodity futures,
crypto trusts) are tagged for exclusion so the ACWI denominator stays coherent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd
import yfinance as yf


# ── Enums ────────────────────────────────────────────────────────────────────

class AssetClass(str, Enum):
    US_EQUITY = "us_equity"
    INTL_DEV = "intl_dev"
    EM = "em"
    SECTOR = "sector"
    THEMATIC_EQUITY = "thematic_equity"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FIXED_INCOME = "fixed_income"


EQUITY_CLASSES = {
    AssetClass.US_EQUITY,
    AssetClass.INTL_DEV,
    AssetClass.EM,
    AssetClass.SECTOR,
    AssetClass.THEMATIC_EQUITY,
}


# Themes — finer label, replaces assign_rs_theme keyword cascade.
# Curated list; admin UI should constrain edits to these values.
THEMES = [
    "US Broad", "US Growth", "US Value / Div", "Small / Mid Cap", "Smart Beta",
    "Tech", "Financials", "Industrials", "Consumer Cyclical", "Defensives",
    "Energy", "Materials", "Real Estate",
    "Int'l Developed", "Emerging Markets",
    "Thematic",
    # Non-equity (tagged so they sort cleanly in admin view)
    "Commodity", "Crypto", "Fixed Income",
]


# ── Hardcoded classifications ────────────────────────────────────────────────

# Broad-market benchmarks — excluded from RS as too generic to carry signal.
BROAD_BENCHMARKS = {
    "SPY", "IVV", "VOO", "SPYM",
    "QQQ", "QQQM",
    "DIA",
    "VTI", "ITOT", "SCHB", "SPTM", "DFUS",
    "VT",
    "ACWI", "ACWX",
    "VXUS", "VEU", "IXUS",
    "IWB", "SCHX", "SCHK", "OEF",
    "RSP",
}

# Symbol-specific overrides where Focus alone is misleading.
ASSET_CLASS_OVERRIDES = {
    "USO": AssetClass.COMMODITY,    # oil futures, not equity
    "GBTC": AssetClass.CRYPTO,       # holds bitcoin directly
    # Note: URA, GDX, COPX, SIL, SILJ, GDXJ hold *miners* (equity).
    # BLOK holds blockchain stocks (equity).
    # AMLP holds MLPs (equity-like).
}


# ── Pattern detectors ────────────────────────────────────────────────────────

LEVERAGED_RE = re.compile(
    r"\b(?:2x|3x|1\.5x|-?\dx|leveraged|ultrapro|ultra\s*short|ultra\s*pro|inverse"
    r"|daily\s+(?:bull|bear|short|long))\b",
    re.IGNORECASE,
)

# Single-stock ETFs (NVDL, TSLL, MSFU, etc.) — name typically references one company.
SINGLE_STOCK_RE = re.compile(
    r"\b(daily target|2x long|2x short|single stock|1\.5x|t-rex)\b"
    r"|\b(?:nvidia|tesla|apple|microsoft|amazon|meta|alphabet|google|netflix)\s+(?:etf|fund|target|daily)\b",
    re.IGNORECASE,
)

YIELD_CARRY_RE = re.compile(
    r"\b(yieldmax|covered call|option income|buffer|defined outcome|target outcome|premium income)\b",
    re.IGNORECASE,
)


# ── Seed classification (used for migration from current Focus column) ───────

_EM_KEYWORDS = ("emerging", "msci em", "china", "korea", "india", "brazil",
                "mexico", "latin", "taiwan", "chile", "saudi", "hong kong",
                "asia ex japan")
_INTL_KEYWORDS = ("international", "intl", "ex us", "ex-us", "ex u.s.",
                  "developed", "eafe", "euro", "japan", "pacific",
                  "all world", "all country", "global", "acwi",
                  "canada", "germany", "australia", "united kingdom")
_FACTOR_KEYWORDS = ("quality", "momentum", "min vol", "minimum volatility",
                    "low volatility", "factor", "profitab", "moat")
_VALUE_KEYWORDS = ("value", "dividend", "div appreciation", "div growth",
                   "income", "cash flow")
_GROWTH_KEYWORDS = ("growth", "nasdaq")

_SECTOR_FOCUS_MAP = {
    "energy": "Energy",
    "materials": "Materials",
    "information technology": "Tech",
    "communication services": "Tech",
    "financials": "Financials",
    "industrials": "Industrials",
    "consumer discretionary": "Consumer Cyclical",
    "health care": "Defensives",
    "utilities": "Defensives",
    "consumer staples": "Defensives",
    "real estate": "Real Estate",
}


def seed_classify(symbol: str, name: str, focus: str) -> tuple[AssetClass, str]:
    """Map (symbol, name, focus) → (asset_class, theme).

    Used once during migration to seed the enriched CSV from the current schema.
    Pure function — no I/O.
    """
    s = symbol.upper()
    f = (focus or "").strip().lower()
    n = (name or "").lower()

    # Symbol overrides for non-equity
    if s in ASSET_CLASS_OVERRIDES:
        ac = ASSET_CLASS_OVERRIDES[s]
        if ac == AssetClass.COMMODITY:
            return ac, "Commodity"
        if ac == AssetClass.CRYPTO:
            return ac, "Crypto"
        if ac == AssetClass.FIXED_INCOME:
            return ac, "Fixed Income"

    # Sector ETFs (well-defined GICS-style focus)
    if f in _SECTOR_FOCUS_MAP:
        return AssetClass.SECTOR, _SECTOR_FOCUS_MAP[f]

    # High-dividend-yield: usually US-focused, but flag intl variants
    if f == "high dividend yield":
        if any(k in n for k in _INTL_KEYWORDS):
            return AssetClass.INTL_DEV, "Int'l Developed"
        return AssetClass.US_EQUITY, "US Value / Div"

    # Size buckets — usually US small/mid unless intl tagged
    if f in ("mid cap", "small cap"):
        if any(k in n for k in _INTL_KEYWORDS):
            return AssetClass.INTL_DEV, "Int'l Developed"
        return AssetClass.US_EQUITY, "Small / Mid Cap"

    if f == "extended market":
        return AssetClass.US_EQUITY, "Small / Mid Cap"

    # Broad / style / regional — disambiguate via name
    if f in ("large cap", "total market"):
        if any(k in n for k in _EM_KEYWORDS):
            return AssetClass.EM, "Emerging Markets"
        if any(k in n for k in _INTL_KEYWORDS):
            return AssetClass.INTL_DEV, "Int'l Developed"
        if any(k in n for k in _FACTOR_KEYWORDS):
            return AssetClass.US_EQUITY, "Smart Beta"
        if any(k in n for k in _GROWTH_KEYWORDS):
            return AssetClass.US_EQUITY, "US Growth"
        if any(k in n for k in _VALUE_KEYWORDS):
            return AssetClass.US_EQUITY, "US Value / Div"
        return AssetClass.US_EQUITY, "US Broad"

    # Thematic — route to sector where exposure is clear, else generic Thematic
    if f == "theme":
        if any(k in n for k in ("uranium", "gold", "silver", "copper",
                                 "metals", "mining", "rare earth", "steel")):
            return AssetClass.THEMATIC_EQUITY, "Materials"
        if any(k in n for k in ("china", "emerging")):
            return AssetClass.EM, "Emerging Markets"
        if any(k in n for k in ("biotech", "genomic", "pharma", "medical")):
            return AssetClass.THEMATIC_EQUITY, "Defensives"
        if any(k in n for k in ("home", "construction", "homebuilder",
                                 "online retail", "e-commerce",
                                 "autonomous", "electric vehicle")):
            return AssetClass.THEMATIC_EQUITY, "Consumer Cyclical"
        if any(k in n for k in ("infrastructure", "aerospace", "defense")):
            return AssetClass.THEMATIC_EQUITY, "Industrials"
        if any(k in n for k in ("payment", "fintech")):
            return AssetClass.THEMATIC_EQUITY, "Financials"
        if any(k in n for k in ("blockchain", "bitcoin")):
            # Blockchain *stocks* (BLOK) are equity; pure crypto trusts overridden above
            return AssetClass.THEMATIC_EQUITY, "Tech"
        if any(k in n for k in ("cyber", "cloud", "artificial intelligence",
                                 "robot", "innovation", "semi", "software",
                                 "internet", "tech", "data center",
                                 "magnificent", "telecommunications", "a.i.")):
            return AssetClass.THEMATIC_EQUITY, "Tech"
        if any(k in n for k in ("clean energy", "solar", "natural gas")):
            return AssetClass.THEMATIC_EQUITY, "Energy"
        return AssetClass.THEMATIC_EQUITY, "Thematic"

    return AssetClass.US_EQUITY, "US Broad"


# ── Metadata enrichment ──────────────────────────────────────────────────────

@dataclass
class Metadata:
    aum: Optional[float] = None
    avg_volume_30d: Optional[float] = None  # dollar volume
    inception_date: Optional[str] = None     # ISO date string
    expense_ratio: Optional[float] = None
    is_leveraged: bool = False
    is_single_stock: bool = False
    last_refreshed: Optional[str] = None     # ISO date string
    long_name: Optional[str] = None
    category: Optional[str] = None           # Morningstar category from yfinance


def enrich_one(symbol: str) -> Metadata:
    """Pull yfinance metadata for one ticker. Failures degrade to None fields."""
    md = Metadata()
    md.last_refreshed = datetime.now(timezone.utc).date().isoformat()
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
    except Exception:
        info = {}

    md.aum = info.get("totalAssets")
    er = info.get("annualReportExpenseRatio") or info.get("netExpenseRatio")
    md.expense_ratio = float(er) if er else None

    inception_ts = info.get("fundInceptionDate")
    if inception_ts:
        try:
            md.inception_date = datetime.fromtimestamp(int(inception_ts), tz=timezone.utc).date().isoformat()
        except Exception:
            md.inception_date = None

    md.long_name = info.get("longName") or info.get("shortName")
    md.category = info.get("category")

    # Pattern-based flags from the long name (info names are sometimes more
    # explicit than the short Description in universe.csv).
    long_name = md.long_name or ""
    md.is_leveraged = bool(LEVERAGED_RE.search(long_name))
    md.is_single_stock = bool(SINGLE_STOCK_RE.search(long_name))

    # Dollar volume: 30d avg shares × latest close
    try:
        hist = t.history(period="60d", auto_adjust=False)
        if not hist.empty and "Volume" in hist.columns:
            recent = hist.tail(30)
            avg_shares = recent["Volume"].mean()
            last_close = recent["Close"].iloc[-1] if len(recent) else None
            if avg_shares and last_close:
                md.avg_volume_30d = float(avg_shares * last_close)
    except Exception:
        pass

    return md


# ── Morningstar category → seed_classify Focus ─────────────────────────────

# yfinance returns Morningstar category names that don't match our Focus values
# 1:1, but most fall into our buckets. Used by classify_new_ticker.
_MORNINGSTAR_TO_FOCUS = {
    "Technology": "Information technology",
    "Communications": "Communication services",
    "Financial": "Financials",
    "Equity Energy": "Energy",
    "Natural Resources": "Materials",
    "Industrials": "Industrials",
    "Consumer Cyclical": "Consumer discretionary",
    "Consumer Defensive": "Consumer staples",
    "Health": "Health care",
    "Utilities": "Utilities",
    "Real Estate": "Real estate",
    "Large Blend": "Large cap", "Large Growth": "Large cap", "Large Value": "Large cap",
    "Mid-Cap Blend": "Mid cap", "Mid-Cap Growth": "Mid cap", "Mid-Cap Value": "Mid cap",
    "Small Blend": "Small cap", "Small Growth": "Small cap", "Small Value": "Small cap",
    "Foreign Large Blend": "Total market", "Foreign Large Growth": "Total market",
    "Foreign Large Value": "Total market", "Foreign Small/Mid Blend": "Total market",
    "Diversified Emerging Mkts": "Total market", "World Stock": "Total market",
    "Miscellaneous Sector": "Theme",
}


def classify_new_ticker(symbol: str, md: Metadata) -> tuple[str, str, str]:
    """For a freshly fetched ticker, propose (name, asset_class, theme).

    Uses Morningstar category → Focus → seed_classify, then falls back to
    name-only keyword matching if the category is unknown.
    """
    name = md.long_name or symbol
    cat = (md.category or "").strip()
    focus = _MORNINGSTAR_TO_FOCUS.get(cat, "Theme")  # unknown → thematic
    ac, theme = seed_classify(symbol, name, focus)
    return name, ac.value, theme


def enrich_metadata(symbols: list[str], progress=None) -> pd.DataFrame:
    """Batch enrichment. Returns a DataFrame indexed by symbol with metadata columns.

    `progress` is an optional callable (i, total, symbol) for reporting.
    """
    rows = []
    total = len(symbols)
    for i, s in enumerate(symbols):
        if progress:
            progress(i, total, s)
        md = enrich_one(s)
        rows.append({
            "symbol": s,
            "aum": md.aum,
            "avg_volume_30d": md.avg_volume_30d,
            "inception_date": md.inception_date,
            "expense_ratio": md.expense_ratio,
            "is_leveraged": md.is_leveraged,
            "is_single_stock": md.is_single_stock,
            "last_refreshed": md.last_refreshed,
        })
    return pd.DataFrame(rows).set_index("symbol")


# ── Inclusion rules ──────────────────────────────────────────────────────────

# Defaults — adjustable.
MIN_AUM = 50_000_000           # $50M
MIN_DOLLAR_VOLUME = 1_000_000  # $1M/day
MIN_HISTORY_YEARS = 2          # need 252d for Z-scores; require 2y headroom


def evaluate_row(row: pd.Series, today: Optional[datetime] = None) -> tuple[bool, Optional[str]]:
    """Return (included, exclude_reason). First failing rule wins."""
    today = today or datetime.now(timezone.utc)

    sym = str(row.get("symbol", "")).upper()
    name = str(row.get("name", ""))
    asset_class = row.get("asset_class")
    canonical_for = row.get("canonical_for")

    # 1. Broad benchmark
    if sym in BROAD_BENCHMARKS:
        return False, "broad index"

    # 2. Redundant (dedup) — this row points to a canonical sibling
    if pd.notna(canonical_for) and str(canonical_for).strip():
        return False, "redundant"

    # 3. Non-equity asset class
    try:
        ac = AssetClass(asset_class) if not isinstance(asset_class, AssetClass) else asset_class
    except ValueError:
        ac = None
    if ac is None or ac not in EQUITY_CLASSES:
        return False, "non-equity"

    # 4. Leveraged / inverse
    if bool(row.get("is_leveraged", False)) or LEVERAGED_RE.search(name):
        return False, "leveraged"

    # 5. Single-stock ETF
    if bool(row.get("is_single_stock", False)) or SINGLE_STOCK_RE.search(name):
        return False, "single-name"

    # 6. Yield-carry strategy
    if YIELD_CARRY_RE.search(name):
        return False, "yield-carry"

    # 7. Inception too recent (need 252d history with headroom)
    inception = row.get("inception_date")
    if pd.notna(inception) and inception:
        try:
            inc_dt = datetime.fromisoformat(str(inception)).replace(tzinfo=timezone.utc)
            years = (today - inc_dt).days / 365.25
            if years < MIN_HISTORY_YEARS:
                return False, "short history"
        except Exception:
            pass  # don't exclude on parse failure

    # 8. AUM floor (only if metadata present — missing AUM is a soft signal)
    aum = row.get("aum")
    if pd.notna(aum) and aum is not None and float(aum) < MIN_AUM:
        return False, "low AUM"

    # Liquidity floor disabled — AUM floor ($50M) is the sole capacity gate.
    # avg_volume_30d is still captured in metadata for admin reference.

    return True, None


def apply_rules(df: pd.DataFrame, preserve_overrides: bool = True) -> pd.DataFrame:
    """Populate `included` and `exclude_reason` for each row.

    If `preserve_overrides` and a row has `override == True`, its existing
    `included`/`exclude_reason` are kept (admin manual override).
    """
    df = df.copy()
    if "override" not in df.columns:
        df["override"] = False

    incs, reasons = [], []
    for _, row in df.iterrows():
        if preserve_overrides and bool(row.get("override", False)):
            incs.append(bool(row.get("included", True)))
            reasons.append(row.get("exclude_reason"))
            continue
        inc, reason = evaluate_row(row)
        incs.append(inc)
        reasons.append(reason)

    df["included"] = incs
    df["exclude_reason"] = reasons
    return df
