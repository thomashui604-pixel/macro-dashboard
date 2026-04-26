"""
Statistical tests for the macro-regime model.

Four test families:
  A. Construct validity — PCA + correlation matrix per pillar.
  B. Predictive validity — OLS of forward returns on pillar scores (HAC errors).
  C. Recession nowcast — logistic regression on NBER USREC.
  D. Regime-conditional return distributions — ANOVA/Kruskal-Wallis.

Plus:
  E. Walk-forward parity check.
  F. Sensitivity analysis on z-window, threshold, weighting.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

import regime_model as rm


# ──────────────────────────────────────────────────────────────────────
# A. Construct validity
# ──────────────────────────────────────────────────────────────────────
def cronbach_alpha(df: pd.DataFrame) -> float:
    """Internal-consistency reliability across columns of df."""
    df = df.dropna()
    k = df.shape[1]
    if k < 2 or len(df) < 3:
        return float("nan")
    item_var = df.var(axis=0, ddof=1).sum()
    total_var = df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_var / total_var)


def pillar_construct_validity(pillar_df: pd.DataFrame, name: str) -> dict:
    """PCA + correlation summary for one pillar's z-scored, sign-adjusted inputs."""
    X = pillar_df.dropna()
    if len(X) < 36:
        return {"name": name, "n": len(X), "error": "insufficient data"}

    pca = PCA(n_components=min(3, X.shape[1]))
    pca.fit(X.values)
    pc1_loadings = pd.Series(pca.components_[0], index=X.columns)

    return {
        "name": name,
        "n_obs": len(X),
        "n_inputs": X.shape[1],
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_pct": float(pca.explained_variance_ratio_[0] * 100),
        "pc1_loadings": pc1_loadings.to_dict(),
        "all_same_sign": bool((pc1_loadings > 0).all() or (pc1_loadings < 0).all()),
        "cronbach_alpha": cronbach_alpha(X),
        "correlation_matrix": X.corr().to_dict(),
    }


# ──────────────────────────────────────────────────────────────────────
# B. Predictive validity — OLS with Newey-West HAC errors
# ──────────────────────────────────────────────────────────────────────
@dataclass
class OLSResult:
    asset: str
    horizon: int
    spec: str
    n: int
    r_squared: float
    adj_r_squared: float
    coefs: dict
    tstats: dict
    pvalues: dict


def _ols_hac(y: pd.Series, X: pd.DataFrame, hac_lags: int) -> sm.regression.linear_model.RegressionResultsWrapper:
    df = pd.concat([y, X], axis=1).dropna()
    yv = df.iloc[:, 0]
    Xv = sm.add_constant(df.iloc[:, 1:])
    return sm.OLS(yv, Xv).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def forward_return_regressions(
    scores: pd.DataFrame,
    forward_returns: dict[int, pd.DataFrame],
    horizons=(1, 3, 6, 12),
    multivariate: bool = True,
) -> pd.DataFrame:
    """OLS of forward k-month returns on contemporaneous pillar scores.

    Newey-West lags = k + 1 to handle overlap from k-month forward returns.
    """
    rows: list[OLSResult] = []
    pillars = ["Growth", "Inflation", "Liquidity"]

    for asset in forward_returns[horizons[0]].columns:
        for k in horizons:
            y = forward_returns[k][asset]
            # Univariate specs — one pillar at a time.
            for p in pillars:
                X = scores[[p]]
                try:
                    res = _ols_hac(y, X, hac_lags=k + 1)
                except Exception:
                    continue
                rows.append(OLSResult(
                    asset=asset, horizon=k, spec=f"univariate:{p}",
                    n=int(res.nobs), r_squared=res.rsquared, adj_r_squared=res.rsquared_adj,
                    coefs=res.params.to_dict(), tstats=res.tvalues.to_dict(),
                    pvalues=res.pvalues.to_dict(),
                ))
            # Multivariate — all three pillars together.
            if multivariate:
                X = scores[pillars]
                try:
                    res = _ols_hac(y, X, hac_lags=k + 1)
                except Exception:
                    continue
                rows.append(OLSResult(
                    asset=asset, horizon=k, spec="multivariate:G+I+L",
                    n=int(res.nobs), r_squared=res.rsquared, adj_r_squared=res.rsquared_adj,
                    coefs=res.params.to_dict(), tstats=res.tvalues.to_dict(),
                    pvalues=res.pvalues.to_dict(),
                ))

    out = []
    for r in rows:
        base = {"asset": r.asset, "horizon": r.horizon, "spec": r.spec,
                "n": r.n, "r2": r.r_squared, "adj_r2": r.adj_r_squared}
        for k, v in r.coefs.items():
            base[f"coef_{k}"] = v
            base[f"t_{k}"] = r.tstats.get(k)
            base[f"p_{k}"] = r.pvalues.get(k)
        out.append(base)
    return pd.DataFrame(out)


# Economic-prior sign table — what the regression coefficient SHOULD be if
# the pillars carry information.
PRIOR_SIGNS = {
    ("SPY", "Growth"):    +1,
    ("SPY", "Inflation"): -1,   # higher inflation → tighter policy → bad for equities
    ("SPY", "Liquidity"): +1,
    ("AGG", "Growth"):    -1,   # strong growth → rates ↑ → bond prices ↓
    ("AGG", "Inflation"): -1,
    ("AGG", "Liquidity"): +1,
    ("GLD", "Growth"):    -1,
    ("GLD", "Inflation"): +1,
    ("GLD", "Liquidity"): +1,
    ("DXY", "Growth"):    +1,   # ambiguous; mild positive prior
    ("DXY", "Inflation"): -1,
    ("DXY", "Liquidity"): -1,
}


def sign_table(reg_df: pd.DataFrame) -> pd.DataFrame:
    """Compare actual coefficient signs to economic priors (univariate only)."""
    rows = []
    for _, r in reg_df[reg_df["spec"].str.startswith("univariate:")].iterrows():
        pillar = r["spec"].split(":")[1]
        prior = PRIOR_SIGNS.get((r["asset"], pillar))
        coef = r.get(f"coef_{pillar}")
        t = r.get(f"t_{pillar}")
        p = r.get(f"p_{pillar}")
        if coef is None or pd.isna(coef):
            continue
        rows.append({
            "asset": r["asset"],
            "horizon": r["horizon"],
            "pillar": pillar,
            "prior_sign": prior,
            "actual_sign": int(np.sign(coef)),
            "matches_prior": prior is not None and np.sign(coef) == prior,
            "coef": coef,
            "t": t,
            "p": p,
            "significant_5pct": (p is not None) and (p < 0.05),
            "n": r["n"],
            "r2": r["r2"],
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# C. Recession nowcast (NBER USREC as ground truth)
# ──────────────────────────────────────────────────────────────────────
def recession_nowcast(scores: pd.DataFrame, usrec: pd.Series, horizons=(6, 12)) -> dict:
    """Logit: P(any USREC=1 in next k months) ~ Growth + Inflation + Liquidity."""
    out: dict = {}
    for k in horizons:
        # Forward indicator: 1 if any recession month in the next k months.
        future_rec = usrec.shift(-1).rolling(k, min_periods=1).max()
        df = pd.concat([scores, future_rec.rename("y")], axis=1).dropna()
        if df["y"].nunique() < 2 or len(df) < 60:
            out[k] = {"error": "insufficient variance / data"}
            continue

        X = df[["Growth", "Inflation", "Liquidity"]].values
        y = df["y"].astype(int).values

        # statsmodels logit for coefficients + pseudo-R².
        Xc = sm.add_constant(X)
        try:
            sm_logit = sm.Logit(y, Xc).fit(disp=0)
            pseudo_r2 = float(sm_logit.prsquared)
            coefs = dict(zip(["const", "Growth", "Inflation", "Liquidity"], sm_logit.params))
            tstats = dict(zip(["const", "Growth", "Inflation", "Liquidity"], sm_logit.tvalues))
            pvals = dict(zip(["const", "Growth", "Inflation", "Liquidity"], sm_logit.pvalues))
        except Exception as e:
            pseudo_r2, coefs, tstats, pvals = float("nan"), {}, {}, {}

        # AUC via sklearn for an interpretable headline.
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        prob = clf.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, prob))

        # Compare to a naive regime-label flag.
        out[k] = {
            "n": len(df), "base_rate": float(y.mean()),
            "pseudo_r2": pseudo_r2, "auc": auc,
            "coefs": coefs, "tstats": tstats, "pvalues": pvals,
            "predicted_prob": pd.Series(prob, index=df.index),
            "actual": pd.Series(y, index=df.index),
        }
    return out


def regime_label_recession_signal(labels: pd.DataFrame, usrec: pd.Series, horizons=(6, 12),
                                  recession_labels=("Recession", "Hard Landing Risk",
                                                    "Slowdown", "Stagflation",
                                                    "Recession (Recovery Setup)")) -> dict:
    """How well does the discrete regime label flag NBER recessions?"""
    out = {}
    flag = labels["Regime"].isin(recession_labels).astype(int).rename("flag")
    flag.index = pd.to_datetime(flag.index)
    for k in horizons:
        future_rec = usrec.shift(-1).rolling(k, min_periods=1).max()
        df = pd.concat([flag, future_rec.rename("y")], axis=1).dropna()
        if len(df) == 0 or df["y"].nunique() < 2:
            out[k] = {"error": "insufficient data"}
            continue
        try:
            auc = float(roc_auc_score(df["y"], df["flag"]))
        except Exception:
            auc = float("nan")
        out[k] = {
            "n": len(df),
            "auc_label_flag": auc,
            "flag_base_rate": float(df["flag"].mean()),
            "recession_base_rate": float(df["y"].mean()),
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# D. Regime-conditional return distributions
# ──────────────────────────────────────────────────────────────────────
def regime_conditional_returns(labels: pd.DataFrame, prices: pd.DataFrame,
                               group_col: str = "Regime",
                               min_obs: int = 6) -> dict:
    """Mean / std / Sharpe of next-month returns conditional on regime label."""
    next_ret = np.log(prices).diff().shift(-1)  # next month's log return
    df = labels[[group_col]].join(next_ret, how="inner").dropna(subset=[group_col])

    summary_rows = []
    anova_results = {}
    groups_kept = (df[group_col].value_counts() >= min_obs)
    valid_labels = groups_kept[groups_kept].index.tolist()
    df = df[df[group_col].isin(valid_labels)]

    for asset in prices.columns:
        for label, sub in df.groupby(group_col):
            r = sub[asset].dropna()
            if len(r) < min_obs:
                continue
            mu, sd = r.mean(), r.std()
            sharpe_annual = (mu / sd) * np.sqrt(12) if sd > 0 else float("nan")
            ci = stats.t.interval(0.95, len(r) - 1, loc=mu, scale=sd / np.sqrt(len(r))) if len(r) > 1 else (np.nan, np.nan)
            summary_rows.append({
                "asset": asset, "regime": label, "n": int(len(r)),
                "mean_log_ret": float(mu), "std": float(sd),
                "sharpe_annualized": float(sharpe_annual),
                "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
            })

        # ANOVA + Kruskal-Wallis on whether regime means/medians differ.
        groups = [sub[asset].dropna().values for _, sub in df.groupby(group_col)
                  if len(sub[asset].dropna()) >= min_obs]
        if len(groups) >= 2:
            try:
                f_stat, p_anova = stats.f_oneway(*groups)
            except Exception:
                f_stat, p_anova = float("nan"), float("nan")
            try:
                h_stat, p_kw = stats.kruskal(*groups)
            except Exception:
                h_stat, p_kw = float("nan"), float("nan")
            anova_results[asset] = {
                "f_stat": float(f_stat), "p_anova": float(p_anova),
                "h_stat": float(h_stat), "p_kruskal": float(p_kw),
                "n_groups": len(groups),
            }

    return {
        "summary": pd.DataFrame(summary_rows),
        "anova": pd.DataFrame(anova_results).T,
    }


def coarse_regime_groups(labels: pd.Series) -> pd.Series:
    """Collapse 27 regimes into 6 macro buckets for higher per-bucket sample size."""
    GOLDILOCKS = {"Goldilocks / Late Cycle", "Goldilocks / Early Cycle",
                  "Disinflationary Boom", "Soft Landing / Recovery"}
    EXPANSION = {"Mid-Cycle Expansion", "Steady Expansion", "Late Cycle",
                 "Reflation", "Reflation / Recovery"}
    OVERHEATING = {"Overheating", "Late Cycle Slowdown Risk",
                   "Late Cycle Disinflation", "Mild Tightening"}
    SLOWDOWN = {"Slowdown", "Disinflationary Slowdown", "Policy Easing",
                "Policy Easing / Trough", "Transition", "Transition (Inflation)",
                "Transition (Disinflation)"}
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

    return labels.apply(bucket)


# ──────────────────────────────────────────────────────────────────────
# E. Walk-forward sanity check
# ──────────────────────────────────────────────────────────────────────
def walk_forward_parity(raw_fred: dict, sample_dates: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """Recompute regime labels using only data up to each test date.

    Confirms that the rolling z-score is genuinely point-in-time. In-sample and
    walk-forward labels at date T should match because z-score is causal.
    """
    full = rm.compute_regime(raw_fred)["combined"]
    if sample_dates is None:
        # Test on every 12th month-end to keep it fast.
        idx = full.dropna(subset=["Regime"]).index
        sample_dates = idx[::12].tolist()

    rows = []
    for d in sample_dates:
        truncated = {sid: s.loc[:d] for sid, s in raw_fred.items()}
        wf = rm.compute_regime(truncated)["combined"]
        if d not in wf.index or d not in full.index:
            continue
        rows.append({
            "date": d,
            "in_sample_label": full.loc[d, "Regime"],
            "walk_forward_label": wf.loc[d, "Regime"],
            "match": full.loc[d, "Regime"] == wf.loc[d, "Regime"],
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# F. Sensitivity analysis on hyperparameters
# ──────────────────────────────────────────────────────────────────────
def sensitivity_grid(raw_fred: dict, forward_returns: dict[int, pd.DataFrame],
                     z_windows=(24, 36, 48),
                     headline_asset: str = "SPY", headline_horizon: int = 3) -> pd.DataFrame:
    """Re-run the headline OLS under different z-score windows.

    Reports R² and pillar coefficients for the multivariate spec at the
    headline (asset, horizon). The classification threshold doesn't enter the
    continuous-score regression, so it's varied separately in
    ``threshold_sensitivity`` against the discrete-label tests.
    """
    rows = []
    y = forward_returns[headline_horizon][headline_asset]
    for w in z_windows:
        out = rm.compute_regime(raw_fred, z_window=w, z_min_periods=max(12, w - 12))
        s = out["combined"][["Growth", "Inflation", "Liquidity"]]
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        try:
            res = _ols_hac(y, s, hac_lags=headline_horizon + 1)
            rows.append({
                "z_window": w,
                "n": int(res.nobs), "r2": res.rsquared,
                **{f"coef_{p}": res.params.get(p) for p in ["Growth", "Inflation", "Liquidity"]},
                **{f"t_{p}": res.tvalues.get(p) for p in ["Growth", "Inflation", "Liquidity"]},
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def threshold_sensitivity(raw_fred: dict, prices: pd.DataFrame,
                          thresholds=(0.25, 0.5, 0.75)) -> pd.DataFrame:
    """Vary the ±σ classification threshold; report ANOVA p-values across regimes.

    Uses the coarse 6-bucket grouping so per-bucket samples are large enough
    to be meaningful at all three thresholds.
    """
    next_ret = np.log(prices).diff().shift(-1)
    rows = []
    for thr in thresholds:
        out = rm.compute_regime(raw_fred, threshold=thr)
        labels = out["combined"][["Regime"]].copy()
        labels.index = pd.to_datetime(labels.index).to_period("M").to_timestamp("M")
        labels["bucket"] = coarse_regime_groups(labels["Regime"])
        df = labels[["bucket"]].join(next_ret, how="inner").dropna(subset=["bucket"])
        for asset in prices.columns:
            groups = [sub[asset].dropna().values for _, sub in df.groupby("bucket")
                      if len(sub[asset].dropna()) >= 6]
            if len(groups) < 2:
                continue
            try:
                f, p = stats.f_oneway(*groups)
            except Exception:
                f, p = float("nan"), float("nan")
            rows.append({
                "threshold": thr, "asset": asset,
                "n_buckets": len(groups), "anova_F": float(f), "anova_p": float(p),
            })
    return pd.DataFrame(rows)
