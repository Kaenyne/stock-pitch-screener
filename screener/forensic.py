"""Forensic / short-side metric library (spec SHORT table; audits 5.1, 5.2,
5.5, 5.6, 3.3).

Single home for the short composite's forensic theme, the false-moat
leverage-propping row, the inflated-growth rows (revenue deceleration,
DSO/DIO expansion, debt-funded buybacks, dilution), the event flags, the
zero/low-revenue cohort tag (audit 5.5), and the SI hump kernel.

Everything here operates on the CompanyData contract from
screener.fundamentals (cd.q / cd.ttm / cd.yoy / cd.inst / cd.meta /
cd.market / cd.flags) and returns NaN/empty gracefully on missing inputs —
never raises. Strings are added to cd.flags whenever a fallback or guard
fires. Cross-sectional normalization (percentile ranks, peer medians,
exclude-and-renormalize) happens downstream in scoring.py; this module emits
raw per-name values only.

Spec literals that are NOT config tunables (they come from the audit text /
module spec, documented here per the audit-4.4 process rule) are module
constants below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .edgar import SubmissionsInfo, has_8k_item
from .fundamentals import CompanyData
from .inflection import theil_sen_slope
from .quarterlyize import align_instant_to_quarters, avg_instant, yoy_ratio

# ---------------------------------------------------------------------------
# Module constants (spec/audit literals with no config counterpart)
# ---------------------------------------------------------------------------
DAYS_PER_YEAR = 365.0
YOY_TOL_DAYS = 45            # same-fiscal-quarter matching tolerance (~1.5mo)
# DSO/DIO applicability materiality (~2% — spec Missing-data policy item 3)
MATERIALITY_FRACTION = 0.02
# "high base = peak YoY in last 8 quarters" (audit 5.1 / spec SHORT table)
DECEL_PEAK_WINDOW_Q = 8
# slope steepness normalizer: -5pp/quarter of YoY-growth slope = full credit
DECEL_SLOPE_FULL = 0.05
# revenue-vs-GP growth divergence threshold (spec: > 5pp)
REV_GP_DIVERGENCE_PP = 0.05
# leverage propping: +5pp ROE-ROA gap widening over 4q = full credit
LEV_GAP_DELTA_FULL = 0.05
# unadjusted-split detector on diluted WAS (audit 5.6: a 2:1 split reads as
# ~100% dilution; >1.9x YoY jump in either direction = suspect)
SPLIT_JUMP_FACTOR = 1.9
# event-flag soft bump weights (spec SHORT table: small soft bump, capped 1.0)
EVENT_BUMP_NT = 0.25
EVENT_BUMP_AUDITOR = 0.25
EVENT_BUMP_NON_RELIANCE = 0.50

# Beneish (1999) M-Score coefficients — model constants, not tunables.
BENEISH_INTERCEPT = -4.84
BENEISH_WEIGHTS = {
    "dsri": 0.92, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892,
    "depi": 0.115, "sgai": -0.172, "tata": 4.679, "lvgi": -0.327,
}
# Neutral values for non-computable components (audit 5.2 fallbacks):
# ratio indices are neutral at 1.0; TATA (an accrual ratio) is neutral at 0.
BENEISH_NEUTRAL = {"dsri": 1.0, "gmi": 1.0, "aqi": 1.0, "sgi": 1.0,
                   "depi": 1.0, "sgai": 1.0, "tata": 0.0, "lvgi": 1.0}


# ---------------------------------------------------------------------------
# Small internal helpers (mirror the quarterlyize._find_prior pattern)
# ---------------------------------------------------------------------------

def _empty() -> pd.Series:
    return pd.Series(dtype=float)


def _ttm(cd: CompanyData, concept: str) -> pd.Series:
    s = cd.ttm.get(concept)
    return s.dropna() if s is not None and len(s) else _empty()


def _inst(cd: CompanyData, concept: str) -> pd.Series:
    s = cd.inst.get(concept)
    return s.dropna() if s is not None and len(s) else _empty()


def _qvals(cd: CompanyData, concept: str) -> pd.Series:
    qs = cd.q.get(concept)
    if qs is None or not len(qs.values):
        return _empty()
    return qs.values.dropna()


def _latest(s: pd.Series) -> float:
    return float(s.iloc[-1]) if s is not None and len(s) else np.nan


def _near(s: pd.Series, target: pd.Timestamp, tol_days: int = YOY_TOL_DAYS
          ) -> float:
    """Value of s at the end date nearest `target` within tolerance
    (the end-date matching pattern of quarterlyize._find_prior)."""
    if s is None or not len(s):
        return np.nan
    idx = s.index
    lo = target - pd.Timedelta(days=tol_days)
    hi = target + pd.Timedelta(days=tol_days)
    cand = idx[(idx >= lo) & (idx <= hi)]
    if len(cand) == 0:
        return np.nan
    t = cand[np.argmin(np.abs((cand - target).days))]
    return float(s.loc[t])


def _yoy_pair(s: pd.Series) -> tuple[float, float, pd.Timestamp | None]:
    """(latest value, fiscal-aligned value ~365d earlier, latest end date)."""
    if s is None or not len(s):
        return np.nan, np.nan, None
    t = s.index.max()
    prev = _near(s, t - pd.Timedelta(days=DAYS_PER_YEAR))
    return float(s.loc[t]), prev, t


def _finite(*vals) -> bool:
    return all(v is not None and np.isfinite(v) for v in vals)


# ---------------------------------------------------------------------------
# 1. Sloan accruals — THE single accruals definition (audit 5.2 / 3.3)
# ---------------------------------------------------------------------------

def sloan_accruals(cd: CompanyData) -> pd.Series:
    """Sloan accruals series = (TTM net income - TTM CFO) / average assets.

    CASH-FLOW method (audit 5.2): balance-sheet accruals are corrupted by
    M&A/divestitures/FX (Hribar & Collins) — endemic in this cap band — so
    accruals are measured from the statement of cash flows. Average assets =
    mean of assets now and 4 quarters ago (quarterlyize.avg_instant).

    This is the ONE accruals construction shared by the forensic theme and
    the false-moat leverage line (audit 3.3: compute Sloan exactly once);
    Beneish TATA below reuses the same numerator. Returns an end-date-indexed
    Series (empty on missing inputs).
    """
    ni = _ttm(cd, "net_income")
    cfo = _ttm(cd, "cfo")
    assets = _inst(cd, "assets")
    if not len(ni) or not len(cfo) or not len(assets):
        return _empty()
    common = ni.index.intersection(cfo.index)
    if len(common) == 0:
        return _empty()
    accruals = (ni[common] - cfo[common]).astype(float)
    assets_q = align_instant_to_quarters(assets, common)
    if not len(assets_q):
        return _empty()
    avg_assets = avg_instant(assets_q, periods_back=4)
    avg_assets = avg_assets[avg_assets > 0]
    if not len(avg_assets):
        return _empty()
    idx = accruals.index.intersection(avg_assets.index)
    if len(idx) == 0:
        return _empty()
    out = (accruals[idx] / avg_assets[idx]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    return out


# ---------------------------------------------------------------------------
# 2. Beneish M-Score — continuous, FY-over-FY (spec SHORT table; audit 5.2)
# ---------------------------------------------------------------------------

def beneish(cd: CompanyData) -> dict:
    """Continuous Beneish M-Score, fiscal-year over fiscal-year.

    Annual sums are the TTM values at the latest available quarter t vs the
    TTM values at t-4 fiscal quarters (end-date matching ~365d back, the
    quarterlyize._find_prior pattern); instants are matched to the same two
    dates. Never emits a -2.22 binary — the continuous score is percentile-
    ranked downstream (audit 5.2).

    Component fallbacks (each fires a cd.flags string):
    - DEPI: needs cd.q['depreciation_only']; else neutral 1.0 + 'depi_neutral'.
      With depreciation available, the canonical PPE denominator is not in the
      CompanyData contract, so the depreciation rate uses total assets
      (dep / (dep + assets)) + 'depi_ppe_proxy_assets'.
    - SGAI: cd.q['sga'] ladder; else neutral 1.0 + 'sgai_neutral'.
    - GMI: gross_profit ladder; else neutral 1.0 + 'gmi_neutral'.
    - AQI: no PPE tag in the contract -> soft-asset proxy
      (1 - AssetsCurrent/Assets) + 'aqi_no_ppe'.
    - LVGI: total_debt missing treated as 0 when current liabilities exist
      (debt-free filers have no debt tags) + 'lvgi_partial_components'.
    - TATA: balance-sheet-free version (TTM NI - TTM CFO) / assets — same
      numerator as sloan_accruals (audit 5.2 shared definition).

    Ratio indices winsorized to config.BENEISH_INDEX_WINSOR (TATA is not an
    index and is not winsorized). Requires >= config.BENEISH_MIN_COMPONENTS
    computed-from-data components, else m_score=NaN with a reason; neutral
    fills enter the formula but do not count toward n_computable.

    Suppressed for financials/REITs (out-of-model, audit 5.2):
    m_score=NaN + 'beneish_suppressed'.

    Returns {'m_score', 'components' (all 8), 'n_computable',
    'sgi_dominant' (SGI contribution > half of |sum of contributions|),
    'reason' (None when computed)}.
    """
    comps = dict(BENEISH_NEUTRAL)
    out = {"m_score": np.nan, "components": comps, "n_computable": 0,
           "sgi_dominant": False, "reason": None}
    if cd.meta.get("is_financial"):
        cd.flags.add("beneish_suppressed")
        out["reason"] = "beneish_suppressed"
        return out

    rev = _ttm(cd, "revenue")
    sales_t, sales_p, t = _yoy_pair(rev)
    if t is None or not _finite(sales_t) or sales_t <= 0:
        out["reason"] = "no_revenue"
        return out
    tp = t - pd.Timedelta(days=DAYS_PER_YEAR)
    have_prior_sales = _finite(sales_p) and sales_p > 0

    lo, hi = config.BENEISH_INDEX_WINSOR

    def winsor(x: float) -> float:
        return float(np.clip(x, lo, hi))

    computed: set[str] = set()

    # --- DSRI: (receivables/sales)_t / (receivables/sales)_{t-1}
    rec = _inst(cd, "receivables")
    rec_t, rec_p = _near(rec, t), _near(rec, tp)
    if have_prior_sales and _finite(rec_t, rec_p) and rec_p > 0:
        comps["dsri"] = winsor((rec_t / sales_t) / (rec_p / sales_p))
        computed.add("dsri")

    # --- GMI: GM_{t-1} / GM_t via the gross_profit ladder
    gp = _ttm(cd, "gross_profit")
    gp_t, gp_p = _near(gp, t), _near(gp, tp)
    if have_prior_sales and _finite(gp_t, gp_p) \
            and gp_t > 0 and gp_p > 0:
        comps["gmi"] = winsor((gp_p / sales_p) / (gp_t / sales_t))
        computed.add("gmi")
    else:
        cd.flags.add("gmi_neutral")

    # --- AQI: soft-asset proxy (no PPE tag in the CompanyData contract)
    assets = _inst(cd, "assets")
    ca = _inst(cd, "assets_current")
    ta_t, ta_p = _near(assets, t), _near(assets, tp)
    ca_t, ca_p = _near(ca, t), _near(ca, tp)
    if _finite(ta_t, ta_p, ca_t, ca_p) and ta_t > 0 and ta_p > 0:
        soft_t = 1.0 - ca_t / ta_t
        soft_p = 1.0 - ca_p / ta_p
        if soft_p > 0 and soft_t >= 0:
            comps["aqi"] = winsor(soft_t / soft_p)
            computed.add("aqi")
            cd.flags.add("aqi_no_ppe")

    # --- SGI: sales_t / sales_{t-1}
    if have_prior_sales:
        comps["sgi"] = winsor(sales_t / sales_p)
        computed.add("sgi")

    # --- DEPI: depreciation-rate index; needs the Depreciation-only tag
    dep = _ttm(cd, "depreciation_only")
    dep_t, dep_p = _near(dep, t), _near(dep, tp)
    if _finite(dep_t, dep_p, ta_t, ta_p) and dep_t > 0 and dep_p > 0 \
            and ta_t > 0 and ta_p > 0:
        rate_t = dep_t / (dep_t + ta_t)
        rate_p = dep_p / (dep_p + ta_p)
        if rate_t > 0:
            comps["depi"] = winsor(rate_p / rate_t)
            computed.add("depi")
            cd.flags.add("depi_ppe_proxy_assets")
    if "depi" not in computed:
        cd.flags.add("depi_neutral")

    # --- SGAI: (SG&A/sales)_t / (SG&A/sales)_{t-1}
    sga = _ttm(cd, "sga")
    sga_t, sga_p = _near(sga, t), _near(sga, tp)
    if have_prior_sales and _finite(sga_t, sga_p) and sga_p > 0:
        comps["sgai"] = winsor((sga_t / sales_t) / (sga_p / sales_p))
        computed.add("sgai")
    else:
        cd.flags.add("sgai_neutral")

    # --- LVGI: leverage index, (debt + current liabilities) / assets
    debt = _inst(cd, "total_debt")
    lc = _inst(cd, "liabilities_current")
    debt_t, debt_p = _near(debt, t), _near(debt, tp)
    lc_t, lc_p = _near(lc, t), _near(lc, tp)
    # zero-substitute a component only when its SERIES is entirely absent
    # (a debt-free filer genuinely has no debt tag); a series that merely
    # lacks an observation near t is a data gap -> NaN, not zero
    if not _finite(debt_t) and not len(debt):
        debt_t = 0.0 if _finite(lc_t) else np.nan
    if not _finite(debt_p) and not len(debt):
        debt_p = 0.0 if _finite(lc_p) else np.nan
    if not _finite(lc_t) and not len(lc):
        lc_t = 0.0 if len(debt) else np.nan
    if not _finite(lc_p) and not len(lc):
        lc_p = 0.0 if len(debt) else np.nan
    if _finite(debt_t, debt_p, lc_t, lc_p, ta_t, ta_p) \
            and ta_t > 0 and ta_p > 0:
        if not len(debt) or not len(lc):
            cd.flags.add("lvgi_partial_components")
        lev_t = (debt_t + lc_t) / ta_t
        lev_p = (debt_p + lc_p) / ta_p
        if lev_p > 0:
            comps["lvgi"] = winsor(lev_t / lev_p)
            computed.add("lvgi")

    # --- TATA: (TTM NI - TTM CFO) / assets (balance-sheet-free; audit 5.2)
    ni_t = _near(_ttm(cd, "net_income"), t)
    cfo_t = _near(_ttm(cd, "cfo"), t)
    if _finite(ni_t, cfo_t, ta_t) and ta_t > 0:
        comps["tata"] = float((ni_t - cfo_t) / ta_t)
        computed.add("tata")

    out["n_computable"] = len(computed)
    if len(computed) < config.BENEISH_MIN_COMPONENTS:
        out["reason"] = "insufficient_components"
        return out

    contributions = {k: BENEISH_WEIGHTS[k] * comps[k] for k in BENEISH_WEIGHTS}
    total = sum(contributions.values())
    out["m_score"] = float(BENEISH_INTERCEPT + total)
    # SGI-dominance flag (audit 5.2: SGI mechanically flags fast growers)
    out["sgi_dominant"] = bool(np.isfinite(total)
                               and contributions["sgi"] > 0.5 * abs(total))
    return out


# ---------------------------------------------------------------------------
# 3. DSO / DIO expansion — YoY same-fiscal-quarter (spec SHORT table;
#    applicability per Missing-data policy item 3)
# ---------------------------------------------------------------------------

def dso_dio(cd: CompanyData) -> dict:
    """DSO/DIO expansion, measured YoY SAME-FISCAL-QUARTER (latest vs the
    quarter ~365d earlier — never sequential, which is seasonality-corrupted).

    DSO = receivables / (TTM revenue / 365)
    DIO = inventory  / (TTM COGS / 365); COGS = revenue - gross_profit when
    no direct tag resolves (the config gross_profit ladder derive rule).

    Applicability masks (spec Missing-data policy item 3; inapplicable ->
    NaN deltas, excluded-and-renormalized downstream):
    - dso_applicable: receivables material (> ~2% of TTM revenue) AND
      non-financial;
    - dio_applicable: inventory material (> ~2% of TTM revenue or of assets).

    Returns {'dso_yoy_delta' (days), 'dio_yoy_delta' (days),
    'dso_applicable', 'dio_applicable'}.
    """
    out = {"dso_yoy_delta": np.nan, "dio_yoy_delta": np.nan,
           "dso_applicable": False, "dio_applicable": False}
    rev = _ttm(cd, "revenue")
    sales_t, sales_p, t = _yoy_pair(rev)
    if t is None or not _finite(sales_t) or sales_t <= 0:
        return out
    tp = t - pd.Timedelta(days=DAYS_PER_YEAR)

    # --- DSO
    rec = _inst(cd, "receivables")
    rec_t, rec_p = _near(rec, t), _near(rec, tp)
    non_financial = not bool(cd.meta.get("is_financial"))
    if non_financial and _finite(rec_t) \
            and rec_t > MATERIALITY_FRACTION * sales_t:
        out["dso_applicable"] = True
        if _finite(rec_p) and _finite(sales_p) and sales_p > 0:
            dso_t = rec_t / (sales_t / DAYS_PER_YEAR)
            dso_p = rec_p / (sales_p / DAYS_PER_YEAR)
            out["dso_yoy_delta"] = float(dso_t - dso_p)

    # --- DIO (COGS = revenue - gross_profit when no direct tag)
    cogs = _ttm(cd, "cogs")
    if not len(cogs):
        gp = _ttm(cd, "gross_profit")
        common = rev.index.intersection(gp.index)
        if len(common):
            cogs = (rev[common] - gp[common]).astype(float)
    inv = _inst(cd, "inventory")
    inv_t, inv_p = _near(inv, t), _near(inv, tp)
    assets_t = _near(_inst(cd, "assets"), t)
    material = _finite(inv_t) and (
        inv_t > MATERIALITY_FRACTION * sales_t
        or (_finite(assets_t) and assets_t > 0
            and inv_t > MATERIALITY_FRACTION * assets_t))
    if material:
        out["dio_applicable"] = True
        cogs_t, cogs_p = _near(cogs, t), _near(cogs, tp)
        if _finite(inv_p, cogs_t, cogs_p) and cogs_t > 0 and cogs_p > 0:
            dio_t = inv_t / (cogs_t / DAYS_PER_YEAR)
            dio_p = inv_p / (cogs_p / DAYS_PER_YEAR)
            out["dio_yoy_delta"] = float(dio_t - dio_p)
    return out


# ---------------------------------------------------------------------------
# 4. Revenue deceleration off a high base (audit 5.1)
# ---------------------------------------------------------------------------

def revenue_decel(cd: CompanyData) -> dict:
    """Revenue growth decelerating off a high base (audit 5.1).

    YoY revenue growth per quarter (quarterlyize.yoy_ratio on the discrete
    quarterly series — fiscal-aligned, seasonality-free) over the last
    config.DECEL_LOOKBACK_Q quarters:
    - high_base: peak YoY over the last 8 quarters >=
      config.DECEL_HIGH_BASE_PEAK_YOY ('peak_yoy' returned raw so the
      top-quintile-of-universe alternative can be applied downstream);
    - decel: latest YoY well below peak ('gap' = peak - latest) AND negative
      Theil-Sen slope of the YoY series over config.DECEL_SLOPE_WINDOW_Q;
    - score_raw in [0,1]: 0 if not high_base or slope >= 0, else
      min(1, gap/peak_yoy) x min(1, -slope/0.05); NaN when the slope is not
      computable (excluded, not zero-filled).

    M&A flag — TAG, not gate (fake decel from lapping deals): goodwill up
    > config.MA_GOODWILL_JUMP YoY OR TTM acquisitions >
    config.MA_ACQ_REVENUE_FRACTION x TTM revenue.

    rev_gp_divergence: TTM revenue YoY growth exceeds TTM gross-profit YoY
    growth by > 5pp — separate revenue-quality flag (audit 5.1).
    """
    out = {"high_base": False, "peak_yoy": np.nan, "latest_yoy": np.nan,
           "gap": np.nan, "decel_slope": np.nan, "score_raw": np.nan,
           "ma_flag": False, "rev_gp_divergence": False}
    g = yoy_ratio(_qvals(cd, "revenue")).dropna()
    if len(g):
        window = g.iloc[-config.DECEL_LOOKBACK_Q:]
        peak = float(window.iloc[-DECEL_PEAK_WINDOW_Q:].max())
        latest = float(window.iloc[-1])
        out["peak_yoy"] = peak
        out["latest_yoy"] = latest
        out["gap"] = peak - latest
        out["high_base"] = bool(peak >= config.DECEL_HIGH_BASE_PEAK_YOY)
        slope = theil_sen_slope(
            window.iloc[-config.DECEL_SLOPE_WINDOW_Q:].values)
        out["decel_slope"] = float(slope) if np.isfinite(slope) else np.nan
        if not out["high_base"]:
            out["score_raw"] = 0.0
        elif np.isfinite(slope):
            if slope >= 0:
                out["score_raw"] = 0.0
            else:
                depth = min(1.0, max(0.0, out["gap"] / peak))
                steep = min(1.0, -slope / DECEL_SLOPE_FULL)
                out["score_raw"] = float(np.clip(depth * steep, 0.0, 1.0))
        # slope NaN with high_base: score stays NaN (exclude, don't zero-fill)

    # --- M&A flag (tag, not gate)
    gw = _inst(cd, "goodwill")
    gw_t, gw_p, _ = _yoy_pair(gw)
    if _finite(gw_t, gw_p) and gw_p > 0 \
            and (gw_t / gw_p - 1.0) > config.MA_GOODWILL_JUMP:
        out["ma_flag"] = True
    acq_t = _latest(_ttm(cd, "acquisitions"))
    rev_t = _latest(_ttm(cd, "revenue"))
    if _finite(acq_t, rev_t) and rev_t > 0 \
            and acq_t > config.MA_ACQ_REVENUE_FRACTION * rev_t:
        out["ma_flag"] = True

    # --- revenue-vs-GP growth divergence (revenue-quality flag)
    rt, rp, _ = _yoy_pair(_ttm(cd, "revenue"))
    gt, gp_prev, _ = _yoy_pair(_ttm(cd, "gross_profit"))
    if _finite(rt, rp, gt, gp_prev) and rp > 0 and gp_prev > 0:
        rev_growth = rt / rp - 1.0
        gp_growth = gt / gp_prev - 1.0
        out["rev_gp_divergence"] = bool(
            rev_growth - gp_growth > REV_GP_DIVERGENCE_PP)
    return out


# ---------------------------------------------------------------------------
# 5. Leverage propping — the false-moat leverage row (audit 3.3)
# ---------------------------------------------------------------------------

def leverage_propping(cd: CompanyData, roic: pd.Series | None = None) -> dict:
    """Real leverage-propping measure replacing the old 'leverage/accruals'
    false-moat row (audit 3.3): returns-propped-by-leverage, not accruals
    (accruals live once in the forensic theme via sloan_accruals).

    (a) ROE-ROA gap widening: gap = TTM NI/avg equity - TTM NI/avg assets;
        'roe_roa_gap_delta' = gap now minus gap 4 quarters ago (>0 = widening).
        Computed only where average equity > 0 ('lev_negative_equity' flag
        when negative-equity endpoints block it).
    (b) Debt/EBITDA rising while ROIC falls: EBITDA = TTM operating income +
        TTM D&A ('ebitda_no_da' flag when D&A missing -> EBIT proxy); ratio
        only where EBITDA > 0; 'roic_falling' from the caller-supplied roic
        series (latest vs ~4q earlier; False when unavailable).

    score_raw in [0,1] = max of the two normalized signals: the gap signal
    min(1, delta/0.05) (0 when narrowing), and the debt signal 1.0 iff
    debt/EBITDA rising AND ROIC falling else 0. NaN when neither signal is
    computable.
    """
    out = {"roe_roa_gap_delta": np.nan, "debt_ebitda_delta": np.nan,
           "roic_falling": False, "score_raw": np.nan}

    # --- (a) ROE - ROA gap widening
    ni = _ttm(cd, "net_income")
    eq = _inst(cd, "equity")
    assets = _inst(cd, "assets")
    if len(ni) and len(eq) and len(assets):
        eq_q = align_instant_to_quarters(eq, ni.index)
        as_q = align_instant_to_quarters(assets, ni.index)
        if len(eq_q) and len(as_q):
            avg_eq = avg_instant(eq_q, periods_back=4)
            avg_as = avg_instant(as_q, periods_back=4)
            idx = ni.index.intersection(avg_eq.index).intersection(
                avg_as.index)
            if len(idx):
                pos = (avg_eq[idx] > 0) & (avg_as[idx] > 0)
                if bool((~pos).any()):
                    cd.flags.add("lev_negative_equity")
                idx = idx[pos]
                if len(idx):
                    gap = (ni[idx] / avg_eq[idx]
                           - ni[idx] / avg_as[idx]).dropna()
                    gap_t, gap_p, pair_idx = _yoy_pair(gap)
                    if _finite(gap_t, gap_p):
                        # gap-widening only means "returns propped by
                        # leverage" when returns are POSITIVE: for a
                        # loss-maker the gap is negative and shrinking
                        # losses raise it mechanically — not propping.
                        ni_ok = bool((ni[idx] > 0).iloc[-1]) if len(idx) else False
                        if ni_ok:
                            out["roe_roa_gap_delta"] = float(gap_t - gap_p)
                        else:
                            cd.flags.add("lev_gap_lossmaker")

    # --- (b) debt/EBITDA rising while ROIC falls
    oi = _ttm(cd, "operating_income")
    da = _ttm(cd, "dep_amort")
    ebitda = _empty()
    if len(oi):
        if len(da):
            common = oi.index.intersection(da.index)
            ebitda = (oi[common] + da[common]).astype(float)
        else:
            ebitda = oi.copy()
            cd.flags.add("ebitda_no_da")
    debt = _inst(cd, "total_debt")
    if len(ebitda) and len(debt):
        debt_q = align_instant_to_quarters(debt, ebitda.index)
        idx = ebitda.index.intersection(debt_q.index)
        idx = idx[ebitda[idx] > 0]
        if len(idx):
            de = (debt_q[idx] / ebitda[idx]).dropna()
            de_t, de_p, _ = _yoy_pair(de)
            if _finite(de_t, de_p):
                out["debt_ebitda_delta"] = float(de_t - de_p)

    roic_known = False
    if roic is not None and len(roic.dropna()):
        r = roic.dropna()
        r_t, r_p, _ = _yoy_pair(r)
        if _finite(r_t, r_p):
            out["roic_falling"] = bool(r_t - r_p < 0)
            roic_known = True

    # --- combine: max of the two normalized signals. The debt/EBITDA leg
    # requires BOTH of its inputs (delta and roic trend); with roic unknown
    # the leg is INCOMPUTABLE (excluded), not "definitely not propping".
    signals = []
    if _finite(out["roe_roa_gap_delta"]):
        signals.append(float(np.clip(
            out["roe_roa_gap_delta"] / LEV_GAP_DELTA_FULL, 0.0, 1.0)))
    if _finite(out["debt_ebitda_delta"]) and roic_known:
        signals.append(1.0 if (out["debt_ebitda_delta"] > 0
                               and out["roic_falling"]) else 0.0)
    if signals:
        out["score_raw"] = float(max(signals))
    return out


# ---------------------------------------------------------------------------
# 6. Debt-funded buybacks (audit 5.6 / spec inflated-growth row)
# ---------------------------------------------------------------------------

def debt_funded_buybacks(cd: CompanyData) -> dict:
    """TTM buybacks exceed free cash flow while total debt rises YoY —
    literally 'growth assumed to persist' (spec inflated-growth row;
    audit 5.6 item 2).

    FCF = TTM CFO - TTM capex (reuses cd.ttm['fcf'] when the fundamentals
    layer built it — same definition, capex-fallback flag included there).
    A missing buybacks tag on a filer WITH a cash-flow statement (CFO
    present) means no repurchases (the cash-flow line only exists when
    nonzero): treated as $0 + 'buybacks_absent_as_zero' flag, scoring 0
    rather than NaN; with no cash-flow data at all the score stays NaN.
    'active' additionally requires buybacks > 0 so a no-buyback name with
    negative FCF cannot fire.

    Returns {'active', 'buyback_excess' (buybacks - FCF, $),
    'score_raw' 1.0 if active else 0.0 (NaN when buybacks exist but FCF or
    debt history is missing)}.
    """
    out = {"active": False, "buyback_excess": np.nan, "score_raw": np.nan}
    bb = _ttm(cd, "buybacks")
    if len(bb):
        bb_t = _latest(bb)
    elif len(_ttm(cd, "cfo")):
        bb_t = 0.0
        cd.flags.add("buybacks_absent_as_zero")
    else:
        return out  # no cash-flow data at all: unknown, not zero

    fcf = _ttm(cd, "fcf")
    if len(fcf):
        fcf_t = _latest(fcf)
    else:
        cfo = _ttm(cd, "cfo")
        capex = _ttm(cd, "capex")
        cfo_t = _latest(cfo)
        capex_t = _latest(capex) if len(capex) else 0.0
        fcf_t = cfo_t - capex_t if _finite(cfo_t, capex_t) else np.nan

    debt_t, debt_p, _ = _yoy_pair(_inst(cd, "total_debt"))
    debt_rising = _finite(debt_t, debt_p) and debt_t > debt_p

    if _finite(bb_t, fcf_t):
        out["buyback_excess"] = float(bb_t - fcf_t)
    if bb_t == 0.0:
        out["active"] = False
        out["score_raw"] = 0.0
    elif _finite(bb_t, fcf_t) and _finite(debt_t, debt_p):
        out["active"] = bool(bb_t > 0 and bb_t > fcf_t and debt_rising)
        out["score_raw"] = 1.0 if out["active"] else 0.0
    # else: buybacks exist but FCF/debt unknown -> NaN (exclude downstream)
    return out


# ---------------------------------------------------------------------------
# 7. Dilution (audit 5.6 item 1)
# ---------------------------------------------------------------------------

def dilution_yoy(cd: CompanyData) -> float:
    """Latest YoY growth in SPLIT-ADJUSTED diluted weighted-average shares
    (audit 5.6: without split adjustment a 2:1 split reads as 100% dilution).

    cd.q['diluted_was'].values is adjusted with cd.market['splits'] (a
    pd.Series of split ratios indexed by date): observations whose period end
    precedes a split date are multiplied by that split's ratio (cumulative
    across multiple splits), restating history onto the current share basis.

    When splits data is None/empty, no adjustment is assumed — but if the
    latest YoY factor jumps more than 1.9x in either direction (a 2:1 split
    or 1:2 reverse split signature), 'dilution_unadjusted_splits' is flagged
    and NaN returned rather than reporting a fake ~100% dilution.
    """
    # diluted WAS lives as a LEVEL series (fundamentals resolves ~90d
    # duration facts directly; never Q4-derived/TTM-summed)
    q = cd.inst.get("diluted_was_q", pd.Series(dtype=float)).dropna()
    if len(q) < 5:
        return np.nan
    splits = cd.market.get("splits")
    have_splits = splits is not None and len(splits) > 0
    adj = q.astype(float).copy()
    if have_splits:
        try:
            for d, r in splits.dropna().items():
                r = float(r)
                if not np.isfinite(r) or r <= 0:
                    continue
                d = pd.Timestamp(d)
                if d.tzinfo is not None:
                    d = d.tz_localize(None)
                mask = adj.index < d
                adj[mask] = adj[mask] * r
        except (TypeError, ValueError):
            have_splits = False  # malformed splits: fall through unadjusted
            adj = q.astype(float).copy()
    g = yoy_ratio(adj).dropna()
    if not len(g):
        return np.nan
    latest = float(g.iloc[-1])
    if not have_splits:
        factor = 1.0 + latest
        if factor > SPLIT_JUMP_FACTOR or 0 < factor < 1.0 / SPLIT_JUMP_FACTOR:
            cd.flags.add("dilution_unadjusted_splits")
            return np.nan
    return latest


# ---------------------------------------------------------------------------
# 8. Event flags (audit 5.6 items 3-4; spec forensic event-flag row)
# ---------------------------------------------------------------------------

def event_flags(cd: CompanyData, sub_info: SubmissionsInfo | None) -> dict:
    """Cheap, high-signal EDGAR event tells (audit 5.6): tear-sheet red
    lines + a SMALL soft bump, not heavy composite weights.

    - restated: an 8-K item 4.02, OR an amended periodic report (10-K/A etc.)
      that actually moved a retained value >3%. Both legs name a filing the
      reader can pull, which is the point: the old definition was the >3%
      value move ALONE, so it fired on 1,495 of 2,138 names (69.9%) while
      1,459 of them had no 4.02 — and the tear-sheet checklist told the reader
      to go read that non-existent filing. A judge falsifies that in one
      question.
    - comparatives_represented: the >3% retained-value move on its own. Real
      and worth showing, but it is ordinary re-presentation (segment
      recasts, discontinued-ops reclassification), not a restatement claim.
    - nt_filer: NT 10-K/10-Q filings in the last ~3 years;
    - auditor_change: 8-K item 4.01 (review prompt — benign rotations exist);
    - non_reliance: 8-K item 4.02 (high signal);
    - score_raw = 0.25*nt + 0.25*auditor_change + 0.5*(non_reliance or
      restated), capped at 1.0.
    """
    represented = "restated" in cd.flags       # >3% retained-value move
    nt = auditor = non_reliance = amended = False
    if sub_info is not None:
        nt = sub_info.nt_filings_recent > 0
        auditor = has_8k_item(sub_info, "4.01")
        non_reliance = has_8k_item(sub_info, "4.02")
        amended = bool(sub_info.amended_periodic)
    restated = bool(non_reliance or (amended and represented))
    score = (EVENT_BUMP_NT * float(nt)
             + EVENT_BUMP_AUDITOR * float(auditor)
             + EVENT_BUMP_NON_RELIANCE * float(non_reliance or restated))
    return {"restated": restated,
            "comparatives_represented": bool(represented),
            "restatement_filing": (sub_info.amended_periodic[0][0]
                                   if amended and sub_info.amended_periodic
                                   else ("8-K 4.02" if non_reliance else "")),
            "nt_filer": bool(nt),
            "auditor_change": bool(auditor),
            "non_reliance": bool(non_reliance),
            "score_raw": float(min(1.0, score))}


# ---------------------------------------------------------------------------
# 8b. Pricing-masking-volume signature (user direction 2026-07-08 — the
#     CAKE tell: price hikes create short-term margin/revenue bumps that
#     mask a declining demand base)
# ---------------------------------------------------------------------------

def pricing_masking_volume(cd: CompanyData) -> dict:
    """Conjunctive signature (kernel in config.PMV): GM expanding while
    revenue growth decays toward low single digits and asset turnover falls
    — revenue growth that is all price, no volume. XBRL cannot see traffic/
    units; this three-way accounting signature is the proxy.

    Components (each [0,1], geometric mean, NO floors — a signature, so all
    three legs must be present):
    - gm_expand: TTM gross margin change over ~4 quarters, full at +2pp
    - growth_decay: latest revenue YoY, 1.0 at <=2%, 0 at >=8%; hard 0 below
      -2% (outright decline belongs to the deceleration/still-declining
      machinery — no double counting)
    - turnover_fall: relative fall in TTM revenue / avg assets over ~4
      quarters, full at -5%

    Returns {'gm_expand_pp', 'rev_yoy_latest', 'turnover_delta_rel',
    'score_raw', 'flag'}; adds 'pricing_masking_volume' to cd.flags when the
    score clears config.PMV['flag_at']."""
    k = config.PMV
    out = {"gm_expand_pp": np.nan, "rev_yoy_latest": np.nan,
           "turnover_delta_rel": np.nan, "score_raw": np.nan, "flag": False,
           "margin_source": None}
    rev_ttm = _ttm(cd, "revenue")
    gp_ttm = _ttm(cd, "gross_profit")
    assets = _inst(cd, "assets")
    rev_q = _qvals(cd, "revenue")
    if len(rev_ttm) < 6 or len(rev_q) < 5:
        return out
    rev_end = rev_ttm.dropna().index.max()

    def _margin_expand(num_ttm: pd.Series) -> float:
        """Margin change: the LARGER of the ~4q and ~8q changes (price-hike
        campaigns span quarters; a hike that started 6 quarters ago still
        counts). Series must be FRESH (ends near the revenue series end)."""
        if len(num_ttm) < 8:
            return np.nan
        common = num_ttm.index.intersection(rev_ttm.index)
        m = (num_ttm[common] / rev_ttm[common].where(rev_ttm[common] > 0)).dropna()
        if len(m) < 8 or (rev_end - m.index.max()).days > 200:
            return np.nan
        t = m.index.max()
        changes = []
        for days_back in (DAYS_PER_YEAR, 2 * DAYS_PER_YEAR):
            prev = _near(m, t - pd.Timedelta(days=days_back))
            if _finite(prev):
                changes.append(float(m.loc[t] - prev))
        return max(changes) if changes else np.nan

    # margin leg: GM when its tags still resolve (fresh), else OPERATING
    # margin — filers who moved COGS to custom extension tags (CAKE) have no
    # standardized GM after ~2022, but the price-hike bump shows in OM too
    out["gm_expand_pp"] = _margin_expand(gp_ttm)
    out["margin_source"] = "gm"
    if not _finite(out["gm_expand_pp"]):
        out["gm_expand_pp"] = _margin_expand(_ttm(cd, "operating_income"))
        out["margin_source"] = "om"
        if _finite(out["gm_expand_pp"]):
            cd.flags.add("pmv_om_fallback")
    # growth_decay: latest revenue YoY
    g = yoy_ratio(rev_q).dropna()
    if len(g):
        out["rev_yoy_latest"] = float(g.iloc[-1])
    # turnover_fall: TTM revenue / avg assets, now vs ~4 quarters ago
    if len(assets):
        as_q = align_instant_to_quarters(assets, rev_ttm.index)
        if len(as_q):
            avg_as = avg_instant(as_q, periods_back=4)
            common = rev_ttm.index.intersection(avg_as.index)
            turn = (rev_ttm[common] / avg_as[common].where(avg_as[common] > 0)).dropna()
            tn_t, tn_p, _ = _yoy_pair(turn)
            if _finite(tn_t, tn_p) and tn_p > 0:
                out["turnover_delta_rel"] = float(tn_t / tn_p - 1.0)

    a = out["gm_expand_pp"]
    b = out["rev_yoy_latest"]
    c = out["turnover_delta_rel"]
    if not (_finite(a) and _finite(b) and _finite(c)):
        return out
    gm_comp = float(np.clip(a / k["gm_expand_full_pp"], 0.0, 1.0))
    if b < k["growth_floor"]:
        growth_comp = 0.0
    else:
        growth_comp = float(np.clip((k["growth_zero"] - b)
                                    / (k["growth_zero"] - k["growth_full"]),
                                    0.0, 1.0))
    turn_comp = float(np.clip(-c / k["turnover_fall_full"], 0.0, 1.0))
    score = float((gm_comp * growth_comp * turn_comp) ** (1.0 / 3.0)) \
        if min(gm_comp, growth_comp, turn_comp) > 0 else 0.0
    out["score_raw"] = score
    out["flag"] = bool(score >= k["flag_at"])
    if out["flag"]:
        cd.flags.add("pricing_masking_volume")
    return out


# ---------------------------------------------------------------------------
# 8b. Non-operating / one-time earnings share (CAKE quality-of-earnings tell)
# ---------------------------------------------------------------------------

def nonop_earnings_share(cd: CompanyData) -> dict:
    """Share of pretax income coming from BELOW the operating line, excluding
    interest expense — gains on sale, other/non-operating income, gift-card
    breakage (the CAKE $4.01 -> $2.66 quality-of-earnings bridge).

    other_nonop = pretax - operating_income + interest_expense (interest added
    back so a normal financing cost isn't miscounted as 'other income');
    share = other_nonop / pretax. score_raw scales on the LEVEL, zeroed when the
    share is clearly falling (improving quality); the flag additionally requires
    a clear YoY rise (the checklist tell).

    Degeneracy guard: the operating_income tag ladder's fallback IS the pretax
    ladder's primary tag (IncomeLossFromContinuingOperationsBeforeIncomeTaxes
    ...), so a filer lacking OperatingIncomeLoss resolves operating_income ==
    pretax and the bridge collapses to exactly interest_expense — detected by
    series equality; returns NaN + 'nonop_oi_is_pretax'.

    Returns {'share','share_yoy_delta','score_raw','flag','degenerate'}.
    """
    out = {"share": np.nan, "share_yoy_delta": np.nan, "score_raw": np.nan,
           "flag": False, "degenerate": False}
    pretax = _ttm(cd, "pretax_income")
    oper = _ttm(cd, "operating_income")
    if pretax.empty or oper.empty:
        return out
    common = pretax.index.intersection(oper.index)
    if len(common) < 4:
        return out
    p = pretax[common]
    o = oper[common]
    # degeneracy: operating_income resolved to the pretax tag (ladder fallback)
    denom = p.abs().clip(lower=1e6)
    if bool((((p - o).abs() / denom) < 1e-6).all()):
        out["degenerate"] = True
        cd.flags.add("nonop_oi_is_pretax")
        return out
    ie = _ttm(cd, "interest_expense").reindex(common)
    other = p - o + ie.fillna(0.0)
    share = (other / p.where(p > 0)).dropna()
    if share.empty:
        return out
    share_now = float(share.iloc[-1])
    out["share"] = share_now
    prev = _near(share, share.index.max() - pd.Timedelta(days=DAYS_PER_YEAR))
    delta = share_now - prev if _finite(prev) else np.nan
    out["share_yoy_delta"] = delta
    # meaningful only with a positive operating core at the latest point
    o_now = float(o.loc[common.max()])
    if not (np.isfinite(o_now) and o_now > 0):
        return out
    k = config.NONOP_SHARE
    lvl = float(np.clip((share_now - k["share_zero"])
                        / (k["share_full"] - k["share_zero"]), 0.0, 1.0))
    falling = np.isfinite(delta) and delta < k["yoy_min"]
    out["score_raw"] = 0.0 if falling else lvl
    out["flag"] = bool(share_now >= k["flag_share"] and np.isfinite(delta)
                       and delta >= k["flag_yoy"])
    return out


# ---------------------------------------------------------------------------
# 8b2. Tax-tailwind EPS (Tier-2: quality-of-earnings)
# ---------------------------------------------------------------------------

def tax_tailwind(cd: CompanyData) -> dict:
    """EPS propped by a below-trend effective tax rate. Builds an UNCLAMPED TTM
    effective-rate series (tax/pretax where pretax>0); boost = max(0,
    median_rate - current_rate) x pretax/net_income = the share of net income
    attributable to paying below the company's OWN normal (median) rate. Fires
    on a material boost with >=8q history, pretax>0, net_income>0.
    (config.TAX_RATE_CLAMP would mask genuine benefits, so this is unclamped.)
    Returns {'eff_rate','median_rate','boost_share','score_raw','flag'}."""
    out = {"eff_rate": np.nan, "median_rate": np.nan, "boost_share": np.nan,
           "score_raw": np.nan, "flag": False}
    pretax = _ttm(cd, "pretax_income")
    tax = _ttm(cd, "tax_expense")
    ni = _ttm(cd, "net_income")
    if len(pretax) < 8 or len(tax) == 0:
        return out
    common = pretax.index.intersection(tax.index)
    if len(common) < 8:
        return out
    p, t = pretax[common], tax[common]
    rate = (t / p.where(p > 0)).dropna()   # unclamped, pretax>0 only
    if len(rate) < 8:
        return out
    eff, med = float(rate.iloc[-1]), float(np.median(rate.values))
    out["eff_rate"], out["median_rate"] = eff, med
    p_now, ni_now = _latest(p), _latest(ni)
    if not (_finite(p_now, ni_now) and p_now > 0 and ni_now > 0):
        return out
    boost = max(0.0, med - eff) * p_now / ni_now
    out["boost_share"] = boost
    k = config.TAX_TAILWIND
    out["score_raw"] = float(np.clip(boost / k["boost_full"], 0.0, 1.0))
    out["flag"] = bool(boost >= k["flag_boost"]
                       and (med - eff) >= k["rate_drop_min"])
    return out


# ---------------------------------------------------------------------------
# 8c. Capacity-turnover decay (Tier-2: CAKE organic-vs-new-unit proxy)
# ---------------------------------------------------------------------------

def capacity_turnover_decay(cd: CompanyData) -> dict:
    """Organic-vs-new-capacity growth proxy (CAKE 'all reported growth is new
    units'). organic_proxy = TTM revenue YoY - net PP&E YoY: the revenue growth
    NOT explained by capacity expansion. Fires when revenue is still growing and
    capacity is expanding but organic_proxy is negative — growth that is just
    added units/stores, not same-unit demand.

    Asset-light guard: NaN when net PP&E < config.CAPACITY_MIN_PPE_ASSETS of
    total assets (the ratio is meaningless for software/services). Sub-universe
    by construction (retail/restaurant/hotel/industrial). Returns
    {'organic_proxy','rev_yoy','ppe_yoy','flag','score_raw'}."""
    out = {"organic_proxy": np.nan, "rev_yoy": np.nan, "ppe_yoy": np.nan,
           "flag": False, "score_raw": np.nan}
    rev = _ttm(cd, "revenue")
    ppe = _inst(cd, "ppe_net")
    if len(rev) < 5 or len(ppe) < 5:
        return out
    ppe_now, as_now = _latest(ppe), _latest(_inst(cd, "assets"))
    if _finite(ppe_now, as_now) and as_now > 0 and \
            ppe_now / as_now < config.CAPACITY_MIN_PPE_ASSETS:
        return out  # asset-light -> ratio is noise
    r_now, r_prev, _ = _yoy_pair(rev)
    p_now, p_prev, _ = _yoy_pair(ppe)
    if not (_finite(r_now, r_prev) and r_prev > 0
            and _finite(p_now, p_prev) and p_prev > 0):
        return out
    rev_yoy = r_now / r_prev - 1.0
    ppe_yoy = p_now / p_prev - 1.0
    organic = rev_yoy - ppe_yoy
    out.update(organic_proxy=organic, rev_yoy=rev_yoy, ppe_yoy=ppe_yoy)
    k = config.CAPACITY_DECAY
    growing_on_capacity = (rev_yoy > k["rev_grow_min"]
                           and ppe_yoy > k["ppe_grow_min"])
    out["flag"] = bool(growing_on_capacity and organic < k["organic_max"])
    out["score_raw"] = (float(np.clip(-organic / k["organic_full"], 0.0, 1.0))
                        if growing_on_capacity else 0.0)
    return out


# ---------------------------------------------------------------------------
# 8d. External output-price signals (Tier-2 FRED: VITL windfall / real revenue)
# ---------------------------------------------------------------------------

def external_price_signal(cd: CompanyData, price_series, asof) -> dict:
    """Given the industry PPI series mapped to this name (SIC crosswalk / ticker
    override), filtered point-in-time to <= asof:
      - ppi_windfall: index >= z_min sigma above its trailing-5yr mean AND
        rolling over (>= rolloff_min below its window peak) — a boom riding an
        external price that is elevated and reverting (VITL/UAMY/DK).
      - real_rev: real_rev_growth = (1+rev_YoY)/(1+ppi_YoY)-1; flag real<0 while
        nominal>0 (a deflated PROXY — the index is not the firm's realized price).
    Returns {'ppi_zscore','ppi_windfall','real_rev_growth','real_rev_deflated'}."""
    out = {"ppi_zscore": np.nan, "ppi_windfall": False,
           "real_rev_growth": np.nan, "real_rev_deflated": False}
    if price_series is None or not len(price_series):
        return out
    ps = price_series[price_series.index <= asof].dropna().sort_index()
    k = config.PPI_WINDFALL
    if len(ps) < k["min_obs_m"]:
        return out
    win = ps.iloc[-k["trend_window_m"]:]
    mu, sd, now = float(win.mean()), float(win.std()), float(ps.iloc[-1])
    if sd > 0:
        z = (now - mu) / sd
        out["ppi_zscore"] = z
        peak = float(win.max())
        rolling_over = peak > 0 and now < peak * (1 - k["rolloff_min"])
        out["ppi_windfall"] = bool(z >= k["z_min"] and rolling_over)
    rev = _ttm(cd, "revenue")
    if len(rev) >= 5:
        r_now, r_prev, _ = _yoy_pair(rev)
        p_prev = _near(ps, ps.index.max() - pd.Timedelta(days=DAYS_PER_YEAR))
        if _finite(r_now, r_prev, p_prev) and r_prev > 0 and p_prev > 0:
            rev_yoy = r_now / r_prev - 1.0
            ppi_yoy = now / p_prev - 1.0
            real = (1 + rev_yoy) / (1 + ppi_yoy) - 1.0
            out["real_rev_growth"] = real
            out["real_rev_deflated"] = bool(real < 0 and rev_yoy > 0)
    return out


# ---------------------------------------------------------------------------
# 9. Zero/low-revenue cohort tag (audit 5.5)
# ---------------------------------------------------------------------------

def zero_low_revenue(cd: CompanyData) -> bool:
    """Zero/low-revenue cohort tag (audit 5.5): TTM revenue <
    config.ZERO_REV_TTM, or TTM revenue / market cap <
    config.ZERO_REV_MKTCAP_FRACTION. Market cap from cd.market; when
    missing, the revenue test alone decides.

    Downstream, tagged names have all sales-denominated short metrics nulled
    and renormalized (EV/Sales, decel, margins, DSO/DIO, Beneish) and rank in
    their own segmented section — this function only emits the tag. A name
    with no revenue data at all is tagged True ('zero_rev_no_revenue_data'
    flag): pre-revenue filers are exactly this cohort.
    """
    rev_t = _latest(_ttm(cd, "revenue"))
    if not _finite(rev_t):
        # annual-only FPIs have NO TTM series by construction (cd.annual is
        # the data home) — they are not pre-revenue; use the latest FY value
        ann = cd.annual.get("revenue") if hasattr(cd, "annual") else None
        if ann is not None and len(ann.dropna()):
            rev_t = float(ann.dropna().iloc[-1])
        else:
            cd.flags.add("zero_rev_no_revenue_data")
            return True
    if rev_t < config.ZERO_REV_TTM:
        return True
    mktcap = None
    for key in ("market_cap", "mktcap", "marketCap"):
        v = cd.market.get(key)
        if v is not None and np.isfinite(v) and v > 0:
            mktcap = float(v)
            break
    if mktcap is not None:
        return bool(rev_t / mktcap < config.ZERO_REV_MKTCAP_FRACTION)
    return False


# ---------------------------------------------------------------------------
# 10. Short-interest hump kernel (spec confirmation row; audit 4.7)
# ---------------------------------------------------------------------------

def si_hump(si_ratio: float) -> float:
    """Hump-shaped short-confirmation kernel on SI = sharesShort /
    sharesOutstanding (spec confirmation row: 'elevated but not crowded';
    breakpoints from config.SI_HUMP).

    Pure function: 0 at 0; linear rise to 1.0 at rise_full_at; flat 1.0
    through flat_to; linear decline to crowded_score at crowded_at; constant
    crowded_score beyond. NaN in -> NaN out.
    """
    k = config.SI_HUMP
    if si_ratio is None or not np.isfinite(si_ratio):
        return np.nan
    si = float(si_ratio)
    if si <= 0.0:
        return 0.0
    rise, flat = k["rise_full_at"], k["flat_to"]
    crowded_at, crowded = k["crowded_at"], k["crowded_score"]
    if si < rise:
        return si / rise
    if si <= flat:
        return 1.0
    if si < crowded_at:
        return 1.0 + (si - flat) / (crowded_at - flat) * (crowded - 1.0)
    return float(crowded)


# ---------------------------------------------------------------------------
# 11. Margin-vs-peer inputs (audit 5.3 — interaction logic lives downstream)
# ---------------------------------------------------------------------------

def margin_vs_peer_inputs(cd: CompanyData) -> dict:
    """The name's own latest TTM gross margin and operating margin —
    just the raw inputs for the margin-vs-peer INTERACTION term (audit 5.3).
    Peer medians (3-digit SIC, fallback 2-digit, >=8 members, SIC 6xxx
    excluded) and the interaction gating live downstream in scoring.py where
    the peer groups exist. Returns {'gm': float, 'om': float} (NaN when the
    inputs are missing).
    """
    rev = _ttm(cd, "revenue")
    out = {"gm": np.nan, "om": np.nan}
    if not len(rev):
        return out
    gp = _ttm(cd, "gross_profit")
    if len(gp):
        common = gp.index.intersection(rev.index)
        common = common[rev[common] > 0]
        if len(common):
            out["gm"] = float(gp[common].iloc[-1] / rev[common].iloc[-1])
    oi = _ttm(cd, "operating_income")
    if len(oi):
        common = oi.index.intersection(rev.index)
        common = common[rev[common] > 0]
        if len(common):
            out["om"] = float(oi[common].iloc[-1] / rev[common].iloc[-1])
    return out


__all__ = [
    "sloan_accruals",
    "beneish",
    "dso_dio",
    "revenue_decel",
    "leverage_propping",
    "debt_funded_buybacks",
    "dilution_yoy",
    "nonop_earnings_share",
    "tax_tailwind",
    "capacity_turnover_decay",
    "external_price_signal",
    "event_flags",
    "zero_low_revenue",
    "si_hump",
    "margin_vs_peer_inputs",
]
