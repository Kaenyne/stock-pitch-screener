"""Metric library: ROIC/WACC/moat spread, F-Score, survivability, and the
financials/REIT substitute metrics.

Implements the spec sections "ROIC, WACC, and guards" (audits 3.5, 3.6),
"Financials/REITs substitute mapping" (audit 3.4), the demonstrated-economics
moat construction (audit 4.2), the survivability context strip (audit 4.5),
and the TTM Piotroski F-Score (audit 4.8).

Contract:
- operates ONLY on the CompanyData object (``cd``) built by
  screener.fundamentals -- cd.q / cd.ttm / cd.yoy / cd.inst / cd.market /
  cd.meta / cd.flags;
- every function gracefully returns NaN / empty (never raises) when inputs
  are missing -- the missing-data policy is exclude-and-renormalize
  downstream (audit 3.1), never zero-fill;
- whenever a clamp / fallback / guard fires, a flag string is added to
  cd.flags (spec: "Tear-sheet flag whenever any clamp/fallback fired");
- all tunables come from screener.config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .quarterlyize import align_instant_to_quarters, avg_instant, ttm_ratio

# ---------------------------------------------------------------------------
# Module constants (values that do NOT exist in config; everything tunable
# lives in screener.config per the spec's process rule, audit 4.4)
# ---------------------------------------------------------------------------
QDAYS = 91.25                    # avg days per fiscal quarter (mirrors inflection.QDAYS)
YEAR_DAYS = 365                  # fiscal-aligned 4-quarter lookback
NEAR_TOL_DAYS = 60               # tolerance when snapping to a target date
MOAT_MAX_QUARTER_GAP_DAYS = 130  # adjacency bound for "consecutive" quarters
GM_EXCLUDE_TRAILING_Q = 7        # spec: robust dispersion excl. trailing 6-8q
CONSISTENCY_MIN_OBS = 8          # earnings-consistency minimum observations
FSCORE_N_SIGNALS = 9             # definitional size of the Piotroski scale
RUNWAY_GREEN_MULT = 2.0          # runway >= 2x warn threshold reads "green"
MAD_TO_STD = 1.4826              # MAD -> sigma consistency constant
ALTMAN_COEFS = (1.2, 1.4, 3.3, 0.6, 1.0)  # canonical Altman (1968) weights


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------

def _fnum(x) -> float:
    """Coerce anything (None, str, np scalar) to a finite float or NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _series(d: dict, key: str) -> pd.Series:
    """Fetch a Series from cd.ttm / cd.inst dicts; empty float Series if
    absent (never raises)."""
    s = d.get(key) if isinstance(d, dict) else None
    if isinstance(s, pd.Series) and len(s):
        return s
    return pd.Series(dtype=float)


def _qvals(cd, concept: str) -> pd.Series:
    """Quarterly flow values for a concept (cd.q[concept].values), empty
    Series when the concept is unresolved."""
    qs = cd.q.get(concept) if isinstance(cd.q, dict) else None
    if qs is not None and isinstance(qs.values, pd.Series) and len(qs.values):
        return qs.values
    return pd.Series(dtype=float)


def _latest(s: pd.Series) -> tuple[pd.Timestamp | None, float]:
    """(index, value) of the last valid observation, (None, NaN) if none."""
    s = s.dropna()
    if s.empty:
        return None, np.nan
    return s.index[-1], float(s.iloc[-1])


def _latest_val(s: pd.Series) -> float:
    return _latest(s)[1]


def _near_val(s: pd.Series, target: pd.Timestamp,
              tol_days: int = NEAR_TOL_DAYS) -> float:
    """Value at the observation nearest to `target` within +/- tol_days."""
    s = s.dropna()
    if s.empty:
        return np.nan
    win = s[(s.index >= target - pd.Timedelta(days=tol_days))
            & (s.index <= target + pd.Timedelta(days=tol_days))]
    if win.empty:
        return np.nan
    pos = int(np.argmin(np.abs((win.index - target).days)))
    return float(win.iloc[pos])


def _last_le(s: pd.Series, cutoff: pd.Timestamp,
             max_age_days: int | None = None
             ) -> tuple[pd.Timestamp | None, float]:
    """Last observation at or before `cutoff` (as-of semantics, audit 1.6).
    Rejected when older than max_age_days (default config.STALE_FACT_DAYS)."""
    if max_age_days is None:
        max_age_days = config.STALE_FACT_DAYS
    s = s.dropna()
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return None, np.nan
    s = s[s.index <= cutoff]
    if s.empty:
        return None, np.nan
    idx = s.index[-1]
    if (cutoff - idx).days > max_age_days:
        return None, np.nan
    return idx, float(s.iloc[-1])


def _nearest_reindex(s: pd.Series, grid: pd.DatetimeIndex,
                     tol_days: int = 45) -> pd.Series:
    """Reindex a series onto `grid` snapping to the nearest observation
    within tol_days (NaN where nothing is close)."""
    if len(grid) == 0:
        return pd.Series(dtype=float)
    s = s.dropna()
    if s.empty:
        return pd.Series(np.nan, index=grid, dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(grid, method="nearest",
                     tolerance=pd.Timedelta(days=tol_days)).astype(float)


def _is_financial(cd) -> bool:
    if bool(cd.meta.get("is_financial", False)):
        return True
    return str(cd.meta.get("fin_sector", "nonfinancial")) != "nonfinancial"


def _fin_sector(cd) -> str:
    return str(cd.meta.get("fin_sector", "nonfinancial"))


def _avg_assets_on(cd, grid: pd.DatetimeIndex) -> pd.Series:
    """4-quarter average total assets aligned onto a quarterly grid."""
    assets = _series(cd.inst, "assets")
    if assets.dropna().empty or len(grid) == 0:
        return pd.Series(dtype=float)
    aq = align_instant_to_quarters(assets, grid)
    return avg_instant(aq) if len(aq) else pd.Series(dtype=float)


def _gm_ttm(cd) -> pd.Series:
    """TTM gross margin = TTM gross profit / TTM revenue, revenue floored at
    config.MARGIN_DENOM_ASSET_FLOOR x assets (audit 2.2)."""
    gp = _series(cd.ttm, "gross_profit")
    rev = _series(cd.ttm, "revenue")
    if gp.dropna().empty or rev.dropna().empty:
        return pd.Series(dtype=float)
    assets = _series(cd.inst, "assets")
    floor = assets * config.MARGIN_DENOM_ASSET_FLOOR if len(assets) else None
    return ttm_ratio(gp, rev, den_floor=floor)


# ---------------------------------------------------------------------------
# Tax rate (shared by NOPAT and WACC; spec "ROIC, WACC, and guards", audit 3.5)
# ---------------------------------------------------------------------------

def _tax_rate_series(cd, grid: pd.DatetimeIndex) -> pd.Series:
    """Per-period effective TTM tax rate on `grid`: TTM tax / TTM pretax
    clamped to config.TAX_RATE_CLAMP; statutory
    config.TAX_RATE_STATUTORY_FALLBACK wherever pretax <= 0 or either input
    is unresolved (flag 'tax_fallback'; 'tax_clamped' when the clamp binds).
    """
    lo, hi = config.TAX_RATE_CLAMP
    tax = _nearest_reindex(_series(cd.ttm, "tax_expense"), grid)
    pretax = _nearest_reindex(_series(cd.ttm, "pretax_income"), grid)
    if len(grid) == 0:
        return pd.Series(dtype=float)
    ok = pretax.notna() & (pretax > 0) & tax.notna()
    raw = (tax / pretax).where(ok)
    clipped = raw.clip(lower=lo, upper=hi)
    if bool(((clipped != raw) & raw.notna()).any()):
        cd.flags.add("tax_clamped")
    if bool((~ok).any()):
        cd.flags.add("tax_fallback")
    return clipped.where(ok, other=config.TAX_RATE_STATUTORY_FALLBACK
                         ).astype(float)


def _current_tax_rate(cd) -> float:
    """Latest effective TTM tax rate, same clamp/fallback logic as NOPAT."""
    pretax = _series(cd.ttm, "pretax_income").dropna()
    tax = _series(cd.ttm, "tax_expense").dropna()
    lo, hi = config.TAX_RATE_CLAMP
    common = pretax.index.intersection(tax.index)
    if len(common) == 0:
        cd.flags.add("tax_fallback")
        return config.TAX_RATE_STATUTORY_FALLBACK
    t_idx = common.max()
    pt, tx = float(pretax[t_idx]), float(tax[t_idx])
    if not np.isfinite(pt) or pt <= 0 or not np.isfinite(tx):
        cd.flags.add("tax_fallback")
        return config.TAX_RATE_STATUTORY_FALLBACK
    raw = tx / pt
    clamped = min(max(raw, lo), hi)
    if clamped != raw:
        cd.flags.add("tax_clamped")
    return clamped


# ---------------------------------------------------------------------------
# 1. NOPAT (spec "ROIC, WACC, and guards"; audit 3.5)
# ---------------------------------------------------------------------------

def nopat_ttm(cd) -> pd.Series:
    """TTM NOPAT = TTM OperatingIncomeLoss x (1 - t).

    Spec "ROIC, WACC, and guards" / audit 3.5: t = TTM tax expense / TTM
    pretax income, clamped to config.TAX_RATE_CLAMP per period; statutory
    config.TAX_RATE_STATUTORY_FALLBACK when pretax <= 0 or the tax inputs
    are unresolved (flag 'tax_fallback'). Empty Series when TTM operating
    income is unavailable.
    """
    oi = _series(cd.ttm, "operating_income").dropna()
    if oi.empty:
        return pd.Series(dtype=float)
    rate = _tax_rate_series(cd, oi.index)
    return (oi * (1.0 - rate)).dropna().sort_index()


# ---------------------------------------------------------------------------
# 2. Invested capital (spec "ROIC, WACC, and guards"; audit 3.5)
# ---------------------------------------------------------------------------

def invested_capital(cd) -> pd.Series:
    """Invested capital = total debt (cd.inst['total_debt'], operating
    leases already excluded by the ladder) + equity - cash, GOODWILL-
    INCLUSIVE (audit 3.5; goodwill-ex variant for the tear sheet is
    invested_capital_ex_goodwill).

    Instant components are snapped onto the quarterly grid of
    operating_income via align_instant_to_quarters. Equity is required; a
    completely absent total-debt or cash series is treated as zero (a filer
    with no debt has no debt tags -- flags 'ic_no_debt_data' /
    'ic_no_cash_data' record the fallback). When a component series exists,
    only quarters where it aligns are kept (never zero-filled).
    """
    grid = _qvals(cd, "operating_income").index
    eq = _series(cd.inst, "equity")
    if len(grid) == 0 or eq.dropna().empty:
        return pd.Series(dtype=float)
    eq_q = align_instant_to_quarters(eq, grid)
    if eq_q.empty:
        return pd.Series(dtype=float)

    debt = _series(cd.inst, "total_debt")
    cash = _series(cd.inst, "cash")
    debt_q = align_instant_to_quarters(debt, grid) if len(debt) else \
        pd.Series(dtype=float)
    cash_q = align_instant_to_quarters(cash, grid) if len(cash) else \
        pd.Series(dtype=float)

    common = eq_q.index
    if len(debt):
        common = common.intersection(debt_q.index)
    else:
        cd.flags.add("ic_no_debt_data")
    if len(cash):
        common = common.intersection(cash_q.index)
    else:
        cd.flags.add("ic_no_cash_data")
    if len(common) == 0:
        return pd.Series(dtype=float)

    ic = eq_q[common].astype(float)
    if len(debt):
        ic = ic + debt_q[common]
    if len(cash):
        ic = ic - cash_q[common]
    return ic.dropna().sort_index()


def invested_capital_ex_goodwill(cd) -> pd.Series:
    """Goodwill-EXCLUSIVE invested capital (tear-sheet variant, audit 3.5).

    Goodwill is subtracted at quarters where a goodwill instant aligns;
    quarters with no goodwill observation keep the inclusive value (an
    absent goodwill tag means no goodwill on the balance sheet).
    """
    ic = invested_capital(cd)
    gw = _series(cd.inst, "goodwill")
    if ic.empty or gw.dropna().empty:
        return ic
    gw_q = align_instant_to_quarters(gw, ic.index)
    common = ic.index.intersection(gw_q.index)
    out = ic.copy()
    if len(common):
        out.loc[common] = out.loc[common] - gw_q[common]
    return out


# ---------------------------------------------------------------------------
# 3-4. ROIC / ROE series (audits 3.4, 3.5)
# ---------------------------------------------------------------------------

def roe_series(cd) -> pd.Series:
    """Quarterly TTM ROE = TTM net income / 4-quarter average equity
    (avg_instant), winsorized to config.ROIC_WINSOR (audit 3.5's winsor
    band shared with ROIC).

    Observations with average equity <= 0 are nulled (ROE off negative
    equity is sign-flipped garbage; flag 'roe_negative_equity').
    """
    ni = _series(cd.ttm, "net_income").dropna()
    eq = _series(cd.inst, "equity")
    if ni.empty or eq.dropna().empty:
        return pd.Series(dtype=float)
    eq_q = align_instant_to_quarters(eq, ni.index)
    if eq_q.empty:
        return pd.Series(dtype=float)
    avg_eq = avg_instant(eq_q)
    common = ni.index.intersection(avg_eq.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    den = avg_eq[common]
    roe = ni[common] / den.where(den > 0)
    roe = roe.replace([np.inf, -np.inf], np.nan)
    if bool((den <= 0).any()):
        cd.flags.add("roe_negative_equity")
    lo, hi = config.ROIC_WINSOR
    clipped = roe.clip(lower=lo, upper=hi)
    if bool(((clipped != roe) & roe.notna()).any()):
        cd.flags.add("roe_winsorized")
    return clipped.dropna().sort_index()


def roic_series(cd) -> pd.Series:
    """Quarterly TTM ROIC = NOPAT(TTM) / 4-quarter average invested capital
    (spec "ROIC, WACC, and guards"; audit 3.5).

    Guard: at any period where |avg IC| < config.IC_MIN_FRACTION_OF_ASSETS x
    average assets, substitute EBIT (TTM operating income) / average assets
    and flag 'ic_guard' (audit 3.5 -- IC crosses zero on buyback-heavy
    names). Each observation winsorized to config.ROIC_WINSOR (flag
    'roic_winsorized' when it binds).

    Banks/insurers (spec 3.4): the whole ROIC/WACC apparatus is provably
    wrong (deposit interest / deposits outside debt tags), so ROE
    substitutes for both the inflection and moat rows -- returns
    roe_series(cd) with flag 'roe_substitute'. Plain ROE is returned here;
    the Ke subtraction (ROE - Ke hurdle) happens in moat_spread / scoring.
    """
    if _fin_sector(cd) in ("bank", "insurer"):
        cd.flags.add("roe_substitute")
        return roe_series(cd)

    nopat = nopat_ttm(cd)
    ic = invested_capital(cd)
    if nopat.empty or ic.empty:
        return pd.Series(dtype=float)
    avg_ic = avg_instant(ic)
    common = nopat.index.intersection(avg_ic.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    den = avg_ic[common]
    roic = nopat[common] / den.where(den != 0)
    roic = roic.replace([np.inf, -np.inf], np.nan)

    # |IC| guard -> EBIT / avg assets substitution
    avg_a = _avg_assets_on(cd, common)
    oi = _series(cd.ttm, "operating_income")
    if len(avg_a) and len(oi):
        guard_fired = False
        for t in common:
            a = float(avg_a.get(t, np.nan))
            d = float(den.get(t, np.nan))
            if (np.isfinite(a) and a > 0 and np.isfinite(d)
                    and abs(d) < config.IC_MIN_FRACTION_OF_ASSETS * a):
                e = float(oi.get(t, np.nan))
                roic.loc[t] = e / a if np.isfinite(e) else np.nan
                guard_fired = True
        if guard_fired:
            cd.flags.add("ic_guard")

    lo, hi = config.ROIC_WINSOR
    clipped = roic.clip(lower=lo, upper=hi)
    if bool(((clipped != roic) & roic.notna()).any()):
        cd.flags.add("roic_winsorized")
    return clipped.dropna().sort_index()


# ---------------------------------------------------------------------------
# 5-6. Cost of debt / WACC (spec "ROIC, WACC, and guards"; audits 3.5, 3.6)
# ---------------------------------------------------------------------------

def cost_of_debt(cd, rf: float) -> float:
    """PRE-tax cost of debt = latest TTM interest expense / latest total
    debt (audit 3.5). wacc() applies (1 - t) with the same tax-rate logic
    as NOPAT.

    Computed ONLY when total debt > config.KD_MIN_DEBT_FRACTION_OF_ASSETS x
    assets -- otherwise returns NaN, meaning D/V ~ 0 (the caller then uses
    WACC = Ke). Clamped to [rf + config.KD_CLAMP_OVER_RF[0],
    rf + config.KD_CLAMP_OVER_RF[1]]; flag 'kd_clamped' when the clamp
    binds; flag 'kd_missing_interest' when debt is material but interest
    expense is unresolved (falls back to NaN => D/V = 0).
    """
    rf = _fnum(rf)
    if not np.isfinite(rf):
        return np.nan
    debt = _latest_val(_series(cd.inst, "total_debt"))
    assets = _latest_val(_series(cd.inst, "assets"))
    if not np.isfinite(debt) or not np.isfinite(assets) or assets <= 0:
        return np.nan
    if debt <= config.KD_MIN_DEBT_FRACTION_OF_ASSETS * assets:
        return np.nan  # D/V ~ 0
    ie = _latest_val(_series(cd.ttm, "interest_expense"))
    if not np.isfinite(ie):
        cd.flags.add("kd_missing_interest")
        return np.nan
    raw = ie / debt
    lo, hi = rf + config.KD_CLAMP_OVER_RF[0], rf + config.KD_CLAMP_OVER_RF[1]
    kd = min(max(raw, lo), hi)
    if kd != raw:
        cd.flags.add("kd_clamped")
    return float(kd)


def current_wacc(cd, rf: float) -> tuple[float, dict]:
    """Full-CAPM current WACC (spec "ROIC, WACC, and guards"; audit 3.5).

    Ke = rf + beta x config.ERP, beta from cd.market['beta'] clamped to
    config.BETA_CLAMP (flag 'beta_clamped' when binding) else
    config.DEFAULT_BETA (flag 'beta_default'). E = market cap, D = latest
    book total debt; after-tax Kd from cost_of_debt (Kd NaN => D/V = 0,
    i.e. WACC = Ke). Tax rate: same clamp/statutory-fallback logic as NOPAT.

    Banks/insurers (spec 3.4): returns (Ke, details) -- Ke is the
    ROE - Ke hurdle; debt-side inputs are out-of-model.
    Missing market cap: falls back to WACC = Ke (flag 'wacc_no_mktcap').

    Returns (wacc, details) with details keys ke/kd/beta/e/d/tax (kd is
    PRE-tax; d is the debt weight actually used).
    """
    rf = _fnum(rf)
    beta_raw = _fnum(cd.market.get("beta"))
    if np.isfinite(beta_raw):
        lo, hi = config.BETA_CLAMP
        beta = min(max(beta_raw, lo), hi)
        if beta != beta_raw:
            cd.flags.add("beta_clamped")
    else:
        beta = config.DEFAULT_BETA
        cd.flags.add("beta_default")
    ke = rf + beta * config.ERP if np.isfinite(rf) else np.nan
    tax = _current_tax_rate(cd)
    e = _fnum(cd.market.get("mktcap"))
    details = {"ke": ke, "kd": np.nan, "beta": beta, "e": e, "d": np.nan,
               "tax": tax}

    if _fin_sector(cd) in ("bank", "insurer"):
        return ke, details

    kd = cost_of_debt(cd, rf)
    details["kd"] = kd
    if not np.isfinite(e) or e <= 0:
        cd.flags.add("wacc_no_mktcap")
        return ke, details
    d = _latest_val(_series(cd.inst, "total_debt"))
    if not np.isfinite(kd):
        d = 0.0  # D/V ~ 0 (immaterial debt, or Kd unresolvable)
    elif not np.isfinite(d) or d < 0:
        d = 0.0
    details["d"] = d
    v = e + d
    if not np.isfinite(v) or v <= 0:
        return ke, details
    kd_at = kd * (1.0 - tax) if np.isfinite(kd) else 0.0
    wacc = (e / v) * ke + (d / v) * kd_at
    return float(wacc) if np.isfinite(wacc) else np.nan, details


# ---------------------------------------------------------------------------
# 7. Demonstrated-economics moat spread (audits 4.2, 3.6)
# ---------------------------------------------------------------------------

def moat_spread(cd, roic: pd.Series, rf_history: pd.Series | None) -> dict:
    """Demonstrated-economics ROIC - WACC moat construction (audit 4.2) with
    a historical-Rf time dimension (audit 3.6).

    Per-quarter spread_t = roic_t - wacc_t where wacc_t re-computes the full
    CAPM WACC varying ONLY the risk-free rate by the calendar year of each
    quarter (rf_history: annual 10-yr Treasury yields indexed by year);
    beta, ERP, tax rate, and the CURRENT E and D are held constant -- the
    time dimension is deliberately Rf-only per audit 3.6(b). For
    banks/insurers `roic` is the ROE series (roic_series substitutes) and
    wacc_t is Ke_t, so the spread is the ROE - Ke hurdle (spec 3.4).

    rf_history empty/None => constant current rf (cd.market['rf']) and flag
    'moat_constant_rf'; the row is then "5-yr avg ROIC minus current WACC",
    scored by percentile downstream (audit 3.6c). Years missing from
    rf_history fall back to the nearest available year.

    - demonstrated: MAX window-mean spread over all rolling windows of
      config.MOAT_DEMONSTRATED_WINDOW_Q consecutive quarters (every adjacent
      pair <= 130 days apart) within the trailing
      config.MOAT_DEMONSTRATED_LOOKBACK_Q quarters; window start/end dates
      recorded (the audit-4.2 tear-sheet tell for COVID/cyclical-top
      windows).
    - trailing_5y: mean spread over the trailing 4 x config.MOAT_TRAILING_YEARS
      quarters (second column; demonstrated - trailing = fallen-angel
      signature).
    - current_spread_negative: snapshot latest ROIC - current WACC < 0
      boolean tag (audit 3.6a).
    - n_obs: spread observations inside the demonstrated lookback;
      confidence shrink for < config.MIN_OBS_5YR obs is applied downstream.
    """
    out = {"demonstrated": np.nan, "window_start": pd.NaT,
           "window_end": pd.NaT, "trailing_5y": np.nan,
           "fallen_angel_spread": np.nan, "current_spread_negative": False,
           "n_obs": 0,
           "spread_slope": np.nan, "spread_now": np.nan,
           "spread_peak_in_window": np.nan, "buffer_consumed": np.nan,
           "crossing_soon": False}
    roic = roic.dropna() if isinstance(roic, pd.Series) else \
        pd.Series(dtype=float)
    if roic.empty:
        return out

    rf_now = _fnum(cd.market.get("rf"))
    rf_map: dict[int, float] = {}
    if isinstance(rf_history, pd.Series) and len(rf_history):
        for k, v in rf_history.dropna().items():
            try:
                y = int(getattr(k, "year", k))
            except (TypeError, ValueError):
                continue
            fv = _fnum(v)
            if np.isfinite(fv):
                rf_map[y] = fv
    if not rf_map:
        if not np.isfinite(rf_now):
            return out
        cd.flags.add("moat_constant_rf")

    wacc_cache: dict[float, float] = {}

    def _wacc_at(rf_t: float) -> float:
        key = round(rf_t, 8)
        if key not in wacc_cache:
            wacc_cache[key] = current_wacc(cd, rf_t)[0]
        return wacc_cache[key]

    vals: dict[pd.Timestamp, float] = {}
    for t in roic.index:
        if rf_map:
            y = int(t.year)
            rf_t = rf_map.get(y)
            if rf_t is None:
                rf_t = rf_map[min(rf_map, key=lambda yy: abs(yy - y))]
        else:
            rf_t = rf_now
        w = _wacc_at(rf_t)
        r = float(roic.loc[t])
        if np.isfinite(w) and np.isfinite(r):
            vals[t] = r - w
    if not vals:
        return out
    s = pd.Series(vals, dtype=float).sort_index()

    latest = s.index.max()
    cut = latest - pd.Timedelta(
        days=QDAYS * config.MOAT_DEMONSTRATED_LOOKBACK_Q + 45)
    sl = s[s.index >= cut]
    out["n_obs"] = int(len(sl))

    w_q = config.MOAT_DEMONSTRATED_WINDOW_Q
    best, b_start, b_end = np.nan, pd.NaT, pd.NaT
    idx = sl.index
    for i in range(0, len(sl) - w_q + 1):
        widx = idx[i:i + w_q]
        gaps = (widx[1:] - widx[:-1]).days
        if len(gaps) and int(max(gaps)) > MOAT_MAX_QUARTER_GAP_DAYS:
            continue  # not 12 CONSECUTIVE quarters
        m = float(sl.iloc[i:i + w_q].mean())
        if not np.isfinite(best) or m > best:
            best, b_start, b_end = m, widx[0], widx[-1]
    out["demonstrated"] = best
    out["window_start"], out["window_end"] = b_start, b_end

    cut5 = latest - pd.Timedelta(
        days=QDAYS * 4 * config.MOAT_TRAILING_YEARS + 45)
    s5 = s[s.index >= cut5]
    if len(s5):
        out["trailing_5y"] = float(s5.mean())
    if np.isfinite(best) and np.isfinite(out["trailing_5y"]):
        out["fallen_angel_spread"] = float(best - out["trailing_5y"])

    if np.isfinite(rf_now):
        w_now = _wacc_at(rf_now)
        if np.isfinite(w_now):
            out["current_spread_negative"] = bool(
                float(roic.iloc[-1]) - w_now < 0)

    # ROIC-WACC spread-compression trajectory (short false-moat "rolling over"
    # signal; CAKE spread 750->130bps). Recent slope of the same per-quarter
    # spread series `s` (inherits roic winsor + ic_guard), how much of the
    # window-peak spread has been consumed, and whether the spread is at/near
    # crossing WACC. This is a RECENT-rollover read (trailing window), distinct
    # from the long-lookback fallen_angel_spread.
    comp_cfg = config.COMPRESSION
    comp = s.iloc[-comp_cfg["slope_window_q"]:]
    now = float(s.iloc[-1])
    out["spread_now"] = now
    if len(comp) >= comp_cfg["min_obs"]:
        from .inflection import theil_sen_slope
        sl_slope = theil_sen_slope(comp.values)
        if np.isfinite(sl_slope):
            out["spread_slope"] = float(sl_slope)
    peak_w = float(comp.max()) if len(comp) else np.nan
    if np.isfinite(peak_w):
        out["spread_peak_in_window"] = peak_w
        if peak_w > 0:
            out["buffer_consumed"] = float(1.0 - now / peak_w)
    if now < 0:
        out["crossing_soon"] = True
    elif np.isfinite(out["spread_slope"]) and out["spread_slope"] < 0:
        out["crossing_soon"] = bool(
            now / (-out["spread_slope"]) <= comp_cfg["cross_horizon_q"])
    return out


# ---------------------------------------------------------------------------
# 8. GM level + stability on the demonstrated window (spec LONG moat row;
#    audit 4.2)
# ---------------------------------------------------------------------------

def gm_stability(cd, window_dates) -> dict:
    """GM level (mean TTM gross margin) and stability (NEGATIVE stdev of TTM
    GM, so higher = better) measured ON the demonstrated-economics window
    (spec LONG moat table; audit 4.2 -- GM stability on the same window the
    moat level is demonstrated on).

    window_dates: (start, end) from moat_spread. When None/NaT, falls back
    to robust dispersion over the full history EXCLUDING the trailing
    GM_EXCLUDE_TRAILING_Q (=7, spec: 6-8q) quarters: level = median,
    stability = -MAD x 1.4826.
    """
    out = {"gm_level": np.nan, "gm_stability": np.nan}
    gm = _gm_ttm(cd).dropna()
    if gm.empty:
        return out

    ws = we = None
    if (window_dates is not None and hasattr(window_dates, "__len__")
            and len(window_dates) >= 2 and not pd.isna(window_dates[0])
            and not pd.isna(window_dates[1])):
        ws, we = pd.Timestamp(window_dates[0]), pd.Timestamp(window_dates[1])

    if ws is not None:
        pad = pd.Timedelta(days=5)
        win = gm[(gm.index >= ws - pad) & (gm.index <= we + pad)]
        if len(win) >= 2:
            out["gm_level"] = float(win.mean())
        if len(win) >= 3:
            out["gm_stability"] = float(-win.std())
        return out

    cutoff = gm.index.max() - pd.Timedelta(days=QDAYS * GM_EXCLUDE_TRAILING_Q)
    hist = gm[gm.index <= cutoff]
    if len(hist) >= 4:
        out["gm_level"] = float(hist.median())
        med = float(np.median(hist.values))
        mad = float(np.median(np.abs(hist.values - med)))
        out["gm_stability"] = float(-mad * MAD_TO_STD)
    return out


# ---------------------------------------------------------------------------
# 9. Earnings / FCF consistency (spec LONG moat row)
# ---------------------------------------------------------------------------

def earnings_consistency(cd) -> float:
    """Earnings/FCF consistency (spec LONG moat table, "Earnings/FCF
    consistency: stable"): fraction of the last up-to-20 TTM quarters
    (4 x config.MOAT_TRAILING_YEARS) with positive continuing income, plus
    the fraction with positive TTM FCF, averaged over whichever of the two
    is computable. NaN when neither series has >= 8 observations
    (CONSISTENCY_MIN_OBS).
    """
    n_q = 4 * config.MOAT_TRAILING_YEARS
    fracs = []
    for concept in ("continuing_income", "fcf"):
        s = _series(cd.ttm, concept).dropna().iloc[-n_q:]
        if len(s) >= CONSISTENCY_MIN_OBS:
            fracs.append(float((s > 0).mean()))
    if not fracs:
        return np.nan
    return float(np.mean(fracs))


# ---------------------------------------------------------------------------
# 10. TTM Piotroski F-Score (audit 4.8)
# ---------------------------------------------------------------------------

def _sig_pos(s: pd.Series, cutoff: pd.Timestamp) -> bool | None:
    _, v = _last_le(s, cutoff)
    return None if not np.isfinite(v) else bool(v > 0)


def _sig_delta(s: pd.Series, cutoff: pd.Timestamp, up: bool = True
               ) -> bool | None:
    idx, v = _last_le(s, cutoff)
    if idx is None:
        return None
    prior = _near_val(s, idx - pd.Timedelta(days=YEAR_DAYS))
    if not np.isfinite(prior):
        return None
    return bool(v > prior) if up else bool(v < prior)


def _sig_no_issuance(sh: pd.Series, cutoff: pd.Timestamp) -> bool | None:
    idx, v = _last_le(sh, cutoff)
    if idx is None:
        return None
    prior = _near_val(sh, idx - pd.Timedelta(days=YEAR_DAYS))
    if not np.isfinite(prior) or prior <= 0:
        return None
    return bool(v / prior - 1.0 <= config.FSCORE_ISSUANCE_THRESHOLD)


def piotroski_ttm(cd, asof) -> dict:
    """TTM Piotroski F-Score updated quarterly (audit 4.8): flow signals on
    trailing-4-quarter sums, stock signals latest balance vs 4 quarters ago.

    Nine signals: ROA > 0 (TTM NI / avg assets); CFO > 0 (TTM); dROA > 0
    (vs 4q ago); accruals CFO > NI (TTM); leverage down (debt/assets --
    NOTE: cd.inst carries total_debt, not a separate LT-debt instant, so
    total debt/assets proxies the classic LT-debt ratio); current ratio up;
    no issuance (shares outstanding YoY growth <=
    config.FSCORE_ISSUANCE_THRESHOLD -- SBC passes, distressed raises
    fail); GM up; asset turnover up.

    Financials (cd.meta['is_financial']): current-ratio and GM components
    dropped (audit 4.8 / spec 3.4 -- verified absent for banks) and the
    score rescales as F = 9 x mean(available signals).

    F is computed at both t (asof) and t - 4 quarters over the SAME
    available-signal set (a signal counts only if computable at both dates);
    'f_rising' = F(t) - F(t-4q), continuous. Returns
    {'f_score': 0-9 float, 'f_rising': float, 'n_signals': int}.
    """
    asof = pd.Timestamp(asof)

    ni = _series(cd.ttm, "net_income")
    cfo = _series(cd.ttm, "cfo")
    rev = _series(cd.ttm, "revenue")
    roa = ttm_ratio(ni, _avg_assets_on(cd, ni.dropna().index)) \
        if len(ni) else pd.Series(dtype=float)
    at = ttm_ratio(rev, _avg_assets_on(cd, rev.dropna().index)) \
        if len(rev) else pd.Series(dtype=float)
    gm = _gm_ttm(cd)

    common = cfo.dropna().index.intersection(ni.dropna().index)
    accr = (cfo[common] - ni[common]) if len(common) else \
        pd.Series(dtype=float)

    debt = _series(cd.inst, "total_debt")
    assets = _series(cd.inst, "assets")
    common = debt.dropna().index.intersection(assets.dropna().index)
    lev = (debt[common] / assets[common].where(assets[common] > 0)).dropna() \
        if len(common) else pd.Series(dtype=float)

    ac = _series(cd.inst, "assets_current")
    lc = _series(cd.inst, "liabilities_current")
    common = ac.dropna().index.intersection(lc.dropna().index)
    cr = (ac[common] / lc[common].where(lc[common] > 0)).dropna() \
        if len(common) else pd.Series(dtype=float)

    sh = _series(cd.inst, "shares_outstanding")

    sigs = {
        "roa_pos": lambda c: _sig_pos(roa, c),
        "cfo_pos": lambda c: _sig_pos(cfo, c),
        "droa_up": lambda c: _sig_delta(roa, c, up=True),
        "accruals": lambda c: _sig_pos(accr, c),
        "lev_down": lambda c: _sig_delta(lev, c, up=False),
        "cr_up": lambda c: _sig_delta(cr, c, up=True),
        "no_issuance": lambda c: _sig_no_issuance(sh, c),
        "gm_up": lambda c: _sig_delta(gm, c, up=True),
        "at_up": lambda c: _sig_delta(at, c, up=True),
    }
    # Banks/insurers: force-drop the two structurally-inapplicable signals
    # (spec 3.4). Other financials (REITs, financial_other) keep whatever
    # their filings can compute — the availability rescale below (audit 4.8:
    # "drop signals whose inputs don't exist, rescale over available")
    # already excludes missing inputs without a blanket drop.
    if cd.meta.get("fin_sector") in ("bank", "insurer"):
        sigs.pop("cr_up", None)
        sigs.pop("gm_up", None)

    cut_prev = asof - pd.Timedelta(days=YEAR_DAYS)
    now = {k: f(asof) for k, f in sigs.items()}
    prev = {k: f(cut_prev) for k, f in sigs.items()}
    avail = [k for k in sigs if now[k] is not None and prev[k] is not None]
    n = len(avail)
    if n == 0:
        return {"f_score": np.nan, "f_rising": np.nan, "n_signals": 0}
    f_now = FSCORE_N_SIGNALS * float(
        np.mean([1.0 if now[k] else 0.0 for k in avail]))
    f_prev = FSCORE_N_SIGNALS * float(
        np.mean([1.0 if prev[k] else 0.0 for k in avail]))
    return {"f_score": f_now, "f_rising": f_now - f_prev, "n_signals": n}


# ---------------------------------------------------------------------------
# 11. Survivability strip (audit 4.5; ZERO score weight, tear-sheet only)
# ---------------------------------------------------------------------------

def survivability(cd) -> dict:
    """Survivability context strip (audit 4.5; spec LONG context-tags row).
    ZERO score weight -- a tear-sheet traffic light.

    - runway_q: (cash + ST investments) / |TTM FCF| per quarter, only when
      TTM FCF < 0; NaN with funded=True when FCF >= 0; suppressed (NaN,
      light 'n/a') for financials.
    - near_term_debt: latest cd.inst['debt_next_12m'] (the tagged
      LongTermDebtMaturities...NextTwelveMonths ladder), else 'n/a'.
    - debt_gt_cash: near-term debt > (cash + ST investments); None when
      either side is unresolved.
    - dilution_2y: 2-year growth in diluted WAS (TTM average vs the TTM
      average 8 quarters earlier -- ratio of 4-quarter sums equals ratio of
      averages); flag 'dilution_flag' when > config.DILUTION_FLAG_2Y.
    - traffic_light: 'red' = runway < config.RUNWAY_WARN_QUARTERS AND
      near-term debt > cash; 'yellow' = burning with runway <
      RUNWAY_GREEN_MULT x warn or debt > cash; 'green' = FCF-positive or
      comfortably funded; 'n/a' = financials or unresolvable ("distressed
      but funded" reads green/yellow -- the narrative amplifier).
    """
    is_fin = _is_financial(cd)

    cash_v = _latest_val(_series(cd.inst, "cash"))
    sti_v = _latest_val(_series(cd.inst, "st_investments"))
    liquidity = np.nan
    if np.isfinite(cash_v):
        liquidity = cash_v + (sti_v if np.isfinite(sti_v) else 0.0)

    fcf_v = _latest_val(_series(cd.ttm, "fcf"))
    runway_q = np.nan
    funded: bool | None = None
    if is_fin:
        funded = None
    elif not np.isfinite(fcf_v):
        funded = None
    elif fcf_v >= 0:
        funded = True
    else:
        funded = False
        burn_per_q = abs(fcf_v) / 4.0
        if np.isfinite(liquidity) and burn_per_q > 0:
            runway_q = liquidity / burn_per_q

    ntd_v = _latest_val(_series(cd.inst, "debt_next_12m"))
    # NaN, not the string "n/a": this is a numeric column, and mixing a str
    # sentinel into it made `raw_rows.parquet` unwritable ("Conversion failed
    # for column surv_near_term_debt with type object"). The write failure was
    # only logged, so the tuning-loop cache silently kept stale rows — a full
    # run would complete, print its summary, and leave rescore.py reading the
    # PREVIOUS run's data. Tear sheets format NaN as "n/a" at display time.
    near_term_debt = float(ntd_v) if np.isfinite(ntd_v) else np.nan

    debt_gt_cash: bool | None = None
    if np.isfinite(ntd_v) and np.isfinite(liquidity):
        debt_gt_cash = bool(ntd_v > liquidity)

    dil = np.nan
    # diluted WAS is a LEVEL series resolved by fundamentals (never TTM-
    # summed — averages aren't additive flows)
    dw = _series(cd.inst, "diluted_was_q").dropna()
    if len(dw):
        t_idx, v = dw.index[-1], float(dw.iloc[-1])
        prior = _near_val(dw, t_idx - pd.Timedelta(days=2 * YEAR_DAYS),
                          tol_days=200)
        if np.isfinite(prior) and prior > 0:
            dil = v / prior - 1.0
    dilution_flag = bool(np.isfinite(dil) and dil > config.DILUTION_FLAG_2Y)
    if dilution_flag:
        cd.flags.add("dilution_flag")

    if is_fin or funded is None:
        light = "n/a"
    elif funded:
        light = "green"
    elif not np.isfinite(runway_q):
        light = "n/a"
    elif runway_q < config.RUNWAY_WARN_QUARTERS and debt_gt_cash is True:
        light = "red"
    elif (runway_q < config.RUNWAY_WARN_QUARTERS * RUNWAY_GREEN_MULT
          or debt_gt_cash is True):
        light = "yellow"
    else:
        light = "green"

    return {"runway_q": runway_q, "funded": funded,
            "near_term_debt": near_term_debt, "debt_gt_cash": debt_gt_cash,
            "dilution_2y": dil, "dilution_flag": dilution_flag,
            "traffic_light": light}


# ---------------------------------------------------------------------------
# 12. Altman Z + REIT substitutes (spec LONG context-tags row; audit 3.4)
# ---------------------------------------------------------------------------

def altman_z(cd) -> float:
    """Standard Altman Z (1968) using latest values (spec LONG context row):
    1.2 x WC/TA + 1.4 x RE/TA + 3.3 x EBIT/TA + 0.6 x MVE/TL + 1.0 x
    Sales/TA. MVE from cd.market['mktcap']; TL derived as assets - equity
    (no total-liabilities concept in cd.inst).

    NaN for financials/REITs -- out-of-model (audit 3.4); the REIT context
    tag downstream uses interest_coverage and debt_to_ebitda instead. Z is
    an absolute-scale tag, so any missing component yields NaN (coefficients
    are not renormalizable without breaking the scale).
    """
    if _is_financial(cd) or _fin_sector(cd) == "reit":
        return np.nan
    ta = _latest_val(_series(cd.inst, "assets"))
    ac = _latest_val(_series(cd.inst, "assets_current"))
    lc = _latest_val(_series(cd.inst, "liabilities_current"))
    re_ = _latest_val(_series(cd.inst, "retained_earnings"))
    eq = _latest_val(_series(cd.inst, "equity"))
    ebit = _latest_val(_series(cd.ttm, "operating_income"))
    sales = _latest_val(_series(cd.ttm, "revenue"))
    mve = _fnum(cd.market.get("mktcap"))
    if not np.isfinite(ta) or ta <= 0 or not np.isfinite(eq):
        return np.nan
    tl = ta - eq
    comps = (ac - lc if np.isfinite(ac) and np.isfinite(lc) else np.nan,
             re_, ebit, mve, sales)
    if not all(np.isfinite(c) for c in comps) or tl <= 0:
        return np.nan
    wc, re_, ebit, mve, sales = comps
    c = ALTMAN_COEFS
    z = (c[0] * wc / ta + c[1] * re_ / ta + c[2] * ebit / ta
         + c[3] * mve / tl + c[4] * sales / ta)
    return float(z) if np.isfinite(z) else np.nan


def interest_coverage(cd) -> float:
    """Interest coverage = latest TTM operating income / latest TTM interest
    expense (spec 3.4: replaces the Altman-Z tag for REITs, alongside
    debt_to_ebitda). NaN when interest expense is unresolved or <= 0
    (coverage undefined off net interest income)."""
    oi = _series(cd.ttm, "operating_income").dropna()
    ie = _series(cd.ttm, "interest_expense").dropna()
    common = oi.index.intersection(ie.index)
    if len(common) == 0:
        return np.nan
    t = common.max()
    ie_v = float(ie[t])
    if not np.isfinite(ie_v) or ie_v <= 0:
        return np.nan
    return float(oi[t] / ie_v)


def debt_to_ebitda(cd) -> float:
    """Debt/EBITDA = latest total debt / latest TTM (operating income +
    dep & amort) (spec 3.4: REIT context tag with interest_coverage).
    An entirely absent debt series counts as zero debt (ratio 0.0);
    EBITDA <= 0 yields NaN (ratio meaningless)."""
    oi = _series(cd.ttm, "operating_income").dropna()
    da = _series(cd.ttm, "dep_amort").dropna()
    common = oi.index.intersection(da.index)
    if len(common) == 0:
        return np.nan
    t = common.max()
    ebitda = float(oi[t] + da[t])
    if not np.isfinite(ebitda) or ebitda <= 0:
        return np.nan
    debt_s = _series(cd.inst, "total_debt")
    debt = _latest_val(debt_s)
    if not np.isfinite(debt):
        if debt_s.dropna().empty:
            return 0.0  # no debt tags => no debt
        return np.nan
    return float(debt / ebitda)


# ---------------------------------------------------------------------------
# 13. Bank efficiency ratio (spec 3.4 / audit 3.4)
# ---------------------------------------------------------------------------

def efficiency_ratio(cd) -> pd.Series:
    """Bank efficiency ratio, TTM: NoninterestExpense /
    (InterestIncomeExpenseNet + NoninterestIncome) (spec 3.4 -- replaces
    op-margin-vs-peak for banks; all three tags verified live in the
    audit). LOWER is better -- the raw ratio is returned, direction is
    handled downstream.

    All three TTM inputs are required (a missing fee-income leg would
    systematically overstate the ratio); returns empty when any is
    unresolved (exclude-and-renormalize downstream). Denominator <= 0
    observations are nulled.
    """
    num = _series(cd.ttm, "noninterest_expense").dropna()
    nii = _series(cd.ttm, "net_interest_income").dropna()
    fee = _series(cd.ttm, "noninterest_income").dropna()
    if num.empty or nii.empty or fee.empty:
        return pd.Series(dtype=float)
    common = num.index.intersection(nii.index).intersection(fee.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    den = (nii[common] + fee[common]).astype(float)
    ratio = num[common] / den.where(den > 0)
    return ratio.replace([np.inf, -np.inf], np.nan).dropna().sort_index()


# ---------------------------------------------------------------------------
# 14. REIT FFO proxy (spec 3.4 / audit 3.4)
# ---------------------------------------------------------------------------

def reit_ffo_proxy(cd) -> pd.Series:
    """Quarterly FFO proxy = net income + dep & amort - gains on sale
    (spec 3.4: REIT earnings-inflection substitute; FundsFromOperations is
    absent from companyfacts).

    Gains come from cd.q['gains_on_sale'] IF the fundamentals layer
    resolved that ladder; the concept is not in fundamentals.FLOW_CONCEPTS
    today, so the series is usually absent -- gains are then omitted and
    flag 'ffo_no_gains' is added (the spec's "else omit gains" fallback).
    Quarters where the gains series exists but has no observation count as
    zero (gain tags are only filed when a sale occurred).
    """
    ni = _qvals(cd, "net_income").dropna()
    da = _qvals(cd, "dep_amort").dropna()
    if ni.empty or da.empty:
        return pd.Series(dtype=float)
    common = ni.index.intersection(da.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    ffo = (ni[common] + da[common]).astype(float)
    gains = _qvals(cd, "gains_on_sale").dropna()
    if len(gains):
        ffo = ffo - gains.reindex(ffo.index).fillna(0.0)
    else:
        cd.flags.add("ffo_no_gains")
    return ffo.dropna().sort_index()


# ---------------------------------------------------------------------------
# 15. Valuation snapshots (spec LONG valuation row / audit 4.6; SHORT
#     EV/Sales richness row)
# ---------------------------------------------------------------------------

def _enterprise_value(cd) -> float:
    """EV = cd.market['enterprise_value'], else the spec fallback chain
    mktcap + latest total debt - latest cash (audit 5.4)."""
    ev = _fnum(cd.market.get("enterprise_value"))
    if np.isfinite(ev):
        return ev
    mkt = _fnum(cd.market.get("mktcap"))
    if not np.isfinite(mkt):
        return np.nan
    debt = _latest_val(_series(cd.inst, "total_debt"))
    cash = _latest_val(_series(cd.inst, "cash"))
    return float(mkt + (debt if np.isfinite(debt) else 0.0)
                 - (cash if np.isfinite(cash) else 0.0))


def ev_sales(cd) -> float:
    """EV / TTM sales snapshot (spec: LONG soft-cheapness row, audit 4.6;
    SHORT EV/Sales-richness row).

    NaN when TTM revenue < config.ZERO_REV_TTM (zero-revenue nulling also
    happens downstream, audit 5.5 -- but don't emit garbage here) and for
    financials (downstream uses p_tbv instead).
    """
    if _is_financial(cd):
        return np.nan
    rev = _latest_val(_series(cd.ttm, "revenue"))
    if not np.isfinite(rev) or rev < config.ZERO_REV_TTM:
        return np.nan
    ev = _enterprise_value(cd)
    if not np.isfinite(ev):
        return np.nan
    return float(ev / rev)


def _normalized_op_margin(cd) -> tuple[float, float, int]:
    """(median, peak, n_obs) of own-history TTM operating margin over the
    trailing LEVEL_HISTORY_CAP_Q window. Shared by ev_normalized (median ->
    normalized EBIT) and reverse_dcf (peak -> basis-consistent margin-ceiling
    comparison). Sets the same 'ev_norm_short_history'/'ev_norm_margin_too_low'
    flags on cd the old inline ev_normalized block did. Returns nan median when
    history is too short or the median margin is below NORMALIZED_MARGIN_MIN
    (peak is still returned when computable). The caller owns the zero-revenue
    and EV guards."""
    rev_ttm = _series(cd.ttm, "revenue")
    oi_ttm = _series(cd.ttm, "operating_income")
    common = oi_ttm.index.intersection(rev_ttm.index)
    if len(common) < config.NORMALIZED_MIN_OBS:
        if len(common):
            cd.flags.add("ev_norm_short_history")
        return np.nan, np.nan, len(common)
    om = (oi_ttm[common] / rev_ttm[common].where(rev_ttm[common] > 0)).dropna()
    om = om.iloc[-config.LEVEL_HISTORY_CAP_Q:]
    if len(om) < config.NORMALIZED_MIN_OBS:
        return np.nan, np.nan, len(om)
    peak_margin = float(np.max(om.values))
    norm_margin = float(np.median(om.values))
    if norm_margin < config.NORMALIZED_MARGIN_MIN:
        cd.flags.add("ev_norm_margin_too_low")
        return np.nan, peak_margin, len(om)
    return norm_margin, peak_margin, len(om)


def ev_normalized(cd) -> float:
    """EV / normalized EBIT, where normalized EBIT = (own full-history
    MEDIAN TTM operating margin) x current TTM revenue (user direction
    2026-07-08: valuation vs MEAN-REVERTED economics, both sides).

    - Peak-margin names (VITL/CAKE archetype) look cheap on current earnings
      but rich here: their median margin is far below today's.
    - Trough names look expensive on current earnings but cheap here.
    NaN for financials, zero-revenue names, medians below
    config.NORMALIZED_MARGIN_MIN (a median-loss business has no meaningful
    normalized earnings), or fewer than config.NORMALIZED_MIN_OBS margin
    observations ('ev_norm_short_history' flag)."""
    if _is_financial(cd):
        return np.nan
    rev = _latest_val(_series(cd.ttm, "revenue"))
    if not np.isfinite(rev) or rev < config.ZERO_REV_TTM:
        return np.nan
    norm_margin, _peak, _n = _normalized_op_margin(cd)
    if not np.isfinite(norm_margin):
        return np.nan
    ev = _enterprise_value(cd)
    if not np.isfinite(ev):
        return np.nan
    return float(ev / (norm_margin * rev))


def trailing_revenue_cagr(cd, years: int) -> tuple[float, int]:
    """Own trailing TTM-revenue CAGR over `years` (needs 4*years+1 quarters),
    basis-consistent with reverse_dcf's Rev0. NaN cagr when history too short
    or an endpoint <= 0."""
    rev = _series(cd.ttm, "revenue").dropna()
    n = len(rev)
    need = 4 * years + 1
    if n < need:
        return np.nan, n
    r_end = float(rev.iloc[-1])
    r_beg = float(rev.iloc[-need])
    if not (np.isfinite(r_end) and np.isfinite(r_beg)) or r_beg <= 0 or r_end <= 0:
        return np.nan, n
    return float((r_end / r_beg) ** (1.0 / years) - 1.0), n


def reverse_dcf(cd, rf: float, wacc: float | None = None) -> dict:
    """Valuation-implied growth/margin vs the company's own trajectory
    (the reverse-DCF 'what's priced in' primitive; CAKE move 6 / VITL move 5).

    Two-stage DCF inverted on EV to back out the stage-1 revenue CAGR the
    current EV requires (holding the normalized op margin flat, FCF ~ NOPAT),
    then growth_gap = implied_growth - own trailing CAGR. Also solves the
    steady-state op margin the price requires (implied_margin_ceiling) and
    compares it to the company's own PEAK op margin (margin_ceiling_gap > 0 =>
    the market requires a margin never achieved in its own history). Gordon
    single-stage implied-g is a display-only cross-check.

    Returns a dict (all finite-or-nan, never raises); financials, zero-rev,
    EV<=0, non-finite WACC, short history, or wacc<=g degeneracy each null the
    relevant fields. Not a fair value: margin is held flat and reinvestment is
    ignored, so implied_growth is a valuation-implied revenue growth only."""
    nan = np.nan
    out = {"implied_growth": nan, "implied_g_gordon": nan, "growth_gap": nan,
           "trailing_cagr": nan, "implied_margin_ceiling": nan,
           "peak_op_margin": nan, "margin_ceiling_gap": nan, "wacc": nan,
           "ev": nan, "rev0": nan, "norm_margin": nan, "flags": []}
    if _is_financial(cd):
        return out
    rev0 = _latest_val(_series(cd.ttm, "revenue"))
    if not np.isfinite(rev0) or rev0 < config.ZERO_REV_TTM:
        return out
    out["rev0"] = float(rev0)
    ev = _enterprise_value(cd)
    if not np.isfinite(ev) or ev <= 0:
        out["flags"].append("rdcf_ev_nonpos")
        return out
    out["ev"] = float(ev)
    if wacc is None:
        wacc = current_wacc(cd, rf)[0]
    if not np.isfinite(wacc):
        return out
    out["wacc"] = float(wacc)
    norm_margin, peak_margin, _ = _normalized_op_margin(cd)
    if np.isfinite(peak_margin):
        out["peak_op_margin"] = float(peak_margin)
    if not np.isfinite(norm_margin):
        return out
    out["norm_margin"] = float(norm_margin)
    tax = _current_tax_rate(cd)
    nopat_margin = norm_margin * (1.0 - tax)
    fcf0 = nopat_margin * rev0
    cfg = config.REVERSE_DCF
    g_term = cfg["terminal_growth"]

    trailing = trailing_revenue_cagr(cd, cfg["trailing_years"])[0]
    if np.isfinite(trailing):
        out["trailing_cagr"] = float(trailing)

    # implied steady-state op margin the price requires (EV = FCF0*(1+g)/(wacc-g)
    # solved for the margin), converted NOPAT->EBIT so it is comparable to the
    # own PEAK operating margin.
    if wacc > g_term and (1.0 - tax) > 0:
        m_req_nopat = ev * (wacc - g_term) / (rev0 * (1.0 + g_term))
        ceiling = m_req_nopat / (1.0 - tax)
        if np.isfinite(ceiling):
            out["implied_margin_ceiling"] = float(ceiling)
            if np.isfinite(peak_margin):
                out["margin_ceiling_gap"] = float(ceiling - peak_margin)

    # 2-stage implied growth via bisection (pv is monotone-increasing in g)
    if wacc <= g_term + cfg["min_wacc_spread"]:
        out["flags"].append("rdcf_wacc_le_g")
    else:
        N = int(cfg["explicit_years"])

        def pv(g: float) -> float:
            s = 0.0
            for t in range(1, N + 1):
                s += fcf0 * (1.0 + g) ** t / (1.0 + wacc) ** t
            tv = fcf0 * (1.0 + g) ** N * (1.0 + g_term) / (wacc - g_term)
            return s + tv / (1.0 + wacc) ** N

        lo, hi = cfg["g_search"]
        fa = pv(lo) - ev
        fb = pv(hi) - ev
        if np.sign(fa) == np.sign(fb):
            out["flags"].append("rdcf_growth_clipped")
            implied = lo if abs(fa) < abs(fb) else hi
        else:
            a, b = lo, hi
            for _ in range(40):
                mid = 0.5 * (a + b)
                fm = pv(mid) - ev
                if fm == 0:
                    a = b = mid
                    break
                if np.sign(fm) == np.sign(fa):
                    a, fa = mid, fm
                else:
                    b = mid
            implied = 0.5 * (a + b)
        out["implied_growth"] = float(implied)
        if np.isfinite(trailing):
            out["growth_gap"] = float(implied - trailing)

    # Gordon single-stage implied-g (display-only cross-check)
    den = ev + fcf0
    if den != 0 and np.isfinite(den):
        out["implied_g_gordon"] = float((wacc * ev - fcf0) / den)
    return out


def ev_ebitda(cd) -> float:
    """EV / TTM EBITDA (operating income + D&A). NaN for financials, zero-rev,
    or EBITDA <= 0 (loss-makers: multiple undefined). Confirmer for the
    peer-multiple-disconnect tag, which leads on EV/Sales."""
    if _is_financial(cd):
        return np.nan
    rev = _latest_val(_series(cd.ttm, "revenue"))
    if not np.isfinite(rev) or rev < config.ZERO_REV_TTM:
        return np.nan
    oi = _series(cd.ttm, "operating_income").dropna()
    da = _series(cd.ttm, "dep_amort").dropna()
    common = oi.index.intersection(da.index)
    if len(common) == 0:
        return np.nan
    t = common.max()
    ebitda = float(oi[t] + da[t])
    if not np.isfinite(ebitda) or ebitda <= 0:
        return np.nan
    ev = _enterprise_value(cd)
    if not np.isfinite(ev):
        return np.nan
    return float(ev / ebitda)


def p_tbv(cd) -> float:
    """Price / tangible book value = mktcap / (equity - goodwill) -- the
    financials substitute for the valuation row (spec 3.4 / audit 4.6:
    "P/TBV for financials"). A wholly absent goodwill series counts as
    zero goodwill. NaN when tangible book <= 0 (multiple undefined)."""
    mkt = _fnum(cd.market.get("mktcap"))
    eq = _latest_val(_series(cd.inst, "equity"))
    if not np.isfinite(mkt) or not np.isfinite(eq):
        return np.nan
    gw = _latest_val(_series(cd.inst, "goodwill"))
    tbv = eq - (gw if np.isfinite(gw) else 0.0)
    if tbv <= 0:
        return np.nan
    return float(mkt / tbv)


# ---------------------------------------------------------------------------
# 16. Insurer loss ratio (spec 3.4 / audit 3.4)
# ---------------------------------------------------------------------------

def insurer_loss_ratio(cd) -> pd.Series:
    """Insurer loss-ratio proxy, TTM: PolicyholderBenefitsAndClaimsIncurredNet
    / PremiumsEarnedNet where tagged (spec 3.4: "loss-ratio proxies where
    tagged, else reduced applicable set"). Empty when either leg is
    unresolved; premiums <= 0 observations are nulled."""
    pb = _series(cd.ttm, "policyholder_benefits").dropna()
    pe = _series(cd.ttm, "premiums_earned").dropna()
    if pb.empty or pe.empty:
        return pd.Series(dtype=float)
    common = pb.index.intersection(pe.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    den = pe[common].astype(float)
    ratio = pb[common] / den.where(den > 0)
    return ratio.replace([np.inf, -np.inf], np.nan).dropna().sort_index()


__all__ = [
    "nopat_ttm",
    "invested_capital",
    "invested_capital_ex_goodwill",
    "roic_series",
    "roe_series",
    "cost_of_debt",
    "current_wacc",
    "moat_spread",
    "gm_stability",
    "earnings_consistency",
    "piotroski_ttm",
    "survivability",
    "altman_z",
    "interest_coverage",
    "debt_to_ebitda",
    "efficiency_ratio",
    "reit_ffo_proxy",
    "ev_sales",
    "ev_normalized",
    "ev_ebitda",
    "trailing_revenue_cagr",
    "reverse_dcf",
    "p_tbv",
    "insurer_loss_ratio",
]
