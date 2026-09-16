"""The inflection engine (spec core; audits 2.1-2.7).

Turnaround signal = second derivative, not level. Per metric:
- level  : percentile of the latest value within the company's own full TTM
           history (capped ~40q), shrunk toward neutral 50 by n/(n+k), k=8
- slope  : Theil-Sen over the YoY-delta series (5-6q window primary), blended
           with a sign-consistency count; unitless via own-history MAD of
           quarterly changes (floored), clipped +/-3
- trough : argmin of TTM over trailing 8-12q; contributes only if >=2q before
           the latest observation AND cumulative recovery >= ~1 MAD; smooth
           recency ramp max(0, 1 - q_since/8) scaled by recovery capped 3 MADs
- combination within a metric: conjunctive geometric mean, per-component
  floors ~0.12 (soft, not a gate); across metrics: additive (weighted)
- the same construction sign-inverted = the short side's PEAK-RECENCY
  primitive (direction=-1)

As-of policy (audit 1.6): windows are calendar quarters back from the run
date; missing recent quarters count as elapsed time; soft staleness decay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import config

QDAYS = 91.25


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def own_history_percentile(series: pd.Series, cap: int = None) -> tuple[float, int]:
    """Percentile (0-100) of the latest value in the company's own history,
    shrunk toward 50 by n/(n+k). Returns (percentile, n_obs)."""
    cap = cap or config.LEVEL_HISTORY_CAP_Q
    s = series.dropna()
    if len(s) < 2:
        return np.nan, len(s)
    s = s.iloc[-cap:]
    n = len(s)
    pct = 100.0 * stats.percentileofscore(s.values, s.values[-1],
                                          kind="mean") / 100.0
    shrunk = 50.0 + (pct - 50.0) * (n / (n + config.LEVEL_SHRINK_K))
    return float(shrunk), n


def theil_sen_slope(y: np.ndarray) -> float:
    """Theil-Sen slope over an evenly-indexed window (per-quarter units)."""
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)[mask]
    y = y[mask]
    slopes = [(y[j] - y[i]) / (x[j] - x[i])
              for i in range(len(y)) for j in range(i + 1, len(y))]
    return float(np.median(slopes))


def slope_score(slope_series: pd.Series, base_series: pd.Series,
                asof: pd.Timestamp, direction: int = 1,
                window_q: int = None, anchor: pd.Timestamp | None = None,
                cross_mad: float | None = None) -> tuple[float, dict]:
    """Unitless slope component in [0,1]. direction=+1 rewards rising,
    -1 rewards falling (short side). base_series supplies the MAD of the
    company's own historical quarterly changes (unit normalization).
    anchor: fit only points at/after this date (trough/peak anchoring)."""
    window_q = window_q or config.SLOPE_WINDOW_Q
    s = slope_series.dropna()
    # calendar as-of: only observations within the lookback window
    cutoff = asof - pd.Timedelta(days=QDAYS * (window_q + 0.7))
    s = s[s.index >= cutoff]
    if anchor is not None:
        anchored = s[s.index >= anchor]
        if len(anchored) >= config.SLOPE_ANCHOR_MIN_POST_TROUGH:
            s = anchored
    if len(s) < 3:
        return np.nan, {}
    win = s.iloc[-window_q:]
    ts = theil_sen_slope(win.values)
    if not np.isfinite(ts):
        return np.nan, {}
    changes = base_series.dropna().diff().dropna().values
    mad = _mad(changes)
    if not np.isfinite(mad) or mad <= 0:
        scale = np.nanstd(win.values)
        mad = scale if scale > 0 else 1.0
    if cross_mad is not None and np.isfinite(cross_mad) and cross_mad > 0:
        mad = 0.7 * mad + 0.3 * cross_mad  # floor/blend w/ cross-sectional
    z = np.clip(ts / mad, -config.SLOPE_MAD_CLIP, config.SLOPE_MAD_CLIP)
    z *= direction
    mapped = (z + config.SLOPE_MAD_CLIP) / (2 * config.SLOPE_MAD_CLIP)
    sign_frac = float(np.mean((win.values * direction) > 0))
    score = 0.6 * mapped + 0.4 * sign_frac
    return float(np.clip(score, 0, 1)), {"theil_sen": ts, "z": float(z),
                                         "sign_frac": sign_frac}


def _robust_scale(resid: np.ndarray) -> float:
    mad = _mad(resid)
    if np.isfinite(mad) and mad > 0:
        return mad
    finite = resid[np.isfinite(resid)]
    if len(finite) < 6:
        return np.nan
    lo, hi = np.percentile(finite, [10, 90])
    core = finite[(finite >= lo) & (finite <= hi)]
    alt = float(np.std(core)) if len(core) else 0.0
    if alt <= 0:
        alt = max(1e-9, float(np.median(np.abs(finite))) * 0.05)
    return alt


def one_off_extremum(raw_q: pd.Series, ttm_extremum_date: pd.Timestamp,
                     direction: int = 1) -> bool:
    """One-off detection on the QUARTERLY series (audit 2.6d).

    TTM smears a single-quarter shock across 4 windows, so the TTM series
    cannot distinguish a one-quarter impairment from a genuine cycle. Test:
    the quarterly extremum feeding the TTM extremum must be (a) k MADs from
    the rolling median AND (b) a feature ONE quarter wide at half-deviation
    (a real cyclical hump/trough is several quarters wide)."""
    q = raw_q.dropna()
    if len(q) < 8 or ttm_extremum_date is None:
        return False
    span = q[(q.index > ttm_extremum_date - pd.Timedelta(days=QDAYS * 3.6))
             & (q.index <= ttm_extremum_date + pd.Timedelta(days=10))]
    if not len(span):
        return False
    ext_t = (span * direction).idxmin()
    med = q.rolling(config.OUTLIER_ROLLING_Q, center=True,
                    min_periods=4).median()
    resid = q - med
    scale = _robust_scale(resid.values)
    if not np.isfinite(scale) or scale <= 0:
        return False
    dev = resid[ext_t]
    if abs(dev) < config.OUTLIER_K_MADS * scale * 1.4826:
        return False
    half = 0.5 * abs(dev)
    same_side = (resid * np.sign(dev)) >= half
    pos = q.index.get_loc(ext_t)
    run = 1
    i = pos - 1
    while i >= 0 and bool(same_side.iloc[i]):
        run += 1
        i -= 1
    i = pos + 1
    while i < len(q) and bool(same_side.iloc[i]):
        run += 1
        i += 1
    return run <= 1


def trough_score(ttm: pd.Series, asof: pd.Timestamp, direction: int = 1,
                 ref_window_q: int = None,
                 raw_q: pd.Series | None = None) -> tuple[float, dict]:
    """Trough-with-bounce (audit 2.3). direction=+1: trough (long);
    direction=-1: PEAK-recency primitive (short) via sign inversion.
    raw_q: underlying quarterly series for one-off outlier damping."""
    ref_window_q = ref_window_q or config.TROUGH_REF_WINDOW_Q
    s = (ttm * direction).dropna()
    if len(s) < 4:
        return np.nan, {}
    cutoff = asof - pd.Timedelta(days=QDAYS * ref_window_q)
    win = s[s.index >= cutoff]
    if len(win) < 4:
        return np.nan, {}
    t_min = win.idxmin()
    latest_t = s.index.max()
    changes = s.diff().dropna().values
    mad = _mad(changes)
    details = {"extremum_date": t_min}
    if not np.isfinite(mad) or mad <= 0:
        # MAD degenerates to 0 when most quarterly changes are exactly zero
        # (long plateaus); fall back to the std of changes
        alt = float(np.nanstd(changes)) if len(changes) else np.nan
        if not np.isfinite(alt) or alt <= 0:
            # essentially-constant series: no trough/peak exists -> 0, not
            # missing (a NaN would let level+slope carry the metric, audit 2.4)
            return 0.0, details
        mad = alt
    # must be >= 2 quarters before the latest OBSERVATION (0.15q slack for
    # calendar-day arithmetic: 182 actual days / 91.25 = 1.995 "quarters")
    q_before_latest = (latest_t - t_min).days / QDAYS
    recovery = (s[latest_t] - s[t_min]) / mad
    details.update({"recovery_mads": float(recovery),
                    "q_before_latest": float(q_before_latest)})
    if q_before_latest < config.TROUGH_MIN_Q_BEFORE_LATEST - 0.15:
        return 0.0, details  # still making lows: scores 0, stays eligible
    if recovery < config.TROUGH_MIN_RECOVERY_MADS:
        return 0.0, details
    # recency in CALENDAR quarters from run date (as-of policy)
    q_since = (asof - t_min).days / QDAYS
    ramp = max(0.0, 1.0 - q_since / config.TROUGH_RECENCY_RAMP_Q)
    magnitude = min(recovery, config.TROUGH_RECOVERY_CAP_MADS) \
        / config.TROUGH_RECOVERY_CAP_MADS
    damp = 1.0
    if raw_q is not None and one_off_extremum(raw_q, t_min, direction):
        damp = config.OUTLIER_TROUGH_DAMP
        details["outlier_trough"] = True
    return float(np.clip(ramp * magnitude * damp, 0, 1)), details


def conjunctive(components: dict[str, float]) -> float:
    """Geometric mean with per-component soft floors (audit 2.4)."""
    vals = [v for v in components.values() if np.isfinite(v)]
    if not vals:
        return np.nan
    floored = [max(v, config.GEOMEAN_COMPONENT_FLOOR) for v in vals]
    return float(np.exp(np.mean(np.log(floored))))


@dataclass
class MetricInflection:
    metric: str
    level: float = np.nan       # component in [0,1] (direction applied)
    slope: float = np.nan
    trough: float = np.nan
    score: float = np.nan       # conjunctive combination
    level_pctile: float = np.nan  # raw own-history percentile of the value
    n_obs: int = 0
    confidence: float = 1.0     # min-obs soft multiplier
    details: dict = field(default_factory=dict)
    flags: set = field(default_factory=set)


def metric_inflection(metric: str, ttm: pd.Series, slope_series: pd.Series,
                      asof: pd.Timestamp, direction: int = 1,
                      slope_window_q: int | None = None,
                      trough_window_q: int | None = None,
                      raw_q: pd.Series | None = None) -> MetricInflection:
    """direction=+1: long turnaround (depressed level, rising, bounced).
    direction=-1: short peak (elevated level, falling, rolled over).
    raw_q: underlying quarterly series (one-off outlier damping)."""
    r = MetricInflection(metric=metric)
    ttm = ttm.dropna()
    if len(ttm) < 4 or slope_series.dropna().empty:
        return r
    # dead-series guard: a concept whose tag stopped resolving years ago
    # (e.g. a filer moving COGS to a custom extension tag) must be MISSING,
    # not scored on its stale level while slope/trough silently drop out
    if (asof - ttm.index.max()).days > 300:
        r.flags.add("stale_series")
        return r
    pct, n = own_history_percentile(ttm)
    r.level_pctile, r.n_obs = pct, n
    if np.isfinite(pct):
        # long: depressed level scores high; short: elevated level scores high
        r.level = (100.0 - pct) / 100.0 if direction > 0 else pct / 100.0
    # trough first (slope may anchor at it)
    t_sc, t_det = trough_score(ttm, asof, direction, trough_window_q,
                               raw_q=raw_q)
    r.trough = t_sc
    r.details.update({f"trough_{k}": v for k, v in t_det.items()})
    if t_det.get("outlier_trough"):
        r.flags.add("outlier_trough")
    anchor = t_det.get("extremum_date")
    s_sc, s_det = slope_score(slope_series, ttm, asof, direction,
                              slope_window_q, anchor=anchor)
    r.slope = s_sc
    r.details.update({f"slope_{k}": v for k, v in s_det.items()})
    r.score = conjunctive({"level": r.level, "slope": r.slope,
                           "trough": r.trough})
    # min-obs soft confidence (audit 2.7)
    if n < config.MIN_OBS_LEVEL:
        r.confidence = max(0.2, n / config.MIN_OBS_LEVEL)
        r.flags.add("short_history_metric")
    return r


# ---------------------------------------------------------------------------
# At-peak primitive (user direction 2026-07-08: catch the VITL/CAKE SETUP —
# margins at an own-history extreme, reached at an unusual rate, right now —
# before the rollover that the peak-recency primitive requires)
# ---------------------------------------------------------------------------

def at_peak_metric(ttm: pd.Series, asof: pd.Timestamp) -> tuple[float, dict]:
    """Strict product of three components (kernel documented in config.AT_PEAK):

    - level gate: latest value's own-full-history percentile, 0 below
      level_lo -> 1 at level_hi. A stable business sits ~50th and scores 0.
    - delta gate: the CURRENT 4-quarter change's percentile within the
      name's OWN history of rolling 4q-changes. A chronic margin expander
      (software transition) is always at a high level but its current delta
      is typical FOR ITSELF -> gate ~0. A price-hiker/cycle-top name got
      here at an unusual rate -> gate high.
    - peak recency: TTM argmax within the trailing ref window, calendar ramp
      to 0 by recency_zero_q (we must be AT/just past the peak, not looking
      back at one).

    Deliberately narrow: this NEVER scores alone downstream — it part-credits
    the false-moat peaked row and opens interaction gates (run-up,
    margin-vs-peer)."""
    k = config.AT_PEAK
    s = ttm.dropna()
    details: dict = {}
    if len(s) < 8:
        return np.nan, details
    pct, n = own_history_percentile(s)
    if not np.isfinite(pct):
        return np.nan, details
    level_gate = float(np.clip((pct - k["level_lo"])
                               / (k["level_hi"] - k["level_lo"]), 0, 1))
    details["level_pctile"] = pct

    # rolling 4q-change history vs the current 4q-change
    deltas = (s - s.shift(4)).dropna()
    if len(deltas) < 6:
        return np.nan, details
    cur_delta = float(deltas.iloc[-1])
    dpct = 100.0 * stats.percentileofscore(deltas.values, cur_delta,
                                           kind="mean") / 100.0
    delta_gate = float(np.clip((dpct - k["delta_lo"])
                               / (k["delta_hi"] - k["delta_lo"]), 0, 1))
    details["delta_pctile"] = float(dpct)

    cutoff = asof - pd.Timedelta(days=QDAYS * k["ref_window_q"])
    win = s[s.index >= cutoff]
    if len(win) < 4:
        return np.nan, details
    t_max = win.idxmax()
    q_since = (asof - t_max).days / QDAYS
    recency = float(np.clip(1.0 - q_since / k["recency_zero_q"], 0, 1))
    details["peak_date"] = t_max
    details["q_since_peak"] = float(q_since)

    score = level_gate * delta_gate * recency
    details.update({"level_gate": level_gate, "delta_gate": delta_gate,
                    "recency": recency})
    return float(score), details


def at_peak_combined(inputs: dict, asof: pd.Timestamp) -> tuple[float, dict]:
    """Weighted mean of at_peak_metric over roic/gm/om (weights in config),
    renormalized over computable metrics. inputs: {metric: ttm_series}."""
    weights = config.AT_PEAK_METRIC_WEIGHTS
    acc, wsum = 0.0, 0.0
    per: dict = {}
    for m, ttm in inputs.items():
        w = weights.get(m)
        if w is None or ttm is None:
            continue
        sc, det = at_peak_metric(ttm, asof)
        per[m] = {"score": sc, **{kk: (str(vv) if isinstance(vv, pd.Timestamp)
                                       else vv) for kk, vv in det.items()}}
        if np.isfinite(sc):
            acc += w * sc
            wsum += w
    if wsum == 0:
        return np.nan, per
    return float(acc / wsum), per


# ---------------------------------------------------------------------------
# Revenue-context classifier + cost-cut haircut (audit 4.1)
# ---------------------------------------------------------------------------

def revenue_context(rev_q: pd.Series, asof: pd.Timestamp) -> str:
    """growing / stabilizing / still-declining over 3-6q + a longer 12q
    corroboration window (audit 4.1). As-of aware: only observations at or
    before the run date count."""
    from .quarterlyize import yoy_ratio
    g = yoy_ratio(rev_q).dropna()
    if len(g) < 3:  # revenue-less names (mREITs etc.): nothing to classify
        return "unknown"
    g = g[g.index <= asof]
    if len(g) < 3:
        return "unknown"
    recent = g.iloc[-config.REVENUE_CONTEXT_SHORT_Q:]
    longer = g.iloc[-config.REVENUE_CONTEXT_LONG_Q:]
    latest = recent.iloc[-1]
    ts = theil_sen_slope(recent.values)
    if latest > 0.0 and recent.mean() > 0:
        return "growing"
    if latest < 0.0 and (not np.isfinite(ts) or ts <= 0
                         or recent.mean() < -0.02):
        # longer-window corroboration: a recent dip inside a longer uptrend
        # is "stabilizing", not "still-declining"
        if len(longer) >= 8 and longer.mean() > 0.02:
            return "stabilizing"
        return "still-declining"
    return "stabilizing"


HAIRCUT_METRICS = {"gm", "om", "roic", "ni", "fcf"}  # not revenue/drawdown/SI


def staleness_multiplier(latest_end: pd.Timestamp | None,
                         asof: pd.Timestamp) -> float:
    """Soft staleness decay (audit 1.6): 1.0 until ~140d, linear to floor."""
    if latest_end is None:
        return config.STALENESS_DECAY_FLOOR
    age = (asof - latest_end).days
    if age <= config.STALENESS_DECAY_START_DAYS:
        return 1.0
    span = config.STALENESS_DECAY_FULL_DAYS - config.STALENESS_DECAY_START_DAYS
    frac = min(1.0, (age - config.STALENESS_DECAY_START_DAYS) / span)
    return 1.0 - frac * (1.0 - config.STALENESS_DECAY_FLOOR)


@dataclass
class ThemeInflection:
    per_metric: dict[str, MetricInflection] = field(default_factory=dict)
    theme_score: float = np.nan     # 0-1, breadth-adjusted, haircut applied
    breadth: int = 0
    revenue_context: str = "unknown"
    cost_cut_haircut_applied: bool = False
    staleness_mult: float = 1.0
    flags: set = field(default_factory=set)


def combine_metrics(per_metric: dict[str, MetricInflection],
                    rev_ctx: str, staleness_mult: float,
                    weights: dict[str, float] | None = None,
                    breadth_n: int | None = None) -> ThemeInflection:
    """Across metrics: weighted additive mean x breadth term (audit 2.6a),
    with the cost-cut haircut on margin/earnings/FCF metrics only.
    breadth_n: the INTENDED metric count for the breadth denominator — 6 for
    the long six-metric theme, 3 for the short peak theme (roic/gm/om), 5
    for the annual FPI variant (no roic). A structurally-3-metric theme must
    not be capped at sqrt(3/6)."""
    weights = weights or config.INFLECTION_METRIC_WEIGHTS
    breadth_n = breadth_n or config.BREADTH_N_METRICS
    th = ThemeInflection(per_metric=per_metric, revenue_context=rev_ctx,
                         staleness_mult=staleness_mult)
    total_w, acc = 0.0, 0.0
    haircut = rev_ctx == "still-declining"
    for m, res in per_metric.items():
        if not np.isfinite(res.score):
            continue
        w = weights.get(m, 0.1) * res.confidence
        sc = res.score
        if haircut and m in HAIRCUT_METRICS:
            sc *= config.COST_CUT_HAIRCUT
        acc += w * sc
        total_w += w
        th.breadth += 1
        th.flags |= res.flags
    if total_w > 0:
        mean_score = acc / total_w
        breadth_term = np.sqrt(th.breadth / breadth_n)
        th.theme_score = float(np.clip(mean_score * min(1.0, breadth_term)
                                       * staleness_mult, 0, 1))
    th.cost_cut_haircut_applied = haircut and th.breadth > 0
    if th.cost_cut_haircut_applied:
        th.flags.add("cost_cut_inflection")
    return th


# ---------------------------------------------------------------------------
# Op-margin-vs-peak + decline-shape guard (audits 4.3, 2.6e)
# ---------------------------------------------------------------------------

def peak_value(series: pd.Series, top_n: int = None) -> float:
    """5-yr peak construct: median of top-N quarters, not max (audit 2.6e)."""
    top_n = top_n or config.PEAK_TOP_N
    s = series.dropna()
    if len(s) < top_n:
        return np.nan
    return float(np.median(np.sort(s.values)[-top_n:]))


def margin_vs_peak(om_ttm: pd.Series, asof: pd.Timestamp) -> tuple[float, dict]:
    """Score in [0,1]: how depressed current op margin is vs own 5-yr peak,
    with the decline-shape modifier (stumble credited, erosion ~0)."""
    s = om_ttm.dropna()
    details = {}
    if len(s) < 8:
        return np.nan, details
    # dead-series guard, mirroring metric_inflection: `cur` below is just the
    # last observation, so without this a series that stopped resolving years
    # ago is scored as if it were current.
    age_days = (asof - s.index.max()).days
    if age_days > config.MARGIN_VS_PEAK_STALE_DAYS:
        details["skipped"] = "stale_series"
        details["age_days"] = int(age_days)
        return np.nan, details
    lookback = s[s.index >= asof - pd.Timedelta(days=QDAYS * 20)]
    if len(lookback) < 8:
        lookback = s.iloc[-20:]
    peak = peak_value(lookback)
    cur = s.iloc[-1]
    if not np.isfinite(peak):
        return np.nan, details
    # plausibility guard: a margin above 100% means the numerator and the
    # denominator are not the same business (equity-method income, a gain, or
    # a revenue base that excludes the earnings stream). The series is
    # definitionally unusable, so null the metric rather than score the gap.
    if peak > config.MARGIN_VS_PEAK_MAX_PLAUSIBLE:
        details["skipped"] = "implausible_peak"
        details["peak"] = float(peak)
        return np.nan, details
    gap = peak - cur  # in margin points
    details["peak"] = peak
    details["current"] = float(cur)
    details["gap_pp"] = float(gap)
    if len(s) >= config.MIN_OBS_5YR:
        conf = 1.0
    else:
        conf = max(0.2, len(s) / config.MIN_OBS_5YR)
    # kernel: 0 below 3pp gap, full credit at 10pp, plateau beyond
    base = float(np.clip((gap - 0.03) / 0.07, 0, 1))
    # decline-shape modifier: stumble vs erosion
    shape, mult = decline_shape(lookback, details)
    details["decline_shape"] = shape
    return float(np.clip(base * mult * conf, 0, 1)), details


def decline_shape(s: pd.Series, details: dict) -> tuple[str, float]:
    """Sharp drop within ~6-10q after a plateau = 'stumble' (full credit);
    monotonic multi-year erosion = 'erosion' (~0 credit)."""
    s = s.dropna()
    if len(s) < 8:
        return "unknown", 0.7
    # decline starts at the LAST near-peak observation, not the first
    # occurrence of the max (plateaus would otherwise read as long declines)
    peak = s.max()
    near_peak = s[s >= peak - max(0.02 * abs(peak), 1e-12)]
    peak_idx = near_peak.index[-1]
    post = s[s.index > peak_idx]
    if len(post) == 0:
        return "at_peak", 0.3
    decline_q = len(post)
    d = post.diff().dropna()
    frac_down = float((d < 0).mean()) if len(d) else 0.0
    if decline_q <= config.STUMBLE_MAX_DECLINE_Q:
        return "stumble", 1.0
    if decline_q >= config.EROSION_MIN_DECLINE_Q and frac_down > 0.7:
        return "erosion", 0.1
    return "mixed", 0.5
