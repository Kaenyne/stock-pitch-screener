"""Quarterlyization module (spec roadmap step 3.5; audit 1.1).

companyfacts does not serve discrete quarterly series for flow items. This
module constructs them:

- key facts by (taxonomy, tag, unit, start, end); dedupe keeping latest
  `filed` (tie-break: accession number); restrict to 10-K/10-Q and /A forms
- NEVER use fy/fp for period identification (they describe the filing)
- classify durations ~90d (Q), ~180d (H1), ~270d (9M), ~360d (FY)
- derive Q4 = FY - (Q1+Q2+Q3); derive quarters by consecutive-YTD differencing
- validate derived quarters sum back to FY within tolerance; on failure NULL
  the derived quarters, never interpolate
- flag derived observations; `restated` flag when retained value differs >3%
  from originally-filed value
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

DAY = pd.Timedelta(days=1)
QUARTER_TOL = pd.Timedelta(days=20)


def classify_duration(days: int) -> str | None:
    for name, (lo, hi) in config.DURATION_BUCKETS.items():
        if lo <= days <= hi:
            return name
    return None


def dedupe_facts(obs: list[dict], allowed_forms: set[str]) -> pd.DataFrame:
    """obs: raw companyfacts unit observations. Returns one row per
    (start, end) with latest-filed value + first-filed value (restatement).
    Instant facts have start=None -> keyed by end only.
    """
    rows = []
    for o in obs:
        form = o.get("form", "")
        if form not in allowed_forms:
            continue
        val = o.get("val")
        end = o.get("end")
        if val is None or end is None:
            continue
        rows.append({
            "start": o.get("start"), "end": end, "val": float(val),
            "filed": o.get("filed", ""), "accn": o.get("accn", ""),
            "form": form,
        })
    if not rows:
        return pd.DataFrame(columns=["start", "end", "val", "first_val",
                                     "filed", "form", "restated"])
    df = pd.DataFrame(rows)
    df["skey"] = df["start"].fillna("__instant__")
    df = df.sort_values(["filed", "accn"])  # ascending; last = retained
    first_vals = (df.groupby(["end", "skey"], as_index=False)["val"].first()
                  .rename(columns={"val": "first_val"}))
    keep = df.groupby(["end", "skey"], sort=False).tail(1).copy()
    keep = keep.merge(first_vals, on=["end", "skey"], how="left")
    keep = keep.drop(columns=["skey"])
    denom = keep["first_val"].abs().clip(lower=1.0)
    keep["restated"] = ((keep["val"] - keep["first_val"]).abs() / denom
                        > config.RESTATED_REL_TOL)
    keep["start"] = pd.to_datetime(keep["start"], errors="coerce")
    keep["end"] = pd.to_datetime(keep["end"], errors="coerce")
    return keep.dropna(subset=["end"]).sort_values("end").reset_index(drop=True)


@dataclass
class QuarterlySeries:
    """Discrete quarterly series for one flow concept."""
    values: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    derived: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))
    derived_q4: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))
    restated_any: bool = False

    def __len__(self):
        return len(self.values)


def build_quarterly(df: pd.DataFrame) -> QuarterlySeries:
    """df: deduped duration facts (from dedupe_facts, start not null).
    Returns discrete quarterly series indexed by period end date.
    """
    out = QuarterlySeries()
    df = df.dropna(subset=["start"]).copy()
    if df.empty:
        return out
    df["days"] = (df["end"] - df["start"]).dt.days
    df["bucket"] = df["days"].map(classify_duration)
    df = df.dropna(subset=["bucket"])
    if df.empty:
        return out
    out.restated_any = bool(df["restated"].any())

    quarters: dict[pd.Timestamp, float] = {}
    derived: dict[pd.Timestamp, bool] = {}
    derived_q4: dict[pd.Timestamp, bool] = {}

    # 1) As-filed discrete quarters win.
    for _, r in df[df["bucket"] == "Q"].iterrows():
        quarters[r["end"]] = r["val"]
        derived[r["end"]] = False
        derived_q4[r["end"]] = False

    # 2) Group YTD-style durations by fiscal-year start; difference
    #    consecutive marks to fill missing quarters.
    ytd = df[df["bucket"].isin(["Q", "H1", "9M", "FY"])].copy()
    ytd["fy_start"] = ytd["start"]
    order = {"Q": 1, "H1": 2, "9M": 3, "FY": 4}
    fy_rows = df[df["bucket"] == "FY"]

    for fy_start, grp in ytd.groupby("fy_start"):
        grp = grp.sort_values("end")
        marks = [(order[b], e, v) for b, e, v in
                 zip(grp["bucket"], grp["end"], grp["val"])]
        marks = sorted(set(marks))
        for i in range(1, len(marks)):
            o_prev, e_prev, v_prev = marks[i - 1]
            o_cur, e_cur, v_cur = marks[i]
            if o_cur - o_prev != 1:
                continue  # non-adjacent YTD marks; gap too wide to difference
            gap = (e_cur - e_prev).days
            if not (60 <= gap <= 120):
                continue
            if e_cur in quarters and not derived.get(e_cur, True):
                continue  # as-filed discrete quarter present
            quarters[e_cur] = v_cur - v_prev
            derived[e_cur] = True
            derived_q4[e_cur] = (o_cur == 4)

    # 3) Q4 = FY - (Q1+Q2+Q3) when 9M mark absent.
    for _, fy in fy_rows.iterrows():
        fy_start, fy_end, fy_val = fy["start"], fy["end"], fy["val"]
        if fy_end in quarters and not derived.get(fy_end, True):
            continue
        if fy_end in quarters and not derived_q4.get(fy_end, False):
            continue
        # Members strictly inside this FY: Q1 ends ~90d after fy_start, and
        # the PRIOR year's Q4 ends ~at fy_start — exclude it with a +40d floor.
        inside = {e: v for e, v in quarters.items()
                  if fy_start + pd.Timedelta(days=40) < e < fy_end - QUARTER_TOL
                  and not derived_q4.get(e, False)}
        if fy_end not in quarters and len(inside) == 3:
            quarters[fy_end] = fy_val - sum(inside.values())
            derived[fy_end] = True
            derived_q4[fy_end] = True

    # 4) Validation: quarters within an FY must sum to FY within tolerance;
    #    else null the DERIVED quarters of that FY (never interpolate).
    for _, fy in fy_rows.iterrows():
        fy_start, fy_end, fy_val = fy["start"], fy["end"], fy["val"]
        members = [e for e in quarters
                   if fy_start + pd.Timedelta(days=40) < e <= fy_end + QUARTER_TOL]
        if len(members) != 4:
            continue
        qsum = sum(quarters[e] for e in members)
        tol = max(abs(fy_val), config.FY_SUM_ABS_FLOOR) * config.FY_SUM_REL_TOL
        if abs(qsum - fy_val) > tol:
            for e in members:
                if derived.get(e, False):
                    quarters.pop(e, None)
                    derived.pop(e, None)
                    derived_q4.pop(e, None)

    if not quarters:
        return out
    idx = pd.DatetimeIndex(sorted(quarters))
    out.values = pd.Series([quarters[e] for e in idx], index=idx, dtype=float)
    out.derived = pd.Series([derived.get(e, False) for e in idx], index=idx)
    out.derived_q4 = pd.Series([derived_q4.get(e, False) for e in idx],
                               index=idx)
    return out


def build_instant(df: pd.DataFrame) -> pd.Series:
    """Balance-sheet (instant) series: latest-filed value per end date."""
    inst = df[df["start"].isna()]
    if inst.empty:
        # some filers file instants WITH duration context rows absent; treat
        # everything without a valid duration bucket as instant-like
        inst = df.copy()
        if inst.empty:
            return pd.Series(dtype=float)
    s = inst.set_index("end")["val"].sort_index()
    return s[~s.index.duplicated(keep="last")].astype(float)


# ---------------------------------------------------------------------------
# Derived series (audit 2.1): TTM and fiscal-aligned YoY-delta
# ---------------------------------------------------------------------------

def _find_prior(idx: pd.DatetimeIndex, t: pd.Timestamp, days_back: int,
                tol_days: int = 45) -> pd.Timestamp | None:
    target = t - pd.Timedelta(days=days_back)
    lo, hi = target - pd.Timedelta(days=tol_days), target + pd.Timedelta(days=tol_days)
    cand = idx[(idx >= lo) & (idx <= hi)]
    if len(cand) == 0:
        return None
    return cand[np.argmin(np.abs((cand - target).days))]


def to_ttm(q: pd.Series) -> pd.Series:
    """Rolling 4-quarter sums; requires 4 CONSECUTIVE fiscal quarters (each
    ~91d apart) else NaN at that end date."""
    if len(q) < 4:
        return pd.Series(dtype=float)
    idx = q.index
    vals = {}
    for t in idx:
        members = [q[t]]
        cur = t
        ok = True
        for _ in range(3):
            prev = _find_prior(idx, cur, 91, 35)
            if prev is None:
                ok = False
                break
            members.append(q[prev])
            cur = prev
        if ok:
            vals[t] = float(np.sum(members))
    return pd.Series(vals, dtype=float).sort_index()


def yoy_delta(q: pd.Series) -> pd.Series:
    """Q_t - Q_{t-4}, fiscal-aligned by end date ~365d earlier."""
    idx = q.index
    vals = {}
    for t in idx:
        prev = _find_prior(idx, t, 365, 45)
        if prev is not None:
            vals[t] = float(q[t] - q[prev])
    return pd.Series(vals, dtype=float).sort_index()


def yoy_ratio(q: pd.Series) -> pd.Series:
    """YoY growth rate (Q_t / Q_{t-4} - 1), fiscal-aligned; NaN when the base
    is <= 0 (growth undefined off a non-positive base)."""
    idx = q.index
    vals = {}
    for t in idx:
        prev = _find_prior(idx, t, 365, 45)
        if prev is not None and q[prev] > 0:
            vals[t] = float(q[t] / q[prev] - 1.0)
    return pd.Series(vals, dtype=float).sort_index()


def ttm_ratio(num_ttm: pd.Series, den_ttm: pd.Series,
              den_floor: pd.Series | float | None = None) -> pd.Series:
    """TTM-numerator / TTM-denominator ratio aligned on shared end dates.
    den_floor: optional floor for the denominator (audit 2.2: TTM revenue
    denominator floored at a small fraction of assets)."""
    common = num_ttm.index.intersection(den_ttm.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    num = num_ttm[common]
    den = den_ttm[common].astype(float)
    if den_floor is not None:
        if isinstance(den_floor, pd.Series):
            fl = den_floor.reindex(common).ffill()
            den = pd.concat([den, fl], axis=1).max(axis=1)
        else:
            den = den.clip(lower=den_floor)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def align_instant_to_quarters(inst: pd.Series,
                              q_index: pd.DatetimeIndex,
                              tol_days: int = 45) -> pd.Series:
    """Snap instant (balance-sheet) observations onto quarterly end dates."""
    if inst is None or len(inst) == 0 or len(q_index) == 0 \
            or not isinstance(inst.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    vals = {}
    for t in q_index:
        window = inst[(inst.index >= t - pd.Timedelta(days=tol_days))
                      & (inst.index <= t + pd.Timedelta(days=tol_days))]
        if len(window):
            nearest = window.index[np.argmin(np.abs((window.index - t).days))]
            vals[t] = float(window[nearest])
    return pd.Series(vals, dtype=float).sort_index()


def avg_instant(inst_q: pd.Series, periods_back: int = 4) -> pd.Series:
    """Average of current and periods_back-ago instant values (avg assets)."""
    idx = inst_q.index
    vals = {}
    for t in idx:
        prev = _find_prior(idx, t, 91 * periods_back, 60)
        if prev is not None:
            vals[t] = float((inst_q[t] + inst_q[prev]) / 2.0)
        else:
            vals[t] = float(inst_q[t])
    return pd.Series(vals, dtype=float).sort_index()
