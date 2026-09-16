"""Per-company fundamentals assembly: companyfacts JSON -> CompanyData.

CompanyData is the contract consumed by the inflection engine, metric
library, and scoring:

- cd.q[concept]      -> QuarterlySeries (discrete quarterly flows)
- cd.ttm[concept]    -> pd.Series (rolling-4Q sums, end-date indexed)
- cd.yoy[concept]    -> pd.Series (fiscal-aligned Q_t - Q_{t-4})
- cd.inst[concept]   -> pd.Series (instant/balance-sheet, end-date indexed)
- cd.annual[concept] -> pd.Series (FY series; annual_only FPI variant)
- cd.meta            -> dict of universe/submissions attributes & tags
- cd.market          -> dict filled by the market-data layer (prices, SI...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config
from .ladders import (derive_gross_profit, derive_interest_expense,
                      derive_sga, derive_total_debt, resolve_annual,
                      resolve_flow, resolve_instant)
from .quarterlyize import QuarterlySeries, to_ttm, yoy_delta

FLOW_CONCEPTS = [
    "revenue", "operating_income", "net_income", "continuing_income",
    "pretax_income", "tax_expense", "cfo", "capex", "dep_amort",
    "depreciation_only", "buybacks", "acquisitions", "gains_on_sale",
    "noninterest_expense", "net_interest_income", "noninterest_income",
    "policyholder_benefits", "premiums_earned",
]
# diluted_was is deliberately NOT a flow concept: weighted-average share
# counts are AVERAGES, not additive flows — Q4 derivation / YTD differencing
# / TTM summing are all category errors on it (FY avg - sum(Q avgs) is
# garbage). It is resolved separately as a level series below.
INSTANT_CONCEPTS = [
    "cash", "st_investments", "assets", "assets_current",
    "liabilities_current", "inventory", "receivables", "goodwill", "equity",
    "retained_earnings", "shares_outstanding", "debt_next_12m",
    "reportable_segments", "ppe_net",
]
# one-off tags resolved individually for the core-series adjustment
ONE_OFF_TAGS = config.TAG_LADDERS["one_off"]


@dataclass
class CompanyData:
    cik: int
    ticker: str
    name: str = ""
    meta: dict = field(default_factory=dict)
    q: dict[str, QuarterlySeries] = field(default_factory=dict)
    ttm: dict[str, pd.Series] = field(default_factory=dict)
    yoy: dict[str, pd.Series] = field(default_factory=dict)
    inst: dict[str, pd.Series] = field(default_factory=dict)
    annual: dict[str, pd.Series] = field(default_factory=dict)
    one_off: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    market: dict = field(default_factory=dict)
    flags: set = field(default_factory=set)

    @property
    def annual_only(self) -> bool:
        return bool(self.meta.get("annual_only", False))

    def qvals(self, concept: str) -> pd.Series:
        qs = self.q.get(concept)
        return qs.values if qs is not None else pd.Series(dtype=float)

    def latest_period_end(self) -> pd.Timestamp | None:
        ends = []
        for s in self.q.values():
            if len(s.values):
                ends.append(s.values.index.max())
        for s in self.annual.values():
            if len(s):
                ends.append(s.index.max())
        return max(ends) if ends else None


def _postbreak_trim(cd: CompanyData) -> None:
    """Structural breaks (audit 2.7): >20x magnitude jumps near the series
    start on NI and ASSETS are a STANDALONE detector (the formerNames
    de-SPAC regex is a separate cheap flag, not a precondition) -> restart
    own-history windows at the first post-break quarter. The jump must be
    sustained (later observations stay at the new magnitude), so a one-off
    spike doesn't trigger a trim."""
    break_date = None
    series_list = [cd.qvals("net_income").abs()]
    assets = cd.inst.get("assets")
    if assets is not None and len(assets):
        series_list.append(assets.abs())
    for v in series_list:
        if len(v) < 8:
            continue
        head = v.iloc[: min(8, len(v) - 4)]
        for i in range(1, len(head)):
            prev = max(float(head.iloc[:i].max()), 1.0)
            cur = float(head.iloc[i])
            if cur > config.DESPAC_MAGNITUDE_JUMP * prev:
                # sustained at the new magnitude?
                later = v[v.index > head.index[i]]
                if len(later) >= 3 and float(later.median()) > 3.0 * prev:
                    cand = head.index[i]
                    break_date = max(break_date, cand) if break_date else cand
                break
    if break_date is not None:
        cd.flags.add("structural_break")
        for concept, qs in cd.q.items():
            mask = qs.values.index >= break_date
            qs.values = qs.values[mask]
            qs.derived = qs.derived[mask]
            qs.derived_q4 = qs.derived_q4[mask]


def _resolve_diluted_was(facts: dict, forms: set[str]) -> pd.Series:
    """Diluted weighted-average shares as a quarterly LEVEL series: only
    ~90d-duration facts (each is that quarter's average), falling back to
    FY-duration values as annual points. Never differenced or summed."""
    from .ladders import _get_tag_obs
    from .quarterlyize import dedupe_facts
    for tag in config.TAG_LADDERS["diluted_was"]:
        obs = _get_tag_obs(facts, tag)
        if not obs:
            continue
        dd = dedupe_facts(obs, forms).dropna(subset=["start"])
        if dd.empty:
            continue
        days = (dd["end"] - dd["start"]).dt.days
        q = dd[(days >= 75) & (days <= 105)]
        if len(q) >= 4:
            s = q.set_index("end")["val"].sort_index()
            return s[~s.index.duplicated(keep="last")].astype(float)
        fy = dd[(days >= 340) & (days <= 380)]
        if len(fy) >= 2:
            s = fy.set_index("end")["val"].sort_index()
            return s[~s.index.duplicated(keep="last")].astype(float)
    return pd.Series(dtype=float)


def build_company(facts: dict, urow: dict) -> CompanyData:
    """facts: companyfacts JSON; urow: universe row (dict) with tags."""
    cd = CompanyData(cik=int(urow["cik"]), ticker=str(urow["ticker"]),
                     name=str(urow.get("name", "")))
    cd.meta = dict(urow)
    annual_only = bool(urow.get("annual_only", False))
    forms = config.ALLOWED_FORMS | (config.FPI_ANNUAL_FORMS if annual_only
                                    else set())

    if annual_only:
        # Annual-frequency variant over ifrs-full (or vestigial us-gaap).
        for concept, ladder in config.IFRS_LADDERS.items():
            s = resolve_annual(facts, ladder, forms)
            if not len(s):
                s = resolve_annual(facts, config.TAG_LADDERS.get(concept, []),
                                   forms)
            if len(s):
                cd.annual[concept] = s
        # gross profit derive for annual
        if "gross_profit" not in cd.annual and "revenue" in cd.annual:
            pass  # keep minimal per audit 1.3 (small ifrs ladder)
        return cd

    # --- flows ---
    for concept in FLOW_CONCEPTS:
        ladder = config.TAG_LADDERS.get(concept)
        if not ladder:
            continue
        qs = resolve_flow(facts, ladder, forms)
        if len(qs):
            cd.q[concept] = qs
    # derive rules
    rev = cd.q.get("revenue", QuarterlySeries())
    gp = derive_gross_profit(facts, rev, forms)
    if len(gp):
        cd.q["gross_profit"] = gp
    sga = derive_sga(facts, forms)
    if len(sga):
        cd.q["sga"] = sga
    ie = derive_interest_expense(facts, forms)
    if len(ie):
        cd.q["interest_expense"] = ie

    # --- instants ---
    for concept in INSTANT_CONCEPTS:
        ladder = config.TAG_LADDERS.get(concept)
        if not ladder:
            continue
        s = resolve_instant(facts, ladder, forms)
        if len(s):
            cd.inst[concept] = s
    debt = derive_total_debt(facts, forms)
    if len(debt):
        cd.inst["total_debt"] = debt

    # --- one-off items (audit 2.6): summed best-effort series ---
    one_off_parts = []
    for tag in ONE_OFF_TAGS:
        qs = resolve_flow(facts, [tag], forms)
        if len(qs):
            sign = -1.0 if "GainLoss" in tag else 1.0  # gains reduce "core"
            one_off_parts.append(sign * qs.values)
    if one_off_parts:
        df = pd.concat(one_off_parts, axis=1)
        cd.one_off = df.sum(axis=1, min_count=1).sort_index()

    # --- diluted WAS: level series (never Q4-derived/TTM-summed) ---
    was = _resolve_diluted_was(facts, forms)
    if len(was):
        cd.inst["diluted_was_q"] = was

    # --- structural breaks then derived series ---
    _postbreak_trim(cd)
    # FCF = CFO - capex on the quarterly grid. When the capex ladder resolves
    # NOTHING, FCF is declared MISSING (excluded + renormalized downstream)
    # rather than zero-filled with CFO — audit 3.1 forbids zero-fill on a
    # scored input. The flag records why the metric is absent.
    cfo = cd.qvals("cfo")
    capex = cd.qvals("capex")
    if len(cfo):
        if len(capex):
            common = cfo.index.intersection(capex.index)
            fcf = (cfo[common] - capex[common]).astype(float)
        else:
            fcf = pd.Series(dtype=float)
            cd.flags.add("fcf_no_capex")
        if len(fcf) >= 2:
            qs = QuarterlySeries()
            qs.values = fcf
            src = cd.q["cfo"]
            qs.derived = src.derived.reindex(fcf.index).fillna(True)
            qs.derived_q4 = src.derived_q4.reindex(fcf.index).fillna(False)
            cd.q["fcf"] = qs

    for concept, qs in cd.q.items():
        if len(qs.values) >= 4:
            t = to_ttm(qs.values)
            if len(t):
                cd.ttm[concept] = t
        if len(qs.values) >= 5:
            y = yoy_delta(qs.values)
            if len(y):
                cd.yoy[concept] = y

    if any(qs.restated_any for qs in cd.q.values()):
        cd.flags.add("restated")
    return cd
