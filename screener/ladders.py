"""Tag-fallback ladder resolution (audit 1.2).

Ladder selection is recency-aware and PER-PERIOD: for every period end date
we take the value from the highest-priority tag that covers it, stitching
series across tag switches (e.g. the ~2018 ASC-606 revenue-tag change).
A concept is "missing" only after the full ladder (and derive rules) fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .quarterlyize import (QuarterlySeries, build_instant, build_quarterly,
                           dedupe_facts)


def _get_tag_obs(facts: dict, tag: str) -> list[dict]:
    """tag may be 'dei:Xxx' or plain us-gaap tag or 'ifrs-full:Xxx'."""
    if ":" in tag:
        taxo, t = tag.split(":", 1)
    else:
        taxo, t = "us-gaap", tag
    units = facts.get("facts", {}).get(taxo, {}).get(t, {}).get("units", {})
    # Prefer USD / shares / pure units; take the unit with most observations.
    if not units:
        return []
    def unit_pref(u: str) -> tuple:
        return (u == "USD" or u == "shares", len(units[u]))
    best = max(units, key=unit_pref)
    obs = units[best]
    for o in obs:
        o["_unit"] = best
    return obs


def resolve_flow(facts: dict, ladder: list[str],
                 allowed_forms: set[str]) -> QuarterlySeries:
    """Stitched discrete-quarterly series across a tag ladder."""
    per_tag: list[QuarterlySeries] = []
    for tag in ladder:
        obs = _get_tag_obs(facts, tag)
        if not obs:
            continue
        dd = dedupe_facts(obs, allowed_forms)
        qs = build_quarterly(dd)
        if len(qs):
            per_tag.append(qs)
    if not per_tag:
        return QuarterlySeries()
    # Per-period stitch: highest-priority tag that has the period wins.
    all_ends = sorted(set().union(*[set(q.values.index) for q in per_tag]))
    vals, der, der4 = {}, {}, {}
    restated = any(q.restated_any for q in per_tag)
    for e in all_ends:
        for q in per_tag:  # priority order
            if e in q.values.index and np.isfinite(q.values[e]):
                vals[e] = q.values[e]
                der[e] = bool(q.derived.get(e, False))
                der4[e] = bool(q.derived_q4.get(e, False))
                break
    out = QuarterlySeries()
    idx = pd.DatetimeIndex(sorted(vals))
    out.values = pd.Series([vals[e] for e in idx], index=idx, dtype=float)
    out.derived = pd.Series([der[e] for e in idx], index=idx)
    out.derived_q4 = pd.Series([der4[e] for e in idx], index=idx)
    out.restated_any = restated
    return out


def resolve_annual(facts: dict, ladder: list[str],
                   allowed_forms: set[str]) -> pd.Series:
    """FY-duration series for annual-only FPIs (audit 1.3)."""
    for tag in ladder:
        obs = _get_tag_obs(facts, tag)
        if not obs:
            continue
        dd = dedupe_facts(obs, allowed_forms)
        dd = dd.dropna(subset=["start"])
        if dd.empty:
            continue
        days = (dd["end"] - dd["start"]).dt.days
        fy = dd[(days >= 340) & (days <= 380)]
        if len(fy) >= 2:
            s = fy.set_index("end")["val"].sort_index()
            return s[~s.index.duplicated(keep="last")].astype(float)
    return pd.Series(dtype=float)


def resolve_instant(facts: dict, ladder: list[str],
                    allowed_forms: set[str]) -> pd.Series:
    """Stitched instant (balance-sheet) series across a tag ladder."""
    per_tag = []
    for tag in ladder:
        obs = _get_tag_obs(facts, tag)
        if not obs:
            continue
        dd = dedupe_facts(obs, allowed_forms)
        s = build_instant(dd)
        if len(s):
            per_tag.append(s)
    if not per_tag:
        return pd.Series(dtype=float)
    all_ends = sorted(set().union(*[set(s.index) for s in per_tag]))
    vals = {}
    for e in all_ends:
        for s in per_tag:
            if e in s.index and np.isfinite(s[e]):
                vals[e] = float(s[e])
                break
    return pd.Series(vals, dtype=float).sort_index()


# ---------------------------------------------------------------------------
# Derive-from-components rules
# ---------------------------------------------------------------------------

def _merge_missing(primary: QuarterlySeries,
                   fallback: pd.Series) -> QuarterlySeries:
    """Fill primary's missing period-ends from a derived fallback series."""
    if primary.values.empty:
        base_idx = fallback.index
    else:
        base_idx = primary.values.index.union(fallback.index)
    vals, der, der4 = {}, {}, {}
    for e in base_idx:
        if e in primary.values.index and np.isfinite(primary.values.get(e, np.nan)):
            vals[e] = primary.values[e]
            der[e] = bool(primary.derived.get(e, False))
            der4[e] = bool(primary.derived_q4.get(e, False))
        elif e in fallback.index and np.isfinite(fallback[e]):
            vals[e] = float(fallback[e])
            der[e] = True
            der4[e] = False
    out = QuarterlySeries()
    idx = pd.DatetimeIndex(sorted(vals))
    out.values = pd.Series([vals[e] for e in idx], index=idx, dtype=float)
    out.derived = pd.Series([der[e] for e in idx], index=idx)
    out.derived_q4 = pd.Series([der4[e] for e in idx], index=idx)
    out.restated_any = primary.restated_any
    return out


def derive_gross_profit(facts: dict, revenue: QuarterlySeries,
                        allowed_forms: set[str]) -> QuarterlySeries:
    """GP = GrossProfit, else revenue - COGS where COGS resolves PER PERIOD
    as CostOfRevenue -> CostOfGoodsAndServicesSold -> CostOfGoodsSold
    (+ CostOfServices only in that last branch — the first two tags already
    include services; adding CoS onto them double-counts)."""
    gp = resolve_flow(facts, config.TAG_LADDERS["gross_profit"], allowed_forms)
    cor = resolve_flow(facts, ["CostOfRevenue"], allowed_forms)
    cogas = resolve_flow(facts, ["CostOfGoodsAndServicesSold"], allowed_forms)
    cogs_only = resolve_flow(facts, ["CostOfGoodsSold"], allowed_forms)
    cos = resolve_flow(facts, ["CostOfServices"], allowed_forms)

    all_ends = sorted(set(cor.values.index) | set(cogas.values.index)
                      | set(cogs_only.values.index))
    cogs_vals = {}
    for e in all_ends:
        if e in cor.values.index and np.isfinite(cor.values[e]):
            cogs_vals[e] = float(cor.values[e])
        elif e in cogas.values.index and np.isfinite(cogas.values[e]):
            cogs_vals[e] = float(cogas.values[e])
        elif e in cogs_only.values.index and np.isfinite(cogs_only.values[e]):
            v = float(cogs_only.values[e])
            if e in cos.values.index and np.isfinite(cos.values[e]):
                v += float(cos.values[e])
            cogs_vals[e] = v
    if revenue.values.empty or not cogs_vals:
        return gp
    cogs_s = pd.Series(cogs_vals, dtype=float).sort_index()
    common = revenue.values.index.intersection(cogs_s.index)
    derived_gp = revenue.values[common] - cogs_s[common]
    return _merge_missing(gp, derived_gp)


def derive_sga(facts: dict, allowed_forms: set[str]) -> QuarterlySeries:
    """SG&A = SGA tag, else Selling + G&A. When the filer tags BOTH
    components, a period missing the selling piece stays MISSING rather
    than silently zero-filling it; GA-only filers (no selling tag anywhere)
    use GA alone as the best available proxy."""
    sga = resolve_flow(facts, config.TAG_LADDERS["sga"], allowed_forms)
    selling = resolve_flow(facts, config.TAG_LADDERS["sga_selling"], allowed_forms)
    ga = resolve_flow(facts, config.TAG_LADDERS["sga_ga"], allowed_forms)
    if ga.values.empty:
        return sga
    if selling.values.empty:
        combined = ga.values.copy()  # GA-only filer
    else:
        common = ga.values.index.intersection(selling.values.index)
        combined = (ga.values[common] + selling.values[common]).astype(float)
    return _merge_missing(sga, combined)


def derive_interest_expense(facts: dict,
                            allowed_forms: set[str]) -> QuarterlySeries:
    for tag in ["InterestExpense", "InterestExpenseNonoperating",
                "InterestAndDebtExpense"]:
        s = resolve_flow(facts, [tag], allowed_forms)
        if len(s) >= 2:
            return s
    # InterestIncomeExpenseNet with sign check: usable only when it is
    # consistently an expense (negative); flip sign. Positive => net income
    # (banks) => unusable as interest expense.
    net = resolve_flow(facts, ["InterestIncomeExpenseNet"], allowed_forms)
    if len(net) >= 2 and net.values.median() < 0:
        net.values = -net.values
        return net
    return QuarterlySeries()


def derive_total_debt(facts: dict, allowed_forms: set[str]) -> pd.Series:
    """Instant total-debt series (operating leases excluded).

    LT piece: LTDNoncurrent else LongTermDebt.
    Current piece: ShortTermBorrowings/CommercialPaper + LTDCurrent when
    granular tags exist; else DebtCurrent alone (it already includes current
    LTD -- avoid double counting).
    + finance leases (total tag if present, else noncurrent + current).
    """
    def inst(tags):
        return resolve_instant(facts, tags, allowed_forms)

    ltd_nc = inst(["LongTermDebtNoncurrent"])
    ltd_combined = inst(["LongTermDebt"])
    ltd_cur = inst(["LongTermDebtCurrent"])
    stb = inst(["ShortTermBorrowings", "OtherShortTermBorrowings",
                "CommercialPaper", "BankOverdrafts"])
    debt_cur = inst(["DebtCurrent"])
    fl_total = inst(["FinanceLeaseLiability"])
    fl_nc = inst(["FinanceLeaseLiabilityNoncurrent"])
    fl_cur = inst(["FinanceLeaseLiabilityCurrent"])
    leases = fl_total if len(fl_total) else _sum_series([fl_nc, fl_cur])

    # Per-period assembly. When the LT piece falls back to the COMBINED
    # LongTermDebt tag (which already includes current maturities), neither
    # LongTermDebtCurrent nor DebtCurrent may be added on top of it — only
    # true short-term borrowings.
    def _get(s: pd.Series, e) -> float:
        return float(s[e]) if e in s.index and np.isfinite(s[e]) else np.nan

    all_ends = sorted(set(ltd_nc.index) | set(ltd_combined.index)
                      | set(ltd_cur.index) | set(stb.index)
                      | set(debt_cur.index))
    vals = {}
    for e in all_ends:
        nc = _get(ltd_nc, e)
        comb = _get(ltd_combined, e)
        cur_ltd = _get(ltd_cur, e)
        st = _get(stb, e)
        dcur = _get(debt_cur, e)
        core = np.nan
        if np.isfinite(nc):
            pieces = [nc]
            if np.isfinite(st) or np.isfinite(cur_ltd):
                pieces += [x for x in (st, cur_ltd) if np.isfinite(x)]
            elif np.isfinite(dcur):
                pieces.append(dcur)  # includes current LTD + ST borrowings
            core = float(np.sum(pieces))
        elif np.isfinite(comb):
            core = comb + (st if np.isfinite(st) else 0.0)
        elif np.isfinite(dcur) or np.isfinite(st) or np.isfinite(cur_ltd):
            # no LT piece at all: current debt only
            core = dcur if np.isfinite(dcur) else float(
                np.nansum([st, cur_ltd]))
        if np.isfinite(core):
            lease_v = _get(leases, e)
            vals[e] = core + (lease_v if np.isfinite(lease_v) else 0.0)
    return pd.Series(vals, dtype=float).sort_index()


def _sum_series(series_list: list[pd.Series]) -> pd.Series:
    nonempty = [s for s in series_list if len(s)]
    if not nonempty:
        return pd.Series(dtype=float)
    all_idx = sorted(set().union(*[set(s.index) for s in nonempty]))
    out = {}
    for e in all_idx:
        total, any_val = 0.0, False
        for s in nonempty:
            if e in s.index and np.isfinite(s[e]):
                total += float(s[e])
                any_val = True
        if any_val:
            out[e] = total
    return pd.Series(out, dtype=float).sort_index()
