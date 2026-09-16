"""Cross-sectional scoring: normalization, missing-data policy, theme
aggregation, and both composites (spec "Scoring foundations" + LONG/SHORT
tables).

Everything lands on a 0-100 scale:
- cross-sectional inputs -> percentile ranks vs the current snapshot universe
  (sector-bucketed for industry-structural level metrics, audit 3.2)
- self-normalized inputs (inflection engine, vs-own-peak, drawdown kernel,
  Piotroski/Beneish constructions) -> their [0,1] score x 100

Missing-data policy (audit 3.1): missing/inapplicable inputs are EXCLUDED and
weights renormalized over computed inputs; never zero- or neutral-filled.
Coverage columns + low-confidence tag + shrink-toward-median guard.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config

NEUTRAL = 50.0


# ---------------------------------------------------------------------------
# Normalization primitives
# ---------------------------------------------------------------------------

def pct_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Percentile rank 0-100 vs the snapshot universe; NaN stays NaN."""
    r = s.rank(pct=True, na_option="keep") * 100.0
    return r if higher_is_better else 100.0 - r


def sector_rank(df: pd.DataFrame, col: str, bucket_col: str = "sector2",
                higher_is_better: bool = True,
                min_bucket: int = None) -> pd.Series:
    """Within-coarse-sector percentile rank, universe-wide fallback for
    buckets with < min_bucket members (audit 3.2)."""
    min_bucket = min_bucket or config.SECTOR_BUCKET_MIN
    out = pd.Series(np.nan, index=df.index)
    universe_ranks = pct_rank(df[col], higher_is_better)
    for bucket, grp in df.groupby(bucket_col):
        valid = grp[col].notna().sum()
        if valid >= min_bucket:
            out.loc[grp.index] = pct_rank(grp[col], higher_is_better)
        else:
            out.loc[grp.index] = universe_ranks.loc[grp.index]
    return out


def weighted_theme(df: pd.DataFrame, components: dict[str, float],
                   shrink: bool = False) -> tuple[pd.Series, pd.Series]:
    """Exclude-and-renormalize weighted average of 0-100 component columns.
    Returns (theme_score, coverage_fraction_of_intended_weight).

    shrink=True applies the thin-basis guard (audit 3.1 item 4): shrink the
    score toward the universe median (~50) in proportion to missing weight.
    Applied at BUCKET level only, so nested themes don't double-shrink."""
    total_w = sum(components.values())
    acc = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)
    for col, w in components.items():
        c = df[col] if col in df else pd.Series(np.nan, index=df.index)
        m = c.notna()
        acc[m] += c[m] * w
        wsum[m] += w
    score = acc / wsum.replace(0.0, np.nan)
    coverage = wsum / total_w
    if shrink:
        # shrink toward the UNIVERSE MEDIAN of the computed scores (audit
        # 3.1: buckets mix percentile ranks with self-normalized kernels, so
        # the median is not 50 — shrinking to 50 would inflate low-coverage
        # names in a bucket whose typical score sits below 50)
        med = float(np.nanmedian(score.values)) if score.notna().any() \
            else NEUTRAL
        score = coverage * score + (1.0 - coverage) * med
    score[wsum == 0] = np.nan
    return score, coverage


# ---------------------------------------------------------------------------
# Kernels (audit 4.4 process rule: shapes documented in config)
# ---------------------------------------------------------------------------

def drawdown_kernel(dd: float) -> float:
    """Monotone-then-plateau: 0 below 20%, linear to full credit at 35%,
    plateau through 80%, gentle taper beyond (95% dd -> 0.85)."""
    k = config.DRAWDOWN_KERNEL
    if not np.isfinite(dd):
        return np.nan
    if dd <= k["zero_below"]:
        return 0.0
    if dd <= k["full_at"]:
        return (dd - k["zero_below"]) / (k["full_at"] - k["zero_below"])
    if dd <= k["plateau_to"]:
        return 1.0
    slope = (1.0 - k["taper_at_95"]) / (0.95 - k["plateau_to"])
    return float(max(k["taper_at_95"], 1.0 - (dd - k["plateau_to"]) * slope))


def runup_kernel(dd: float, ret_12m: float) -> float:
    """Run-up evidence in [0,1] (user 2026-07-08; shape in config.RUNUP_KERNEL):
    max(proximity-to-2-3yr-high component, 12-mo appreciation component).
    Either is evidence the stock 'ran'; the interaction gate downstream
    decides whether the run rests on temporary success."""
    k = config.RUNUP_KERNEL
    comps = []
    if np.isfinite(dd):
        comps.append(float(np.clip(
            (k["dd_zero_beyond"] - dd) / (k["dd_zero_beyond"] - k["dd_full_within"]),
            0.0, 1.0)))
    if np.isfinite(ret_12m):
        comps.append(float(np.clip(
            (ret_12m - k["ret12_zero"]) / (k["ret12_full"] - k["ret12_zero"]),
            0.0, 1.0)))
    if not comps:
        return np.nan
    return float(max(comps))


def compression_kernel(slope: float, buffer_consumed: float,
                       spread_peak: float, crossing_soon: bool,
                       spread_now: float = np.nan) -> float:
    """ROIC-WACC spread-compression evidence in [0,1] (config.COMPRESSION).
    NaN when there was no real moat to lose (window-peak spread below
    min_peak_spread) so the fm_peaked max() excludes it. crossing_soon (spread
    already negative or about to cross WACC) is a confirmed rollover -> 1.0.
    Otherwise max of a buffer-consumed and a downward-slope component, GATED by
    how thin the CURRENT spread is (now_ceiling) — a spread that shrank from a
    high peak but is still large is not a false moat and earns ~0."""
    k = config.COMPRESSION
    if not np.isfinite(spread_peak) or spread_peak < k["min_peak_spread"]:
        return np.nan
    if crossing_soon:
        return 1.0
    comps = []
    if np.isfinite(buffer_consumed):
        comps.append(float(np.clip(
            (buffer_consumed - k["buffer_zero_below"])
            / (k["buffer_full_at"] - k["buffer_zero_below"]), 0.0, 1.0)))
    if np.isfinite(slope):
        comps.append(float(np.clip((-slope) / k["slope_full_bps_q"], 0.0, 1.0))
                     if slope < 0 else 0.0)
    if not comps:
        return np.nan
    score = max(comps)
    # current-spread gate: only reward compression when the live spread is
    # actually thin (near crossing WACC), not merely down from a high peak
    if np.isfinite(spread_now):
        score *= float(np.clip(
            (k["now_ceiling"] - spread_now) / k["now_ceiling"], 0.0, 1.0))
    return float(score)


def valuation_reset_row(dd: float, pct_above_low: float,
                        slope_3m: float) -> float:
    """Drawdown kernel + small price-confirming term (audit 4.4)."""
    base = drawdown_kernel(dd)
    if not np.isfinite(base):
        return np.nan
    confirm_parts = []
    if np.isfinite(pct_above_low):
        confirm_parts.append(min(1.0, max(0.0, pct_above_low / 0.30)))
    if np.isfinite(slope_3m):
        confirm_parts.append(1.0 if slope_3m > 0 else 0.0)
    confirm = float(np.mean(confirm_parts)) if confirm_parts else np.nan
    w = config.PRICE_CONFIRM_WEIGHT
    if np.isfinite(confirm):
        return float((1 - w) * base + w * confirm)
    return float(base)


# ---------------------------------------------------------------------------
# Peer machinery (short side margin-vs-peer; audit 5.3 / long 4.3 guard)
# ---------------------------------------------------------------------------

def peer_median(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Median of `col` within 3-digit-SIC peer group (fallback 2-digit),
    >= PEER_GROUP_MIN members, SIC 6xxx excluded from margin peer pools.
    Returns (peer_median_series, peer_count_series)."""
    sic = df["sic"].fillna(-1).astype(int)
    non_fin = ~sic.floordiv(1000).eq(6)
    med = pd.Series(np.nan, index=df.index)
    cnt = pd.Series(0, index=df.index)
    for digits in (10, 100):  # sic//10 = 3-digit group, sic//100 = 2-digit
        grp_key = sic.floordiv(digits)
        pool = df.loc[non_fin, col].groupby(grp_key[non_fin])
        med_map = pool.median()
        cnt_map = pool.count()
        cand_med = grp_key.map(med_map)
        cand_cnt = grp_key.map(cnt_map).fillna(0)
        use = med.isna() & (cand_cnt >= config.PEER_GROUP_MIN)
        med[use] = cand_med[use]
        cnt[use] = cand_cnt[use].astype(int)
    return med, cnt


def peer_cross_section(df: pd.DataFrame, col: str,
                       higher_is_better: bool = True,
                       min_members: int | None = None
                       ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """peer_median + the WITHIN-peer-group percentile rank of `col`, in one
    pass. Returns (within_peer_pct_rank, peer_median, peer_count) over the
    3-digit-SIC cohort (2-digit fallback), SIC-6xxx excluded. Names below
    min_members after both tiers stay NaN (never fabricated). Shared by the
    peer-disconnect and relative-share signals."""
    min_members = min_members or config.PEER_GROUP_MIN
    sic = df["sic"].fillna(-1).astype(int)
    non_fin = ~sic.floordiv(1000).eq(6)
    rank = pd.Series(np.nan, index=df.index)
    med = pd.Series(np.nan, index=df.index)
    cnt = pd.Series(0, index=df.index)
    for digits in (10, 100):
        grp_key = sic.floordiv(digits)
        pool = df.loc[non_fin, col].groupby(grp_key[non_fin])
        cand_med = grp_key.map(pool.median())
        cand_cnt = grp_key.map(pool.count()).fillna(0)
        rnk = pool.rank(pct=True) * 100.0
        if not higher_is_better:
            rnk = 100.0 - rnk
        cand_rank = rnk.reindex(df.index)
        use = med.isna() & (cand_cnt >= min_members)
        med[use] = cand_med[use]
        cnt[use] = cand_cnt[use].astype(int)
        rank[use] = cand_rank[use]
    return rank, med, cnt


# ---------------------------------------------------------------------------
# Composite assembly
# ---------------------------------------------------------------------------

def score_universe(df: pd.DataFrame) -> pd.DataFrame:
    """df: one row per company with RAW inputs (see pipeline.py for the
    column contract). Returns df with all sub-score/theme/composite columns
    appended. Raw columns are kept — every output row carries everything."""
    df = df.copy()

    zero_rev = df["zero_low_revenue"].fillna(False).astype(bool)
    is_fin = df["is_financial"].fillna(False).astype(bool)

    # Zero/low-revenue nulling (audit 5.5) — ALL sales-denominated inputs
    # out, both sides (margin-based moat rows and margin-vs-peak included).
    sales_denominated = ["ev_sales", "ev_normalized", "ev_ebitda",
                         "decel_score_raw", "nonop_share_score_raw",
                         "gm_latest", "om_latest", "dso_yoy_delta",
                         "dio_yoy_delta", "beneish_m", "gm_level",
                         "gm_stability", "margin_vs_peak", "pmv_score_raw"]
    for c in sales_denominated:
        if c in df:
            df.loc[zero_rev, c] = np.nan

    # capacity_decay is a PP&E-based proxy that FALSE-FIRES on (a) acquisition
    # PP&E jumps and (b) financials/REITs (deep-dive 2026-07-08: HNI=Steelcase,
    # DKS=Foot Locker were pure M&A artifacts). Null the tag there so the
    # capacity cohort + consumer/industrial cohorts stay clean.
    if "capacity_decay" in df:
        contaminated = df.get("ma_flag", pd.Series(False, index=df.index)) \
            .fillna(False).astype(bool) | is_fin
        df.loc[contaminated, "capacity_decay"] = False

    # pricing-masking-volume is a price-vs-volume construct on a unit-selling
    # business; a lender has neither, so the row is meaningless for financials
    # (34 of 103 pmv flag fires were financials, putting a price/volume
    # checklist on bank tear sheets). Same treatment capacity_decay gets above.
    for c in ("pmv_score_raw", "pricing_masking_volume", "ran_on_pricing_success"):
        if c in df:
            df.loc[is_fin, c] = (False if df[c].dtype == bool else np.nan)

    # ----------------- LONG: turnaround bucket -----------------
    # inflection theme: self-normalized [0,1] x100; op-margin-vs-peak folded
    df["s_inflection"] = df["inflection_score"] * 100.0
    df["s_margin_vs_peak"] = df["margin_vs_peak"] * 100.0
    infl_row, infl_cov = weighted_theme(
        df, {"s_inflection": 0.85, "s_margin_vs_peak": 0.15})
    df["cov_inflection"] = infl_cov
    # Thin-basis guard. Exclude-and-renormalize assumes the missing component
    # is missing at random; a NaN inflection_score means the opposite — the
    # inflection could not be measured at all (breadth 0). Without this the
    # 0.15-weight margin_vs_peak row carried the whole theme, and NRP reached
    # t_inflection 100 / LONG RANK #7 on it. Shrink toward the universe median
    # in proportion to the weight actually present, as the buckets do.
    if config.INFLECTION_THIN_BASIS_SHRINK:
        med = (float(np.nanmedian(infl_row.values))
               if infl_row.notna().any() else NEUTRAL)
        thin = infl_cov < 1.0
        infl_row = infl_row.where(
            ~thin, infl_cov * infl_row + (1.0 - infl_cov) * med)
    df["t_inflection"] = infl_row

    df["t_fscore"] = pct_rank(df["f_rising"])
    df["t_valuation_reset"] = df["valuation_reset"] * 100.0
    # squeeze: SI rank x inflection sub-score (multiplicative; audit 4.7)
    si_rank = pct_rank(df["si_ratio"])
    df["t_squeeze"] = si_rank * df["inflection_score"].clip(0, 1)
    # annual_only FPIs: cap yfinance-only momentum signals (audit 1.3)
    ann = df["annual_only"].fillna(False).astype(bool)
    df.loc[ann, "t_valuation_reset"] *= config.ANNUAL_ONLY_MOMENTUM_CAP
    df.loc[ann, "t_squeeze"] *= config.ANNUAL_ONLY_MOMENTUM_CAP

    turnaround, cov_turn = weighted_theme(df, {
        "t_inflection": config.TURNAROUND_THEMES["inflection"],
        "t_fscore": config.TURNAROUND_THEMES["fscore"],
        "t_valuation_reset": config.TURNAROUND_THEMES["valuation_reset"],
        "t_squeeze": config.TURNAROUND_THEMES["squeeze"],
    }, shrink=True)
    df["bucket_turnaround"] = turnaround
    df["cov_turnaround"] = cov_turn

    # ----------------- LONG: moat bucket -----------------
    # ROIC-WACC demonstrated level: absolute + sector-relative blend
    spread = df["moat_demonstrated"]
    abs_comp = (50.0 + 500.0 * spread).clip(0, 100)
    rel_comp = sector_rank(df, "moat_demonstrated")
    ab = config.MOAT_SPREAD_ABS_BLEND
    df["m_roic_spread"] = ab * abs_comp + (1 - ab) * rel_comp
    # confidence shrink for thin moat histories (audit 2.7)
    conf = (df["moat_n_obs"] / config.MIN_OBS_5YR).clip(0, 1)
    df["m_roic_spread"] = conf * df["m_roic_spread"] + (1 - conf) * NEUTRAL
    df.loc[spread.isna(), "m_roic_spread"] = np.nan

    gm_l = sector_rank(df, "gm_level")
    gm_s = sector_rank(df, "gm_stability")
    # nan-tolerant mean: one missing half must renormalize, not propagate
    df["m_gm"] = pd.concat([gm_l, gm_s], axis=1).mean(axis=1)
    # GM rows masked for banks/insurers (audit 3.1 applicability)
    bank_ins = df["fin_sector"].isin(["bank", "insurer"])
    df.loc[bank_ins, "m_gm"] = np.nan

    df["m_consistency"] = df["earnings_consistency"] * 100.0
    # moat valuation row: cheapness blend of EV/Sales AND EV/normalized-EBIT
    # (user 2026-07-08: a good business at a trough is cheap on MEAN-REVERTED
    # earnings even when optically expensive on trough earnings); P/TBV for
    # financials unchanged. nan-tolerant mean renormalizes a missing half.
    ev_cheap = pct_rank(df["ev_sales"], higher_is_better=False)
    evn_cheap = pct_rank(df["ev_normalized"], higher_is_better=False)
    nonfin_cheap = pd.concat([ev_cheap, evn_cheap], axis=1).mean(axis=1)
    ptbv_cheap = pct_rank(df["p_tbv"], higher_is_better=False)
    df["m_valuation"] = nonfin_cheap.where(~is_fin, ptbv_cheap)

    moat, cov_moat = weighted_theme(df, {
        "m_roic_spread": config.MOAT_THEMES["roic_spread"],
        "m_gm": config.MOAT_THEMES["gm_stability"],
        "m_consistency": config.MOAT_THEMES["consistency"],
        "m_valuation": config.MOAT_THEMES["valuation"],
    }, shrink=True)
    df["bucket_moat"] = moat
    df["cov_moat"] = cov_moat

    # renormalize over available buckets (permissive: a name with only one
    # computable bucket still surfaces rather than going NaN)
    long_comp, _ = weighted_theme(df, {
        "bucket_turnaround": config.LONG_TURNAROUND_WEIGHT,
        "bucket_moat": config.LONG_MOAT_WEIGHT,
    })
    df["long_composite"] = long_comp

    # ----------------- SHORT: false moat -----------------
    # pre-compute the Beneish percentile once (used both as the forensic
    # component and as the "active forensic flag" gate below)
    df["beneish_pct_pre"] = pct_rank(df["beneish_m"])
    # peaked row = max(confirmed rollover, at-peak SETUP at partial credit)
    # (user 2026-07-08: catch the VITL/CAKE setup before the rollover)
    at_peak = df["at_peak_score"].clip(0, 1) if "at_peak_score" in df \
        else pd.Series(np.nan, index=df.index)
    # spread-compression evidence folded into the peaked/rolling-over row via
    # max() (NOT a new weighted component — it correlates with peak_score and
    # leverage_propping.roic_falling; max combines without double-counting)
    if "spread_slope" in df:
        df["compression_score"] = df.apply(
            lambda r: compression_kernel(
                r.get("spread_slope", np.nan), r.get("buffer_consumed", np.nan),
                r.get("spread_peak_window", np.nan),
                bool(r.get("crossing_soon", False)),
                r.get("spread_now", np.nan)), axis=1)
    else:
        df["compression_score"] = np.nan
    df["fm_compression"] = (df["compression_score"] * 100.0
                            * config.COMPRESSION_FM_CREDIT)
    df["fm_peaked"] = pd.concat(
        [df["peak_score"] * 100.0,
         at_peak * 100.0 * config.AT_PEAK_FM_CREDIT,
         df["fm_compression"]], axis=1).max(axis=1)
    df.loc[df["peak_score"].isna() & at_peak.isna()
           & df["compression_score"].isna(), "fm_peaked"] = np.nan
    # margin-vs-peer INTERACTION term (audit 5.3): contributes only alongside
    # own deterioration or an active forensic flag; unconditioned high margin
    # contributes zero.
    peer_med, peer_cnt = peer_median(df, "om_latest")
    excess = df["om_latest"] - peer_med
    excess_rank = pct_rank(excess)
    # deterioration gate: the conjunctive peak score of a stable healthy
    # business floors around ~0.35 (component floors + neutral slope), so
    # rescale [gate_lo, gate_hi] -> [0,1] — otherwise "unconditioned high
    # margin contributes ZERO" (audit 5.3) would leak ~35% of the rank.
    # The at-peak SETUP also opens the gate (it is strict-product, no floor).
    g_lo, g_hi = config.PEER_INTERACTION_GATE
    deterioration = ((df["peak_score"] - g_lo) / (g_hi - g_lo)).clip(0, 1)
    forensic_active = ((df["beneish_pct_pre"].fillna(0) > 80)
                       | df["event_score_raw"].fillna(0).gt(0.4)).astype(float)
    interaction_gate = pd.concat([deterioration, at_peak, forensic_active],
                                 axis=1).max(axis=1)
    df["fm_margin_vs_peer"] = excess_rank * interaction_gate
    df.loc[peer_med.isna(), "fm_margin_vs_peer"] = np.nan
    df["peer_count"] = peer_cnt

    df["fm_leverage"] = df["leverage_prop_raw"] * 100.0

    false_moat_raw, cov_fm = weighted_theme(df, {
        "fm_peaked": config.FALSE_MOAT_COMPONENTS["peaked_rolling_over"],
        "fm_margin_vs_peer": config.FALSE_MOAT_COMPONENTS["margin_vs_peer_interaction"],
        "fm_leverage": config.FALSE_MOAT_COMPONENTS["leverage_propping"],
    })
    # RAW (pre-sector-haircut) score is what the audit-4.2 long tear-sheet
    # cross-check and the false-moat-collision tag consume.
    df["false_moat_raw"] = false_moat_raw
    df["cov_false_moat"] = cov_fm
    haircut = df["sector_margin_declining"].fillna(False).map(
        lambda x: config.SECTOR_DECLINE_HAIRCUT if x else 1.0)
    df["false_moat_scored"] = false_moat_raw * haircut

    # ----------------- SHORT: inflated growth -----------------
    df["ig_decel"] = df["decel_score_raw"] * 100.0
    # richness: EV/Sales pctile (universe-wide, spec) blended with
    # EV/normalized-EBIT pctile (mean-reverted earnings; user 2026-07-08)
    df["ig_ev_sales"] = pct_rank(df["ev_sales"])
    df["ig_ev_norm"] = pct_rank(df["ev_normalized"])
    df["ig_richness"] = pd.concat([df["ig_ev_sales"], df["ig_ev_norm"]],
                                  axis=1).mean(axis=1)
    # run-up row: the stock ran (kernel) x fundamental-peak evidence gate —
    # a compounding business at deserved highs scores ~0
    runup_raw = df.apply(
        lambda r: runup_kernel(r.get("drawdown", np.nan),
                               r.get("ret_12m", np.nan)), axis=1)
    # "temporary success" gate: at-peak margins, confirmed rollover, OR the
    # pricing-masking-volume signature (price-hike-driven success is
    # temporary success even when margins aren't at records — CAKE)
    g_lo2, g_hi2 = config.PEER_INTERACTION_GATE
    pmv_gate = df["pmv_score_raw"].clip(0, 1) if "pmv_score_raw" in df \
        else pd.Series(np.nan, index=df.index)
    peak_gate = pd.concat(
        [((df["peak_score"] - g_lo2) / (g_hi2 - g_lo2)).clip(0, 1),
         at_peak, pmv_gate], axis=1).max(axis=1)
    df["runup_raw"] = runup_raw
    df["ig_runup"] = runup_raw * peak_gate * 100.0
    df.loc[runup_raw.isna(), "ig_runup"] = np.nan
    # pricing-masking-volume row (the CAKE tell)
    df["ig_pmv"] = df["pmv_score_raw"] * 100.0
    dso_r = pct_rank(df["dso_yoy_delta"])
    dio_r = pct_rank(df["dio_yoy_delta"])
    df["ig_dso_dio"] = pd.concat([dso_r, dio_r], axis=1).mean(axis=1)
    df["ig_buybacks"] = df["buyback_score_raw"] * 100.0
    df["ig_dilution"] = pct_rank(df["dilution_yoy"])
    # expectations-vs-trajectory row: relative-share loss (own rev growth below
    # the peer-cohort median) + reverse-DCF margin-ceiling gap. Computed here so
    # rel_share_now is available both to this weighted row and to the
    # losing_share_priced_rich tag downstream. nan-tolerant mean (like ig_richness).
    if "rev_yoy_latest" in df:
        _, _peer_g, _ = peer_cross_section(
            df, "rev_yoy_latest", True, config.RELATIVE_SHARE["min_members"])
        df["rel_share_now"] = df["rev_yoy_latest"] - _peer_g
    else:
        df["rel_share_now"] = np.nan
    # expectations row = the reverse-DCF margin-ceiling gap ONLY. Relative-share
    # is deliberately NOT blended in: the CAKE/VITL replay (2026-07-08) showed
    # revenue-growth-vs-peers is wrong-sign for the pricing-masking-volume /
    # windfall archetype — windfall pricing inflates revenue so a share LOSER
    # screens as a gainer (VITL: +21% rev growth = apparent share gain), which
    # diluted the correct margin-ceiling signal (VITL 80th -> 49th pctile under
    # the mean). rel_share_now is kept (computed above) ONLY for the
    # richness-gated losing_share_priced_rich discovery tag, not the composite.
    df["ig_expectations"] = (pct_rank(df["margin_ceiling_gap"])
                             if "margin_ceiling_gap" in df
                             else pd.Series(np.nan, index=df.index))

    ig, cov_ig = weighted_theme(df, {
        "ig_decel": config.INFLATED_GROWTH_COMPONENTS["revenue_decel"],
        "ig_richness": config.INFLATED_GROWTH_COMPONENTS["richness"],
        "ig_runup": config.INFLATED_GROWTH_COMPONENTS["runup"],
        "ig_pmv": config.INFLATED_GROWTH_COMPONENTS["pricing_masking_volume"],
        "ig_dso_dio": config.INFLATED_GROWTH_COMPONENTS["dso_dio_expansion"],
        "ig_buybacks": config.INFLATED_GROWTH_COMPONENTS["debt_funded_buybacks"],
        "ig_dilution": config.INFLATED_GROWTH_COMPONENTS["dilution"],
        "ig_expectations": config.INFLATED_GROWTH_COMPONENTS["expectations"],
    })
    df["theme_inflated_growth"] = ig
    df["cov_inflated_growth"] = cov_ig

    # ----------------- SHORT: forensic (single theme, audit 3.3) ----------
    df["fo_beneish"] = df["beneish_pct_pre"]
    df["fo_sloan"] = pct_rank(df["sloan_latest"])
    df["fo_events"] = df["event_score_raw"] * 100.0
    # non-operating earnings share (CAKE QoE bridge); the operating-to-pretax
    # bridge is meaningless for financials -> null there (weighted_theme then
    # renormalizes over the remaining forensic rows)
    if "nonop_share_score_raw" in df:
        df["fo_nonop"] = df["nonop_share_score_raw"] * 100.0
        df.loc[is_fin, "fo_nonop"] = np.nan
    else:
        df["fo_nonop"] = np.nan
    fo, cov_fo = weighted_theme(df, {
        "fo_beneish": config.FORENSIC_COMPONENTS["beneish"],
        "fo_sloan": config.FORENSIC_COMPONENTS["sloan"],
        "fo_events": config.FORENSIC_COMPONENTS["event_flags"],
        "fo_nonop": config.FORENSIC_COMPONENTS["nonop_share"],
    })
    df["theme_forensic"] = fo
    df["cov_forensic"] = cov_fo

    # ----------------- SHORT: confirmation (additive bonus ONLY) ----------
    # user 2026-07-08: SI supports/points to a short but its ABSENCE is not
    # evidence against one ("the market hasn't priced it in yet") — so SI is
    # excluded from the weighted composite and applied as a pure bonus.
    # Missing SI -> 0 bonus, 0 penalty. The hump kernel still tapers the
    # bonus for crowded shorts.
    df["theme_confirmation"] = df["si_hump"] * 100.0  # display column
    df["si_bonus"] = df["si_hump"].fillna(0.0).clip(0, 1) \
        * config.SI_CONFIRMATION_BONUS

    short, cov_short = weighted_theme(df, {
        "false_moat_scored": config.SHORT_THEMES["false_moat"],
        "theme_inflated_growth": config.SHORT_THEMES["inflated_growth"],
        "theme_forensic": config.SHORT_THEMES["forensic"],
    }, shrink=True)
    df["cov_short"] = cov_short
    # conviction blend (user 2026-07-09): reward the TOP-K strongest on-thesis
    # signals across all themes so a name strong on ONE genuine mechanism isn't
    # diluted to mid-pack by the components it doesn't fire (absence of a signal
    # is not evidence against a short).
    grp = []
    for _g, cols in config.CONVICTION_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if present:
            grp.append(df[present].max(axis=1))          # strongest signal in the group
    if grp:
        G = np.column_stack([s.to_numpy(dtype=float) for s in grp])
        Gs = np.sort(np.where(np.isnan(G), -np.inf, G), axis=1)[:, ::-1]
        k = min(config.CONVICTION_TOPK, Gs.shape[1])
        topk = np.where(np.isneginf(Gs[:, :k]), np.nan, Gs[:, :k])
        # rows firing zero groups are all-NaN -> nanmean warns "empty slice";
        # errstate(invalid) doesn't catch that RuntimeWarning, so silence it.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df["conviction"] = np.nanmean(topk, axis=1)   # mean of top-K distinct mechanisms
    else:
        df["conviction"] = np.nan
    w = config.CONVICTION_BLEND
    conv = df["conviction"].where(df["conviction"].notna(), short)
    # the composite as it stands BEFORE the additive bonuses — this is what
    # the shortlist threshold is actually applied to
    pre_bonus = (1 - w) * short + w * conv
    df["short_pre_bonus"] = pre_bonus

    # catalyst-timing bonus: additive, gated on the PRE-BONUS composite already
    # firing (timing only re-prioritizes an existing short, never manufactures
    # one) and ramped by proximity to the projected next print. Mirrors the
    # si_bonus additive-only pattern (missing -> 0 bonus, 0 penalty).
    #
    # This used to gate on `short` — the pre-BLEND theme mean — which is a
    # different and much harsher quantity than the composite the threshold is
    # applied to: the conviction blend exists precisely because the theme mean
    # dilutes a name strong on one mechanism. The result was that 649 of 674
    # names with a live catalyst got zero timing credit and the largest bonus
    # in the run was 3.33 of a nominal 8. Timing is the modal judge question,
    # so the best-evidenced timing signal in the build was effectively off.
    # Still strictly pre-bonus, so no short is manufactured by its own timing.
    if "days_to_catalyst" in df:
        dtc = df["days_to_catalyst"]
        win = config.CATALYST_WINDOW_DAYS
        prox = ((win - dtc) / win).clip(lower=0.0, upper=1.0)
        prox = prox.where(dtc.notna(), 0.0)
        guide = df.get("guide_reset_window",
                       pd.Series(False, index=df.index)).fillna(False).astype(float)
        fired = (pre_bonus >= config.SHORTLIST_SCORE_THRESHOLD)
        df["catalyst_bonus"] = fired.astype(float) * prox * (
            config.CATALYST_BONUS_MAX + config.CATALYST_GUIDE_RESET_BONUS * guide)
        df["catalyst_imminent"] = fired & dtc.notna() & (dtc <= win)
    else:
        df["catalyst_bonus"] = 0.0
        df["catalyst_imminent"] = False

    df["short_composite"] = (pre_bonus + df["si_bonus"]
                             + df["catalyst_bonus"])

    # ----------------- coverage, tags -----------------
    # (the 2026-07-07 commodity x0.85 discount was rescinded by the user on
    # 2026-07-08 — the `is_commodity` tag remains on every row; config's
    # COMMODITY_DISCOUNT is 1.0 and no multiplier is applied)

    df["cov_long"] = (config.LONG_TURNAROUND_WEIGHT * df["cov_turnaround"].fillna(0)
                      + config.LONG_MOAT_WEIGHT * df["cov_moat"].fillna(0))
    df["low_confidence"] = (df["cov_long"] < config.LOW_CONFIDENCE_COVERAGE) \
        | (df["cov_short"].fillna(0) < config.LOW_CONFIDENCE_COVERAGE)

    # cyclical tag: high full-history margin volatility (reuses stability calc)
    vol = -df["gm_stability"]  # stability is negative stdev
    df["cyclical"] = vol > vol.quantile(0.80)

    # sector-driven tag: name's inflection matches sector-group median move
    sec_med = df.groupby("sector2")["inflection_score"].transform("median")
    sec_n = df.groupby("sector2")["inflection_score"].transform("count")
    df["sector_driven"] = ((sec_med > 0.45)
                           & ((df["inflection_score"] - sec_med).abs() < 0.15)
                           & (sec_n >= config.PEER_GROUP_MIN))

    # secular/industry-decline tag (audit 4.3 second half): the name's
    # op-margin change since its own peak vs the sector-group median of the
    # same quantity — an industry that fell in lockstep is not a stumble.
    if "om_change_since_peak" in df:
        chg = df["om_change_since_peak"]
        sec_chg = df.groupby("sector2")["om_change_since_peak"].transform("median")
        sec_chg_n = df.groupby("sector2")["om_change_since_peak"].transform("count")
        df["no_peer_data"] = sec_chg_n < config.PEER_MIN_MEMBERS
        df["secular_decline"] = (~df["no_peer_data"]
                                 & (chg < -0.03) & (sec_chg < -0.03)
                                 & ((chg - sec_chg).abs() < 0.05))
    else:
        df["no_peer_data"] = True
        df["secular_decline"] = False

    # story-stock short score, ZERO-REV SEGMENT ONLY (user 2026-07-09, ASTS
    # exemplar): run-up gated by dilution evidence, issuance, P/TBV richness
    # within the segment, accruals; + the same additive SI bonus. The main
    # composite is untouched (audit 5.5's segmentation stands).
    df["story_short_score"] = np.nan
    zr_mask = zero_rev.copy()
    # guards (ASTS calibration): no ETF/trust entities, no fresh IPOs (a
    # 12-mo "run" off an IPO base is meaningless), and dilution must be
    # computable — issuance is the core of the story-stock archetype
    if "bdc_or_trust" in df:
        zr_mask &= ~df["bdc_or_trust"].fillna(False).astype(bool)
    if "history_quarters" in df:
        zr_mask &= df["history_quarters"].fillna(0) >= 12
    zr_idx = df.index[zr_mask]
    if len(zr_idx) >= 8:
        sub = df.loc[zr_idx]
        w = config.STORY_SHORT_WEIGHTS
        dil_rank = pct_rank(sub["dilution_yoy"])          # within segment
        ptbv_rank = pct_rank(sub["p_tbv"])
        sloan_rank = pct_rank(sub["sloan_latest"])
        runup_gated = sub["runup_raw"].clip(0, 1) * (dil_rank / 100.0)
        comps = pd.DataFrame({
            "runup_gated": runup_gated * 100.0,
            "dilution": dil_rank,
            "ptbv_rich": ptbv_rank,
            "sloan": sloan_rank,
        })
        acc = pd.Series(0.0, index=sub.index)
        wsum = pd.Series(0.0, index=sub.index)
        for c, wt in w.items():
            m = comps[c].notna()
            acc[m] += comps.loc[m, c] * wt
            wsum[m] += wt
        score = acc / wsum.replace(0.0, np.nan)
        score[sub["dilution_yoy"].isna()] = np.nan  # dilution is required
        df.loc[zr_idx, "story_short_score"] = score + df.loc[zr_idx, "si_bonus"]

    # multiple-disconnect tag (user 2026-07-09): cyclical/windfall-driven
    # earnings wearing a NON-cyclical multiple. A cyclical priced as a
    # cyclical (CENX at 2x sales, BLBD at 1.5x) is not the archetype; a
    # story priced for durability (VITL, UAMY at 27x sales) is.
    md = config.MULTIPLE_DISCONNECT
    g_lo3, g_hi3 = config.PEER_INTERACTION_GATE
    evidence = pd.concat([
        df["at_peak_score"].clip(0, 1) if "at_peak_score" in df
        else pd.Series(np.nan, index=df.index),
        df["pmv_score_raw"].clip(0, 1) if "pmv_score_raw" in df
        else pd.Series(np.nan, index=df.index),
        ((df["peak_score"] - g_lo3) / (g_hi3 - g_lo3)).clip(0, 1),
        df.get("rev_gp_divergence", pd.Series(False, index=df.index))
          .fillna(False).astype(float) * md["divergence_evidence"],
    ], axis=1).max(axis=1)
    df["disconnect_evidence"] = evidence
    # The richness leg ranks EV/Sales WITHIN the coarse sector, not across the
    # whole universe. Universe-wide, the top EV/S decile is structurally
    # software and pharma, so this gate quietly turned a general
    # "cyclical earnings on a durable multiple" tag into a tech/pharma tag and
    # excluded the consumer names the archetype is actually pitched on — CAKE
    # cleared the mechanism evidence (0.628) and was killed by this leg alone.
    # `ig_ev_sales` (universe-wide) is deliberately left untouched because it
    # feeds the SCORED `ig_richness` row; this tag carries no composite weight,
    # so re-basing only the gate cannot move any composite.
    df["ev_sales_sector_pctl"] = sector_rank(df, "ev_sales")
    df["multiple_disconnect"] = (
        (evidence >= md["evidence_min"])
        & (df["ev_sales_sector_pctl"].fillna(0) >= md["multiple_min"]))
    df["disconnect_strength"] = np.where(
        df["multiple_disconnect"],
        evidence * df["ev_sales_sector_pctl"].fillna(0), np.nan)

    # directional PEER multiple-disconnect (CAKE 9.8x vs declining peers): rich
    # vs the peer cohort AND worse trajectory than those same peers. Discovery
    # tag — correlates with ig_ev_sales/ig_decel already weighted in
    # inflated_growth, so it carries no composite weight.
    pdc = config.PEER_DISCONNECT
    mmin = pdc["min_members"]
    val_pctl, evs_med, pcnt = peer_cross_section(df, "ev_sales", True, mmin)
    traj_parts = [peer_cross_section(df, "decel_score_raw", False, mmin)[0]]
    if "om_ttm_delta_4q" in df:
        traj_parts.insert(0, peer_cross_section(df, "om_ttm_delta_4q",
                                                True, mmin)[0])
    traj_pctl = pd.concat(traj_parts, axis=1).mean(axis=1)  # nan-tolerant
    if "ev_ebitda" in df:
        eveb_pctl = peer_cross_section(df, "ev_ebitda", True, mmin)[0]
    else:
        eveb_pctl = pd.Series(np.nan, index=df.index)
    df["peer_val_pctl"] = val_pctl
    df["peer_traj_pctl"] = traj_pctl
    df["peer_disconnect_count"] = pcnt
    df["peer_disconnect_strength"] = val_pctl * (1 - traj_pctl / 100.0)
    eveb_ok = eveb_pctl.isna() | (eveb_pctl >= pdc["confirm_ev_ebitda_pctl"])
    df["peer_multiple_disconnect"] = (
        (val_pctl >= pdc["val_pctl_min"])
        & (traj_pctl <= pdc["traj_pctl_max"])
        & (df["peer_disconnect_strength"] >= pdc["strength_min"])
        & eveb_ok & (pcnt >= mmin)).fillna(False)
    df["peer_rerating_downside_pct"] = evs_med / df["ev_sales"] - 1.0
    df.loc[val_pctl.isna() | df["ev_sales"].isna(),
           "peer_rerating_downside_pct"] = np.nan

    # relative-share tag (VITL): losing share to the peer cohort while priced
    # rich = a mispriced-durability short. Reuses rel_share_now computed in the
    # inflated_growth block (also a weighted row there). The fade-slope variant
    # is a follow-up.
    rsc = config.RELATIVE_SHARE
    if "rel_share_now" in df:
        df["losing_share_priced_rich"] = (
            (df["rel_share_now"] < 0)
            & (df["ig_richness"].fillna(0) >= rsc["rich_min"]))
        df["share_durability_strength"] = np.where(
            df["losing_share_priced_rich"],
            (-df["rel_share_now"]).clip(0, rsc["gap_cap"])
            * df["ig_richness"].fillna(0) / 100.0, np.nan)
    else:
        df["losing_share_priced_rich"] = False
        df["share_durability_strength"] = np.nan

    # "ran-on-temporary-success" archetype tag (user 2026-07-08): the full
    # CAKE/VITL signature regardless of composite rank — discovery cohort
    ar = config.ARCHETYPE_ROTS
    temp_success = (df["pmv_score_raw"].fillna(0) >= ar["pmv_min"]) \
        | (df["at_peak_score"].fillna(0) >= ar["at_peak_min"])
    df["archetype_ran_on_temp_success"] = (
        temp_success
        & (df["runup_raw"].fillna(0) >= ar["runup_min"])
        & (df["ig_ev_norm"].fillna(0) >= ar["ev_norm_rich_min"]))

    # ran-on-pricing-success cohort (2026-07-08, CAKE replay gap): the mechanism
    # setup (pricing-masking-volume / at-peak + stock ran) WITHOUT the
    # normalized-EBIT richness gate. CAKE @2026-04-10 fired pmv 0.63 + runup 0.91
    # but screened ig_ev_norm 53 (<70) — its expensiveness is FORWARD (margin
    # collapse to come) and peer-EV/EBITDA-based, neither visible in filed
    # normalized earnings — so it missed the richer archetype cohort. This is a
    # mechanism-only discovery list; valuation is the manual next step (a
    # superset of archetype_ran_on_temp_success — the ig_ev_norm column shows
    # which members also screen rich).
    _pr_ev = (pd.concat([df["pmv_score_raw"].fillna(0),
                         df["at_peak_score"].fillna(0)], axis=1).max(axis=1)
              if "at_peak_score" in df else df["pmv_score_raw"].fillna(0))
    # exclude financials: the pmv/margin mechanism is meaningless for banks/
    # insurers (BOKF/CBSH-type leakage) — user 2026-07-08
    df["ran_on_pricing_success"] = (
        temp_success & (df["runup_raw"].fillna(0) >= ar["runup_min"]) & ~is_fin)
    df["ran_on_pricing_strength"] = np.where(
        df["ran_on_pricing_success"],
        _pr_ev * df["runup_raw"].fillna(0), np.nan)

    # reverse-DCF "priced for impossible growth" cohort (CAKE move 6 / VITL
    # move 5): EV implies a revenue CAGR well above the name's own trailing
    # CAGR AND a steady-state op margin above its own historical peak, with
    # corroborating EV/normalized-EBIT richness. DISCOVERY TAG only — reads
    # pipeline-persisted rdcf columns, adds no weighted term (would double-count
    # ig_decel/ig_richness). Missing rdcf -> fillna sentinel -> tag False.
    rd = config.REVERSE_DCF_TAG
    if "rdcf_growth_gap" in df:
        df["priced_for_impossible_growth"] = (
            (df["rdcf_growth_gap"].fillna(-9) >= rd["growth_gap_min"])
            & (df["margin_ceiling_gap"].fillna(-9) >= rd["margin_ceiling_gap_min"])
            & (df["ig_ev_norm"].fillna(0) >= rd["ev_norm_rich_min"]))
        df["rdcf_disconnect_strength"] = np.where(
            df["priced_for_impossible_growth"],
            df["rdcf_growth_gap"].fillna(0) * (df["ig_ev_norm"].fillna(0) / 100.0),
            np.nan)
    else:
        df["priced_for_impossible_growth"] = False
        df["rdcf_disconnect_strength"] = np.nan

    # --- cohort sort keys that could not rank (2026-07-25) -----------------
    # A discovery cohort is only as useful as its ordering: the reader works
    # down the list. Two of the six sort keys could not order their cohorts.
    # Both fixes are sort keys ONLY — the tags themselves are unchanged, and
    # these cohorts carry no composite weight, so nothing is re-scored.
    #
    # capacity_score_raw = clip(-organic / organic_full, 0, 1) — a HARD CLIP.
    # Every name whose organic decay exceeds the full-credit threshold lands on
    # exactly 1.0: 22 of 80 members (28%) tied at the max, so the cohort's worst
    # offenders were unorderable and fell through to an arbitrary tie-break.
    # Rank on the UNCLIPPED magnitude instead; the clipped score still gates.
    if "capacity_organic_proxy" in df:
        organic = pd.to_numeric(df["capacity_organic_proxy"], errors="coerce")
        # ...but an unbounded magnitude just surfaces accounting discontinuities
        # at the TOP of the list (RJET: net PP&E grew 6,558pp more than revenue,
        # ma_flag False). Null those rather than rank them — the clip used to
        # hide them among the tied names, which is not the same as handling them.
        implausible = organic < -config.CAPACITY_ORGANIC_IMPLAUSIBLE
        df.loc[implausible, "capacity_decay"] = False
        df["capacity_decay_strength"] = np.where(
            df["capacity_decay"].fillna(False).astype(bool), -organic, np.nan)
    else:
        df["capacity_decay_strength"] = np.nan

    # ppi_zscore comes from the 2-digit-SIC crosswalk, so it is a SECTOR
    # constant: all 61 cohort members shared just TWO distinct values (52 in
    # SIC 13, 9 in SIC 29) and the list was effectively unordered. Rank on the
    # COMPANY-level windfall — the share of revenue growth attributable to the
    # price move (nominal YoY minus PPI-deflated YoY), scaled by how stretched
    # the index is. Same thesis, measured per name instead of per sector.
    if "real_rev_growth" in df and "rev_yoy_latest" in df:
        price_driven = (pd.to_numeric(df["rev_yoy_latest"], errors="coerce")
                        - pd.to_numeric(df["real_rev_growth"], errors="coerce"))
        df["ppi_price_driven_growth"] = price_driven
        df["ppi_windfall_strength"] = np.where(
            df.get("ppi_windfall", pd.Series(False, index=df.index))
              .fillna(False).astype(bool),
            pd.to_numeric(df.get("ppi_zscore"), errors="coerce")
            * price_driven.clip(lower=0.0), np.nan)
    else:
        df["ppi_price_driven_growth"] = np.nan
        df["ppi_windfall_strength"] = np.nan

    # false-moat-collision tag (audit 4.2): top-quartile demonstrated-moat
    # rank AND top-quartile RAW false-moat theme, both vs snapshot universe
    # (financials ranked within their own section), suppressed below the
    # low-confidence coverage cutoff.
    df["false_moat_collision"] = False
    for seg in (True, False):
        seg_mask = is_fin == seg
        sub = df[seg_mask]
        if len(sub) < 8:
            continue
        moat_rank = pct_rank(sub["moat_demonstrated"])
        fm_rank = pct_rank(sub["false_moat_raw"])
        q = config.COLLISION_QUARTILE * 100
        collide = (moat_rank >= q) & (fm_rank >= q) \
            & (sub["cov_false_moat"].fillna(0) >= config.COLLISION_MIN_COVERAGE)
        df.loc[collide[collide].index, "false_moat_collision"] = True

    return df
