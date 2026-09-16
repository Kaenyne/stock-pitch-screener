"""Shortlist assembly & outputs (spec: "surface MORE, not fewer").

- shortlist = all names clearing a tunable soft score threshold, hard-capped
  at ~100 per side; soft per-sector cap ~12-15 with spill to next-ranked
  other-sector names; capped names remain visible in the full ranked CSV
- flags: `contested` (top decile of BOTH composites), `on-both-lists`
  (clears the threshold on both sides, threshold-based PRE-cap — audit 4.7),
  `false_moat_collision` (theme-level, audit 4.2 — set in scoring.py)
- financials/REITs and the zero/low-revenue cohort are ranked in their own
  segmented sections; every row carries BOTH composites + all sub-scores
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _stable_sort(df: pd.DataFrame, by: list[str],
                 ascending: list[bool] | None = None) -> pd.DataFrame:
    """Sort with a deterministic tie-break chain.

    Several cohort strength columns SATURATE — `capacity_score_raw` pins 26 of
    83 names at exactly 1.0 — so a single-key sort leaves the order inside the
    tied block decided by input row order. Re-scoring identical data then
    reshuffles the cohort (VITL moved rank 25 -> 8 that way), which makes a
    before/after diff unreadable. Appending short_composite then ticker makes
    every cohort ranking reproducible."""
    by = list(by)
    ascending = list(ascending) if ascending is not None else [False] * len(by)
    for col, asc in (("short_composite", False), ("ticker", True)):
        if col in df.columns and col not in by:
            by.append(col)
            ascending.append(asc)
    return df.sort_values(by, ascending=ascending, kind="mergesort")


def add_overlap_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lt = df["long_composite"].quantile(config.CONTESTED_DECILE)
    st = df["short_composite"].quantile(config.CONTESTED_DECILE)
    df["contested"] = (df["long_composite"] >= lt) & (df["short_composite"] >= st)
    thr = config.SHORTLIST_SCORE_THRESHOLD
    df["on_both_lists"] = (df["long_composite"] >= thr) \
        & (df["short_composite"] >= thr)
    return df


def _sector_capped(sub: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Soft per-sector cap with spill: walk down the ranking; a name in a
    sector already at cap is skipped (spilled over by the next-ranked
    other-sector name). Capped names simply don't make the shortlist — they
    remain visible in the full ranked CSV, which is always emitted."""
    counts: dict = {}
    keep_idx = []
    for idx, row in _stable_sort(sub, [score_col]).iterrows():
        sec = row.get("sector2", -1)
        if counts.get(sec, 0) >= config.SECTOR_CAP_PER_SIDE:
            continue
        counts[sec] = counts.get(sec, 0) + 1
        keep_idx.append(idx)
        if len(keep_idx) >= config.SHORTLIST_HARD_CAP:
            break
    return sub.loc[keep_idx].copy()


def assemble(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """side in {'long','short'}. Returns the shortlist (rank order)."""
    score_col = f"{side}_composite"
    thr = config.SHORTLIST_SCORE_THRESHOLD
    elig = df[df[score_col] >= thr].copy()
    elig = _stable_sort(elig, [score_col])
    return _sector_capped(elig, score_col)


def rank_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Full ranked universe + shortlists + segmented sections."""
    df = add_overlap_flags(df)
    zero_rev = df["zero_low_revenue"].fillna(False).astype(bool)
    is_fin = df["is_financial"].fillna(False).astype(bool)

    core = df[~is_fin & ~zero_rev]
    fin = df[is_fin & ~zero_rev]
    zr = df[zero_rev]

    out = {
        "universe_ranked": _stable_sort(df, ["long_composite"]),
        "long_shortlist": assemble(core, "long"),
        "short_shortlist": assemble(core, "short"),
        "long_financials": assemble(fin, "long") if len(fin) else fin,
        "short_financials": assemble(fin, "short") if len(fin) else fin,
        # zero-rev segment ranks by the STORY-STOCK score (user 2026-07-09,
        # ASTS exemplar) — its main-composite themes are nulled by design
        "segment_zero_revenue": _stable_sort(zr, [
            "story_short_score" if "story_short_score" in zr.columns
            else "short_composite"]),
    }
    # "ran-on-temporary-success" discovery cohort (user 2026-07-08): the
    # full CAKE/VITL signature regardless of composite rank, sorted by
    # signature strength (pmv/at-peak evidence x normalized richness)
    if "archetype_ran_on_temp_success" in df:
        cohort = df[df["archetype_ran_on_temp_success"].fillna(False)].copy()
        if len(cohort):
            evidence = cohort[["pmv_score_raw", "at_peak_score"]].max(axis=1)
            cohort["signature_strength"] = (
                evidence.fillna(0) * cohort["ig_ev_norm"].fillna(0))
            out["archetype_ran_on_temp_success"] = _stable_sort(
                cohort, ["signature_strength"])
    # additional pitch-derived discovery cohorts (2026-07-08): each surfaces
    # its archetype as its OWN list regardless of composite rank, sorted by
    # signature strength — so a mid-composite VITL/CAKE-shape name still surfaces
    for tag, strength in [
            ("priced_for_impossible_growth", "rdcf_disconnect_strength"),
            ("peer_multiple_disconnect", "peer_disconnect_strength"),
            ("losing_share_priced_rich", "share_durability_strength"),
            ("ran_on_pricing_success", "ran_on_pricing_strength"),
            # both re-pointed 2026-07-25: capacity_score_raw is clipped at 1.0
            # (28% of the cohort tied there) and ppi_zscore is a SECTOR
            # constant (61 members, 2 distinct values) — neither could order
            # its own cohort. See the sort-key block in scoring.py.
            ("capacity_decay", "capacity_decay_strength"),
            ("ppi_windfall", "ppi_windfall_strength")]:
        if tag in df:
            c = df[df[tag].fillna(False)].copy()
            if len(c):
                sort_col = strength if strength in c.columns else "short_composite"
                out[tag] = _stable_sort(c, [sort_col])

    # sector/attribute short cohorts (user pitch-preference 2026-07-08): filter
    # by sector/attribute, rank by ON-THESIS tag count (not the forensic-heavy
    # composite) so simple consumer / commodity-disconnect / industrial shorts
    # surface even when the composite buries them.
    sic2 = df["sic"].fillna(-1).astype(float) // 100
    for key, mask, tags in [
            ("consumer_shorts", sic2.isin(config.CONSUMER_SICS),
             config.ON_THESIS_SHORT_TAGS),
            ("industrials_shorts", sic2.isin(config.INDUSTRIAL_SICS),
             config.ON_THESIS_SHORT_TAGS),
            ("commodity_disconnect_shorts", df["is_commodity"].fillna(False),
             config.COMMODITY_DISCONNECT_TAGS)]:
        coh = tag_filtered_cohort(df, mask, tags)
        if len(coh):
            out[key] = coh

    # pitchable_shorts (user 2026-07-09): the conviction-ranked short list with
    # the hard-to-pitch sectors removed (semis + pharma/biotech), capped, with a
    # high-confidence top tier. This is the "30-40 names, 3-5 high confidence"
    # deliverable — the composite still surfaces everything; this is the curated
    # pitchable view.
    sic = df["sic"].fillna(-1)
    pitch = df[(~df["is_financial"].fillna(False))
               & (~df["zero_low_revenue"].fillna(False))
               & (~sic.isin(config.PITCHABLE_EXCLUDE_SICS))].copy()
    if len(pitch):
        pitch = _stable_sort(pitch, ["short_composite"]) \
            .head(config.PITCHABLE_CAP)
        tagcols = [t for t in config.ON_THESIS_SHORT_TAGS if t in pitch.columns]
        pitch["n_thesis_tags"] = (pitch[tagcols].fillna(False).astype(bool)
                                  .sum(axis=1) if tagcols else 0)
        pitch["high_confidence"] = pitch["n_thesis_tags"] >= config.PITCHABLE_HICONF_TAGS
        out["pitchable_shorts"] = pitch

    # long-side discovery cohorts — additive CSVs, no composite pre-cut; the
    # long side otherwise has exactly one surfacing route (long_composite)
    out.update(long_cohorts(df))
    pl = pitchable_longs(df)
    if len(pl):
        out["pitchable_longs"] = pl
    return out


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric view of a column, NaN if absent (CSV round-trips make object
    dtypes; every cohort leg must be NaN-safe rather than raising)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def long_cohorts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long-side discovery cohorts — the fix for the one-route asymmetry.

    Each surfaces a long archetype ranked on its OWN evidence with no
    composite pre-cut, mirroring what the short side already has. Purely
    additive: no score, weight or existing ranking is touched, so these
    cannot move an exemplar or a shortlist.

    Every leg is NaN-safe and NaN never counts as passing — a name missing the
    evidence is absent from the cohort, never silently included.
    """
    cfg = config.LONG_COHORTS
    out: dict[str, pd.DataFrame] = {}

    def _emit(name: str, coh: pd.DataFrame, by: list[str]) -> None:
        """Rank, cap, and record how many names were dropped by the cap.
        `inflecting_thin_moat` in particular is a RANKED list (~1.1k names
        clear its floor), not a filter — a reader must be able to see that
        the file is a top-N view rather than the whole cohort."""
        if not len(coh):
            return
        ranked = _stable_sort(coh, by)
        ranked["n_qualified"] = len(ranked)
        ranked["capped_at"] = config.LONG_COHORT_CAP
        out[name] = ranked.head(config.LONG_COHORT_CAP)
    core = df[~df["is_financial"].fillna(False).astype(bool)
              & ~df["zero_low_revenue"].fillna(False).astype(bool)].copy()
    if not len(core):
        return out
    light = core.get("surv_traffic_light", pd.Series("", index=core.index)) \
        .astype(str).str.lower()

    val_rank, dd = _num(core, "m_valuation"), _num(core, "drawdown")

    # Each block is opt-in on its config key, so a cohort can be disabled by
    # removing its entry without the run raising.

    # --- derated_compounder: quality intact, price de-rated ---------------
    c = cfg.get("derated_compounder")
    if c:
        moat_rank = _num(core, "m_roic_spread")
        demo, trail, now = (_num(core, "moat_demonstrated"),
                            _num(core, "moat_trailing_5y"),
                            _num(core, "spread_now"))
        # retention is only meaningful against a POSITIVE demonstrated spread:
        # a ratio of two negatives would read as "moat retained" nonsensically
        retention = (trail / demo.where(demo > 0)).where(demo > 0)
        economics_intact = (now > 0).fillna(False) | \
            (retention >= c["retention_min"]).fillna(False)
        derated = (dd >= c["drawdown_min"]).fillna(False) | \
            (val_rank >= c["valuation_rank_min"]).fillna(False)
        mask = (moat_rank >= c["moat_rank_min"]).fillna(False) \
            & economics_intact & derated
        coh = core[mask].copy()
        if len(coh):
            # quality x cheapness — both already 0-100 percentiles, and the
            # whole thesis as one number a judge can be walked through
            coh["derated_compounder_strength"] = (
                _num(coh, "m_roic_spread") / 100.0
                * _num(coh, "m_valuation").fillna(0.0) / 100.0)
            coh["moat_retention"] = retention.reindex(coh.index)
            _emit("derated_compounder", coh, ["derated_compounder_strength"])

    # --- inflecting_thin_moat: the fix is real, the moat is ordinary ------
    c = cfg.get("inflecting_thin_moat")
    if c:
        infl = _num(core, "inflection_score")
        mask = (infl >= c["min_inflection"]).fillna(False) \
            & ~light.isin(c["exclude_traffic_light"])
        _emit("inflecting_thin_moat", core[mask].copy(), ["inflection_score"])

    # --- surviving_distressed_value: cheap, beaten down, DEMONSTRABLY solvent
    c = cfg.get("surviving_distressed_value")
    if c:
        f = _num(core, "f_score")
        mask = (dd.between(c["drawdown_lo"], c["drawdown_hi"])).fillna(False) \
            & (val_rank >= c["valuation_rank_min"]).fillna(False) \
            & (light == c["require_traffic_light"]) \
            & (f >= c["f_score_min"]).fillna(False)
        coh = core[mask].copy()
        if len(coh):
            coh["distressed_value_strength"] = (
                _num(coh, "m_valuation") / 100.0 * _num(coh, "f_score") / 9.0)
            _emit("surviving_distressed_value", coh,
                  ["distressed_value_strength"])
    return out


def moat_rank_effective(df: pd.DataFrame) -> pd.Series:
    """Moat rank that survives a short filing history.

    `m_roic_spread` ranks `moat_demonstrated`, which needs 12 quarters inside a
    28-quarter window — so it is NaN for 36% of core names whose economics are
    perfectly well measured. LII carries spread_now = +0.41 (a 41-point
    ROIC-WACC spread) and still fails every moat gate in the build. Fall back
    to trailing, then current, spread and rank that.

    COVERAGE FIX, NOT A CALIBRATION ONE — used only by cohort gates, never by
    the composite, so it cannot move a score.
    """
    from .scoring import pct_rank
    rank = _num(df, "m_roic_spread")
    proxy = (_num(df, "moat_demonstrated")
             .fillna(_num(df, "moat_trailing_5y"))
             .fillna(_num(df, "spread_now")))
    filled = pct_rank(proxy)
    return rank.where(rank.notna(), filled)


def pitchable_longs(df: pd.DataFrame) -> pd.DataFrame:
    """The long mirror of `pitchable_shorts`: a good business at a temporary
    trough, in a story a judge can follow without a primer.

    Selection is on PITCHABILITY, not on realized return — those are different
    objectives and conflating them picks the wrong name (see the APH/BIRK note
    in config.LONG_PITCHABILITY).
    """
    c = config.LONG_PITCHABILITY
    core = df[~df["is_financial"].fillna(False).astype(bool)
              & ~df["zero_low_revenue"].fillna(False).astype(bool)].copy()
    sic = core["sic"].fillna(-1)
    core = core[~sic.isin(config.LONG_PITCHABLE_EXCLUDE_SICS)]
    if not len(core):
        return core

    moat = moat_rank_effective(core)
    now, demo, trail = (_num(core, "spread_now"), _num(core, "moat_demonstrated"),
                        _num(core, "moat_trailing_5y"))
    retention = (trail / demo.where(demo > 0)).where(demo > 0)
    economics_intact = (now > 0).fillna(False) | (retention >= 0.60).fillna(False)

    # --- story ARC: drawdown into a business whose economics still work ------
    arc = ((_num(core, "drawdown") >= c["min_drawdown"]).fillna(False)
           & (moat >= c["min_moat_rank"]).fillna(False)
           & economics_intact)
    core = core[arc].copy()
    if not len(core):
        return core

    # --- story QUALITY: the legs that separate a BIRK from an APH -----------
    sic2 = core["sic"].fillna(-1).astype(float) // 100
    segs = _num(core, "reportable_segments_display")
    analysts = _num(core, "n_analysts_display")
    core["story_consumer"] = sic2.isin(config.CONSUMER_SICS)
    # NaN segments are unknown, not simple — do not award the point on absence
    core["story_simple"] = (segs <= c["max_segments"]).fillna(False)
    core["story_known"] = analysts.between(c["analyst_lo"], c["analyst_hi"]).fillna(False)
    core["story_score"] = (core["story_consumer"].astype(int)
                           + core["story_simple"].astype(int)
                           + core["story_known"].astype(int))
    core["moat_rank_effective"] = moat.reindex(core.index)
    # rank by how pitchable the story is, then by the strength of the setup
    core["turnaround_evidence"] = _num(core, "t_inflection").fillna(0.0)
    return _stable_sort(core, ["story_score", "turnaround_evidence"]) \
        .head(config.LONG_PITCHABLE_CAP)


def tag_filtered_cohort(df: pd.DataFrame, mask: pd.Series,
                        tags: list[str]) -> pd.DataFrame:
    """Core (non-financial, non-zero-rev) names matching `mask`, keeping only
    those firing >= 1 of `tags`, ranked by the number of on-thesis tags fired
    (then short_composite). The n_thesis_tags column is added for display."""
    sub = df[mask & ~df["is_financial"].fillna(False)
             & ~df["zero_low_revenue"].fillna(False)
             & ~df["sic"].fillna(-1).isin(config.SEMI_EXCLUDE_SICS)].copy()
    present = [t for t in tags if t in sub.columns]
    if not present or not len(sub):
        return sub.iloc[0:0]
    sub["n_thesis_tags"] = sub[present].fillna(False).astype(bool).sum(axis=1)
    sub = sub[sub["n_thesis_tags"] >= 1]
    return _stable_sort(sub, ["n_thesis_tags"])


OUTPUT_COLUMNS_FIRST = [
    "ticker", "name", "sic", "sic_desc", "fin_sector", "mktcap_build",
    "long_composite", "short_composite",
    "bucket_turnaround", "bucket_moat",
    "t_inflection", "t_fscore", "t_valuation_reset", "t_squeeze",
    "m_roic_spread", "m_gm", "m_consistency", "m_valuation",
    "false_moat_raw", "false_moat_scored", "theme_inflated_growth",
    "theme_forensic", "theme_confirmation",
    "cov_long", "cov_turnaround", "cov_moat", "cov_short",
    # timing is the modal judge question — keep it visible, not buried in the
    # column tail of a 200-column CSV
    "days_to_catalyst", "catalyst_imminent", "catalyst_bonus",
    "contested", "on_both_lists", "false_moat_collision",
    "low_confidence", "is_commodity", "cyclical", "sector_driven",
    "zero_low_revenue", "annual_only",
]


def write_csvs(outputs: dict[str, pd.DataFrame], outdir: str) -> list[str]:
    from pathlib import Path
    paths = []
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        if frame is None or len(frame) == 0:
            continue
        cols = [c for c in OUTPUT_COLUMNS_FIRST if c in frame.columns]
        rest = [c for c in frame.columns if c not in cols]
        p = str(Path(outdir) / f"{name}.csv")
        frame[cols + rest].to_csv(p, index=False)
        paths.append(p)
    return paths
