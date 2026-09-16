"""End-to-end pipeline orchestration (spec roadmap steps 2-5).

Stages:
  1. universe   — tickers + hygiene + light zip extraction + closes -> band
  2. market     — chunked 3y closes, beta, drawdown inputs, quote summaries
  3. company    — per-CIK facts -> CompanyData -> raw metric row (parallel)
  4. scoring    — cross-sectional ranks, themes, composites, flags
  5. outputs    — ranked CSVs, shortlists, tear-sheet detail JSONs
"""

from __future__ import annotations

import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, market, metrics, forensic, scoring, shortlist
from .edgar import BulkZip, parse_submissions, SubmissionsInfo, catalyst_calendar
from .fundamentals import CompanyData, build_company
from .inflection import (combine_metrics, margin_vs_peak, metric_inflection,
                         revenue_context, staleness_multiplier)
from .quarterlyize import to_ttm, ttm_ratio, yoy_delta
from .universe import (build_universe, extract_light_batch, hygiene_filter,
                       load_ticker_table)

ROOT = Path(__file__).resolve().parent.parent
FACTS_ZIP = str(ROOT / config.EDGAR_DIR / "companyfacts.zip")
SUBS_ZIP = str(ROOT / config.EDGAR_DIR / "submissions.zip")
TICKERS_JSON = str(ROOT / config.EDGAR_DIR / "company_tickers_exchange.json")
CACHE = ROOT / config.CACHE_DIR
OUT = ROOT / config.OUTPUT_DIR


# ---------------------------------------------------------------------------
# Stage 1: universe
# ---------------------------------------------------------------------------

def _log(msg: str):
    import datetime
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def stage_universe(refresh: bool = False, workers: int = 10) -> pd.DataFrame:
    cache_p = CACHE / "universe.parquet"
    if cache_p.exists() and not refresh:
        return pd.read_parquet(cache_p)
    tickers = hygiene_filter(load_ticker_table(TICKERS_JSON))
    _log(f"universe: {len(tickers)} listings after hygiene; light extraction...")
    light = extract_light_batch(tickers["cik"].tolist(), FACTS_ZIP, SUBS_ZIP,
                                workers=workers)
    _log(f"universe: light extraction done ({int(light['has_facts'].sum())} "
         f"with facts); fetching closes...")
    closes = market.chunked_download(tickers["ticker"].tolist(), period="5d",
                                     cache_path=str(CACHE / "closes_5d.parquet"))
    last = market.last_closes(closes)
    mc_tickers = tickers.loc[tickers["multi_class"], "ticker"].tolist()
    _log(f"universe: {len(mc_tickers)} multi-class names -> fast_info caps...")
    mc_caps = market.fast_marketcaps(mc_tickers) if mc_tickers else None
    uni = build_universe(ROOT / config.EDGAR_DIR, last, light, tickers,
                         multi_class_caps=mc_caps)
    CACHE.mkdir(parents=True, exist_ok=True)
    uni.to_parquet(cache_p)
    _log(f"universe: {len(uni)} names in build band "
         f"({int(uni['is_financial'].sum())} financials, "
         f"{int(uni['annual_only'].sum())} annual-only FPIs)")
    return uni


# ---------------------------------------------------------------------------
# Stage 2: market data
# ---------------------------------------------------------------------------

def stage_market(uni: pd.DataFrame, refresh: bool = False) -> dict:
    tick = uni["ticker"].tolist()
    _log(f"market: 3y closes for {len(tick)} tickers...")
    closes = market.chunked_download(tick, period="3y",
                                     cache_path=str(CACHE / "closes_3y.parquet"))
    spx = market.fetch_spx("3y")
    betas = market.beta_weekly(closes, spx)
    dd = market.drawdown_inputs(closes)
    _log(f"market: quote summaries for {len(tick)} tickers (throttled, "
         f"cached, resumable)...")
    qs = market.fetch_quote_summaries(
        tick, cache_dir=str(CACHE / "qs"),
        progress_cb=lambda i, n: _log(f"market: quote summaries {i}/{n}")
        if i % 250 == 0 else None)
    rf = market.rf_current()
    rf_hist = market.rf_history_annual()
    ppi_ids = sorted(set(config.SIC_PPI_CROSSWALK.values())
                     | set(config.PPI_TICKER_OVERRIDE.values()))
    ppi = market.fetch_ppi_series(ppi_ids)
    _log(f"market: fetched {len(ppi)}/{len(ppi_ids)} FRED PPI series")
    _log("market: done")
    return {"closes": closes, "betas": betas, "dd": dd, "qs": qs,
            "rf": rf, "rf_hist": rf_hist, "ppi": ppi}


# ---------------------------------------------------------------------------
# Stage 3: per-company raw rows (parallelized over the universe)
# ---------------------------------------------------------------------------

_W_FACTS: BulkZip | None = None
_W_SUBS: BulkZip | None = None
_W_CTX: dict = {}


def _init_company_worker(facts_path: str, subs_path: str, ctx: dict):
    global _W_FACTS, _W_SUBS, _W_CTX
    _W_FACTS = BulkZip(facts_path)
    _W_SUBS = BulkZip(subs_path)
    _W_CTX = ctx


def _q_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    common = num.index.intersection(den.index)
    den_c = den[common]
    ok = den_c > 0
    out = (num[common][ok] / den_c[ok]).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _build_metric_inputs(cd: CompanyData, roic: pd.Series) -> dict:
    """Per-metric (ttm_series_for_level_trough, slope_series) pairs."""
    rev_q = cd.qvals("revenue")
    rev_ttm = cd.ttm.get("revenue", pd.Series(dtype=float))
    assets = cd.inst.get("assets", pd.Series(dtype=float))
    floor = None
    if len(assets) and len(rev_ttm):
        from .quarterlyize import align_instant_to_quarters
        fl = align_instant_to_quarters(assets, rev_ttm.index)
        floor = fl * config.MARGIN_DENOM_ASSET_FLOOR

    out = {}  # metric -> (ttm_for_level_trough, slope_series, raw_quarterly)
    gp_q = cd.qvals("gross_profit")
    if len(gp_q) and len(rev_q):
        gm_q = _q_ratio(gp_q, rev_q)
        gm_ttm = ttm_ratio(cd.ttm.get("gross_profit", pd.Series(dtype=float)),
                           rev_ttm, floor)
        if len(gm_ttm) >= 4 and len(gm_q) >= 5:
            out["gm"] = (gm_ttm, yoy_delta(gm_q), gm_q)
    oi_q = cd.qvals("operating_income")
    if len(oi_q) and len(rev_q):
        om_q = _q_ratio(oi_q, rev_q)
        om_ttm = ttm_ratio(cd.ttm.get("operating_income", pd.Series(dtype=float)),
                           rev_ttm, floor)
        if len(om_ttm) >= 4 and len(om_q) >= 5:
            out["om"] = (om_ttm, yoy_delta(om_q), om_q)
    # NI: prefer continuing ops; REITs use the FFO proxy (audit 3.4)
    ni_q = cd.qvals("continuing_income")
    if cd.meta.get("fin_sector") == "reit":
        ffo = metrics.reit_ffo_proxy(cd)
        if len(ffo) >= 5:
            ni_q = ffo
    if len(ni_q) and len(rev_q):
        nim_q = _q_ratio(ni_q, rev_q)
        nim_ttm = ttm_ratio(to_ttm(ni_q), rev_ttm, floor)
        if len(nim_ttm) >= 4 and len(nim_q) >= 5:
            out["ni"] = (nim_ttm, yoy_delta(nim_q), nim_q)
    # FCF special case: level/trough on TTM FCF; slope = YoY of TTM FCF margin
    fcf_ttm = cd.ttm.get("fcf", pd.Series(dtype=float))
    if len(fcf_ttm) >= 4 and len(rev_ttm) >= 4:
        fcf_margin_ttm = ttm_ratio(fcf_ttm, rev_ttm, floor)
        if len(fcf_margin_ttm) >= 5:
            out["fcf"] = (fcf_ttm, yoy_delta(fcf_margin_ttm),
                          cd.qvals("fcf"))
    # Revenue as sixth input (audit 4.1)
    if len(rev_ttm) >= 4 and len(rev_q) >= 5:
        out["revenue"] = (rev_ttm, yoy_delta(rev_q), rev_q)
    # ROIC (banks/insurers: ROE substitute inside roic_series). ROIC is
    # already TTM-based, so no raw-quarterly one-off check for it.
    if len(roic) >= 5:
        out["roic"] = (roic, yoy_delta(roic), None)
    return out


def _build_metric_inputs_annual(cd: CompanyData) -> dict:
    """FPI annual-frequency variant (audit 1.3): level/trough on the FY
    series, slope on FY-over-FY deltas. Window params are supplied in
    quarter-equivalents by the caller (1 FY = 4 quarters of calendar time,
    which the engine's calendar-day as-of math handles natively)."""
    out = {}
    rev = cd.annual.get("revenue", pd.Series(dtype=float))
    for m, concept in (("revenue", "revenue"), ("om", "operating_income"),
                       ("ni", "net_income"), ("fcf", "cfo"),
                       ("gm", "gross_profit")):
        s = cd.annual.get(concept, pd.Series(dtype=float))
        if len(s) < 3:
            continue
        if m in ("om", "ni", "gm", "fcf") and len(rev) >= 3:
            common = s.index.intersection(rev.index)
            base = rev[common]
            s = (s[common][base > 0] / base[base > 0]).dropna()
            if len(s) < 3:
                continue
        d = yoy_delta(s)
        if len(d) >= 2:
            out[m] = (s, d, s)
    return out


def build_raw_row(cd: CompanyData, sub: SubmissionsInfo | None,
                  mkt_row: dict, rf: float, rf_hist: pd.Series,
                  asof: pd.Timestamp,
                  ppi: dict | None = None) -> tuple[dict, dict]:
    """Returns (row_for_scoring, tear_sheet_details)."""
    cd.market.update(mkt_row)
    row: dict = {k: cd.meta.get(k) for k in
                 ("cik", "ticker", "name", "sic", "sic_desc", "fin_sector",
                  "is_financial", "is_commodity", "sector2", "annual_only",
                  "mktcap_build", "nt_recent", "is_despac_name")}
    details: dict = {"ticker": cd.ticker, "cik": cd.cik}

    latest_end = cd.latest_period_end()
    stale_mult = staleness_multiplier(latest_end, asof)
    row["data_as_of"] = str(latest_end.date()) if latest_end is not None else ""
    row["staleness_days"] = (asof - latest_end).days if latest_end is not None else np.nan
    row["stale"] = (row["staleness_days"] or 0) > config.STALE_FACT_DAYS

    # ---- inflection (long) + peak primitive (short) ----
    roic = metrics.roic_series(cd)
    annual_variant = cd.annual_only
    if annual_variant:
        inputs = _build_metric_inputs_annual(cd)
        slope_wq = config.FPI_SLOPE_WINDOW_FY * 4
        trough_wq = config.FPI_TROUGH_WINDOW_FY * 4
        rev_ctx = revenue_context(cd.annual.get("revenue",
                                                pd.Series(dtype=float)), asof)
    else:
        inputs = _build_metric_inputs(cd, roic)
        slope_wq, trough_wq = None, None
        rev_ctx = revenue_context(cd.qvals("revenue"), asof)
    per_metric = {}
    per_metric_short = {}
    for m, (ttm_s, slope_s, raw_s) in inputs.items():
        per_metric[m] = metric_inflection(m, ttm_s, slope_s, asof, direction=1,
                                          slope_window_q=slope_wq,
                                          trough_window_q=trough_wq,
                                          raw_q=raw_s)
        if m in ("roic", "gm", "om") and not annual_variant:
            per_metric_short[m] = metric_inflection(
                m, ttm_s, slope_s, asof, direction=-1, raw_q=raw_s)
    theme = combine_metrics(per_metric, rev_ctx, stale_mult,
                            breadth_n=(config.BREADTH_N_ANNUAL
                                       if annual_variant else None))
    row["inflection_score"] = theme.theme_score
    row["inflection_breadth"] = theme.breadth
    row["revenue_context"] = rev_ctx
    row["cost_cut_inflection"] = theme.cost_cut_haircut_applied
    details["inflection"] = {
        m: {"level": r.level, "slope": r.slope, "trough": r.trough,
            "score": r.score, "level_pctile": r.level_pctile,
            "n_obs": r.n_obs, "flags": sorted(r.flags),
            **{k: (str(v) if isinstance(v, pd.Timestamp) else v)
               for k, v in r.details.items()}}
        for m, r in per_metric.items()}

    peak_theme = combine_metrics(
        per_metric_short, "unknown", stale_mult,
        weights={"roic": 0.4, "gm": 0.3, "om": 0.3},
        breadth_n=config.BREADTH_N_PEAK)
    row["peak_score"] = peak_theme.theme_score
    details["peak"] = {
        m: {"level": r.level, "slope": r.slope, "peak_recency": r.trough,
            "score": r.score}
        for m, r in per_metric_short.items()}

    # at-peak primitive (user 2026-07-08): the setup, before the rollover
    from .inflection import at_peak_combined
    if not annual_variant:
        at_inputs = {m: inputs[m][0] for m in ("roic", "gm", "om")
                     if m in inputs}
        row["at_peak_score"], at_det = at_peak_combined(at_inputs, asof) \
            if at_inputs else (np.nan, {})
        details["at_peak"] = at_det
    else:
        row["at_peak_score"] = np.nan

    # op-margin-vs-peak row (banks: efficiency ratio replaces it, inverted)
    if cd.meta.get("fin_sector") == "bank":
        eff = metrics.efficiency_ratio(cd)
        mvp, mvp_det = margin_vs_peak(-eff, asof) if len(eff) else (np.nan, {})
    else:
        om_ttm = inputs.get("om", (pd.Series(dtype=float),))[0]
        mvp, mvp_det = margin_vs_peak(om_ttm, asof)
    row["margin_vs_peak"] = mvp
    row["decline_shape"] = mvp_det.get("decline_shape", "unknown")
    details["margin_vs_peak"] = {k: (str(v) if isinstance(v, pd.Timestamp) else v)
                                 for k, v in mvp_det.items()}
    # sector-relative peer check input (audit 4.3): margin change since peak
    row["om_change_since_peak"] = (
        mvp_det.get("current", np.nan) - mvp_det.get("peak", np.nan)
        if mvp_det else np.nan)

    # ---- moat ----
    spread = metrics.moat_spread(cd, roic, rf_hist if len(rf_hist) else
                                 pd.Series(dtype=float))
    row["moat_demonstrated"] = spread.get("demonstrated", np.nan)
    row["moat_trailing_5y"] = spread.get("trailing_5y", np.nan)
    row["fallen_angel_spread"] = spread.get("fallen_angel_spread", np.nan)
    row["moat_window_start"] = str(spread.get("window_start", ""))[:10]
    row["moat_window_end"] = str(spread.get("window_end", ""))[:10]
    row["current_spread_negative"] = spread.get("current_spread_negative")
    row["moat_n_obs"] = spread.get("n_obs", 0)
    row["spread_slope"] = spread.get("spread_slope", np.nan)
    row["spread_now"] = spread.get("spread_now", np.nan)
    row["spread_peak_window"] = spread.get("spread_peak_in_window", np.nan)
    row["buffer_consumed"] = spread.get("buffer_consumed", np.nan)
    row["crossing_soon"] = bool(spread.get("crossing_soon", False))
    gmstab = metrics.gm_stability(
        cd, (spread.get("window_start"), spread.get("window_end")))
    row["gm_level"] = gmstab.get("gm_level", np.nan)
    row["gm_stability"] = gmstab.get("gm_stability", np.nan)
    row["earnings_consistency"] = metrics.earnings_consistency(cd)

    # ---- F-score ----
    pio = metrics.piotroski_ttm(cd, asof)
    row["f_score"] = pio.get("f_score", np.nan)
    row["f_rising"] = pio.get("f_rising", np.nan)

    # ---- valuation / market rows ----
    dd = mkt_row.get("drawdown", np.nan)
    row["drawdown"] = dd
    row["valuation_reset"] = scoring.valuation_reset_row(
        dd, mkt_row.get("pct_above_low", np.nan),
        mkt_row.get("slope_3m", np.nan))
    row["si_ratio"] = mkt_row.get("si_ratio", np.nan)
    row["date_short_interest"] = mkt_row.get("dateShortInterest")
    row["short_pct_float_display"] = mkt_row.get("shortPercentOfFloat")
    si_v = row["si_ratio"]
    # NOTE: `si_v or np.nan` would turn a real 0.0 into NaN (truthiness) —
    # sharesShort == 0 is a defined value and must score si_hump(0.0) == 0.
    row["si_hump"] = forensic.si_hump(float(si_v)) \
        if isinstance(si_v, (int, float, np.floating)) and np.isfinite(si_v) \
        else np.nan
    row["ev_sales"] = metrics.ev_sales(cd)
    row["ev_normalized"] = metrics.ev_normalized(cd)
    row["ev_ebitda"] = metrics.ev_ebitda(cd)
    # persist per-company WACC (full CAPM, also computed inside moat_spread);
    # needed by the reverse-DCF primitive + useful for diagnostics
    wacc_cur, _wdet = metrics.current_wacc(cd, rf)
    row["current_wacc"] = wacc_cur
    rdcf = metrics.reverse_dcf(cd, rf, wacc=wacc_cur)
    row["rdcf_implied_growth"] = rdcf.get("implied_growth", np.nan)
    row["implied_g_gordon"] = rdcf.get("implied_g_gordon", np.nan)
    row["rdcf_growth_gap"] = rdcf.get("growth_gap", np.nan)
    row["rdcf_trailing_cagr"] = rdcf.get("trailing_cagr", np.nan)
    row["implied_margin_ceiling"] = rdcf.get("implied_margin_ceiling", np.nan)
    row["rdcf_peak_op_margin"] = rdcf.get("peak_op_margin", np.nan)
    row["margin_ceiling_gap"] = rdcf.get("margin_ceiling_gap", np.nan)
    details["reverse_dcf"] = rdcf
    row["p_tbv"] = metrics.p_tbv(cd)
    row["peg_display_only"] = mkt_row.get("trailingPegRatio")
    # coverage/neglect + skin-in-the-game, display-only (no scoring weight):
    # already in the cached quote summaries, previously discarded.
    row["n_analysts_display"] = mkt_row.get("numberOfAnalystOpinions")
    row["insider_pct_display"] = mkt_row.get("heldPercentInsiders")
    row["ret_12m"] = mkt_row.get("ret_12m", np.nan)

    seg = cd.inst.get("reportable_segments")
    row["reportable_segments_display"] = (
        float(seg.dropna().iloc[-1]) if seg is not None and len(seg.dropna())
        else np.nan)

    # ---- survivability / context tags (zero score weight) ----
    surv = metrics.survivability(cd)
    row.update({f"surv_{k}": v for k, v in surv.items()})
    row["altman_z"] = metrics.altman_z(cd)
    row["interest_coverage"] = metrics.interest_coverage(cd)
    row["debt_to_ebitda"] = metrics.debt_to_ebitda(cd)

    # ---- short side ----
    sloan = forensic.sloan_accruals(cd)
    row["sloan_latest"] = float(sloan.iloc[-1]) if len(sloan) else np.nan
    ben = forensic.beneish(cd)
    row["beneish_m"] = ben.get("m_score", np.nan)
    row["beneish_sgi_dominant"] = ben.get("sgi_dominant", False)
    details["beneish_components"] = ben.get("components", {})
    dso = forensic.dso_dio(cd)
    row["dso_yoy_delta"] = dso.get("dso_yoy_delta", np.nan) \
        if dso.get("dso_applicable") else np.nan
    row["dio_yoy_delta"] = dso.get("dio_yoy_delta", np.nan) \
        if dso.get("dio_applicable") else np.nan
    dec = forensic.revenue_decel(cd)
    row["decel_score_raw"] = dec.get("score_raw", np.nan)
    row["ma_flag"] = dec.get("ma_flag", False)
    row["rev_gp_divergence"] = dec.get("rev_gp_divergence", False)
    details["revenue_decel"] = {k: v for k, v in dec.items()}
    pmv = forensic.pricing_masking_volume(cd)
    row["pmv_score_raw"] = pmv.get("score_raw", np.nan)
    row["pricing_masking_volume"] = pmv.get("flag", False)
    details["pricing_masking_volume"] = {k: v for k, v in pmv.items()}
    lev = forensic.leverage_propping(cd, roic)
    row["leverage_prop_raw"] = lev.get("score_raw", np.nan)
    bb = forensic.debt_funded_buybacks(cd)
    row["buyback_score_raw"] = bb.get("score_raw", np.nan)
    row["dilution_yoy"] = forensic.dilution_yoy(cd)
    nos = forensic.nonop_earnings_share(cd)
    row["nonop_share"] = nos.get("share", np.nan)
    row["nonop_share_score_raw"] = nos.get("score_raw", np.nan)
    row["nonop_share_yoy_delta"] = nos.get("share_yoy_delta", np.nan)
    row["nonop_income_propping"] = nos.get("flag", False)
    # TTM revenue YoY growth (for the relative-share vs peer-cohort signal)
    _rev_ttm = cd.ttm.get("revenue")
    _rs = _rev_ttm.dropna() if _rev_ttm is not None else pd.Series(dtype=float)
    row["rev_yoy_latest"] = (float(_rs.iloc[-1] / _rs.iloc[-5] - 1.0)
                             if len(_rs) >= 5 and float(_rs.iloc[-5]) > 0
                             else np.nan)
    # catalyst calendar (EDGAR filing-cadence next-print estimate)
    cat = catalyst_calendar(sub, latest_end, asof, annual_only=cd.annual_only) \
        if sub is not None else {}
    row["days_to_catalyst"] = cat.get("days_to_catalyst", np.nan)
    row["guide_reset_window"] = bool(cat.get("guide_reset_window", False))
    row["catalyst_next_print"] = cat.get("projected_next_filing", "")
    row["catalyst_lag_days"] = cat.get("lag_days", np.nan)
    cap = forensic.capacity_turnover_decay(cd)
    row["capacity_organic_proxy"] = cap.get("organic_proxy", np.nan)
    row["capacity_score_raw"] = cap.get("score_raw", np.nan)
    row["capacity_decay"] = cap.get("flag", False)
    # Tier-2 external-price (FRED PPI): resolve this name's series via ticker
    # override then 2-digit-SIC crosswalk; external_price_signal filters the
    # series to <= asof (point-in-time safe for replay)
    price_series = None
    if ppi:
        _sid = config.PPI_TICKER_OVERRIDE.get(cd.ticker)
        if _sid is None:
            try:  # sic may be NaN/None/non-numeric -> no PPI mapping
                _sid = config.SIC_PPI_CROSSWALK.get(int(cd.meta.get("sic")) // 100)
            except (TypeError, ValueError):
                _sid = None
        if _sid:
            price_series = ppi.get(_sid)
    ext = forensic.external_price_signal(cd, price_series, asof)
    row["ppi_zscore"] = ext.get("ppi_zscore", np.nan)
    row["ppi_windfall"] = ext.get("ppi_windfall", False)
    row["real_rev_growth"] = ext.get("real_rev_growth", np.nan)
    row["real_rev_deflated"] = ext.get("real_rev_deflated", False)
    txt = forensic.tax_tailwind(cd)
    row["tax_eff_rate"] = txt.get("eff_rate", np.nan)
    row["tax_boost_share"] = txt.get("boost_share", np.nan)
    row["tax_tailwind"] = txt.get("flag", False)
    # Provenance for the as-of filing guard: how many filings were hidden as
    # filed-after-asof. 0 on a live run; >0 on a replay proves the guard ran.
    row["n_filings_suppressed"] = int(getattr(sub, "n_filings_suppressed", 0)) \
        if sub is not None else 0
    ev = forensic.event_flags(cd, sub) if sub is not None else {}
    row["event_score_raw"] = ev.get("score_raw", np.nan)
    row["restated"] = ev.get("restated", False)
    # the raw >3% retained-value move, kept as its own (non-forensic) column
    row["comparatives_represented"] = ev.get(
        "comparatives_represented", "restated" in cd.flags)
    row["restatement_filing"] = ev.get("restatement_filing", "")
    row["nt_filer"] = ev.get("nt_filer", False)
    row["auditor_change"] = ev.get("auditor_change", False)
    row["non_reliance"] = ev.get("non_reliance", False)
    row["zero_low_revenue"] = forensic.zero_low_revenue(cd)

    # latest margins for peer machinery
    gm_in = inputs.get("gm")
    om_in = inputs.get("om")
    row["gm_latest"] = float(gm_in[0].iloc[-1]) if gm_in is not None and len(gm_in[0]) else np.nan
    row["om_latest"] = float(om_in[0].iloc[-1]) if om_in is not None and len(om_in[0]) else np.nan
    # TTM om 4q change (for the sector-margin-declining haircut input)
    if om_in is not None and len(om_in[0]) >= 5:
        s = om_in[0]
        row["om_ttm_delta_4q"] = float(s.iloc[-1] - s.iloc[-5])
    else:
        row["om_ttm_delta_4q"] = np.nan

    # ---- first-run diagnostics inputs (audit 6.4) ----
    # parameter sensitivity: inflection theme under slope x trough combos
    if not annual_variant:
        for sw in config.SLOPE_WINDOWS_SENSITIVITY:
            for tw in config.TROUGH_REF_WINDOWS_SENSITIVITY:
                pm = {m: metric_inflection(m, t, sl, asof, direction=1,
                                           slope_window_q=sw,
                                           trough_window_q=tw, raw_q=rq)
                      for m, (t, sl, rq) in inputs.items()}
                th = combine_metrics(pm, rev_ctx, stale_mult)
                row[f"sens_s{sw}_t{tw}"] = th.theme_score
                if sw == config.SLOPE_WINDOWS_SENSITIVITY[0]:
                    row[f"trough_fired_t{tw}"] = any(
                        np.isfinite(r.trough) and r.trough > 0
                        for r in pm.values())
        row["trough_fired_primary"] = any(
            np.isfinite(r.trough) and r.trough > 0 for r in per_metric.values())

    # ---- tier-2 tear-sheet payload ----
    if sub is not None:
        details["tier2"] = {
            "recent_8k_items": sub.recent_8k_items[:24],
            "sic_description": sub.sic_description,
            "latest_10k_accn": sub.latest_10k_accn,
            "latest_10q_accn": sub.latest_10q_accn,
            "nt_filings_recent": sub.nt_filings_recent,
            "fiscal_year_end": sub.fiscal_year_end,
            # the filing a `restated` flag points the reader at
            "amended_periodic": sub.amended_periodic[:6],
        }
    details["tier2_market"] = {
        "longBusinessSummary": str(mkt_row.get("longBusinessSummary", ""))[:500],
        "nextEarningsDate": str(mkt_row.get("nextEarningsDate", "")),
        "trailingPegRatio_display_only": mkt_row.get("trailingPegRatio"),
        "shortPercentOfFloat_display_only": mkt_row.get("shortPercentOfFloat"),
        "numberOfAnalystOpinions": mkt_row.get("numberOfAnalystOpinions"),
        "heldPercentInsiders": mkt_row.get("heldPercentInsiders"),
    }

    # filing-history depth (story-stock guards; user 2026-07-09): longest
    # quarterly series the company has — a 2025 IPO's "12-mo run" is
    # meaningless and its dilution history is post-IPO noise
    row["history_quarters"] = max(
        [len(qs.values) for qs in cd.q.values()] +
        [len(s) for s in cd.annual.values()] + [0])

    row["flags"] = ",".join(sorted(cd.flags | theme.flags))
    details["quarterly_series"] = {
        m: {"ttm": {str(k.date()): float(v) for k, v in ttm_s.items()},
            "slope_series": {str(k.date()): float(v) for k, v in slope_s.items()},
            "quarterly": ({str(k.date()): float(v) for k, v in raw_s.items()}
                          if raw_s is not None else {})}
        for m, (ttm_s, slope_s, raw_s) in inputs.items()}
    return row, details


def _company_task(args) -> tuple[dict, dict] | None:
    cik, urow = args
    try:
        facts = _W_FACTS.get(cik)
        if not facts:
            return None
        cutoff = _W_CTX.get("filed_cutoff")
        if cutoff:
            from .edgar import filter_facts_asof
            facts = filter_facts_asof(facts, cutoff)
        subs_raw = _W_SUBS.get(cik)
        # `cutoff` is None on live runs -> today's view; on a replay it hides
        # every filing made after the as-of date (form history, 8-K items,
        # NT counts, latest-accession pointers).
        sub = parse_submissions(subs_raw, asof=cutoff) if subs_raw else None
        cd = build_company(facts, urow)
        mkt_row = dict(_W_CTX["market_rows"].get(urow["ticker"], {}))
        if cutoff:
            # point-in-time market cap: as-of dei shares x as-of close;
            # today's quote-summary EV/mktcap are invalid at the replay date
            mkt_row.pop("enterprise_value", None)
            close = mkt_row.get("last")
            sh = cd.inst.get("shares_outstanding")
            if close and sh is not None and len(sh.dropna()) \
                    and np.isfinite(close):
                mkt_row["mktcap"] = float(sh.dropna().iloc[-1]) * float(close)
        return build_raw_row(cd, sub, mkt_row, _W_CTX["rf"],
                             _W_CTX["rf_hist"], _W_CTX["asof"],
                             _W_CTX.get("ppi"))
    except Exception as e:  # permissive: one bad name never kills the run
        return ({"cik": cik, "ticker": urow.get("ticker"), "error": str(e)},
                {})


def stage_companies(uni: pd.DataFrame, mkt: dict, workers: int = 10,
                    asof: pd.Timestamp | None = None,
                    filed_cutoff: str | None = None) -> tuple[pd.DataFrame, dict]:
    """filed_cutoff (ISO date): point-in-time replay — only facts FILED on
    or before this date are visible, and market caps are reconstructed from
    as-of shares x as-of close (replay harness, user 2026-07-08)."""
    asof = asof or pd.Timestamp.today().normalize()
    market_rows = _assemble_market_rows(uni, mkt)
    ctx = {"market_rows": market_rows, "rf": mkt["rf"],
           "rf_hist": mkt["rf_hist"], "asof": asof,
           "ppi": mkt.get("ppi", {}), "filed_cutoff": filed_cutoff}
    tasks = [(int(r["cik"]), dict(r)) for _, r in uni.iterrows()]
    _log(f"companies: building {len(tasks)} names with {workers} workers...")
    rows, det_map = [], {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_company_worker,
                             initargs=(FACTS_ZIP, SUBS_ZIP, ctx)) as ex:
        for res in ex.map(_company_task, tasks, chunksize=20):
            done += 1
            if done % 250 == 0:
                _log(f"companies: {done}/{len(tasks)}")
            if res is None:
                continue
            row, det = res
            rows.append(row)
            if det:
                det_map[row["ticker"]] = det
    _log(f"companies: done ({len(rows)} rows)")
    df = pd.DataFrame(rows)
    # sector-margin-declining input for the short-side haircut
    if "om_ttm_delta_4q" in df:
        sec_med = df.groupby("sector2")["om_ttm_delta_4q"].transform("median")
        sec_n = df.groupby("sector2")["om_ttm_delta_4q"].transform("count")
        df["sector_margin_declining"] = (sec_med < 0) & (sec_n >= config.PEER_GROUP_MIN)
    else:
        df["sector_margin_declining"] = False
    return df, det_map


def _assemble_market_rows(uni: pd.DataFrame, mkt: dict) -> dict[str, dict]:
    out = {}
    dd: pd.DataFrame = mkt["dd"]
    qs: pd.DataFrame = mkt["qs"]
    betas: pd.Series = mkt["betas"]
    for _, r in uni.iterrows():
        t = r["ticker"]
        row = {"mktcap": r.get("mktcap_build"), "last_close": r.get("last_close"),
               "rf": mkt["rf"]}
        if t in dd.index:
            row.update(dd.loc[t].to_dict())
        if betas is not None and t in betas.index:
            row["beta"] = betas[t]
        if qs is not None and t in qs.index:
            q = qs.loc[t].to_dict()
            row.update(q)
            if pd.notna(q.get("marketCap")):
                row["mktcap"] = q["marketCap"]
            if pd.notna(q.get("enterpriseValue")):
                row["enterprise_value"] = q["enterpriseValue"]
        out[t] = row
    return out


# ---------------------------------------------------------------------------
# Stage 4+5: score and write
# ---------------------------------------------------------------------------

def run(refresh_universe: bool = False, workers: int = 10,
        universe_override: pd.DataFrame | None = None) -> dict:
    uni = universe_override if universe_override is not None \
        else stage_universe(refresh=refresh_universe, workers=workers)
    mkt = stage_market(uni)
    raw, details = stage_companies(uni, mkt, workers=workers)
    if "error" in raw.columns:
        errs = raw[raw["error"].notna()]
        raw = raw[raw["error"].isna()].drop(columns=["error"])
    else:
        errs = pd.DataFrame()
    # cache the raw rows (the ~23-min per-company build) so scoring-only changes
    # can be re-run in seconds via scripts/rescore.py instead of a full rebuild
    _raw_path = CACHE / "raw_rows.parquet"
    try:
        raw.to_parquet(_raw_path)
    except Exception as e:  # never let a cache-write failure kill the run
        # ...but never leave a STALE cache behind either. This failed silently
        # on a mixed-dtype column and left rescore.py reading a previous run's
        # rows while the full run reported success — the worst kind of bug,
        # because every downstream "before/after" was then measured against
        # data that had nothing to do with the change under test.
        _log(f"(!! raw-row cache write FAILED: {e})")
        try:
            if _raw_path.exists():
                _raw_path.unlink()
                _log("(!! removed the now-stale raw_rows.parquet — "
                     "rescore.py will fall back to output/universe_ranked.csv)")
        except OSError as e2:
            _log(f"(!! could not remove stale raw_rows.parquet: {e2})")
    scored = scoring.score_universe(raw)
    outputs = shortlist.rank_outputs(scored)
    paths = shortlist.write_csvs(outputs, str(OUT))
    # persist tear-sheet details for surfaced names
    surfaced = set()
    for key in ("long_shortlist", "short_shortlist", "long_financials",
                "short_financials"):
        if key in outputs and len(outputs[key]):
            surfaced |= set(outputs[key]["ticker"])
    det_dir = OUT / "details"
    det_dir.mkdir(parents=True, exist_ok=True)
    for t in surfaced:
        if t in details:
            (det_dir / f"{t}.json").write_text(
                json.dumps(details[t], default=str))
    # tear sheets + first-run diagnostics
    from . import diagnostics, tearsheet
    # shortlist-stage insider enrichment (spec: surfaced names only,
    # nullable — yfinance insider tables don't scale universe-wide)
    surfaced_pre = set()
    for key in ("long_shortlist", "short_shortlist", "long_financials",
                "short_financials"):
        if key in outputs and len(outputs[key]):
            surfaced_pre |= set(outputs[key]["ticker"])
    for t in surfaced_pre:
        if t in details:
            try:
                ins = market.insider_summary(t)
            except Exception:
                ins = None
            details[t].setdefault("tier2_market", {})["insider"] = ins
    n_sheets = tearsheet.write_tearsheets(outputs, details,
                                          str(OUT / "tearsheets"))
    diag_path = diagnostics.run_all(scored, str(OUT / "diagnostics"))
    with open(CACHE / "last_run.pkl", "wb") as fh:
        pickle.dump({"scored": scored, "outputs": {k: v for k, v in outputs.items()},
                     "errors": errs}, fh)
    return {"universe_n": len(uni), "scored_n": len(scored),
            "errors_n": len(errs), "paths": paths, "outputs": outputs,
            "scored": scored, "details": details, "n_tearsheets": n_sheets,
            "diagnostics": diag_path}
