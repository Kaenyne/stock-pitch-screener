"""All tunables, weights, windows, kernel shapes, and tag ladders.

Per the spec's process rule (audit 4.4): the metric->score kernel shape
(direction, breakpoints, plateau/taper) is documented here for every row in
both composites BEFORE the code that consumes it. Nothing in the scoring path
may hard-code a number that belongs here.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# EDGAR / data layer
# ---------------------------------------------------------------------------
# SEC EDGAR requires every request to identify who is asking ("Name email").
# Resolution order: EDGAR_USER_AGENT env var -> user_agent.txt at the repo root
# (written by scripts/run_screener.py on first run) -> a placeholder that the
# runner refuses to start with.
def _load_user_agent() -> str:
    env = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if env:
        return env
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "user_agent.txt")
    try:
        with open(p, encoding="utf-8") as fh:
            v = fh.read().strip()
            if v:
                return v
    except OSError:
        pass
    return ""

EDGAR_USER_AGENT = _load_user_agent()
# FRED API key (api.stlouisfed.org). OPTIONAL: the keyless fredgraph CSV is
# tried first and ^TNX is the final fallback for the risk-free rate, so leave
# this empty unless the keyless host is blocked on your network. Free key at
# https://fred.stlouisfed.org/docs/api/api_key.html ; set FRED_API_KEY env var.
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
# FRED series (DGS10 + the PPI/CPI crosswalk) are persisted to
# data/cache/fred_series.parquet. Nothing about the PPI subsystem was
# reproducible before this: every run re-fetched live, so a replay of a past
# date could not be reproduced or run offline at all.
FRED_CACHE_MAX_AGE_DAYS = 7
EDGAR_MAX_RPS = 8  # courtesy limit is ~10/s; stay under
DATA_DIR = "data"
EDGAR_DIR = "data/edgar"
CACHE_DIR = "data/cache"
OUTPUT_DIR = "output"

# Only these forms feed the quarterlyizer (spec: 10-K/10-Q and /A forms).
ALLOWED_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
# Annual-only FPI variant additionally consumes these.
FPI_ANNUAL_FORMS = {"20-F", "40-F", "20-F/A", "40-F/A"}
# Amended periodic reports — the filed evidence that a restatement actually
# happened. `restated` used to mean only "a retained value moved >3% from the
# originally-filed one", which fired on 1,495 of 2,138 names (69.9%) with no
# 8-K 4.02 behind 1,459 of them, while the tear sheet told the reader to go
# pull that 4.02. A revision without an amendment or a 4.02 is a represented
# comparative, not a restatement.
AMENDED_PERIODIC_FORMS = {"10-K/A", "10-Q/A", "20-F/A", "40-F/A"}
AMENDMENT_LOOKBACK_YEARS = 3

# Duration classification windows (days) -> bucket.
DURATION_BUCKETS = {
    "Q": (75, 105),      # ~90d  discrete quarter
    "H1": (165, 195),    # ~180d H1 YTD
    "9M": (255, 285),    # ~270d 9M YTD
    "FY": (340, 380),    # ~360d fiscal year
}
# Derived-quarter validation: |sum(quarters) - FY| tolerance, relative to
# max(|FY|, floor). On failure: null the quarter, never interpolate.
FY_SUM_REL_TOL = 0.02
FY_SUM_ABS_FLOOR = 1e6  # dollars; avoids false failures on tiny FY values
# Restatement flag: retained value differs from originally-filed by more than
RESTATED_REL_TOL = 0.03  # spec: >2-5%
# Staleness: latest quarterly fact older than this many days -> stale flag.
STALE_FACT_DAYS = 200
# Soft staleness decay starts once last-period-end older than this (audit 1.6)
STALENESS_DECAY_START_DAYS = 140
STALENESS_DECAY_FULL_DAYS = 400  # linear decay to floor by here
STALENESS_DECAY_FLOOR = 0.5

# margin_vs_peak guards (2026-07-25). Two defects it had none against:
# (a) no dead-series guard — `cur` was just the last observation however old,
#     unlike metric_inflection which drops a series stale by >300d;
# (b) no plausibility bound on the peak — NRP peaked at a 104.95% "operating
#     margin" (equity-method income over a revenue base that excludes it),
#     giving a 21pp gap, full kernel credit, t_inflection 100 and LONG RANK #7
#     on a row with inflection_breadth 0. A margin above 100% means the
#     numerator and denominator are not the same business, so the series is
#     definitionally unusable rather than merely extreme -> null the metric.
# Banks pass a NEGATED efficiency ratio here, so only the upper bound binds.
MARGIN_VS_PEAK_STALE_DAYS = 300
MARGIN_VS_PEAK_MAX_PLAUSIBLE = 1.0
# When inflection_score is unmeasurable, exclude-and-renormalize handed 100%
# of t_inflection to the 0.15-weight margin_vs_peak row. Shrink toward the
# universe median in proportion to the weight actually present instead — the
# same thin-basis guard the buckets already use (audit 3.1 item 4).
INFLECTION_THIN_BASIS_SHRINK = True

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
CAP_BAND_BUILD = (0.8e9, 25e9)   # widened build-time band (dei shares stale)
CAP_BAND_FINAL = (1.0e9, 20e9)   # final band, re-checked with fast_info
CAP_CORE_FOCUS = (1.0e9, 10e9)
DERIVATIVE_SUFFIX_RE = r"[-.](?:WT|WS|W|U|UN|RT|R|P[A-Z]?|PR[A-Z]?)$"
# Structural non-operating detection: recent forms dominated by these => fund
FUND_FORMS = {"N-CSR", "N-CSRS", "N-PORT", "N-2", "N-1A", "NPORT-P", "N-CEN"}
DESPAC_FORMER_NAME_RE = r"acquisition\s+(corp|co|company|corporation)"
DESPAC_MAGNITUDE_JUMP = 20.0  # >20x jump near series start on NI/assets

# Financials / REIT sector detection (SIC major groups via submissions JSON)
SIC_BANKS = range(6020, 6200)        # depository institutions + related
SIC_INSURERS = range(6300, 6500)
SIC_REITS = {6798}
SIC_FINANCIAL_ALL = range(6000, 6800)  # margin-peer exclusion (SIC 6xxx)

# Commodity price-taker sectors: the x0.85 discount was RESCINDED by the
# user 2026-07-08 (cycle-top commodity names at peak margins are prime
# shorts under the ran-on-temporary-success system — HL/DK-type setups were
# being suppressed). The `commodity` TAG remains on every output row.
COMMODITY_DISCOUNT = 1.0
def is_commodity_sic(sic: int | None) -> bool:
    if sic is None:
        return False
    m2 = sic // 100  # 2-digit major group
    if m2 in (10, 12, 13, 14, 24, 26, 29, 33, 44, 1, 2):
        return True
    m3 = sic // 10  # 3-digit for chemicals
    return m3 in (281, 286, 287)

# ---------------------------------------------------------------------------
# Inflection engine
# ---------------------------------------------------------------------------
SLOPE_WINDOW_Q = 5           # primary 5-6q; tunable {3,4,5,6}
SLOPE_WINDOWS_SENSITIVITY = (3, 4, 6)
TROUGH_REF_WINDOW_Q = 10     # argmin reference window on TTM, 8-12q
TROUGH_REF_WINDOWS_SENSITIVITY = (8, 12)
TROUGH_MIN_Q_BEFORE_LATEST = 2
TROUGH_MIN_RECOVERY_MADS = 1.0
TROUGH_RECENCY_RAMP_Q = 8    # score = max(0, 1 - q_since/8)
TROUGH_RECOVERY_CAP_MADS = 3.0
SLOPE_ANCHOR_MIN_POST_TROUGH = 3
LEVEL_HISTORY_CAP_Q = 40
LEVEL_SHRINK_K = 8           # shrink toward 50 by n/(n+k)
SLOPE_MAD_CLIP = 3.0
GEOMEAN_COMPONENT_FLOOR = 0.12   # soft floor 0.1-0.15, not a gate
# Metric weights inside the profitability-inflection theme (margin-led).
# Revenue folded in per tuning; spec baseline GM30/OM25/ROIC15/NI15/FCF15.
INFLECTION_METRIC_WEIGHTS = {
    "gm": 0.25, "om": 0.20, "roic": 0.15, "ni": 0.125, "fcf": 0.125,
    "revenue": 0.15,
}
BREADTH_N_METRICS = 6        # sqrt(breadth/6) breadth term
# Cost-cut haircut (audit 4.1): applied to margin/earnings/FCF inflection
# components only when revenue-context = still-declining.
COST_CUT_HAIRCUT = 0.6       # 0.5-0.7x
REVENUE_CONTEXT_SHORT_Q = 4  # 3-6q YoY window for classifier
REVENUE_CONTEXT_LONG_Q = 12
# One-off outlier damping (audit 2.6)
OUTLIER_K_MADS = 4.0         # k MADs from rolling median => outlier
OUTLIER_ROLLING_Q = 8
OUTLIER_TROUGH_DAMP = 0.3    # trough component multiplier when trough=outlier
# 5-yr peak constructs: median of top-3 quarters, not max.
PEAK_TOP_N = 3
# Minimum-observation soft confidence multipliers (audit 2.7)
MIN_OBS_LEVEL = 12
MIN_OBS_5YR = 16
SHORT_HISTORY_YEARS = 3
# Dollar->margin conversion: TTM revenue denominator floored at fraction of assets
MARGIN_DENOM_ASSET_FLOOR = 0.05
# FPI annual variant
FPI_SLOPE_WINDOW_FY = 4      # slope over 3-5 FYs
# Trough SEARCH window in FYs (the argmin reference); "trough within 2 FYs"
# is enforced by the recency ramp (zero past 8 quarter-equivalents), not by
# the search window — 2 FYs of annual points can't clear the >=4-obs guard.
FPI_TROUGH_WINDOW_FY = 4
# Peak theme (short side) intentionally uses 3 metrics (roic/gm/om); its
# breadth term is sqrt(breadth/3), not /6.
BREADTH_N_PEAK = 3
BREADTH_N_ANNUAL = 5         # annual FPI variant has no roic input
# Margin-vs-peer interaction gate (audit 5.3): the conjunctive peak score of
# a stable healthy business floors around ~0.35 (component floors + neutral
# slope), so the gate rescales [gate_lo, gate_hi] -> [0, 1] to make
# "unconditioned high margin contributes ZERO" actually hold.
PEER_INTERACTION_GATE = (0.35, 0.85)
# Cap the weight of yfinance-only signals for annual_only names so ADRs
# can't rank on momentum alone (audit 1.3): multiplier on the drawdown and
# squeeze rows.
ANNUAL_ONLY_MOMENTUM_CAP = 0.6

# ---------------------------------------------------------------------------
# ROIC / WACC (audit 3.5, 3.6)
# ---------------------------------------------------------------------------
TAX_RATE_CLAMP = (0.0, 0.30)
TAX_RATE_STATUTORY_FALLBACK = 0.23
IC_MIN_FRACTION_OF_ASSETS = 0.10   # |IC| < 10% assets -> EBIT/Assets + flag
ROIC_WINSOR = (-0.50, 0.50)
SPREAD_WINSOR_PCTL = (1, 99)
KD_MIN_DEBT_FRACTION_OF_ASSETS = 0.02
KD_CLAMP_OVER_RF = (0.01, 0.08)
ERP = 0.05                    # 4.5-5.5% single assumed figure
BETA_LOOKBACK = "2y"          # weekly returns vs S&P 500
BETA_CLAMP = (0.2, 3.0)
DEFAULT_BETA = 1.0
# Demonstrated-economics moat window (audit 4.2)
MOAT_DEMONSTRATED_WINDOW_Q = 12   # best rolling 12 consecutive quarters
MOAT_DEMONSTRATED_LOOKBACK_Q = 28 # trailing ~7 years
MOAT_TRAILING_YEARS = 5
# ROIC-WACC absolute/relative blend for moat level (audit 3.2 exception)
MOAT_SPREAD_ABS_BLEND = 0.4   # weight on is-spread-positive component

# ---------------------------------------------------------------------------
# Long composite kernels & weights
# ---------------------------------------------------------------------------
LONG_TURNAROUND_WEIGHT = 0.55
LONG_MOAT_WEIGHT = 0.45
# Turnaround theme weights (of turnaround bucket)
TURNAROUND_THEMES = {
    "inflection": 0.50,   # six metrics + op-margin-vs-peak
    "fscore": 0.15,
    "valuation_reset": 0.20,
    "squeeze": 0.15,
}
# Moat theme weights (of moat bucket). Valuation row carved from moat:
# ~5-10% of the LONG composite = ~0.07/0.45 of the moat bucket.
MOAT_THEMES = {
    "roic_spread": 0.40,
    "gm_stability": 0.25,
    "consistency": 0.20,
    "valuation": 0.15,    # 0.15*0.45 = 6.75% of composite (in 5-10% band)
}
# Drawdown kernel (audit 4.4): monotone-then-plateau on drawdown from 2-3yr
# high: 0 below 20%, linear to full credit at 35%, plateau to 80%, gentle
# taper beyond (slope such that 95% dd scores 0.85).
DRAWDOWN_LOOKBACK = "3y"
DRAWDOWN_KERNEL = {"zero_below": 0.20, "full_at": 0.35, "plateau_to": 0.80,
                   "taper_at_95": 0.85}
PRICE_CONFIRM_WEIGHT = 0.15   # small "price confirming" term inside the row
# F-score
FSCORE_ISSUANCE_THRESHOLD = 0.02   # shares YoY growth >2% fails issuance
# Squeeze: SI percentile x inflection sub-score (multiplicative)
# Survivability strip
RUNWAY_WARN_QUARTERS = 4
DILUTION_FLAG_2Y = 0.15
# Op-margin-vs-peak guards (audit 4.3)
STUMBLE_MAX_DECLINE_Q = 10    # sharp drop within ~6-10q after plateau
EROSION_MIN_DECLINE_Q = 12    # monotonic multi-year erosion
PEER_MIN_MEMBERS = 8

# ---------------------------------------------------------------------------
# Short composite kernels & weights
# ---------------------------------------------------------------------------
# SI confirmation is NOT a weighted theme anymore (user 2026-07-08): a name
# with no short interest must not be dragged down — "no SI could just mean
# the market hasn't priced it in yet." The composite is built from the three
# fundamental themes; SI adds an additive-only BONUS of up to
# SI_CONFIRMATION_BONUS points through the hump kernel (crowded shorts get
# a smaller bonus; absence gets zero bonus and zero penalty).
SHORT_THEMES = {
    "false_moat": 0.39,
    "inflated_growth": 0.39,
    "forensic": 0.22,
}
# --- Conviction blend (user 2026-07-09: the short composite over-penalized
# names strong on ONE mechanism — VITL/CAKE sat mid-pack because their real
# signals were diluted by non-firing components, and *absence* of a signal was
# treated as evidence against the short). The final short composite blends the
# theme-mean with a CONVICTION score = mean of the top-K strongest on-thesis
# signals across ALL themes, so a genuine single-mechanism short (pricing-
# masking-volume, priced-for-impossible, one-time-benefit-extrapolated, forensic)
# surfaces. TUNABLE — these drive how many names surface and where CAKE/VITL land.
# Grouped so conviction rewards firing DISTINCT mechanisms, not many correlated
# signals (a pure-richness biotech fires every valuation-linked signal and would
# otherwise dominate). conviction = mean of the top-K group maxes.
CONVICTION_GROUPS = {
    "valuation": ["ig_richness", "ig_expectations"],        # priced for unsustainable earnings
    "pricing_ran": ["ig_pmv", "ig_runup"],                  # ran on pricing / temp success
    "forensic": ["fo_beneish", "fo_sloan"],                 # accrual quality
    "capital": ["fm_leverage", "ig_buybacks", "ig_dilution"],  # financing prop
    "deterioration": ["ig_decel", "fm_peaked", "fm_margin_vs_peer",
                      "ig_dso_dio"],  # margin/growth rolling over (fm_compression
                      # dropped: subsumed by fm_peaked = max(peak, at_peak, compression))
    "one_time_income": ["fo_nonop"],                        # one-time income modeled forward
}
CONVICTION_TOPK = 3
CONVICTION_BLEND = 0.50
SI_CONFIRMATION_BONUS = 8.0
# --- Catalyst calendar (CAKE Apr-29 print / VITL Q4-Q1 floor drop): additive,
# gated bonus that re-prioritizes an already-fired short near its next print —
# it NEVER manufactures a short (gated on the pre-bonus short >= threshold).
# Mirrors the si_bonus additive pattern. QUARTER_DAYS/MIN_PERIODIC/
# LAG_FALLBACK/FY_MATCH_TOL feed the edgar.catalyst_calendar projection;
# WINDOW_DAYS = proximity ramp; BONUS_MAX + GUIDE_RESET_BONUS <= SI's +8.
CATALYST_QUARTER_DAYS = 91.25
CATALYST_MIN_PERIODIC = 4
CATALYST_LAG_FALLBACK_DAYS = 40
CATALYST_FY_MATCH_TOL_DAYS = 25
CATALYST_WINDOW_DAYS = 45
CATALYST_BONUS_MAX = 5.0
CATALYST_GUIDE_RESET_BONUS = 3.0

# --- Archetype verification checklist (tear-sheet render only; no scoring).
# Each fired short-side tag emits an ordered set of manual checks against the
# deep-linked filing — encodes the UAMY/CHH lesson: a fired flag is a prompt to
# VERIFY, not a conclusion. Ordered dict: render order = insertion order; keys
# must be TAG_COLS booleans.
ARCHETYPE_VERIFY_CHECKLIST = {
    "archetype_ran_on_temp_success": {
        "title": "Ran on temporary success (CAKE/VITL signature)",
        "primary_filing": "10-K",
        "steps": [
            "Read the MD&A price-vs-volume bridge: is the revenue run a one-off "
            "demand/price pull-forward, not a durable step-change?",
            "Confirm the peak quarters are not a COVID/commodity-spike window.",
            "Check whether the EV/normalized-EBIT richness rests on trailing-"
            "peak EBIT that is already rolling over."]},
    "priced_for_impossible_growth": {
        "title": "Reverse-DCF: priced above own history",
        "primary_filing": "10-K",
        "steps": [
            "Sanity-check the implied growth vs the company's own trailing "
            "revenue CAGR — is the gap real or a mix/acquisition artifact?",
            "Confirm the required steady-state margin exceeds every margin the "
            "company has actually printed."]},
    "multiple_disconnect": {
        "title": "Multiple-disconnect (cyclical earnings, durable multiple)",
        "primary_filing": "10-K",
        "steps": [
            "Confirm the disconnect runs rich-multiple-on-cyclical-earnings, "
            "not a commodity name where a low multiple is structural."]},
    "peer_multiple_disconnect": {
        "title": "Expensive AND deteriorating vs. peers",
        "primary_filing": "10-K",
        "steps": [
            "Verify the peer cohort is genuinely comparable (same 3-digit SIC).",
            "Confirm the trajectory is worse than the cheaper peers, not just a "
            "sector-wide slowdown."]},
    "pricing_masking_volume": {
        "title": "Pricing masking volume",
        "primary_filing": "10-Q",
        "steps": [
            "Read the latest MD&A price/volume decomposition; confirm unit "
            "volumes are declining while ASP props revenue."]},
    "ran_on_pricing_success": {
        "title": "Ran on pricing-driven success (may not screen expensive)",
        "primary_filing": "10-Q",
        "steps": [
            "Mechanism fired (pricing-masking-volume/at-peak + the stock ran) "
            "but this cohort has NO valuation gate — the name may be cheap on "
            "filed metrics (CAKE case: expensiveness is forward + peer-EV/EBITDA).",
            "Do the valuation work manually: peer EV/EBITDA vs the name's, and "
            "forward EPS if pricing reverses / margins normalize.",
            "Confirm the revenue run is price, not durable volume."]},
    "capacity_decay": {
        "title": "Growth is new capacity, not same-unit demand",
        "primary_filing": "10-K",
        "steps": [
            "PP&E-proxy says revenue growth trails capacity growth — pull the "
            "10-K unit/store count + same-store/organic comps (Item 2 / earnings "
            "release) to confirm the split; the PP&E proxy is confounded by M&A, "
            "remodels, and inflation.",
            "Check whether new-unit economics (AUV, returns) are below the base."]},
    "tax_tailwind": {
        "title": "EPS propped by a below-trend tax rate",
        "primary_filing": "10-K",
        "steps": [
            "Read the tax-rate reconciliation note; identify what dropped the "
            "effective rate (one-time credits, discrete items, jurisdiction mix) "
            "and whether it recurs.",
            "Re-strike EPS at the company's normalized/statutory rate."]},
    "ppi_windfall": {
        "title": "Margin/revenue riding an external price windfall that is reverting",
        "primary_filing": "10-K",
        "steps": [
            "The mapped industry PPI is elevated vs its 5yr trend and rolling "
            "over — confirm the company's margin/revenue boom is driven by that "
            "external price (MD&A price/volume), not durable.",
            "Verify the SIC->series mapping fits this company's actual output "
            "(the crosswalk is coarse/sector-level)."]},
    "real_rev_deflated": {
        "title": "Real (deflated) revenue growth negative while nominal positive",
        "primary_filing": "10-K",
        "steps": [
            "Nominal growth minus the industry PPI implies negative real (volume) "
            "growth — a deflated PROXY (index != the firm's realized price); "
            "confirm against MD&A price vs volume."]},
    "losing_share_priced_rich": {
        "title": "Losing share to peers while priced rich",
        "primary_filing": "10-K",
        "steps": [
            "Confirm the below-peer growth is share loss, not the company "
            "lapping a one-off; check the competitive/private-label context."]},
    "nonop_income_propping": {
        "title": "Earnings propped by below-operating-line income",
        "primary_filing": "10-Q",
        "steps": [
            "Tie the non-operating income to a specific line (gains on sale, "
            "breakage, other income); rule out a recurring source.",
            "Confirm operating income (not total EPS) is the deteriorating line."]},
    "false_moat_collision": {
        "title": "False-moat collision (demonstrated moat + short flags)",
        "primary_filing": "10-K",
        "steps": [
            "Re-examine the demonstrated-economics window — is a best-12q window "
            "ending 2021-22 a COVID/cyclical spike?",
            "Confirm the ROIC-WACC spread is not a fallen-angel artifact."]},
    "rev_gp_divergence": {
        "title": "Revenue / gross-profit divergence",
        "primary_filing": "10-K",
        "steps": [
            "Tie the divergence to a statement line; rule out an acquisition or "
            "accounting-change explanation in the notes."]},
    "restated": {
        "title": "Restatement / reliability flag",
        "primary_filing": "8-K 4.02 or 10-K/A",
        "steps": [
            "Pull the filing named in `restatement_filing` — an 8-K item 4.02 "
            "or the amended periodic report; this flag only fires when one "
            "exists, so there is always a document to read.",
            "Read the reason and distinguish a benign reclassification (the "
            "CHH lesson) from a real controls problem."]},
    "comparatives_represented": {
        "title": "Comparatives re-presented (not a restatement)",
        "primary_filing": "10-K",
        "steps": [
            "A retained value moved >3% from the originally-filed one with no "
            "amendment or 4.02 behind it — usually a segment recast or "
            "discontinued-operations reclassification. Confirm which, and do "
            "NOT pitch it as a restatement."]},
    "ma_flag": {
        "title": "Acquisition-driven growth",
        "primary_filing": "10-K",
        "steps": [
            "Confirm growth is organic, not roll-up/acquisition-driven; check "
            "goodwill and the acquisitions note."]},
    "secular_decline": {
        "title": "Secular / industry-wide decline",
        "primary_filing": "10-K",
        "steps": [
            "Confirm the sector fell in lockstep (a real short) vs a "
            "company-specific stumble that may mean-revert."]},
}
FALSE_MOAT_COMPONENTS = {  # within false-moat theme
    "peaked_rolling_over": 0.45,
    "margin_vs_peer_interaction": 0.30,
    "leverage_propping": 0.25,
}
INFLATED_GROWTH_COMPONENTS = {
    "revenue_decel": 0.20,
    "richness": 0.20,          # blend: EV/Sales pctile + EV/normalized-EBIT pctile
    "runup": 0.16,             # stock ran on temporary success (user 2026-07-08)
    "pricing_masking_volume": 0.10,
    "dso_dio_expansion": 0.08,
    "debt_funded_buybacks": 0.08,
    "dilution": 0.08,
    # expectations-vs-trajectory (2026-07-08): valuation-implied steady-state
    # op margin ABOVE the name's own historical peak (reverse-DCF margin-ceiling
    # gap) — the market prices margins the business has never achieved. Weighted
    # row, orthogonal to decel (rho .01). NOTE: this is the margin-ceiling gap
    # ONLY. Relative-share loss was initially blended in but REMOVED after the
    # CAKE/VITL replay showed revenue-vs-peers is wrong-sign for windfall/
    # pricing-masking-volume names (it now drives only the losing_share_priced_
    # rich tag). rdcf_growth_gap also NOT here (rho .76 vs ev_normalized = redundant).
    "expectations": 0.10,
}
# --- Run-up row (user direction 2026-07-08: "the stock ran recently on
# temporary success") — mirror of the long drawdown kernel, interaction-gated
# by fundamental-peak evidence so a deserved compounder high scores ~0.
# proximity: full credit within 10% of the 2-3yr high, tapering to 0 at 35%
# below it; appreciation: 0 below +30% 12-mo return, full at +80%.
# run-up evidence = max(proximity, appreciation); row = evidence x gate.
RUNUP_KERNEL = {"dd_full_within": 0.10, "dd_zero_beyond": 0.35,
                "ret12_zero": 0.30, "ret12_full": 0.80}
# --- At-peak primitive (catches the VITL/CAKE SETUP before the rollover the
# peak-recency primitive requires). Strict PRODUCT of three components
# (discovery variant — narrow by design):
#   level gate  : own-history margin/ROIC percentile, 0 below lo -> 1 at hi
#   delta gate  : percentile of the CURRENT 4q-change within the name's own
#                 history of rolling 4q-changes (kills chronic margin
#                 expanders: their current delta is typical for themselves)
#   peak recency: TTM argmax within the trailing window, ramp to 0 by zero_q
# recency_zero_q = 6 (not 4): the as-of clock makes even a live filing ~1q
# old, so a 4q ramp would mechanically dock every name ~25%; at 6q a name
# whose latest data IS the peak keeps ~0.8+, a year-stale one decays to ~0.2
# level gates are on the SHRUNK own-history percentile (n/(n+8) toward 50),
# whose ceiling is ~91 even for a maximal spike on 40q of history — so
# level_hi is 90, not 95+
AT_PEAK = {"level_lo": 75.0, "level_hi": 90.0,
           "delta_lo": 60.0, "delta_hi": 90.0,
           "recency_zero_q": 6.0, "ref_window_q": 10}
AT_PEAK_METRIC_WEIGHTS = {"roic": 0.4, "gm": 0.3, "om": 0.3}
# at-peak contributes to the false-moat "peaked" row at partial credit
# (setup, not confirmation); rolling-over keeps full credit
AT_PEAK_FM_CREDIT = 0.7
# --- ROIC-WACC spread compression (CAKE spread 750->130bps): the moat is
# rolling over. Extends moat_spread (reuses its per-quarter spread series);
# folded into the fm_peaked row via max() (NOT a new weighted component —
# would double-count peak_score op-margin-rollover and leverage_propping's
# roic_falling). slope_window_q = trailing window for Theil-Sen slope + peak;
# min_peak_spread = the window-peak spread must have been a real moat (>=200bps)
# to count as one being lost (anti-double-count / applicability gate);
# buffer band = fraction of the peak spread consumed; slope_full_bps_q =
# -50bps/quarter -> full slope credit; cross_horizon_q = projected quarters to
# cross zero for crossing_soon.
COMPRESSION = {"slope_window_q": 12, "min_obs": 6, "min_peak_spread": 0.02,
               "buffer_zero_below": 0.30, "buffer_full_at": 0.80,
               "slope_full_bps_q": 0.005, "cross_horizon_q": 6.0,
               # now_ceiling: the compression geometry only earns credit when
               # the CURRENT spread is actually thin — a spread that shrank
               # from a high peak but is still large (40%->20%) is NOT a false
               # moat. Credit scales (now_ceiling - spread_now)/now_ceiling,
               # 0 once the live spread exceeds now_ceiling (8% = 800bps). The
               # crossing_soon case (already-negative / about to cross) bypasses
               # this and keeps full credit. Added 2026-07-08: first-run
               # diagnostics showed 46% of names scoring >0.9 without it.
               "now_ceiling": 0.08}
# credit applied to compression_score before it enters the fm_peaked max();
# 0.7 = a rollover SETUP (same tier as at_peak's 0.7), not full confirmation —
# so compression alone cannot set fm_peaked to 100 and dominate the theme.
COMPRESSION_FM_CREDIT = 0.7
# --- Pricing-masking-volume signature (the CAKE tell): GM expanding while
# revenue growth decays toward low single digits and asset turnover falls —
# revenue growth that is all price, no volume. Conjunctive geometric mean of:
#   gm_expand : TTM GM change over 4q, full credit at +2pp
#   growth_decay: revenue YoY, 1.0 at <=2%, 0 at >=8%; 0 below -2% (outright
#                 decline is the deceleration row's territory)
#   turnover_fall: relative fall in TTM revenue/avg assets over 4q, full at -5%
PMV = {"gm_expand_full_pp": 0.02, "growth_full": 0.02, "growth_zero": 0.08,
       "growth_floor": -0.02, "turnover_fall_full": 0.05, "flag_at": 0.5}
# --- Normalized-margin valuation (both sides; user 2026-07-08): EV over
# (own full-history MEDIAN TTM op margin x current TTM revenue) — richness
# vs mean-reverted earnings. Catches peak-margin names that look cheap on
# current earnings (short) and trough names cheap on normal earnings (long).
NORMALIZED_MARGIN_MIN = 0.005   # median margin below this -> ratio undefined
NORMALIZED_MIN_OBS = 12
# --- "Ran-on-temporary-success" archetype cohort (user 2026-07-08): names
# carrying the full CAKE/VITL signature regardless of composite rank —
# price-masking-volume OR at-peak setup, AND the stock ran, AND rich on
# normalized earnings. Emitted as its own output section + tag; the
# composite stays honest (accounting evidence may be thin), the cohort is
# the discovery list.
ARCHETYPE_ROTS = {"pmv_min": 0.40, "at_peak_min": 0.50, "runup_min": 0.60,
                  "ev_norm_rich_min": 70.0}
# --- Reverse-DCF (CAKE move 6 / VITL move 5): back out the revenue growth and
# steady-state op margin the current EV requires and compare to the company's
# own trajectory. Discovery tag only (correlates with decel/ev_normalized which
# are already weighted in inflated_growth); no composite weight. explicit_years
# = stage-1 horizon; terminal_growth = Gordon perpetuity g; min_wacc_spread
# guards the wacc<=g degeneracy; g_search brackets the bisection; trailing_years
# = own-CAGR lookback for the growth_gap.
REVERSE_DCF = {"explicit_years": 5, "terminal_growth": 0.025,
               "min_wacc_spread": 0.01, "g_search": (-0.5, 1.0),
               "trailing_years": 3}
# Fire gates for the priced_for_impossible_growth cohort: implied growth >= 5pp
# above own trailing CAGR AND required op margin >= 3pp above own peak AND
# corroborating EV/normalized-EBIT richness percentile.
# thresholds tightened 2026-07-08: 169(7.9%)->131(6.1%)->104(4.9%) at these
# values. This is as tight as the exemplar allows — VITL @2025-10-15 (growth_gap
# .102, margin_ceiling .076, ig_ev_norm 91) clears here; growth_gap>=.12 would
# hit ~4.8% but DROP VITL, so 4.9% is the floor that keeps the validated short.
REVERSE_DCF_TAG = {"growth_gap_min": 0.10, "margin_ceiling_gap_min": 0.06,
                   "ev_norm_rich_min": 80.0}
# --- Story-stock short score for the ZERO/LOW-REVENUE segment (user
# 2026-07-09, ASTS exemplar: pitched short ~2026-01-27 while buried at
# 85/304 in the segment). Pre-revenue names have no margins, so the main
# composite's themes are mostly nulled by design (audit 5.5 stands for the
# MAIN list) — but a story stock that RAN while funding itself with paper
# is exactly "ran on temporary success". Segment-only ranking score:
#   runup x dilution-gate : the run, gated by issuance evidence (dilution
#                           rank/100) instead of the margin gate that
#                           cannot exist here
#   dilution rank         : share-count growth percentile (the machine)
#   p_tbv rank in segment : richness vs book — the only valuation anchor
#                           left without revenue
#   sloan rank            : accrual quality where computable
# SI adds the same additive-only bonus as the main composite.
STORY_SHORT_WEIGHTS = {"runup_gated": 0.35, "dilution": 0.30,
                       "ptbv_rich": 0.20, "sloan": 0.15}
# --- Multiple-disconnect tag (user 2026-07-09, refining on CENX/BLBD vs
# UAMY/VITL): the short archetype is NOT "cyclical at a top" — it is a name
# whose earnings are cyclical/windfall-driven but whose MULTIPLE says the
# market is pricing a durable story. A smelter at 2x sales already trades
# like a cyclical (no disconnect, no short); an egg brand at premium
# multiples or a $1B story at 27x sales does not.
#   cyclical-earnings evidence = max(at-peak setup, pricing-masking-volume,
#     rescaled rolling-over score, 0.6 x rev-vs-GP-divergence flag)
#   disconnect fires when evidence >= evidence_min AND the EV/Sales
#     percentile >= multiple_min (the market is NOT applying a cyclical
#     multiple)
MULTIPLE_DISCONNECT = {"evidence_min": 0.40, "multiple_min": 70.0,
                       "divergence_evidence": 0.60}
# --- Directional peer multiple-disconnect (CAKE 9.8x vs declining peers
# 4.2-6.8x): expensive vs the peer cohort AND worse trajectory than those same
# peers. Complements the internal-evidence multiple_disconnect (which has no
# peer group / no direction). Discovery tag only. min_members = cohort floor
# (looser than PEER_GROUP_MIN=8 used for margin pools; passed explicitly);
# val_pctl_min = expensive-vs-peer gate; traj_pctl_max = weak-trajectory gate;
# strength_min = floor on val*(1-traj) so BOTH conditions bind; confirm_ev_
# ebitda_pctl = EV/EBITDA confirmer, applied only when computable.
PEER_DISCONNECT = {"min_members": 6, "val_pctl_min": 70.0, "traj_pctl_max": 40.0,
                   "strength_min": 45.0, "confirm_ev_ebitda_pctl": 60.0}
# --- Relative-share (VITL 76%->54% but Street presumes durable): own TTM
# revenue growth minus the peer-median growth = whether the name is losing
# share to its cohort, gated by valuation richness (a mispriced-durability
# short). Pure cross-sectional, no external data. Discovery tag. rich_min =
# EV-richness percentile gate; gap_cap caps the below-peer growth gap fed into
# the strength. (The multi-quarter FADE-SLOPE variant is a follow-up needing a
# per-quarter peer panel; this ships the LEVEL/loser term.)
RELATIVE_SHARE = {"min_members": 6, "rich_min": 70.0, "gap_cap": 0.20}
FORENSIC_COMPONENTS = {
    "beneish": 0.35,
    "sloan": 0.35,
    "event_flags": 0.15,   # restated/NT/4.01/4.02 small soft bump
    "nonop_share": 0.15,   # earnings propped by below-operating-line income
}
# --- Non-operating / one-time earnings share (CAKE QoE bridge): share of
# pretax income from below the operating line (ex-interest). score scales on
# the LEVEL, zeroed when clearly falling (improving quality). share_zero/full
# = the level kernel band; yoy_min = the "not falling" gate; flag_share/flag_yoy
# = the checklist tell (high AND clearly rising).
NONOP_SHARE = {"share_zero": 0.05, "share_full": 0.30, "yoy_min": 0.0,
               "flag_share": 0.12, "flag_yoy": 0.04}
# --- Tier-2: capacity-turnover decay (CAKE 'all growth is new units'). organic
# proxy = revenue YoY - net PP&E YoY. Asset-light guard: net PP&E must be >=
# CAPACITY_MIN_PPE_ASSETS of total assets (else the ratio is noise). Fires when
# revenue grows > rev_grow_min AND capacity expands > ppe_grow_min AND organic
# proxy < organic_max (growth is capacity-driven, not same-unit demand).
CAPACITY_MIN_PPE_ASSETS = 0.15
CAPACITY_DECAY = {"rev_grow_min": 0.02, "ppe_grow_min": 0.05,
                  "organic_max": 0.0, "organic_full": 0.15}
# Plausibility bound on the organic proxy (2026-07-25). rev_yoy - ppe_yoy is a
# capacity-turnover measure; at -65.6 (RJET: net PP&E grew 6,558pp more than
# revenue in a year) the input is an accounting discontinuity — fresh-start
# accounting, a fleet purchase, a lease capitalization — not overcapacity.
# ma_flag did not catch it, and the old clip at 1.0 HID it by lumping it in
# with 21 other names at the cohort max. Same class as
# MARGIN_VS_PEAK_MAX_PLAUSIBLE: null the reading rather than score it.
CAPACITY_ORGANIC_IMPLAUSIBLE = 1.5   # 150pp of PP&E-over-revenue growth
# --- Tier-2: tax-tailwind EPS (QoE). Effective rate computed UNCLAMPED (the
# TAX_RATE_CLAMP above would hide genuine benefits). boost = fraction of net
# income from paying below the company's own median rate. boost_full = full
# score; flag when boost >= flag_boost AND the rate dropped >= rate_drop_min.
# flag_boost/rate_drop tightened 2026-07-08 in two steps: 0.05/0.05 -> 11.5%
# (noise; TTM rates wander ~5pp) -> 0.10/0.10 -> 6.0% -> 0.15/0.13 (rate 13pp
# below the name's own median AND the benefit >=15% of NI = an unambiguous,
# large tax event). boost_full raised to 0.25 so the score keeps range above
# the flag point.
TAX_TAILWIND = {"boost_full": 0.25, "flag_boost": 0.15, "rate_drop_min": 0.13}
# --- Tier-2 FRED signals (enabled 2026-07-08 via the FRED API key). SIC major
# group (2-digit) -> FRED PPI series (all verified to resolve). Coarse by design
# (sector-level, not name-specific) — a research prompt, tag not weight.
SIC_PPI_CROSSWALK = {
    1: "WPU01", 2: "WPU01",       # agriculture -> farm products
    10: "WPU10", 14: "WPU10", 33: "WPU10",  # metal mining / minerals / primary metals
    12: "WPU051",                 # coal
    13: "WPU0561",                # oil & gas extraction / field svcs -> crude
    20: "WPU02",                  # food & kindred -> processed foods
    24: "WPU08",                  # lumber & wood
    26: "WPU09",                  # paper
    28: "WPU06",                  # chemicals
    29: "WPU057",                 # petroleum refining -> refined petroleum
}
# Per-name overrides where the company's specific output commodity is known and
# the SIC group is too broad (curated, mechanism-based — extend as needed). VITL
# is generic food (SIC 2000) but sells eggs, so the egg PPI is its real
# output-price series.
PPI_TICKER_OVERRIDE = {"VITL": "WPU017107"}   # eggs
# windfall: index >= z_min sigma above its trailing-trend_window_m-month mean AND
# rolling over (>= rolloff_min below its window peak). real-rev deflates nominal
# revenue YoY by the index YoY.
PPI_WINDFALL = {"z_min": 1.0, "rolloff_min": 0.03, "trend_window_m": 60,
                "min_obs_m": 24}

# --- Sector/attribute short cohorts (2026-07-08, user pitch-preference). The
# main composite is forensic-heavy and skews pharma/semi, so simple consumer
# shorts sit mid-pack. These cohorts filter by sector/attribute and rank by how
# many ON-THESIS discovery tags fire (not the composite) — surfacing pitchable
# names the ranking buries. SIC sets are 2-digit major groups (tunable).
CONSUMER_SICS = {20, 21, 22, 23, 25, 31, 39,          # food/bev, apparel, furniture, misc consumer mfg
                 52, 53, 54, 55, 56, 57, 58, 59,      # retail (58 = restaurants)
                 70, 78, 79}                          # lodging, movies, recreation
INDUSTRIAL_SICS = {15, 16, 17,                        # construction
                   34, 35, 37,                        # fabricated metal, machinery, transport equip
                   50}                                # durable-goods wholesale/distribution
# NEVER-semis (taxonomy rule 7): chips/PCBs/semicap-equipment/semi-test SICs,
# excluded from ALL sector cohorts (e.g. VECO/ACMR are SIC 3559 semicap that
# would otherwise leak into industrials via the 2-digit 35 group).
SEMI_EXCLUDE_SICS = {3559, 3572, 3576, 3577, 3670, 3671, 3672, 3674, 3675,
                     3676, 3677, 3678, 3679, 3825, 3826, 3827}
# pharma/biotech — "hard to pitch" (user 2026-07-09), excluded from the
# pitchable short list (still scored + in the raw universe/shortlist)
PHARMA_BIOTECH_SICS = {2834, 2835, 2836, 8731}
# pitchable_shorts: the conviction-ranked short list with the hard-to-pitch
# sectors removed (semis + pharma/biotech). Commodity names are KEPT (the
# CENX-style exited-cycle disconnect is on-thesis). Capped, with a high-
# confidence top tier by conviction.
PITCHABLE_EXCLUDE_SICS = SEMI_EXCLUDE_SICS | PHARMA_BIOTECH_SICS
PITCHABLE_CAP = 40
# high-confidence tier: fires >= this many DISTINCT on-thesis mechanism tags
# (multiple independent props > one strong-but-lonely signal). ~3-5 names clear it.
PITCHABLE_HICONF_TAGS = 3
# on-thesis short mechanisms (the "unsustainable prop" tags) — the cohort rank key
ON_THESIS_SHORT_TAGS = ["pricing_masking_volume", "ran_on_pricing_success",
                        "capacity_decay", "priced_for_impossible_growth",
                        "multiple_disconnect", "losing_share_priced_rich",
                        "real_rev_deflated"]
# for the commodity cohort: the DISCONNECT tags (market NOT pricing it as a pure
# cyclical) — the CENX "exited the cycle but financials look mid-cycle" shape
COMMODITY_DISCONNECT_TAGS = ["multiple_disconnect", "ran_on_pricing_success",
                             "pricing_masking_volume", "real_rev_deflated",
                             "priced_for_impossible_growth"]
# ---------------------------------------------------------------------------
# LONG-side discovery cohorts (2026-07-25 audit, top recommendation)
# ---------------------------------------------------------------------------
# The structural asymmetry the audit found: the short side has 10+ cohort CSVs
# that surface an archetype REGARDLESS of composite rank, while the long side
# had exactly one route — long_composite >= threshold. So a long archetype the
# composite is not built to reward is not merely low-ranked, it has no output
# file at all and is invisible. 13 of the top 20 names by inflection_score
# missed the shortlist entirely (AXTI at rank 1161/2138).
#
# These three mirror the short-side pattern: ranked on their OWN evidence, no
# composite pre-cut, emitted as additive CSVs. They change no score and no
# existing ranking — a name appearing here is a candidate to research, exactly
# as `consumer_shorts` is on the other side.
LONG_COHORTS = {
    # "great business the market has de-rated" — quality intact, price is not.
    # Economics-intact leg is an OR because a moat can persist either as a
    # still-positive current spread or as most of the demonstrated spread
    # retained; requiring both would demand the trough not have happened.
    "derated_compounder": {
        "moat_rank_min": 70.0,       # m_roic_spread percentile
        "retention_min": 0.60,       # moat_trailing_5y / moat_demonstrated
        "drawdown_min": 0.20,        # de-rated on price...
        "valuation_rank_min": 60.0,  # ...or cheap on the valuation composite
    },
    # "the fix is real, the moat is ordinary" — the archetype the moat-first
    # composite structurally cannot reward, since t_inflection is range-
    # compressed (inflection_score caps ~0.58) against 0-100 percentile rows.
    # No moat gate by design. Red traffic light excluded: an inflection on a
    # name that may not survive to realize it is a falling knife, not a pitch.
    "inflecting_thin_moat": {
        "exclude_traffic_light": ("red",),
        "min_inflection": 0.20,
    },
    # The one long shape the adverse literature endorses: deep drawdown, cheap,
    # and DEMONSTRABLY SOLVENT. The survivability legs are the whole point —
    # without them this is a falling-knife screen.
    "surviving_distressed_value": {
        "drawdown_lo": 0.35,
        "drawdown_hi": 0.80,
        "valuation_rank_min": 70.0,
        "require_traffic_light": "green",
        "f_score_min": 7.0,
    },
}
# Raised 2026-07-25 for the point-in-time calibration run: at 60 the CAP was
# doing the selecting rather than the gates (MTN cleared every derated_compounder
# leg and was cut at row 60 of 203; POOL/PATK/TYL likewise in pitchable_longs).
# Set the production value from the replay evidence, not from taste.
LONG_COHORT_CAP = 150  # rows per cohort CSV; discovery lists, not portfolios

# ---------------------------------------------------------------------------
# PITCHABLE LONGS (2026-07-25) — the long mirror of `pitchable_shorts`
# ---------------------------------------------------------------------------
# The objective here is winning a stock pitch COMPETITION, which is not the same
# objective as making money, and the two must not be conflated. Measured on the
# 21 longs that placed 1st or 2nd at a sponsored/flagship competition (UNC Alpha
# Challenge, Point72, Balyasny, UIC, Sohn):
#     consumer-facing SIC   40% of winners vs 14% of the universe
#     <= 2 reportable segs  73% of winners vs 54% of the universe
# The non-consumer winners are still single-line businesses a judge can picture
# in one sentence (POOL pool supplies, CLH hazardous waste, WMS drainage pipe,
# PATK RV parts, MIDD commercial ovens). The hard-to-explain names (ENTG
# semiconductor materials, TYL government software) are the minority.
#
# User's own evidence for why this axis and not realized return: they pitched
# APH (long) and lost only to BIRK the same day. APH went on to +25% vs BIRK +5%
# and APH was added to a $7B book at their firm — but BIRK won the room because
# it was the easier story. Selecting on returns would have picked the loser here.
#
# So: the STORY ARC is the screener's original thesis (a good business at a
# temporary trough — reversal / turnaround / counterconsensus), and the STORY
# QUALITY legs are what make it pitchable in 90 seconds.
LONG_PITCHABILITY = {
    # --- story arc: something went wrong, to a business still worth owning ---
    "min_drawdown": 0.20,        # the counterconsensus entry point
    "min_moat_rank": 60.0,       # on moat_rank_effective (NaN-tolerant, see below)
    # --- story quality: can a judge follow it without a primer? ---
    "max_segments": 2,           # single-line business
    "analyst_lo": 3.0,           # covered enough that the name is recognizable...
    "analyst_hi": 30.0,          # ...but not so consensus there is no variant view
}
# Same hard-to-pitch sectors the short side already excludes (semis + pharma /
# biotech): "no more insanely difficult to understand companies."
LONG_PITCHABLE_EXCLUDE_SICS = PITCHABLE_EXCLUDE_SICS
LONG_PITCHABLE_CAP = 100

# Revenue deceleration definition (audit 5.1)
DECEL_HIGH_BASE_PEAK_YOY = 0.20     # or top quintile of universe peaks
DECEL_LOOKBACK_Q = 10               # 8-12q of YoY growth
DECEL_SLOPE_WINDOW_Q = 4
MA_GOODWILL_JUMP = 0.20             # goodwill +>20% YoY -> M&A flag
MA_ACQ_REVENUE_FRACTION = 0.05
# SI hump kernel (short confirmation): 0 at 0%, rising to full credit at
# ~10-15% SI, flat to ~20%, declining to 0.25 by 35%+ (crowded).
SI_HUMP = {"rise_full_at": 0.12, "flat_to": 0.20, "crowded_at": 0.35,
           "crowded_score": 0.25}
# Beneish
BENEISH_INDEX_WINSOR = (0.1, 10.0)
BENEISH_MIN_COMPONENTS = 6
# Zero/low-revenue cohort (audit 5.5)
ZERO_REV_TTM = 10e6
ZERO_REV_MKTCAP_FRACTION = 0.01
# Sector-wide-decline haircut on false-moat theme (short composite only;
# the audit-4.2 long tear-sheet cross-check reads the RAW pre-haircut score)
SECTOR_DECLINE_HAIRCUT = 0.75

# ---------------------------------------------------------------------------
# Scoring foundations
# ---------------------------------------------------------------------------
SECTOR_BUCKET_MIN = 15        # fallback universe-wide below this
PEER_GROUP_MIN = 8
LOW_CONFIDENCE_COVERAGE = 0.60
# Shrink bucket scores toward universe median in proportion to missing weight
# (implemented as score = cov*score + (1-cov)*50).
# Leverage-propping definition: ROE-ROA gap widening OR debt/EBITDA rising
# while ROIC falls (both computed; max of the two rank-scores).

# ---------------------------------------------------------------------------
# Shortlist assembly
# ---------------------------------------------------------------------------
SHORTLIST_HARD_CAP = 100
SHORTLIST_SCORE_THRESHOLD = 60.0   # tunable soft threshold (percentile pts)
SECTOR_CAP_PER_SIDE = 14           # soft per-sector cap ~12-15
CONTESTED_DECILE = 0.90
COLLISION_QUARTILE = 0.75          # false-moat-collision: top-quartile both
COLLISION_MIN_COVERAGE = 0.60      # tag suppressed below low-confidence cutoff

# ---------------------------------------------------------------------------
# Tag-fallback ladders (audit 1.2). Order matters. "derive" entries are
# handled in ladders.py. us-gaap unless prefixed.
# ---------------------------------------------------------------------------
TAG_LADDERS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "RevenuesNetOfInterestExpense",           # financials
        "InterestAndDividendIncomeOperating",     # financials
    ],
    "cogs": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",  # summed with CostOfGoodsSold in derive rule
    ],
    "gross_profit": ["GrossProfit"],  # else revenue - cogs (derive)
    "sga": [
        "SellingGeneralAndAdministrativeExpense",
        # else SellingAndMarketingExpense/SellingExpense + G&A (derive)
    ],
    "sga_selling": ["SellingAndMarketingExpense", "SellingExpense"],
    "sga_ga": ["GeneralAndAdministrativeExpense"],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "IncomeLossFromContinuingOperationsNetOfTax",
    ],
    "continuing_income": [
        "IncomeLossFromContinuingOperationsNetOfTax",
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestAndDebtExpense",
        "InterestIncomeExpenseNet",  # sign check in derive rule
    ],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
    ],
    "dep_amort": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ],
    "depreciation_only": ["Depreciation"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "st_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent"],
    "assets": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "inventory": ["InventoryNet", "InventoryGross"],
    "ppe_net": ["PropertyPlantAndEquipmentNet", "PropertyPlantAndEquipmentGross"],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsNotesAndLoansReceivableNetCurrent",
    ],
    "goodwill": ["Goodwill"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "lt_debt_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "lt_debt_current": ["LongTermDebtCurrent"],
    "st_borrowings": ["ShortTermBorrowings", "DebtCurrent",
                      "OtherShortTermBorrowings", "CommercialPaper"],
    "finance_lease": ["FinanceLeaseLiability",
                      "FinanceLeaseLiabilityNoncurrent"],
    "debt_next_12m": [
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
        "DebtCurrent",  # spec fallback: NextTwelveMonths -> DebtCurrent -> n/a
    ],
    "shares_outstanding": [
        "dei:EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
    "diluted_was": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "acquisitions": ["PaymentsToAcquireBusinessesNetOfCashAcquired"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "ebit_proxy": ["OperatingIncomeLoss"],
    # Banks / insurers
    "noninterest_expense": ["NoninterestExpense"],
    "net_interest_income": ["InterestIncomeExpenseNet"],
    "noninterest_income": ["NoninterestIncome"],
    "policyholder_benefits": ["PolicyholderBenefitsAndClaimsIncurredNet"],
    "premiums_earned": ["PremiumsEarnedNet"],
    # REIT FFO proxy gains fallback chain
    "gains_on_sale": [
        "GainLossOnSaleOfProperties",
        "GainsLossesOnSalesOfInvestmentRealEstate",
        "GainLossOnDispositionOfAssets1",
        "GainLossOnDispositionOfAssets",
    ],
    # Display-only (tear sheet tier 2; audit 6.5 — never scored)
    "reportable_segments": ["NumberOfReportableSegments"],
    # One-off / core-series tag list (audit 2.6)
    "one_off": [
        "GoodwillImpairmentLoss",
        "AssetImpairmentCharges",
        "ImpairmentOfIntangibleAssetsExcludingGoodwill",
        "RestructuringCharges",
        "RestructuringSettlementAndImpairmentProvisions",
        "GainLossOnDispositionOfAssets1",
        "GainLossOnDispositionOfAssets",
        "LitigationSettlementExpense",
        "GainLossRelatedToLitigationSettlement",
    ],
}

# ifrs-full ladder for FPI annual variant (small, per audit 1.3)
IFRS_LADDERS: dict[str, list[str]] = {
    "revenue": ["ifrs-full:Revenue", "ifrs-full:RevenueFromContractsWithCustomers"],
    "gross_profit": ["ifrs-full:GrossProfit"],
    "operating_income": ["ifrs-full:ProfitLossFromOperatingActivities"],
    "net_income": ["ifrs-full:ProfitLoss",
                   "ifrs-full:ProfitLossAttributableToOwnersOfParent"],
    "cfo": ["ifrs-full:CashFlowsFromUsedInOperatingActivities"],
    "assets": ["ifrs-full:Assets"],
    "equity": ["ifrs-full:Equity"],
    "cash": ["ifrs-full:CashAndCashEquivalents"],
}

# ASC-606 revenue-tag switch stitching: ladders are recency-aware per-period;
# the resolver picks, per period, the highest-priority tag with coverage in
# that period's neighborhood and stitches series (see ladders.py).
ASC606_SWITCH_YEAR = 2018

# 8-K item code legend (tear sheet tier 2)
ITEM_8K_LEGEND = {
    "2.02": "results of operations",
    "5.02": "director/officer departure or election",
    "2.05": "costs associated with exit/disposal (restructuring)",
    "2.06": "material impairments",
    "1.01": "entry into material agreement",
    "4.01": "auditor change (review prompt; benign rotations exist)",
    "4.02": "non-reliance on previously issued financials (high signal)",
    "3.01": "delisting notice",
}
