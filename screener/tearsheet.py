"""Per-name tear sheets (spec audit 6.3; audit 4.2 cross-check is Tier 1).

Tier 1 (must-have): composites + every theme/sub-score; per inflection metric
the level/slope/trough triplet; quarterly + TTM series; all tags; coverage %
with missing metrics listed by name; guard/clamp flags; and — on every
LONG-side tear sheet — the short-side false-moat cross-check (raw pre-haircut
theme score + components + demonstrated-window dates) rendered next to the
moat rows.

Tier 2 (best-effort, nullable): trailing 8-K item codes with legend, next
earnings date, short interest + staleness stamp + shortPercentOfFloat
(display only), EDGAR links, business summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

TAG_COLS = ["fin_sector", "is_commodity", "cyclical", "sector_driven",
            "secular_decline", "no_peer_data",
            "cost_cut_inflection", "decline_shape", "restated",
            "comparatives_represented", "nt_filer",
            "auditor_change", "non_reliance", "low_confidence",
            "annual_only", "contested", "on_both_lists",
            "false_moat_collision", "zero_low_revenue", "stale",
            "ma_flag", "rev_gp_divergence", "beneish_sgi_dominant",
            "is_despac_name", "cap_fallback_dei_multiclass",
            "pricing_masking_volume", "archetype_ran_on_temp_success",
            "multiple_disconnect", "peer_multiple_disconnect",
            "priced_for_impossible_growth", "nonop_income_propping",
            "losing_share_priced_rich", "catalyst_imminent",
            "guide_reset_window", "ran_on_pricing_success",
            "capacity_decay", "tax_tailwind",
            "ppi_windfall", "real_rev_deflated"]

RAW_INPUT_COLS = ["inflection_score", "margin_vs_peak", "f_rising",
                  "valuation_reset", "si_ratio", "moat_demonstrated",
                  "gm_level", "gm_stability", "earnings_consistency",
                  "ev_sales", "ev_normalized", "p_tbv", "peak_score",
                  "at_peak_score", "sloan_latest",
                  "beneish_m", "decel_score_raw", "pmv_score_raw",
                  "dso_yoy_delta", "dio_yoy_delta", "dilution_yoy",
                  "leverage_prop_raw", "buyback_score_raw", "ret_12m"]


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "YES" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
        return "n/a" if not np.isfinite(f) else f"{f:.{nd}f}"
    except (TypeError, ValueError):
        return str(v) if str(v) not in ("nan", "None", "") else "n/a"


def _archetype_checklist_block(row: pd.Series, details: dict) -> list[str]:
    """Ordered verification checklist for each FIRED short-side archetype tag,
    each deep-linked to the relevant filing. Returns markdown lines (empty when
    no tag fired). Consumes only tags already on the row + the Tier-2 accn
    links — no new computation, no scoring impact."""
    t2 = details.get("tier2", {})
    cik = row.get("cik")

    def _link(kind: str) -> str:
        accn = (t2.get("latest_10q_accn") if kind == "10-Q"
                else t2.get("latest_10k_accn"))
        if accn and cik is not None:
            try:
                return ("https://www.sec.gov/Archives/edgar/data/"
                        f"{int(cik)}/{str(accn).replace('-', '')}")
            except (ValueError, TypeError):
                return ""
        return ""

    lines: list[str] = []
    for tag, spec in config.ARCHETYPE_VERIFY_CHECKLIST.items():
        if tag not in row.index:
            continue
        v = row.get(tag)
        if not (bool(v) and str(v) not in
                ("nan", "n/a", "no", "False", "unknown", "-1")):
            continue
        lines.append(f"### ☐ {spec['title']}  `{tag}`")
        pf = spec.get("primary_filing", "10-K")
        link = _link(pf)
        lines.append(f"Primary filing: [{pf}]({link})" if link
                     else f"Primary filing: {pf} (link unavailable)")
        for i, step in enumerate(spec["steps"], 1):
            lines.append(f"{i}. {step}")
        lines.append("")
    return lines


def render(row: pd.Series, details: dict, side: str) -> str:
    """row: scored universe row; details: per-name detail blob; side:
    'long'|'short'. Returns markdown."""
    L = []
    t = row.get("ticker", "?")
    L.append(f"# {t} — {row.get('name', '')}  [{side.upper()} tear sheet]")
    L.append(f"*{row.get('sic_desc', '')} (SIC {row.get('sic', '?')}) | "
             f"mktcap ~${(row.get('mktcap_build') or 0) / 1e9:.1f}B | "
             f"data as of {row.get('data_as_of', '?')} "
             f"({_fmt(row.get('staleness_days'), 0)} days old)*")
    L.append("")

    # ---- composites & themes ----
    L.append("## Composites")
    L.append(f"| | score | coverage |")
    L.append(f"|---|---|---|")
    L.append(f"| LONG composite | {_fmt(row.get('long_composite'), 1)} | "
             f"{_fmt(row.get('cov_long'), 2)} |")
    L.append(f"| SHORT composite | {_fmt(row.get('short_composite'), 1)} | "
             f"{_fmt(row.get('cov_short'), 2)} |")
    L.append("")
    L.append("### Long themes")
    for c, lbl in [("bucket_turnaround", "Turnaround bucket (55%)"),
                   ("bucket_moat", "Moat bucket (45%)"),
                   ("t_inflection", "· profitability inflection"),
                   ("t_fscore", "· F-score rising"),
                   ("t_valuation_reset", "· valuation reset (drawdown)"),
                   ("t_squeeze", "· squeeze (SI x inflection)"),
                   ("m_roic_spread", "· moat: ROIC-WACC demonstrated"),
                   ("m_gm", "· moat: GM level+stability"),
                   ("m_consistency", "· moat: earnings/FCF consistency"),
                   ("m_valuation", "· moat: valuation (EV/S / P/TBV)")]:
        L.append(f"- {lbl}: **{_fmt(row.get(c), 1)}**")
    L.append("")
    L.append("### Short themes")
    for c, lbl in [("false_moat_scored", "False moat (scored, post-haircut)"),
                   ("false_moat_raw", "False moat RAW (pre-haircut)"),
                   ("theme_inflated_growth", "Inflated growth"),
                   ("theme_forensic", "Forensic"),
                   ("si_bonus", "SI bonus (additive only, max +8)"),
                   ("theme_confirmation", "· SI hump (display)"),
                   ("fm_peaked", "· peaked (rollover or at-peak setup)"),
                   ("fm_margin_vs_peer", "· margin vs peer (interaction)"),
                   ("fm_leverage", "· leverage propping"),
                   ("ig_decel", "· revenue deceleration"),
                   ("ig_richness", "· richness (EV/S + EV/norm-EBIT)"),
                   ("ig_ev_sales", "· · EV/Sales pctile"),
                   ("ig_ev_norm", "· · EV/normalized-EBIT pctile"),
                   ("ig_runup", "· run-up x peak-evidence"),
                   ("ig_pmv", "· pricing-masking-volume"),
                   ("ig_dso_dio", "· DSO/DIO expansion"),
                   ("ig_buybacks", "· debt-funded buybacks"),
                   ("ig_dilution", "· dilution"),
                   ("fo_beneish", "· Beneish pctile"),
                   ("fo_sloan", "· Sloan accruals pctile"),
                   ("fo_events", "· event flags (NT/4.01/4.02)")]:
        L.append(f"- {lbl}: **{_fmt(row.get(c), 1)}**")
    L.append("")

    # ---- audit-4.2 cross-check: ALWAYS on long tear sheets ----
    if side == "long":
        L.append("## ⚠ Short-side false-moat cross-check (audit 4.2 guardrail)")
        L.append(f"- demonstrated-economics window: "
                 f"**{row.get('moat_window_start', '?')} → "
                 f"{row.get('moat_window_end', '?')}** "
                 f"(a best-12q window ending 2021-22 is the COVID-spike tell)")
        L.append(f"- demonstrated ROIC-WACC: {_fmt(row.get('moat_demonstrated'))} "
                 f"| trailing 5-yr: {_fmt(row.get('moat_trailing_5y'))} "
                 f"| fallen-angel spread: {_fmt(row.get('fallen_angel_spread'))}")
        L.append(f"- false-moat theme RAW (pre-haircut): "
                 f"**{_fmt(row.get('false_moat_raw'), 1)}** — components: "
                 f"peaked&rolling {_fmt(row.get('fm_peaked'), 1)}, "
                 f"margin-vs-peer {_fmt(row.get('fm_margin_vs_peer'), 1)}, "
                 f"leverage-propping {_fmt(row.get('fm_leverage'), 1)}")
        collision = row.get("false_moat_collision", False)
        L.append(f"- `false-moat-collision` tag: "
                 f"**{'FIRED — review the window' if collision else 'not fired'}**")
        L.append("")

    # ---- inflection triplets ----
    infl = details.get("inflection", {})
    if infl:
        L.append("## Inflection engine (level / slope / trough per metric)")
        L.append("| metric | level | slope | trough | conj. score | own-hist pctile | n | flags |")
        L.append("|---|---|---|---|---|---|---|---|")
        for m, d in infl.items():
            L.append(f"| {m} | {_fmt(d.get('level'))} | {_fmt(d.get('slope'))} "
                     f"| {_fmt(d.get('trough'))} | {_fmt(d.get('score'))} "
                     f"| {_fmt(d.get('level_pctile'), 1)} | {d.get('n_obs', '?')} "
                     f"| {','.join(d.get('flags', [])) or '-'} |")
        L.append(f"\n- revenue context: **{row.get('revenue_context', '?')}**"
                 + (" → cost-cut haircut applied"
                    if row.get("cost_cut_inflection") else ""))
        L.append(f"- op-margin vs 5-yr peak: {_fmt(row.get('margin_vs_peak'))} "
                 f"(decline shape: {row.get('decline_shape', '?')}"
                 + ("; SECULAR/INDUSTRY DECLINE — sector fell in lockstep"
                    if row.get("secular_decline") else "")
                 + ("; no-peer-data" if row.get("no_peer_data") else "") + ")")
        L.append("")
        # quarterly + TTM series per metric (Tier 1: pipeline retains
        # per-metric intermediates) — trailing 8 TTM points, compact
        qs_det = details.get("quarterly_series", {})
        if qs_det:
            L.append("### TTM series (trailing 8 per metric)")
            for m, blob in qs_det.items():
                ttm_map = blob.get("ttm", {})
                if not ttm_map:
                    continue
                tail = sorted(ttm_map.items())[-8:]
                vals = " | ".join(f"{d[2:7]}: {v:,.3g}" for d, v in tail)
                L.append(f"- **{m}**: {vals}")
            L.append("")

    # ---- survivability strip (context, no score weight) ----
    L.append("## Survivability strip (context only)")
    L.append(f"- traffic light: **{row.get('surv_traffic_light', 'n/a')}** | "
             f"runway: {_fmt(row.get('surv_runway_q'), 1)}q | "
             f"near-term debt > cash: {_fmt(row.get('surv_debt_gt_cash'))} | "
             f"2-yr dilution: {_fmt(row.get('surv_dilution_2y'))}")
    z = row.get("altman_z")
    if row.get("fin_sector") == "reit":
        L.append(f"- REIT substitutes: interest coverage "
                 f"{_fmt(row.get('interest_coverage'), 1)}x, "
                 f"debt/EBITDA {_fmt(row.get('debt_to_ebitda'), 1)}x")
    else:
        L.append(f"- Altman Z: {_fmt(z, 2)}")
    L.append("")

    # ---- tags ----
    tags = [c for c in TAG_COLS if bool(row.get(c)) and str(row.get(c)) not in
            ("nan", "n/a", "no", "False", "unknown", "-1")]
    L.append("## Tags")
    L.append(", ".join(f"`{c}={row.get(c)}`" if not isinstance(row.get(c), (bool, np.bool_))
                       else f"`{c}`" for c in tags) or "(none)")
    L.append("")

    # ---- archetype verification checklist (short side only) ----
    if side == "short":
        block = _archetype_checklist_block(row, details)
        L.append("## Archetype verification checklist (verify before pitching)")
        L.extend(block if block else ["(no short-side archetype tags fired)"])
        L.append("")

    # ---- coverage / missing metrics by name ----
    missing = [c for c in RAW_INPUT_COLS
               if c in row.index and (row.get(c) is None or
                                      (isinstance(row.get(c), float) and not np.isfinite(row.get(c))))]
    L.append("## Coverage")
    L.append(f"- long bucket coverage {_fmt(row.get('cov_long'), 2)}, "
             f"short {_fmt(row.get('cov_short'), 2)}"
             + (" — **low-confidence**" if row.get("low_confidence") else ""))
    L.append(f"- missing/inapplicable inputs: "
             f"{', '.join(missing) if missing else 'none'}")
    flags = row.get("flags", "")
    if flags:
        L.append(f"- guard/clamp flags fired: {flags}")
    L.append("")

    # ---- Tier 2 ----
    t2 = details.get("tier2", {})
    t2m = details.get("tier2_market", {})
    L.append("## Tier 2 (best-effort)")
    items = t2.get("recent_8k_items", [])
    if items:
        L.append("Recent 8-K items (date, items — see legend):")
        for dt, it in items[:12]:
            notes = "; ".join(config.ITEM_8K_LEGEND.get(code.strip(), "")
                              for code in str(it).split(",")
                              if code.strip() in config.ITEM_8K_LEGEND)
            L.append(f"- {dt}: {it}" + (f" ({notes})" if notes else ""))
    si = row.get("si_ratio")
    L.append(f"- short interest: {_fmt(si)} of shares outstanding "
             f"(as of {row.get('date_short_interest', 'n/a')}); "
             f"shortPercentOfFloat (display only): "
             f"{_fmt(t2m.get('shortPercentOfFloat_display_only'))}")
    ins = t2m.get("heldPercentInsiders")
    try:
        ins_s = f"{float(ins) * 100:.1f}%"
    except (TypeError, ValueError):
        ins_s = "n/a"
    L.append(f"- analyst coverage (display only): "
             f"{_fmt(t2m.get('numberOfAnalystOpinions'), 0)} opinions"
             f" | insider ownership: {ins_s}")
    L.append(f"- trailing PEG (display only, sparse): "
             f"{_fmt(t2m.get('trailingPegRatio_display_only'))}")
    L.append(f"- next earnings: {t2m.get('nextEarningsDate') or 'n/a'}")
    seg = row.get("reportable_segments_display")
    if seg is not None and str(seg) not in ("nan", "None", ""):
        L.append(f"- NumberOfReportableSegments (display only, sparse "
                 f"coverage): {_fmt(seg, 0)}")
    ins = t2m.get("insider")
    if ins:
        L.append(f"- insider activity (yfinance, shortlist-stage, nullable): "
                 f"{ins}")
    else:
        L.append("- insider activity: n/a (enrichment unavailable)")
    cik = row.get("cik")
    for label, accn in [("latest 10-K", t2.get("latest_10k_accn")),
                        ("latest 10-Q", t2.get("latest_10q_accn"))]:
        if accn:
            L.append(f"- {label}: https://www.sec.gov/Archives/edgar/data/"
                     f"{int(cik)}/{str(accn).replace('-', '')}")
    summ = t2m.get("longBusinessSummary", "")
    if summ and summ not in ("nan", "None"):
        L.append("")
        L.append(f"> {summ}")
    L.append("")
    L.append("*Simple-business judgment is manual: segment/geography mix is "
             "not computable from companyfacts (audit 6.5) — use the links "
             "above.*")
    return "\n".join(L)


def write_tearsheets(outputs: dict, details: dict, outdir: str,
                     max_per_side: int | None = None) -> int:
    """Render tear sheets for every surfaced name. Returns count written."""
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)
    n = 0
    for key, side in (("long_shortlist", "long"), ("short_shortlist", "short"),
                      ("long_financials", "long"), ("short_financials", "short")):
        frame = outputs.get(key)
        if frame is None or not len(frame):
            continue
        rows = frame if max_per_side is None else frame.head(max_per_side)
        for _, row in rows.iterrows():
            t = row["ticker"]
            det = details.get(t, {})
            md = render(row, det, side)
            (outp / f"{t}_{side}.md").write_text(md, encoding="utf-8")
            n += 1
    return n
