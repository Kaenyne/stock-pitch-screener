"""First-run diagnostics (spec audit 6.4 + item 6 for the 4.2 collision tag).

Required BEFORE locking windows/weights — this is the only validation a
no-backtest project gets:
1. parameter sensitivity: slope {3,4,6}q x trough {8,12}q(TTM-ref) ->
   top-50 Jaccard overlap AND full-universe Spearman between combos
2. sub-score health: distribution, stdev, correlation with composite
3. trough-detector fire rate per window (>~half the universe = can't
   discriminate)
4. per-metric coverage counts, split by financials tag
5. top-decile sector mix vs universe mix
6. false-moat-collision fire rate + joint distribution of demonstrated-moat
   rank x false-moat theme rank (calibrates the audit-4.2 thresholds)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config

SUBSCORES_LONG = ["t_inflection", "t_fscore", "t_valuation_reset",
                  "t_squeeze", "m_roic_spread", "m_gm", "m_consistency",
                  "m_valuation"]
SUBSCORES_SHORT = ["fm_peaked", "fm_margin_vs_peer", "fm_leverage",
                   "ig_decel", "ig_richness", "ig_ev_sales", "ig_ev_norm",
                   "ig_runup", "ig_pmv", "ig_dso_dio", "ig_buybacks",
                   "ig_dilution", "fo_beneish", "fo_sloan", "fo_events",
                   "theme_confirmation"]
RAW_METRIC_COLS = ["inflection_score", "margin_vs_peak", "f_rising",
                   "valuation_reset", "si_ratio", "moat_demonstrated",
                   "gm_level", "earnings_consistency", "ev_sales", "p_tbv",
                   "peak_score", "sloan_latest", "beneish_m",
                   "decel_score_raw", "dso_yoy_delta", "dio_yoy_delta",
                   "dilution_yoy", "leverage_prop_raw", "at_peak_score",
                   "ev_normalized", "pmv_score_raw", "ret_12m"]


def param_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Jaccard(top-50) + Spearman(full universe) between window combos,
    measured on the inflection theme score (the only window-dependent part
    of the composite)."""
    combos = [c for c in df.columns if c.startswith("sens_s")]
    combos.append("inflection_score")  # the primary (5,10) run
    rows = []
    for i, a in enumerate(combos):
        for b in combos[i + 1:]:
            sub = df[[a, b]].dropna()
            if len(sub) < 20:
                continue
            top_a = set(sub[a].nlargest(50).index)
            top_b = set(sub[b].nlargest(50).index)
            jac = len(top_a & top_b) / max(1, len(top_a | top_b))
            rho = stats.spearmanr(sub[a], sub[b]).statistic
            rows.append({"combo_a": a, "combo_b": b, "n": len(sub),
                         "jaccard_top50": round(jac, 3),
                         "spearman": round(float(rho), 3)})
    return pd.DataFrame(rows)


def subscore_health(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, cols, comp in (("long", SUBSCORES_LONG, "long_composite"),
                             ("short", SUBSCORES_SHORT, "short_composite")):
        for c in cols:
            if c not in df:
                continue
            s = df[c]
            valid = s.dropna()
            corr = s.corr(df[comp], method="spearman") if len(valid) > 20 else np.nan
            rows.append({
                "side": side, "subscore": c, "n": len(valid),
                "coverage": round(len(valid) / max(1, len(df)), 3),
                "mean": round(float(valid.mean()), 2) if len(valid) else np.nan,
                "std": round(float(valid.std()), 2) if len(valid) else np.nan,
                "p10": round(float(valid.quantile(0.1)), 2) if len(valid) else np.nan,
                "p90": round(float(valid.quantile(0.9)), 2) if len(valid) else np.nan,
                "spearman_vs_composite": round(float(corr), 3) if np.isfinite(corr) else np.nan,
                "flag": ("DEAD" if len(valid) and valid.std() < 2.0 else
                         "DOMINANT" if np.isfinite(corr) and abs(corr) > 0.92
                         else ""),
            })
    return pd.DataFrame(rows)


def trough_fire_rate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in [c for c in df.columns if c.startswith("trough_fired")]:
        s = df[c].dropna()
        if not len(s):
            continue
        rate = float(s.astype(bool).mean())
        rows.append({"window": c, "n": len(s), "fire_rate": round(rate, 3),
                     "flag": "CANNOT_DISCRIMINATE" if rate > 0.5 else ""})
    return pd.DataFrame(rows)


def coverage_by_financials(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grp = df.groupby(df["is_financial"].fillna(False))
    for is_fin, g in grp:
        for c in RAW_METRIC_COLS:
            if c not in g:
                continue
            rows.append({"is_financial": bool(is_fin), "metric": c,
                         "n": len(g), "n_computed": int(g[c].notna().sum()),
                         "coverage": round(g[c].notna().mean(), 3)})
    return pd.DataFrame(rows)


def sector_mix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    uni_mix = df["sector2"].value_counts(normalize=True)
    for side in ("long", "short"):
        comp = df[f"{side}_composite"]
        top = df[comp >= comp.quantile(0.9)]
        top_mix = top["sector2"].value_counts(normalize=True)
        for sec in top_mix.index:
            rows.append({"side": side, "sector2": sec,
                         "top_decile_share": round(float(top_mix[sec]), 3),
                         "universe_share": round(float(uni_mix.get(sec, 0)), 3),
                         "ratio": round(float(top_mix[sec] /
                                              max(uni_mix.get(sec, 1e-9), 1e-9)), 2)})
    return pd.DataFrame(rows).sort_values(["side", "top_decile_share"],
                                          ascending=[True, False])


def collision_diagnostic(df: pd.DataFrame) -> dict:
    """Diagnostic 6: collision fire rate in the long top decile / shortlist +
    the joint distribution of demonstrated-moat rank x false-moat rank."""
    out = {}
    n = len(df)
    fired = df["false_moat_collision"].fillna(False).astype(bool)
    out["universe_fire_rate"] = round(float(fired.mean()), 4)
    comp = df["long_composite"]
    top_decile = df[comp >= comp.quantile(0.9)]
    out["long_top_decile_fire_rate"] = round(
        float(top_decile["false_moat_collision"].fillna(False).mean()), 4) \
        if len(top_decile) else np.nan
    thr = config.SHORTLIST_SCORE_THRESHOLD
    sl = df[comp >= thr]
    out["long_shortlist_fire_rate"] = round(
        float(sl["false_moat_collision"].fillna(False).mean()), 4) \
        if len(sl) else np.nan
    # joint distribution: quartile x quartile counts
    mr = df["moat_demonstrated"].rank(pct=True)
    fr = df["false_moat_raw"].rank(pct=True)
    joint = pd.crosstab(pd.cut(mr, [0, .25, .5, .75, 1.0]),
                        pd.cut(fr, [0, .25, .5, .75, 1.0]))
    out["joint_quartiles"] = joint
    out["verdict"] = ("DEAD_GUARDRAIL" if out["universe_fire_rate"] == 0 else
                      "NOISE" if (out["long_shortlist_fire_rate"] or 0) > 0.5
                      else "ok")
    return out


def run_all(df: pd.DataFrame, outdir: str) -> str:
    """Write diagnostics CSVs + a summary markdown; returns the md path."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ps = param_sensitivity(df)
    sh = subscore_health(df)
    tf = trough_fire_rate(df)
    cov = coverage_by_financials(df)
    sm = sector_mix(df)
    col = collision_diagnostic(df)
    ps.to_csv(out / "diag1_param_sensitivity.csv", index=False)
    sh.to_csv(out / "diag2_subscore_health.csv", index=False)
    tf.to_csv(out / "diag3_trough_fire_rate.csv", index=False)
    cov.to_csv(out / "diag4_coverage.csv", index=False)
    sm.to_csv(out / "diag5_sector_mix.csv", index=False)
    col["joint_quartiles"].to_csv(out / "diag6_collision_joint.csv")

    lines = ["# First-run diagnostics (audit 6.4)", ""]
    lines += ["## 1. Parameter sensitivity (inflection theme rankings)",
              ps.to_string(index=False) if len(ps) else "(insufficient data)",
              ""]
    lines += ["## 2. Sub-score health",
              sh.to_string(index=False) if len(sh) else "(none)", ""]
    flagged = sh[sh["flag"] != ""] if len(sh) else pd.DataFrame()
    if len(flagged):
        lines += ["**FLAGGED SUB-SCORES:** " +
                  ", ".join(f"{r.subscore}({r.flag})"
                            for r in flagged.itertuples()), ""]
    lines += ["## 3. Trough-detector fire rate",
              tf.to_string(index=False) if len(tf) else "(none)", ""]
    lines += ["## 4. Coverage by financials tag (lowest 15)",
              cov.nsmallest(15, "coverage").to_string(index=False)
              if len(cov) else "(none)", ""]
    lines += ["## 5. Top-decile sector mix vs universe (top 10 rows/side)",
              sm.groupby("side").head(10).to_string(index=False)
              if len(sm) else "(none)", ""]
    lines += ["## 6. False-moat-collision calibration (audit 4.2)",
              f"universe fire rate: {col['universe_fire_rate']}",
              f"long top-decile fire rate: {col['long_top_decile_fire_rate']}",
              f"long shortlist fire rate: {col['long_shortlist_fire_rate']}",
              f"verdict: {col['verdict']}", "",
              "joint quartile counts (moat rank x false-moat rank):",
              col["joint_quartiles"].to_string(), ""]
    md = out / "first_run_diagnostics.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return str(md)
