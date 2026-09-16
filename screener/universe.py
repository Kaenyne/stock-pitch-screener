"""Universe builder (spec roadmap step 2).

EDGAR tickers -> entity hygiene -> shares x price cap band -> tagged list.

Hygiene per audit 1.5:
- drop OTC exchange rows (company_tickers_exchange.json)
- regex-drop derivative suffixes (-WT, -WS, -U, -UN, -RT, preferreds)
- dedupe to one row per CIK keeping the FIRST-listed ticker (primary class)
- structural non-operating detection: no us-gaap facts in companyfacts, or
  recent forms are N-series (SIC unreliable for funds); tag-don't-drop
  BDC/royalty-trusts that still file 10-K/10-Q
- multi-class names: never cap from dei shares; use primary ticker's yfinance
  marketCap (fetched separately)

Cap band at build time is widened to ~$0.8B-$25B (dei shares can be a quarter
stale); final band re-checked with fast_info for surfaced names only.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .edgar import BulkZip, parse_submissions

# ---------------------------------------------------------------------------
# Light per-CIK extraction (pass 1): shares, taxonomy presence, submissions
# ---------------------------------------------------------------------------

_FACTS_ZIP: BulkZip | None = None
_SUBS_ZIP: BulkZip | None = None


def _init_worker(facts_path: str, subs_path: str):
    global _FACTS_ZIP, _SUBS_ZIP
    _FACTS_ZIP = BulkZip(facts_path)
    _SUBS_ZIP = BulkZip(subs_path)


def _latest_fact_value(units: dict) -> tuple[float, str] | None:
    """Latest (by end date) value across unit types; returns (val, end)."""
    best = None
    for _unit, obs in units.items():
        for o in obs:
            end = o.get("end", "")
            val = o.get("val")
            if val is None or not end:
                continue
            if best is None or end > best[1]:
                best = (float(val), end)
    return best


def _extract_light(cik: int) -> dict:
    out = {"cik": cik, "has_facts": False, "has_usgaap": False,
           "is_ifrs": False, "shares": np.nan, "shares_asof": "",
           "sic": None, "sic_desc": "", "filer_type": "domestic_quarterly",
           "is_fund": False, "is_despac_name": False, "nt_recent": 0,
           "fiscal_year_end": "", "sub_name": ""}
    facts = _FACTS_ZIP.get(cik)
    if facts:
        out["has_facts"] = True
        f = facts.get("facts", {})
        out["has_usgaap"] = bool(f.get("us-gaap"))
        out["is_ifrs"] = bool(f.get("ifrs-full"))
        # Shares ladder: dei -> us-gaap common shares -> diluted WAS
        for taxo, tag in [("dei", "EntityCommonStockSharesOutstanding"),
                          ("us-gaap", "CommonStockSharesOutstanding"),
                          ("us-gaap", "CommonStockSharesIssued"),
                          ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")]:
            units = f.get(taxo, {}).get(tag, {}).get("units")
            if units:
                got = _latest_fact_value(units)
                if got and got[0] > 0:
                    out["shares"], out["shares_asof"] = got
                    break
    subs = _SUBS_ZIP.get(cik)
    if subs:
        info = parse_submissions(subs)
        out["sic"] = info.sic
        out["sic_desc"] = info.sic_description
        out["filer_type"] = info.filer_type
        out["is_fund"] = info.is_fund
        out["is_despac_name"] = info.is_despac_name
        out["nt_recent"] = info.nt_filings_recent
        out["fiscal_year_end"] = info.fiscal_year_end
        out["sub_name"] = info.name
        # tag-don't-drop BDCs / royalty trusts (audit 1.5): ANY recent
        # N-2/N-CSR/N-54 presence while still filing 10-K/10-Q — presence,
        # not dominance (ARCC-style BDCs file mostly 10-K/Q with occasional
        # N-2; dominance is the DROP criterion, handled via is_fund)
        recent_forms = {fm.split("/")[0] for fm, _, _, _ in info.forms[:60]}
        any_n = bool(recent_forms & (config.FUND_FORMS | {"N-54A", "N-54C"}))
        files_ops = bool({"10-K", "10-Q"} & recent_forms)
        out["bdc_or_trust"] = any_n and files_ops
    return out


def extract_light_batch(ciks: list[int], facts_path: str, subs_path: str,
                        workers: int = 8) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker,
                             initargs=(facts_path, subs_path)) as ex:
        for r in ex.map(_extract_light, ciks, chunksize=50):
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ticker table hygiene
# ---------------------------------------------------------------------------

def load_ticker_table(path: str | Path) -> pd.DataFrame:
    d = json.loads(Path(path).read_text())
    df = pd.DataFrame(d["data"], columns=d["fields"])
    return df


def hygiene_filter(df: pd.DataFrame) -> pd.DataFrame:
    """OTC drop, derivative-suffix drop, CIK dedupe keeping first-listed."""
    df = df.copy()
    df["row_order"] = range(len(df))
    # Drop OTC (exchange is None/'' or 'OTC')
    df = df[df["exchange"].isin(["Nasdaq", "NYSE", "NYSE American",
                                 "NYSE Arca", "CBOE", "BATS"])]
    # Derivative suffixes on ticker
    pat = re.compile(config.DERIVATIVE_SUFFIX_RE, re.IGNORECASE)
    df = df[~df["ticker"].str.contains(pat, na=False)]
    # Units/warrants often plain 'U'/'W' suffix on 5-char tickers (SPAC style)
    df = df[~df["ticker"].str.match(r"^[A-Z]{4}[UW]$", na=False)]
    # Multi-class flag before dedupe
    counts = df.groupby("cik")["ticker"].transform("count")
    df["multi_class"] = counts > 1
    # Keep first-listed ticker per CIK (verified: primary class lists first)
    df = df.sort_values("row_order").drop_duplicates("cik", keep="first")
    return df.drop(columns=["row_order"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def classify_sector(sic: int | None) -> str:
    """Coarse tag: bank / insurer / reit / financial_other / nonfinancial."""
    if sic is None:
        return "nonfinancial"
    if sic in config.SIC_REITS:
        return "reit"
    if sic in config.SIC_BANKS:
        return "bank"
    if sic in config.SIC_INSURERS:
        return "insurer"
    if sic in config.SIC_FINANCIAL_ALL:
        return "financial_other"
    return "nonfinancial"


def build_universe(edgar_dir: str | Path, prices: pd.Series,
                   light: pd.DataFrame, tickers: pd.DataFrame,
                   multi_class_caps: pd.Series | None = None) -> pd.DataFrame:
    """Join hygiene-filtered tickers + light extraction + last closes into the
    build-time universe with tags.

    prices: Series ticker -> last close (from chunked yf.download)
    multi_class_caps: Series ticker -> yfinance marketCap for multi-class names
    """
    u = tickers.merge(light, on="cik", how="left")
    u["last_close"] = u["ticker"].map(prices)

    u["mktcap_build"] = u["shares"] * u["last_close"]
    u["cap_fallback_dei_multiclass"] = False
    if multi_class_caps is not None:
        mc_ok = u["multi_class"] & u["ticker"].isin(multi_class_caps.index) \
            & u["ticker"].map(multi_class_caps).notna()
        u.loc[mc_ok, "mktcap_build"] = u.loc[mc_ok, "ticker"].map(multi_class_caps)
        # yfinance cap unavailable for a multi-class name: fall back to
        # dei shares x close but TAG it (audit 1.4: dei caps are unreliable
        # for multi-class) so the number is visibly suspect, not silent
        u.loc[u["multi_class"] & ~mc_ok, "cap_fallback_dei_multiclass"] = True
    else:
        u.loc[u["multi_class"], "cap_fallback_dei_multiclass"] = True

    # Structural non-operating drop: no facts at all, or fund without 10-K/Q
    operating = u["has_facts"] & (u["has_usgaap"] | u["is_ifrs"])
    keep_fund = ~u["is_fund"].fillna(False) | u["bdc_or_trust"].fillna(False)
    u = u[operating.fillna(False) & keep_fund]

    lo, hi = config.CAP_BAND_BUILD
    u = u[(u["mktcap_build"] >= lo) & (u["mktcap_build"] <= hi)]

    u["fin_sector"] = u["sic"].map(classify_sector)
    u["is_financial"] = u["fin_sector"] != "nonfinancial"
    u["is_commodity"] = u["sic"].map(config.is_commodity_sic)
    u["sector2"] = u["sic"].map(
        lambda s: int(s) // 100 if pd.notna(s) and s is not None else -1)
    u["annual_only"] = u["filer_type"] == "annual_only"
    return u.reset_index(drop=True)
