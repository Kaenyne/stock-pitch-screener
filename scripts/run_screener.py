"""The one command. Checks your setup, downloads SEC data if needed, runs the
screener over every US-listed $1-20B company, and writes output/SUMMARY.md.

    python scripts/run_screener.py                 # full run (60-90 min first time)
    python scripts/run_screener.py --test          # 5-min install check on ~12 names
    python scripts/run_screener.py --refresh-data  # re-download the SEC files first
    python scripts/run_screener.py --workers 4     # fewer CPU cores (laptops)

You normally launch this through run_screener.bat / run_screener.sh, which
also create the Python environment for you.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_TICKERS = ["PTON", "CVNA", "W", "CROX", "DECK", "ETSY", "KEY", "BXP",
                "CLF", "CAKE", "VITL", "GIS"]


# ---------------------------------------------------------------------------
# setup checks
# ---------------------------------------------------------------------------

class _Tee:
    """Echo stdout to a log file so a long run can be reviewed afterwards."""
    def __init__(self, path: Path):
        self.fh = open(path, "a", encoding="utf-8")
        self.term = sys.__stdout__

    def write(self, s):
        self.term.write(s)
        self.fh.write(s)

    def flush(self):
        self.term.flush()
        self.fh.flush()


def _check_python() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required (you have "
                         + sys.version.split()[0] + "). Install from python.org.")
    missing = []
    for mod in ("pandas", "numpy", "scipy", "yfinance", "requests", "pyarrow"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit("Missing packages: " + ", ".join(missing)
                         + "\nRun:  pip install -r requirements.txt")


def _ensure_user_agent() -> str:
    """SEC EDGAR refuses anonymous bulk requests. Ask once, remember forever."""
    from screener import config
    if config.EDGAR_USER_AGENT and "@" in config.EDGAR_USER_AGENT:
        return config.EDGAR_USER_AGENT
    print("\nThe SEC requires every EDGAR download to say who is asking.")
    print("This is only sent to sec.gov; it is saved in user_agent.txt.\n")
    try:
        name = input("  Your name:  ").strip()
        email = input("  Your email: ").strip()
    except EOFError:
        raise SystemExit("No terminal input available. Set the EDGAR_USER_AGENT "
                         "environment variable to 'Your Name you@email.com'.")
    if not name or "@" not in email:
        raise SystemExit("Need a name and a valid email. Please run again.")
    ua = name + " " + email
    (ROOT / "user_agent.txt").write_text(ua + "\n", encoding="utf-8")
    os.environ["EDGAR_USER_AGENT"] = ua      # worker processes re-import config
    config.EDGAR_USER_AGENT = ua
    return ua


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _test_universe(workers: int):
    from screener import market, pipeline
    from screener.universe import (build_universe, extract_light_batch,
                                   hygiene_filter, load_ticker_table)
    tick = hygiene_filter(load_ticker_table(pipeline.TICKERS_JSON))
    tick = tick[tick["ticker"].isin(TEST_TICKERS)].reset_index(drop=True)
    light = extract_light_batch(tick["cik"].tolist(), pipeline.FACTS_ZIP,
                                pipeline.SUBS_ZIP, workers=min(workers, 6))
    closes = market.chunked_download(tick["ticker"].tolist(), period="5d")
    return build_universe(None, market.last_closes(closes), light, tick)


def _run(test: bool, workers: int) -> dict:
    from screener import pipeline
    if test:
        pipeline.OUT = ROOT / "output" / "test_run"
        pipeline.OUT.mkdir(parents=True, exist_ok=True)
        uni = _test_universe(workers)
        print("  test universe: " + ", ".join(sorted(uni["ticker"])))
        return pipeline.run(universe_override=uni, workers=min(workers, 6))
    return pipeline.run(refresh_universe=True, workers=workers)


# ---------------------------------------------------------------------------
# SUMMARY.md - the human-readable view of the CSV outputs
# ---------------------------------------------------------------------------

LONG_COHORTS = {
    "pitchable_longs": "Pitchable longs (good business at a temporary trough, simple story)",
    "derated_compounder": "De-rated compounders (proven moat, economics intact, price down)",
    "inflecting_thin_moat": "Inflecting, thin moat (margins/returns turning up, no moat gate)",
    "surviving_distressed_value": "Surviving distressed value (35-80% drawdown, cheap, green survivability, F-score 7+)",
    "long_shortlist": "Long composite shortlist (highest overall long score)",
    "long_financials": "Long: banks, insurers, REITs (own metrics)",
}
SHORT_COHORTS = {
    "pitchable_shorts": "Pitchable shorts (composite rank, semis + biotech removed, HIGH_CONFIDENCE flag)",
    "consumer_shorts": "Consumer shorts (ranked by number of on-thesis mechanisms)",
    "industrials_shorts": "Industrial shorts (ranked by number of on-thesis mechanisms)",
    "commodity_disconnect_shorts": "Commodity names the market is NOT pricing as cyclicals",
    "archetype_ran_on_temp_success": "Ran on temporary success (the CAKE / VITL shape)",
    "ran_on_pricing_success": "Growth came from price hikes, not volume",
    "pricing_masking_volume": "Pricing masking volume declines",
    "priced_for_impossible_growth": "Priced for growth the business cannot deliver",
    "peer_multiple_disconnect": "Trading far above peers on multiples",
    "losing_share_priced_rich": "Losing share while priced as a winner",
    "capacity_decay": "Capacity / asset turnover decaying",
    "ppi_windfall": "Input-price windfall that is not in the P&L yet",
    "short_shortlist": "Short composite shortlist (highest overall short score)",
    "short_financials": "Short: banks, insurers, REITs (own metrics)",
}
SHORT_TAGS = ["pricing_masking_volume", "ran_on_pricing_success", "capacity_decay",
              "priced_for_impossible_growth", "multiple_disconnect",
              "losing_share_priced_rich", "real_rev_deflated",
              "archetype_ran_on_temp_success", "ppi_windfall"]


def _truthy(v) -> bool:
    return v is True or (isinstance(v, (int, float)) and v == 1) or str(v) == "True"


def _mktcap(v) -> str:
    try:
        return "$" + format(float(v) / 1e9, ".1f") + "B"
    except (TypeError, ValueError):
        return ""


def _score(v) -> str:
    try:
        return format(float(v), ".0f")
    except (TypeError, ValueError):
        return ""


def _table(df, side: str, outputs: dict, out_dir: Path, n: int) -> list[str]:
    cohorts = LONG_COHORTS if side == "long" else SHORT_COHORTS
    members = {k: set(v["ticker"]) for k, v in outputs.items() if k in cohorts}
    score_col = "long_composite" if side == "long" else "short_composite"
    lines = ["| ticker | company | sector | mkt cap | " + side + " score"
             + " | also on | tags | tear sheet |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.head(n).iterrows():
        t = r["ticker"]
        also = sum(1 for v in members.values() if t in v)
        tags = []
        if side == "short":
            if "high_confidence" in r.index and _truthy(r["high_confidence"]):
                tags.append("HIGH_CONFIDENCE")
            tags += [c for c in SHORT_TAGS if c in r.index and _truthy(r[c])]
        sheet = out_dir / "tearsheets" / (t + "_" + side + ".md")
        link = "[open](tearsheets/" + t + "_" + side + ".md)" if sheet.exists() else ""
        sector = str(r.get("sic_desc", "") or "")[:40]
        name = str(r.get("name", ""))[:34]
        lines.append("| " + " | ".join([
            t, name, sector, _mktcap(r.get("mktcap_build")),
            _score(r.get(score_col)), str(also) + (" list" if also == 1 else " lists"),
            ", ".join(tags), link,
        ]) + " |")
    return lines


def write_summary(res: dict, out_dir: Path, test: bool, minutes: float) -> Path:
    outputs = res["outputs"]
    L = ["# Screener run - " + format(dt.date.today(), "%Y-%m-%d"), ""]
    if test:
        L += ["**TEST MODE.** Only ~12 companies were scored, so every score and "
              "rank below is meaningless. This run only proves the install works. "
              "Run without `--test` for the real thing.", ""]
    L += ["Scored " + str(res["scored_n"]) + " companies in "
          + format(minutes, ".0f") + " min (" + str(res["errors_n"])
          + " build errors, " + str(res["n_tearsheets"]) + " tear sheets).", "",
          "## How to read this", "",
          "- Every list is a **pool of names worth a look, not a ranking**. "
          "The screener is good at surfacing candidates and bad at ordering the "
          "top of the list. Treat position 3 and position 25 as equally worth "
          "opening.",
          "- The **also on** column counts how many separate lists (lenses) flagged "
          "the name. A name on 3+ lists has several independent reasons to exist; "
          "that is a better use of your first hour than the top row of any one list.",
          "- **Tags** name the mechanism the screener thinks it sees. Open the tear "
          "sheet and verify the mechanism in the filings before you believe it.",
          "- Every list here is also a CSV in this folder with every column the "
          "screener computed. `universe_ranked.csv` has all scored companies.", ""]
    for side, cohorts in (("long", LONG_COHORTS), ("short", SHORT_COHORTS)):
        L += ["## " + side.upper() + " ideas", ""]
        for key, title in cohorts.items():
            df = outputs.get(key)
            if df is None or not len(df):
                continue
            n = 40 if key.startswith("pitchable") else 15
            L += ["### " + title, "",
                  "`" + key + ".csv` - " + str(len(df)) + " names; first "
                  + str(min(n, len(df))) + " shown.", ""]
            L += _table(df, side, outputs, out_dir, n)
            L.append("")
    p = out_dir / "SUMMARY.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true",
                    help="quick install check on ~12 companies (~5 min)")
    ap.add_argument("--refresh-data", action="store_true",
                    help="re-download the SEC bulk files before running")
    ap.add_argument("--workers", type=int,
                    default=max(2, min(10, (os.cpu_count() or 4) - 1)),
                    help="parallel worker processes (default: CPU cores - 1, max 10)")
    a = ap.parse_args()

    _check_python()
    ua = _ensure_user_agent()

    from screener import pipeline
    from scripts.download_data import ensure_data

    out_dir = ROOT / "output" / ("test_run" if a.test else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / ("run_" + format(dt.date.today(), "%Y-%m-%d") + ".log"))

    print("=== Pitch screener - " + format(dt.datetime.now(), "%Y-%m-%d %H:%M")
          + " (" + ("TEST" if a.test else "FULL") + " run, " + str(a.workers)
          + " workers) ===")
    print("step 1/3  SEC data")
    ensure_data(ua, refresh=a.refresh_data)
    print("step 2/3  building and scoring the universe"
          + ("" if a.test else " (60-90 minutes; progress lines below)"))
    t0 = time.time()
    res = _run(a.test, a.workers)
    minutes = (time.time() - t0) / 60
    print("step 3/3  writing summary")
    p = write_summary(res, pipeline.OUT, a.test, minutes)
    print("\n=== DONE in " + format(minutes, ".0f") + " min ===")
    print("scored " + str(res["scored_n"]) + " companies, "
          + str(res["errors_n"]) + " errors")
    print("\nOpen this file first:\n  " + str(p))
    print("All CSVs and tear sheets are in:\n  " + str(pipeline.OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
