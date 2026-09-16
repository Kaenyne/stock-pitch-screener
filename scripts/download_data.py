"""Fetch the three SEC EDGAR bulk files the screener reads (~3 GB total).

    python scripts/download_data.py            # download whatever is missing
    python scripts/download_data.py --refresh  # re-download everything

The SEC rebuilds these files nightly; the screener only sees filings that
were in the zips when you downloaded them, so refresh every few weeks.
Downloads are resumable: an interrupted file continues from where it stopped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from screener import config  # noqa: E402

EDGAR_DIR = ROOT / config.EDGAR_DIR
FILES = {
    "company_tickers_exchange.json":
        "https://www.sec.gov/files/company_tickers_exchange.json",
    "companyfacts.zip":
        "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
    "submissions.zip":
        "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
}
STALE_DAYS = 45


def _fmt_mb(n: float) -> str:
    return f"{n / 1e6:,.0f} MB"


def _download(url: str, dest: Path, user_agent: str) -> None:
    import requests
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
    have = tmp.stat().st_size if tmp.exists() else 0
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:            # server says we already have it all
            tmp.replace(dest)
            return
        r.raise_for_status()
        resumed = r.status_code == 206
        total = int(r.headers.get("Content-Length", 0)) + (have if resumed else 0)
        mode = "ab" if resumed else "wb"
        done = have if resumed else 0
        t0, last = time.time(), 0.0
        with open(tmp, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if time.time() - last > 2:
                    last = time.time()
                    pct = f"{100 * done / total:5.1f}%" if total else ""
                    rate = (done - have) / max(time.time() - t0, 1e-6)
                    print(f"\r    {dest.name}: {_fmt_mb(done)}"
                          f"{' / ' + _fmt_mb(total) if total else ''} {pct}"
                          f"  ({_fmt_mb(rate)}/s)   ", end="", flush=True)
    print()
    tmp.replace(dest)


def status() -> dict[str, str]:
    """name -> 'missing' | 'stale' | 'ok'."""
    out = {}
    for name in FILES:
        p = EDGAR_DIR / name
        if not p.exists() or p.stat().st_size == 0:
            out[name] = "missing"
        elif (time.time() - p.stat().st_mtime) > STALE_DAYS * 86400:
            out[name] = "stale"
        else:
            out[name] = "ok"
    return out


def ensure_data(user_agent: str, refresh: bool = False) -> None:
    """Download anything missing (or everything, with refresh=True)."""
    if not user_agent:
        raise SystemExit("EDGAR needs a User-Agent ('Your Name you@email.com'). "
                         "Run scripts/run_screener.py once, or set EDGAR_USER_AGENT.")
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    st = status()
    todo = [n for n, s in st.items() if refresh or s == "missing"]
    if not todo:
        stale = [n for n, s in st.items() if s == "stale"]
        if stale:
            print(f"  EDGAR data is older than {STALE_DAYS} days "
                  f"({', '.join(stale)}). Newer filings will be missing; "
                  f"run with --refresh-data when convenient.")
        return
    print(f"  Downloading {len(todo)} SEC file(s) into {EDGAR_DIR} "
          f"(about 3 GB total; this can take 10-40 minutes)...")
    for name in todo:
        dest = EDGAR_DIR / name
        for attempt in range(1, 4):
            try:
                _download(FILES[name], dest, user_agent)
                break
            except Exception as e:  # network blip: retry, resuming the .part
                print(f"\n    {name}: attempt {attempt} failed ({e}); retrying...")
                time.sleep(5 * attempt)
        else:
            raise SystemExit(f"Could not download {name}. Check your internet "
                             f"connection and re-run; the download resumes.")
    print("  SEC data ready.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download all files")
    a = ap.parse_args()
    ensure_data(config.EDGAR_USER_AGENT, refresh=a.refresh)
