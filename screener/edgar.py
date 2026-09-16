"""EDGAR access layer: rate-limited HTTP client + bulk-zip readers +
submissions-JSON parsing (SIC, form history, 8-K items, formerNames, FPI
classification).
"""

from __future__ import annotations

import json
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config


class RateLimiter:
    def __init__(self, max_rps: float):
        self.min_interval = 1.0 / max_rps
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class EdgarClient:
    """Rate-limited EDGAR HTTP client for incremental (non-bulk) fetches."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.EDGAR_USER_AGENT
        self.limiter = RateLimiter(config.EDGAR_MAX_RPS)

    def get_json(self, url: str, retries: int = 3) -> dict | None:
        for attempt in range(retries):
            self.limiter.wait()
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    return None
                if r.status_code in (403, 429, 503):
                    time.sleep(2.0 * (attempt + 1))
                    continue
            except (requests.RequestException, json.JSONDecodeError):
                time.sleep(1.0 * (attempt + 1))
        return None

    def companyfacts(self, cik: int) -> dict | None:
        return self.get_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")

    def submissions(self, cik: int) -> dict | None:
        return self.get_json(
            f"https://data.sec.gov/submissions/CIK{cik:010d}.json")


class BulkZip:
    """Reader over an EDGAR bulk zip (companyfacts.zip / submissions.zip).

    Thread-safe for reads via a lock; for multiprocessing, open one instance
    per process (pass the path, not the object).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._zf = zipfile.ZipFile(self.path)
        self._members = set(self._zf.namelist())
        self._lock = threading.Lock()

    def has(self, cik: int) -> bool:
        return f"CIK{cik:010d}.json" in self._members

    def get(self, cik: int) -> dict | None:
        name = f"CIK{cik:010d}.json"
        if name not in self._members:
            return None
        with self._lock:
            raw = self._zf.read(name)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def ciks(self) -> list[int]:
        out = []
        for m in self._members:
            if m.startswith("CIK") and m.endswith(".json") and "-" not in m:
                try:
                    out.append(int(m[3:13]))
                except ValueError:
                    pass
        return out


# ---------------------------------------------------------------------------
# Submissions parsing
# ---------------------------------------------------------------------------

@dataclass
class SubmissionsInfo:
    cik: int
    name: str = ""
    sic: int | None = None
    sic_description: str = ""
    tickers: list[str] = field(default_factory=list)
    exchanges: list[str] = field(default_factory=list)
    former_names: list[str] = field(default_factory=list)
    forms: list[tuple[str, str, str, str]] = field(default_factory=list)
    # (form, filingDate, accessionNumber, items) — items only set for 8-K
    fiscal_year_end: str = ""  # "MMDD"
    periodic_filings: list[tuple[str, str, str]] = field(default_factory=list)
    # (form, filingDate, reportDate) for 10-K/10-Q(+/A) and FPI annuals —
    # drives the catalyst-calendar filing-lag + next-print projection

    # Derived classification
    filer_type: str = "domestic_quarterly"  # | annual_only
    is_ifrs: bool = False
    is_fund: bool = False
    is_despac_name: bool = False
    nt_filings_recent: int = 0
    recent_8k_items: list[tuple[str, str]] = field(default_factory=list)
    # (filingDate, items) for last ~2y of 8-Ks
    amended_periodic: list[tuple[str, str, str]] = field(default_factory=list)
    # (form, filingDate, accessionNumber) for amended 10-K/10-Q/20-F/40-F —
    # the filed evidence a restatement claim can actually be checked against
    latest_10k_accn: str = ""
    latest_10q_accn: str = ""
    # As-of provenance (replay leak guard). asof_iso == "" means "today's view".
    asof_iso: str = ""
    n_filings_suppressed: int = 0   # filings dropped as filed-after-asof


def _asof_iso(asof) -> str:
    """Normalize an as-of date (ISO string or Timestamp) to 'YYYY-MM-DD'.
    Empty string means no upper bound (live run, today's view)."""
    if asof is None or asof == "":
        return ""
    if isinstance(asof, str):
        return asof[:10]
    try:
        return asof.date().isoformat()      # pd.Timestamp / datetime
    except AttributeError:
        return str(asof)[:10]


def _former_name_visible(fn: dict, asof_iso: str) -> bool:
    """A formerName is only *former* once the rename happened. At a replay
    date before the rename the company still carried that name, so the entry
    is not yet in formerNames as of that date."""
    if not asof_iso:
        return True
    to = str(fn.get("to") or "")[:10]
    return (not to) or to <= asof_iso


def parse_submissions(d: dict, asof=None) -> SubmissionsInfo:
    """Parse an EDGAR submissions JSON into a SubmissionsInfo.

    `asof` (ISO date string or Timestamp) makes the parse point-in-time: any
    filing with filingDate > asof is dropped before anything downstream sees
    it, so form history, 8-K items, NT counts, latest-accession pointers and
    the fund/de-SPAC classifiers are all restricted to what was on file at
    that date. Without it the parse is today's view (live runs).

    Known residual as-of limitations (submissions JSON carries no history for
    these): `sic`, `tickers`, `exchanges` and `name` are always current."""
    info = SubmissionsInfo(cik=int(d.get("cik", 0) or 0))
    asof_iso = _asof_iso(asof)
    info.asof_iso = asof_iso
    info.name = d.get("name", "") or ""
    sic_raw = d.get("sic", "")
    info.sic = int(sic_raw) if str(sic_raw).strip().isdigit() else None
    info.sic_description = d.get("sicDescription", "") or ""
    info.tickers = d.get("tickers", []) or []
    info.exchanges = d.get("exchanges", []) or []
    info.fiscal_year_end = d.get("fiscalYearEnd", "") or ""
    info.former_names = [fn.get("name", "") for fn in d.get("formerNames", [])
                         if _former_name_visible(fn, asof_iso)]

    recent = d.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []
    accns = recent.get("accessionNumber", []) or []
    items = recent.get("items", []) or []
    rdates = recent.get("reportDate", []) or []
    periodic_forms = config.ALLOWED_FORMS | config.FPI_ANNUAL_FORMS
    n = len(forms)
    for i in range(n):
        form = forms[i]
        date = dates[i] if i < len(dates) else ""
        accn = accns[i] if i < len(accns) else ""
        item = items[i] if i < len(items) else ""
        if asof_iso and date and date > asof_iso:
            info.n_filings_suppressed += 1   # filed after asof: not yet known
            continue
        info.forms.append((form, date, accn, item))
        if form in periodic_forms:
            info.periodic_filings.append(
                (form, date, rdates[i] if i < len(rdates) else ""))

    _classify(info, asof_iso)
    return info


def catalyst_calendar(info: SubmissionsInfo, latest_period_end, asof,
                      annual_only: bool = False) -> dict:
    """Universe-wide next-print estimate from EDGAR filing cadence. Median
    filing lag L = median(filingDate - reportDate) over periodic filings filed
    on/before `asof` (the as-of leak guard for the replay harness); the next
    print is projected from latest_period_end + cadence + L. guide_reset_window
    is True when the next print is the FY (10-K) print where annual guidance
    resets. Returns {days_to_catalyst, projected_next_filing, next_period_end,
    lag_days, guide_reset_window, n_periodic}; days_to_catalyst NaN when there
    is no period-end anchor."""
    out = {"days_to_catalyst": np.nan, "projected_next_filing": "",
           "next_period_end": "", "lag_days": np.nan,
           "guide_reset_window": bool(annual_only), "n_periodic": 0}
    asof_iso = asof.date().isoformat()
    lags, rdates = [], []
    for _form, filed, rep in info.periodic_filings:
        if not filed or not rep or filed > asof_iso:  # as-of leak guard
            continue
        try:
            fd, rd = pd.Timestamp(filed), pd.Timestamp(rep)
        except (ValueError, TypeError):
            continue
        d = (fd - rd).days
        if 0 <= d <= 200:
            lags.append(d)
            rdates.append(rd)
    out["n_periodic"] = len(lags)
    L = (float(np.median(lags)) if len(lags) >= config.CATALYST_MIN_PERIODIC
         else float(config.CATALYST_LAG_FALLBACK_DAYS))
    out["lag_days"] = L
    cadence = 365.0 if annual_only else config.CATALYST_QUARTER_DAYS
    base_end = latest_period_end
    if base_end is None and rdates:
        base_end = max(rdates)
    if base_end is None:
        return out
    next_end = base_end + pd.Timedelta(days=cadence)
    proj = next_end + pd.Timedelta(days=L)
    out["next_period_end"] = next_end.date().isoformat()
    out["projected_next_filing"] = proj.date().isoformat()
    out["days_to_catalyst"] = max(0, (proj - asof).days)
    if not annual_only and len(info.fiscal_year_end) == 4 \
            and info.fiscal_year_end.isdigit():
        mm, dd = int(info.fiscal_year_end[:2]), int(info.fiscal_year_end[2:])
        try:
            fy_end = pd.Timestamp(year=next_end.year, month=mm, day=dd)
        except ValueError:
            fy_end = None
        if fy_end is not None:
            cands = [fy_end, fy_end + pd.Timedelta(days=365),
                     fy_end - pd.Timedelta(days=365)]
            gap = min(abs((next_end - c).days) for c in cands)
            out["guide_reset_window"] = bool(
                gap <= config.CATALYST_FY_MATCH_TOL_DAYS)
    return out


def _classify(info: SubmissionsInfo, asof_iso: str = "") -> None:
    form_set = {f for f, _, _, _ in info.forms}
    has_10q = any(f in ("10-Q", "10-Q/A") for f in form_set)
    has_fpi_annual = any(f.split("/")[0] in ("20-F", "40-F") for f in form_set)
    if has_fpi_annual and not has_10q:
        info.filer_type = "annual_only"

    # Structural fund detection (audit 1.5): recent forms are N-series.
    recent_forms = [f for f, _, _, _ in info.forms[:40]]
    n_fund = sum(1 for f in recent_forms
                 if f.split("/")[0] in config.FUND_FORMS)
    n_operating = sum(1 for f in recent_forms
                      if f.split("/")[0] in ("10-K", "10-Q", "20-F", "40-F",
                                             "8-K", "6-K"))
    info.is_fund = n_fund > 0 and n_fund >= n_operating

    info.is_despac_name = any(
        re.search(config.DESPAC_FORMER_NAME_RE, nm, re.IGNORECASE)
        for nm in info.former_names)

    # NT filings in last ~3 years. Both windows are anchored on the as-of date
    # (wall clock only when running live) — anchoring them on today made every
    # replay see post-asof NT/8-K events, the leak that contaminated the
    # nt_filer / auditor_change / non_reliance / restated base rates.
    cutoff = _years_ago_iso(3, asof_iso)
    info.nt_filings_recent = sum(
        1 for f, dt, _, _ in info.forms
        if f.startswith("NT 10") and dt >= cutoff)

    cutoff_8k = _years_ago_iso(2, asof_iso)
    info.recent_8k_items = [
        (dt, it) for f, dt, _, it in info.forms
        if f.split("/")[0] == "8-K" and dt >= cutoff_8k and it]

    cutoff_amd = _years_ago_iso(config.AMENDMENT_LOOKBACK_YEARS, asof_iso)
    info.amended_periodic = [
        (f, dt, accn) for f, dt, accn, _ in info.forms
        if f in config.AMENDED_PERIODIC_FORMS and dt >= cutoff_amd]

    for f, _, accn, _ in info.forms:
        if not info.latest_10k_accn and f in ("10-K", "10-K/A", "20-F", "40-F"):
            info.latest_10k_accn = accn
        if not info.latest_10q_accn and f in ("10-Q", "10-Q/A"):
            info.latest_10q_accn = accn
        if info.latest_10k_accn and info.latest_10q_accn:
            break


def _years_ago_iso(years: int, ref_iso: str = "") -> str:
    """`years` before `ref_iso` (the as-of date), or before today when the
    reference is empty — a live run."""
    import datetime as _dt
    d = _dt.date.today()
    if ref_iso:
        try:
            d = _dt.date.fromisoformat(ref_iso[:10])
        except ValueError:
            pass
    try:
        return d.replace(year=d.year - years).isoformat()
    except ValueError:  # Feb 29
        return d.replace(year=d.year - years, day=28).isoformat()


def has_8k_item(info: SubmissionsInfo, code: str) -> bool:
    return any(code in it for _, it in info.recent_8k_items)


def edgar_filing_url(cik: int, accn: str) -> str:
    if not accn:
        return ""
    return (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={cik:010d}&type=&dateb=&owner=include&count=40")\
        if not accn else \
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}"


def filter_facts_asof(facts_json: dict, filed_cutoff: str) -> dict:
    """Point-in-time view of a companyfacts JSON: keep only observations
    FILED on or before `filed_cutoff` (ISO date string). This is what makes
    the exemplar-replay harness honest — every fact in companyfacts carries
    its `filed` date, so 'what did the market know on date D' is exactly
    'facts with filed <= D'. Restated values filed after D revert to the
    originally-filed number, as they should."""
    out = {"cik": facts_json.get("cik"), "facts": {}}
    for taxo, tags in (facts_json.get("facts") or {}).items():
        new_tags = {}
        for tag, body in tags.items():
            units = {}
            for unit, obs in (body.get("units") or {}).items():
                kept = [o for o in obs if o.get("filed", "") <= filed_cutoff]
                if kept:
                    units[unit] = kept
            if units:
                new_tags[tag] = {**{k: v for k, v in body.items()
                                    if k != "units"}, "units": units}
        if new_tags:
            out["facts"][taxo] = new_tags
    return out
