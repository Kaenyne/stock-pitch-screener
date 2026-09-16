"""Market-data layer — yfinance only (spec: "yfinance = market data only").

Price history, market cap, shares, short interest, splits, FX pairs, and the
risk-free rate. Fundamentals never come from here (shallow, not
point-in-time). Prices are fetched via CHUNKED ``yf.download`` (chart
endpoint), NOT per-ticker ``.info`` at universe scale (rate limits, audit
1.4); per-ticker quoteSummary calls are reserved for smaller cohorts with
caching + retries + throttling (short interest per audit 4.7, drawdown
inputs per audit 4.4 feed from here).

yfinance 1.3.0 API notes (probed 2026-07-07):
- ``yf.download`` defaults: ``auto_adjust=True`` (so 'Close' IS the adjusted
  close), ``multi_level_index=True`` (columns are (Price, Ticker) MultiIndex
  even for a single symbol), ``group_by='column'``, ``progress=True`` (we pass
  False).
- Failed symbols come back as all-NaN columns and are recorded in
  ``yfinance.shared._ERRORS``; the call does not raise.
- ``Ticker.get_info()`` returns a plain dict (~180 keys) incl. sharesShort,
  sharesOutstanding, dateShortInterest (epoch seconds), shortPercentOfFloat,
  floatShares, enterpriseValue, marketCap, trailingPegRatio, currency,
  longBusinessSummary.
- ``Ticker.fast_info`` is a dict-like FastInfo accepting both camelCase and
  snake_case keys ('marketCap' / 'market_cap').
- ``Ticker.get_calendar()`` returns a dict with 'Earnings Date' as a list of
  ``datetime.date``.
- ``Ticker.splits`` returns a float Series on a tz-aware DatetimeIndex.
- ``Ticker.earnings_dates`` scrapes HTML and requires lxml — avoided.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from . import config
from typing import Callable

import numpy as np
import pandas as pd

from . import config

try:  # keep the module importable without yfinance / network
    import yfinance as yf
except Exception as _e:  # pragma: no cover
    yf = None
    _YF_IMPORT_ERROR: Exception | None = _e
else:
    _YF_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state & local tunables (numbers with a home in config live
# there; these are market-layer plumbing constants)
# ---------------------------------------------------------------------------

#: Original (EDGAR-form) tickers that failed to download; accumulates across
#: calls — callers may ``FAILED_TICKERS.clear()`` before a run.
FAILED_TICKERS: list[str] = []

PRICE_CACHE_MAX_AGE_DAYS = 3          # parquet price cache freshness
CHUNK_SLEEP_RANGE_S = (1.0, 2.0)      # polite pause between download chunks
SLOPE_3M_DAYS = 63                    # ~3 months of trading days
MIN_SLOPE_OBS = 10                    # min non-NaN closes for slope_3m
MIN_WEEKLY_OBS = 40                   # min weekly return pairs for beta
QS_SUMMARY_TRUNC = 500                # longBusinessSummary truncation
FRED_DGS10_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
FRED_TIMEOUT_S = (10, 30)             # (connect, read) — FRED can stall
# FRED's bot filter times out spoofed browser strings from non-browser
# clients but serves honest ones immediately (probed 2026-09-16).
FRED_HEADERS = {"User-Agent": "pitch-screener/1.0 (python-requests)"}

_RF_CACHE: dict[str, float] = {}
_RF_HISTORY_CACHE: dict[str, pd.Series] = {}
_FX_CACHE: dict[str, float] = {}

_QS_NUMERIC_FIELDS = [
    "sharesShort", "sharesOutstanding", "shortPercentOfFloat", "floatShares",
    "enterpriseValue", "marketCap", "trailingPegRatio",
    # Display-only, no scoring weight. Both were already in every cached
    # payload (94% / 98% coverage) and thrown away by this keep-list, so
    # surfacing them costs zero network calls: analyst count is the coverage/
    # neglect pillar winning long decks open with, insider ownership is the
    # skin-in-the-game one.
    "numberOfAnalystOpinions", "heldPercentInsiders",
]
_QS_COLUMNS = _QS_NUMERIC_FIELDS + [
    "si_ratio", "dateShortInterest", "nextEarningsDate", "currency",
    "longBusinessSummary",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _require_yf() -> None:
    if yf is None:
        raise RuntimeError(
            f"yfinance is unavailable (import failed: {_YF_IMPORT_ERROR!r})")


def _to_yahoo(ticker: str) -> str:
    """EDGAR ticker -> Yahoo symbol. EDGAR class shares use '.', Yahoo '-'."""
    return ticker.strip().upper().replace(".", "-")


def _period_years(period: str) -> int | None:
    """Parse a yfinance-style period like '3y' / '2y'; None if not year-form."""
    m = re.fullmatch(r"(\d+)y", str(period).strip().lower())
    return int(m.group(1)) if m else None


def _to_float(v) -> float:
    if v is None or isinstance(v, bool):
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _sleep_between_chunks() -> None:
    time.sleep(random.uniform(*CHUNK_SLEEP_RANGE_S))


# ---------------------------------------------------------------------------
# 1. Chunked price download
# ---------------------------------------------------------------------------

def _chunk_closes(chunk: list[str], period: str) -> pd.DataFrame | None:
    """Download one chunk of Yahoo symbols; return adjusted-close frame
    (columns = yahoo symbols) or None on hard failure."""
    try:
        raw = yf.download(chunk, period=period, interval="1d",
                          auto_adjust=True, group_by="column",
                          threads=True, progress=False)
    except Exception as e:
        logger.debug("chunk download raised for %d tickers: %s", len(chunk), e)
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return None
        closes = raw["Close"]
    else:  # defensive: flat columns only happen for single-symbol frames
        if "Close" not in raw.columns or len(chunk) != 1:
            return None
        closes = raw[["Close"]].rename(columns={"Close": chunk[0]})
    closes = closes.loc[:, ~closes.columns.duplicated()]
    return closes


def chunked_download(tickers: list[str], period: str = "3y",
                     chunk_size: int = 400,
                     cache_path: str | None = None) -> pd.DataFrame:
    """Adjusted daily closes for a large ticker list via chunked yf.download.

    Returns a DataFrame: DatetimeIndex x tickers, where the columns use the
    ORIGINAL (EDGAR-form) ticker strings — '.'-class tickers are converted to
    Yahoo's '-' form only for the request. Failed tickers are appended to the
    module-level ``FAILED_TICKERS`` list and simply omitted. Each failed
    chunk is retried once at half size. If ``cache_path`` exists and is less
    than ``PRICE_CACHE_MAX_AGE_DAYS`` old, the parquet is loaded instead of
    downloading; otherwise the result is written there.
    """
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            age_s = time.time() - p.stat().st_mtime
            if age_s < PRICE_CACHE_MAX_AGE_DAYS * 86400:
                out = pd.read_parquet(p)
                out.index = pd.to_datetime(out.index)
                # the cache is only usable if it actually COVERS this request
                # (a prior narrower run must not silently starve a wider one;
                # ~2% slack for genuinely dead/delisted tickers)
                want = {t for t in tickers if t}
                have = set(out.columns)
                if len(want - have) <= max(2, 0.02 * len(want)):
                    logger.debug("price cache hit (%s, %.1fh old)",
                                 cache_path, age_s / 3600)
                    return out
                logger.debug("price cache MISS: %d requested tickers absent",
                             len(want - have))

    _require_yf()

    # De-dupe preserving order; map yahoo symbol -> original EDGAR ticker.
    yahoo_map: dict[str, str] = {}
    for t in tickers:
        if not t:
            continue
        y = _to_yahoo(t)
        if y in yahoo_map:
            if yahoo_map[y] != t:
                logger.debug("yahoo symbol collision: %s and %s -> %s",
                             yahoo_map[y], t, y)
            continue
        yahoo_map[y] = t
    yahoos = list(yahoo_map)

    frames: list[pd.DataFrame] = []

    def _absorb(requested: list[str], closes: pd.DataFrame) -> None:
        good = [y for y in requested
                if y in closes.columns and closes[y].notna().any()]
        bad = [y for y in requested if y not in good]
        if bad:
            FAILED_TICKERS.extend(yahoo_map[y] for y in bad)
            logger.debug("missing/empty tickers in chunk: %s", bad)
        if good:
            frames.append(closes[good])

    chunks = [yahoos[i:i + chunk_size]
              for i in range(0, len(yahoos), chunk_size)]
    for k, chunk in enumerate(chunks):
        closes = _chunk_closes(chunk, period)
        if closes is None:
            # Retry once with halved chunk size.
            half = max(1, len(chunk) // 2)
            logger.debug("chunk %d failed; retrying in halves of %d", k, half)
            for sub in (chunk[j:j + half] for j in range(0, len(chunk), half)):
                _sleep_between_chunks()
                sub_closes = _chunk_closes(list(sub), period)
                if sub_closes is None:
                    FAILED_TICKERS.extend(yahoo_map[y] for y in sub)
                    logger.debug("retry sub-chunk failed: %s", list(sub))
                else:
                    _absorb(list(sub), sub_closes)
        else:
            _absorb(chunk, closes)
        if k < len(chunks) - 1:
            _sleep_between_chunks()

    if frames:
        out = pd.concat(frames, axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        out = out.rename(columns=yahoo_map)  # back to EDGAR ticker strings
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
    else:
        out = pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))

    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(p)
        logger.debug("wrote price cache %s (%d x %d)",
                     cache_path, len(out), out.shape[1])
    return out


# ---------------------------------------------------------------------------
# 2. Last closes
# ---------------------------------------------------------------------------

def last_closes(closes: pd.DataFrame) -> pd.Series:
    """Last non-NaN close per ticker (index = tickers)."""
    if closes.empty:
        return pd.Series(dtype=float, name="last_close")
    s = closes.sort_index().ffill().iloc[-1]
    s.name = "last_close"
    return s


# ---------------------------------------------------------------------------
# 3. Drawdown inputs (audit 4.4 — valuation-reset row)
# ---------------------------------------------------------------------------

def drawdown_inputs(closes: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker inputs for the drawdown kernel + price-confirming term.

    Rows = tickers; columns: 'drawdown' (1 - last/high over the
    config.DRAWDOWN_LOOKBACK window), 'pct_above_low' (last/low - 1),
    'slope_3m' (OLS slope of log price per trading day over the last ~63
    trading days), 'high_3y', 'low_3y', 'last'.
    """
    cols = ["drawdown", "pct_above_low", "slope_3m", "high_3y", "low_3y",
            "last", "ret_12m"]
    if closes.empty:
        return pd.DataFrame(columns=cols)
    closes = closes.sort_index()

    last = last_closes(closes)

    # High/low window: config.DRAWDOWN_LOOKBACK (frame is fetched at 3y, so
    # this is normally the full frame; restrict anyway if a longer frame is
    # passed).
    years = _period_years(config.DRAWDOWN_LOOKBACK)
    if years is not None:
        cutoff = closes.index.max() - pd.DateOffset(years=years)
        window = closes[closes.index > cutoff]
    else:
        window = closes
    high = window.max()
    low = window.min()

    drawdown = (1.0 - last / high).where(high > 0)
    pct_above_low = (last / low - 1.0).where(low > 0)

    # slope_3m: per-day OLS slope of log price over the last ~63 sessions.
    tail = closes.tail(SLOPE_3M_DAYS)
    x_all = np.arange(len(tail), dtype=float)
    slopes = {}
    for c in tail.columns:
        y = tail[c].to_numpy(dtype=float)
        mask = np.isfinite(y) & (y > 0)
        if mask.sum() < MIN_SLOPE_OBS:
            slopes[c] = np.nan
            continue
        slopes[c] = float(np.polyfit(x_all[mask], np.log(y[mask]), 1)[0])

    # ret_12m: 12-month price appreciation (run-up row input) — close ~252
    # sessions back, first valid within a small window for names with gaps.
    if len(closes) > 260:
        base = closes.iloc[-260:-245].mean()
    else:
        base = closes.iloc[0]
    ret_12m = (last / base - 1.0).where(base > 0)

    out = pd.DataFrame({
        "drawdown": drawdown,
        "pct_above_low": pct_above_low,
        "slope_3m": pd.Series(slopes),
        "high_3y": high,
        "low_3y": low,
        "last": last,
        "ret_12m": ret_12m,
    })
    out.index.name = "ticker"
    return out[cols]


# ---------------------------------------------------------------------------
# 4. Weekly beta (CAPM input, audit 3.6)
# ---------------------------------------------------------------------------

def beta_weekly(closes: pd.DataFrame, spx: pd.Series) -> pd.Series:
    """Per-ticker beta from config.BETA_LOOKBACK (2y) of weekly returns
    (W-FRI last, pct_change) vs the S&P; cov/var; NaN below
    ``MIN_WEEKLY_OBS`` paired weekly observations. Unclamped — the WACC
    layer applies config.BETA_CLAMP.
    """
    if closes.empty or spx is None or len(spx) == 0:
        return pd.Series(dtype=float, name="beta")
    if isinstance(spx, pd.DataFrame):  # tolerate a 1-col frame
        spx = spx.iloc[:, 0]

    w = closes.sort_index().resample("W-FRI").last()
    ws = spx.sort_index().resample("W-FRI").last()

    years = _period_years(config.BETA_LOOKBACK) or 2
    cutoff = w.index.max() - pd.DateOffset(years=years)
    w = w[w.index > cutoff]
    ws = ws[ws.index > cutoff]

    rets = w.pct_change()
    mkt = ws.pct_change().reindex(rets.index)

    mkt_arr = mkt.to_numpy(dtype=float)
    betas = {}
    for c in rets.columns:
        y = rets[c].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(mkt_arr)
        n = int(mask.sum())
        if n < MIN_WEEKLY_OBS:
            betas[c] = np.nan
            continue
        xm = mkt_arr[mask]
        ym = y[mask]
        var = xm.var(ddof=1)
        if not np.isfinite(var) or var <= 0:
            betas[c] = np.nan
            continue
        cov = np.cov(xm, ym, ddof=1)[0, 1]
        betas[c] = float(cov / var)
    out = pd.Series(betas, name="beta")
    out.index.name = "ticker"
    return out


# ---------------------------------------------------------------------------
# 5. S&P 500 series
# ---------------------------------------------------------------------------

def fetch_spx(period: str = "3y") -> pd.Series:
    """^GSPC daily closes (auto-adjusted) as a Series."""
    _require_yf()
    raw = yf.download("^GSPC", period=period, interval="1d",
                      auto_adjust=True, progress=False,
                      multi_level_index=False)
    if raw is None or raw.empty or "Close" not in raw.columns:
        logger.debug("fetch_spx returned no data")
        return pd.Series(dtype=float, name="^GSPC")
    s = raw["Close"].dropna()
    s.name = "^GSPC"
    return s


# ---------------------------------------------------------------------------
# 6/7. Risk-free rate — FRED DGS10 keyless CSV, ^TNX fallback
# ---------------------------------------------------------------------------

def _fred_series(series_id: str, extra_params: str = "") -> pd.Series | None:
    """Any FRED series from the keyless fredgraph CSV as a date-indexed float
    Series (missing '.' rows dropped), or None on any failure. Generalizes the
    DGS10 fetch — used for the risk-free rate AND the Tier-2 PPI/CPI series."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}" \
        + extra_params
    try:
        import requests
        r = requests.get(url, headers=FRED_HEADERS, timeout=FRED_TIMEOUT_S)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
    except Exception as e:
        logger.debug("FRED fetch failed (%s): %s", url, e)
        return None
    if df.shape[1] < 2:
        return None
    date_col = df.columns[0]
    val_col = series_id if series_id in df.columns else df.columns[-1]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    vals = pd.to_numeric(df[val_col], errors="coerce")  # '.' -> NaN
    s = pd.Series(vals.to_numpy(), index=dates).dropna()
    return s if len(s) else None


def _fred_dgs10(extra_params: str = "") -> pd.Series | None:
    """DGS10 (10-yr Treasury) — the risk-free rate series."""
    return _fred_series("DGS10", extra_params)


def _fred_api_series(series_id: str) -> pd.Series | None:
    """A FRED series via the JSON API (api.stlouisfed.org) + key — the graph-CSV
    host is firewall-blocked here, but the API host is reachable. Date-indexed
    float Series, None on any failure or when no key is configured."""
    key = config.FRED_API_KEY
    if not key:
        return None
    try:
        import requests
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": series_id, "api_key": key,
                                 "file_type": "json"},
                         headers=FRED_HEADERS, timeout=FRED_TIMEOUT_S)
        r.raise_for_status()
        obs = r.json().get("observations", [])
    except Exception as e:
        logger.debug("FRED API %s failed: %s", series_id, e)
        return None
    if not obs:
        return None
    idx = pd.to_datetime([o.get("date") for o in obs], errors="coerce")
    vals = pd.to_numeric([o.get("value") for o in obs], errors="coerce")  # '.'->NaN
    s = pd.Series(vals, index=idx).dropna()
    return s if len(s) else None


# ---------------------------------------------------------------------------
# FRED disk cache — data/cache/fred_series.parquet, long format
# (series_id, date, value). Without it no replay is reproducible: DGS10 and
# every PPI/CPI series were re-fetched live on each run, so the same replay
# date produced different inputs on different days and could not run offline.
# ---------------------------------------------------------------------------

FRED_CACHE_PATH = (Path(__file__).resolve().parent.parent
                   / config.CACHE_DIR / "fred_series.parquet")
_FRED_DISK: dict[str, pd.Series] | None = None
_FRED_DIRTY: set[str] = set()
_FRED_OFFLINE = False


def set_fred_offline(flag: bool = True) -> None:
    """Forbid FRED network access — cache or nothing. Used by the replay
    harness so a rerun of a past date is byte-reproducible."""
    global _FRED_OFFLINE
    _FRED_OFFLINE = bool(flag)


def _fred_cache_read() -> dict[str, pd.Series]:
    global _FRED_DISK
    if _FRED_DISK is not None:
        return _FRED_DISK
    _FRED_DISK = {}
    try:
        if FRED_CACHE_PATH.exists():
            df = pd.read_parquet(FRED_CACHE_PATH)
            for sid, g in df.groupby("series_id"):
                s = pd.Series(g["value"].to_numpy(dtype=float),
                              index=pd.to_datetime(g["date"])).sort_index()
                _FRED_DISK[str(sid)] = s[~s.index.duplicated(keep="last")]
            logger.debug("FRED cache loaded: %d series from %s",
                         len(_FRED_DISK), FRED_CACHE_PATH)
    except Exception as e:
        logger.debug("FRED cache read failed: %s", e)
        _FRED_DISK = {}
    return _FRED_DISK


def fred_cache_flush() -> None:
    """Persist newly fetched series. No-op when nothing changed."""
    if not _FRED_DIRTY or not _FRED_DISK:
        return
    try:
        FRED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        frames = [pd.DataFrame({"series_id": sid, "date": s.index,
                                "value": s.to_numpy(dtype=float)})
                  for sid, s in _FRED_DISK.items() if len(s)]
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(
                FRED_CACHE_PATH, index=False)
            logger.info("FRED cache written: %s (%d series, %d new)",
                        FRED_CACHE_PATH, len(frames), len(_FRED_DIRTY))
        _FRED_DIRTY.clear()
    except Exception as e:
        logger.debug("FRED cache write failed: %s", e)


def _fred_cache_age_days() -> float:
    try:
        return (time.time() - FRED_CACHE_PATH.stat().st_mtime) / 86400.0
    except OSError:
        return float("inf")


def fred_series_cached(series_id: str, offline: bool | None = None,
                       max_age_days: float | None = None) -> pd.Series | None:
    """A FRED series, disk-cache first when fresh (or offline), network
    otherwise, cache fallback when the network fails. Fresh observations win
    over cached ones on overlapping dates, so FRED revisions land without
    dropping history the cache already holds."""
    offline = _FRED_OFFLINE if offline is None else offline
    max_age = (config.FRED_CACHE_MAX_AGE_DAYS if max_age_days is None
               else max_age_days)
    disk = _fred_cache_read()
    cached = disk.get(series_id)
    if cached is not None and (offline or _fred_cache_age_days() <= max_age):
        return cached.copy()
    if offline:
        return None
    fresh = _fred_api_series(series_id)
    if fresh is None:
        fresh = _fred_series(series_id)
    if fresh is None or not len(fresh):
        return cached.copy() if cached is not None else None
    merged = (fresh if cached is None
              else fresh.combine_first(cached).sort_index())
    disk[series_id] = merged
    _FRED_DIRTY.add(series_id)
    return merged.copy()


def fetch_ppi_series(series_ids, offline: bool | None = None) -> dict:
    """Fetch a set of FRED PPI/CPI series once (Tier-2 external-price signals),
    disk-cache first, then the network (API if a key is set, else the
    keyless CSV). Returns {series_id: Series} for those that resolve;
    silently omits any that fail (a missing series just means that sector's
    signal is absent, never a crash or a false fire)."""
    out = {}
    for sid in series_ids:
        s = fred_series_cached(sid, offline=offline)
        if s is not None and len(s):
            out[sid] = s
    fred_cache_flush()
    return out


def _tnx_closes(period: str) -> pd.Series | None:
    """^TNX (CBOE 10-yr yield, quoted in %) closes, or None."""
    try:
        _require_yf()
        raw = yf.download("^TNX", period=period, interval="1d",
                          auto_adjust=True, progress=False,
                          multi_level_index=False)
    except Exception as e:
        logger.debug("^TNX download failed: %s", e)
        return None
    if raw is None or raw.empty or "Close" not in raw.columns:
        return None
    s = raw["Close"].dropna()
    return s if len(s) else None


def rf_current() -> float:
    """Current 10-yr treasury yield as a decimal. FRED DGS10 keyless CSV
    first (recent window only), ^TNX last close fallback. Cached in-module.
    """
    if "rf" in _RF_CACHE:
        return _RF_CACHE["rf"]
    # Disk-cached full DGS10 — shared with rf_history_annual, so the two
    # together cost one fetch instead of two, and a replay can run offline.
    s = fred_series_cached("DGS10")
    if (s is None or not len(s)) and not _FRED_OFFLINE:
        start = (pd.Timestamp.today() - pd.Timedelta(days=30)).date().isoformat()
        s = _fred_dgs10(extra_params=f"&cosd={start}")
    if s is not None and len(s):
        rf = float(s.iloc[-1]) / 100.0
        _RF_CACHE["rf"] = rf
        fred_cache_flush()
        return rf
    if _FRED_OFFLINE:
        logger.debug("rf_current: offline with no cached DGS10")
        return float("nan")
    tnx = _tnx_closes("5d")
    if tnx is not None:
        rf = float(tnx.iloc[-1]) / 100.0
        _RF_CACHE["rf"] = rf
        return rf
    logger.debug("rf_current: FRED and ^TNX both failed")
    return float("nan")


def rf_history_annual() -> pd.Series:
    """Annual mean 10-yr treasury yield (decimals), index = year (int).

    Used by metrics.moat_spread to vary Rf by year (audit 3.6). FRED DGS10
    full history first; ^TNX period='max' fallback (spec names both
    "^TNX history / FRED DGS10"); empty Series when both fail. Cached.
    """
    if "annual" in _RF_HISTORY_CACHE:
        return _RF_HISTORY_CACHE["annual"].copy()
    s = fred_series_cached("DGS10")
    if (s is None or not len(s)) and not _FRED_OFFLINE:
        s = _tnx_closes("max")
    if s is None or not len(s):
        logger.debug("rf_history_annual: FRED and ^TNX both failed")
        return pd.Series(dtype=float, name="rf")
    fred_cache_flush()
    ann = s.groupby(s.index.year).mean() / 100.0
    ann.index = ann.index.astype(int)
    ann.index.name = "year"
    ann.name = "rf"
    _RF_HISTORY_CACHE["annual"] = ann
    return ann.copy()


# ---------------------------------------------------------------------------
# 8. Per-ticker quoteSummary fields (small cohorts only; audit 4.7)
# ---------------------------------------------------------------------------

def _qs_cache_file(cache_dir: Path, ticker: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    return cache_dir / f"{safe}.json"


def _qs_load_cache(cache_dir: Path, ticker: str,
                   max_age_days: int) -> dict | None:
    f = _qs_cache_file(cache_dir, ticker)
    try:
        if not f.exists():
            return None
        if time.time() - f.stat().st_mtime > max_age_days * 86400:
            return None
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("qs cache read failed for %s: %s", ticker, e)
        return None


def _qs_fetch(ticker: str) -> dict | None:
    """Network fetch of one ticker's info dict + best-effort earnings date.
    Returns a JSON-serializable payload, or None on failure."""
    try:
        tk = yf.Ticker(_to_yahoo(ticker))
        info = tk.get_info()
        if not isinstance(info, dict) or not info:
            return None
    except Exception as e:
        logger.debug("get_info failed for %s: %s", ticker, e)
        return None
    next_earnings = None
    try:
        cal = tk.get_calendar()
        if isinstance(cal, dict):
            eds = cal.get("Earnings Date") or []
            if eds:
                next_earnings = str(eds[0])  # datetime.date -> 'YYYY-MM-DD'
    except Exception as e:  # best-effort, nullable
        logger.debug("get_calendar failed for %s: %s", ticker, e)
    return {
        "_fetched_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "nextEarningsDate": next_earnings,
        "info": info,
    }


def _qs_row(payload: dict | None) -> dict:
    """Payload -> flat row of the scored/displayed fields (NaN/NaT/None when
    missing — shortPercentOfFloat and trailingPegRatio are display-only)."""
    row: dict = {f: float("nan") for f in _QS_NUMERIC_FIELDS}
    row.update({"si_ratio": float("nan"), "dateShortInterest": pd.NaT,
                "nextEarningsDate": pd.NaT, "currency": None,
                "longBusinessSummary": None})
    if not payload:
        return row
    info = payload.get("info") or {}
    for f in _QS_NUMERIC_FIELDS:
        row[f] = _to_float(info.get(f))
    dsi = info.get("dateShortInterest")
    if isinstance(dsi, (int, float)) and not isinstance(dsi, bool):
        row["dateShortInterest"] = pd.to_datetime(dsi, unit="s",
                                                  errors="coerce")
    elif isinstance(dsi, str):
        row["dateShortInterest"] = pd.to_datetime(dsi, errors="coerce")
    ed = payload.get("nextEarningsDate")
    if ed:
        row["nextEarningsDate"] = pd.to_datetime(ed, errors="coerce")
    cur = info.get("currency")
    row["currency"] = cur if isinstance(cur, str) else None
    lbs = info.get("longBusinessSummary")
    if isinstance(lbs, str):
        row["longBusinessSummary"] = lbs[:QS_SUMMARY_TRUNC]
    ss, so = row["sharesShort"], row["sharesOutstanding"]
    if math.isfinite(ss) and math.isfinite(so) and so > 0:
        row["si_ratio"] = ss / so  # spec: denominator = sharesOutstanding
    return row


def fetch_quote_summaries(tickers: list[str], cache_dir: str,
                          max_age_days: int = 7, throttle_s: float = 0.4,
                          progress_cb: Callable[[int, int], None] | None = None,
                          ) -> pd.DataFrame:
    """Slow per-ticker quoteSummary fields for a SMALL cohort (never the full
    universe — chunked_download covers prices).

    Columns: sharesShort, sharesOutstanding, shortPercentOfFloat
    (display-only, often None), floatShares, enterpriseValue, marketCap,
    trailingPegRatio (display-only, sparse), si_ratio
    (= sharesShort/sharesOutstanding), dateShortInterest, nextEarningsDate
    (best-effort), currency, longBusinessSummary (truncated to 500 chars).
    Rows indexed by the original ticker strings.

    Each ticker's raw payload is persisted to ``cache_dir/<ticker>.json``;
    files younger than ``max_age_days`` are served from cache (making a long
    interrupted fetch resumable). Every per-ticker call is wrapped so a
    failure yields a NaN row, never an exception. ``progress_cb(i, n)`` is
    invoked every 25 names (and at completion).
    """
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    n = len(tickers)
    rows: dict[str, dict] = {}
    for i, t in enumerate(tickers):
        payload = _qs_load_cache(cdir, t, max_age_days)
        if payload is None:
            if yf is None:
                logger.debug("yfinance unavailable; NaN row for %s", t)
            else:
                payload = _qs_fetch(t)
                if payload is not None:
                    try:
                        _qs_cache_file(cdir, t).write_text(
                            json.dumps(payload, default=str),
                            encoding="utf-8")
                    except Exception as e:
                        logger.debug("qs cache write failed for %s: %s", t, e)
                time.sleep(throttle_s)  # throttle network path only
        rows[t] = _qs_row(payload)
        if progress_cb is not None and ((i + 1) % 25 == 0 or i + 1 == n):
            try:
                progress_cb(i + 1, n)
            except Exception as e:
                logger.debug("progress_cb raised: %s", e)
    df = pd.DataFrame.from_dict(rows, orient="index")
    if df.empty:
        df = pd.DataFrame(columns=_QS_COLUMNS)
    df.index.name = "ticker"
    return df[_QS_COLUMNS]


# ---------------------------------------------------------------------------
# 9. fast_info market caps (multi-class names + final-shortlist cap recheck)
# ---------------------------------------------------------------------------

def fast_marketcaps(tickers: list[str],
                    throttle_s: float = 0.3) -> pd.Series:
    """Per-ticker fast_info market cap; NaN on any failure. Used for
    multi-class names at universe build and the CAP_BAND_FINAL recheck."""
    _require_yf()
    out: dict[str, float] = {}
    for i, t in enumerate(tickers):
        val = float("nan")
        try:
            v = yf.Ticker(_to_yahoo(t)).fast_info["marketCap"]
            if v is not None:
                val = float(v)
        except Exception as e:
            logger.debug("fast_info marketCap failed for %s: %s", t, e)
        out[t] = val
        if i < len(tickers) - 1:
            time.sleep(throttle_s)
    s = pd.Series(out, dtype=float, name="marketCap")
    s.index.name = "ticker"
    return s


# ---------------------------------------------------------------------------
# 10. Splits
# ---------------------------------------------------------------------------

def splits_series(ticker: str) -> pd.Series:
    """Split history (date-indexed ratio Series); empty Series on failure."""
    try:
        _require_yf()
        s = yf.Ticker(_to_yahoo(ticker)).splits
        if isinstance(s, pd.Series):
            return s
    except Exception as e:
        logger.debug("splits failed for %s: %s", ticker, e)
    return pd.Series(dtype=float)


def insider_summary(ticker: str) -> str | None:
    """Shortlist-stage insider enrichment (spec: yfinance insider_purchases /
    netSharePurchaseActivity for surfaced names only, nullable). Returns a
    one-line human-readable summary or None."""
    try:
        _require_yf()
        t = yf.Ticker(_to_yahoo(ticker))
        for attr in ("insider_purchases", "get_insider_purchases"):
            obj = getattr(t, attr, None)
            df = obj() if callable(obj) else obj
            if df is not None and hasattr(df, "empty") and not df.empty:
                # yfinance returns a small table; render its first rows
                flat = df.head(4).to_string(index=False).replace("\n", "; ")
                return flat[:300]
    except Exception as e:
        logger.debug("insider fetch failed for %s: %s", ticker, e)
    return None


# ---------------------------------------------------------------------------
# 11. FX
# ---------------------------------------------------------------------------

def fx_rate(currency: str) -> float:
    """Spot rate to USD for a currency code ('EUR' -> EURUSD=X last close).
    1.0 for USD, NaN on failure; successful lookups cached in-module.
    'GBp'/'GBX' (pence) handled as GBP/100."""
    if not currency or not isinstance(currency, str):
        return float("nan")
    cur = currency.strip()
    if cur.upper() == "USD":
        return 1.0
    if cur == "GBp" or cur.upper() == "GBX":
        gbp = fx_rate("GBP")
        return gbp / 100.0 if math.isfinite(gbp) else float("nan")
    cur = cur.upper()
    if cur in _FX_CACHE:
        return _FX_CACHE[cur]
    rate = float("nan")
    try:
        _require_yf()
        raw = yf.download(f"{cur}USD=X", period="5d", interval="1d",
                          auto_adjust=True, progress=False,
                          multi_level_index=False)
        if raw is not None and not raw.empty and "Close" in raw.columns:
            vals = raw["Close"].dropna()
            if len(vals):
                rate = float(vals.iloc[-1])
    except Exception as e:
        logger.debug("fx_rate failed for %s: %s", currency, e)
    if math.isfinite(rate):
        _FX_CACHE[cur] = rate
    return rate
