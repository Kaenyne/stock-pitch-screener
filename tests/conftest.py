"""Shared synthetic-fixture builders for the screener test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def qdates(n: int, end: str = "2026-03-31") -> pd.DatetimeIndex:
    """n quarterly period-end dates ending at `end` (~91d spacing)."""
    e = pd.Timestamp(end)
    return pd.DatetimeIndex(sorted(e - pd.Timedelta(days=91) * i
                                   for i in range(n)))


def qseries(values, end: str = "2026-03-31") -> pd.Series:
    v = list(values)
    return pd.Series(v, index=qdates(len(v), end), dtype=float)


def make_duration_obs(quarterly: pd.Series, form_cycle=("10-Q", "10-Q", "10-Q", "10-K"),
                      ytd_style: bool = False, filed_lag_days: int = 40):
    """Build companyfacts-style unit observations from a quarterly series.

    Standard (income-statement) style: 10-Qs carry discrete quarters
    (~90d durations), the 10-K carries ONLY the FY duration (no Q4 fact) —
    this reproduces the audit-1.1 reality.

    ytd_style=True (cash-flow style): 10-Qs carry ONLY YTD durations
    (90/181/272d), 10-K carries FY.
    """
    obs = []
    idx = quarterly.index
    # group into fiscal years of 4 starting from the first quarter
    for start_i in range(0, len(idx) - 3, 4):
        chunk = idx[start_i:start_i + 4]
        if len(chunk) < 4:
            break
        fy_start = chunk[0] - pd.Timedelta(days=90)
        vals = quarterly[chunk]
        # Q1..Q3 filings
        for j in range(3):
            end = chunk[j]
            filed = (end + pd.Timedelta(days=filed_lag_days)).date().isoformat()
            if ytd_style:
                start = fy_start
                val = float(vals.iloc[:j + 1].sum())
            else:
                start = chunk[j] - pd.Timedelta(days=89)
                val = float(vals.iloc[j])
            obs.append({"start": start.date().isoformat(),
                        "end": end.date().isoformat(),
                        "val": val, "form": "10-Q",
                        "filed": filed, "accn": f"a{start_i}{j}",
                        "fy": 2000, "fp": "Q9"})  # fy/fp deliberately WRONG
        # FY filing (10-K): FY duration only — no standalone Q4 fact
        end = chunk[3]
        filed = (end + pd.Timedelta(days=60)).date().isoformat()
        obs.append({"start": fy_start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "val": float(vals.sum()), "form": "10-K",
                    "filed": filed, "accn": f"a{start_i}k",
                    "fy": 1999, "fp": "FY"})
    return obs


def make_instant_obs(series: pd.Series, form: str = "10-Q"):
    obs = []
    for i, (end, val) in enumerate(series.items()):
        obs.append({"end": end.date().isoformat(), "val": float(val),
                    "form": form,
                    "filed": (end + pd.Timedelta(days=40)).date().isoformat(),
                    "accn": f"i{i}"})
    return obs


def facts_json(tag_obs: dict[str, list], taxonomy: str = "us-gaap",
               dei_shares: list | None = None) -> dict:
    """Assemble a companyfacts-shaped JSON from {tag: [observations]}."""
    facts = {taxonomy: {}}
    for tag, obs in tag_obs.items():
        unit = "USD" if not tag.endswith("Shares") else "shares"
        facts[taxonomy][tag] = {"units": {unit: obs}}
    if dei_shares:
        facts["dei"] = {"EntityCommonStockSharesOutstanding":
                        {"units": {"shares": dei_shares}}}
    return {"cik": 123456, "facts": facts}


@pytest.fixture
def asof() -> pd.Timestamp:
    return pd.Timestamp("2026-07-07")


# --- canonical synthetic shapes (10 years of quarters) ---

def shape_seasonal(n: int = 40, base: float = 100.0) -> pd.Series:
    """Strong Q4-heavy seasonality, zero trend: the engine must NOT fire."""
    pattern = np.array([0.8, 0.9, 0.95, 1.35])
    vals = base * np.tile(pattern, n // 4 + 1)[:n]
    return qseries(vals)


def shape_trough_recovery(n: int = 40, base: float = 100.0) -> pd.Series:
    """Healthy -> decline -> trough ~4q ago -> genuine recovery (long hit)."""
    vals = np.full(n, base)
    vals[: n - 12] = base                     # long healthy plateau
    vals[n - 12: n - 4] = np.linspace(base, base * 0.45, 8)  # decline
    vals[n - 4:] = np.linspace(base * 0.45, base * 0.70, 4)  # recovery
    return qseries(vals)


def shape_falling_knife(n: int = 40, base: float = 100.0) -> pd.Series:
    """Monotonic decline into the present: must NOT earn trough credit."""
    vals = np.concatenate([np.full(n - 12, base),
                           np.linspace(base, base * 0.3, 12)])
    return qseries(vals)


def shape_covid_spike(n: int = 40, base: float = 100.0) -> pd.Series:
    """Long-flat -> 2x spike -> rolling over for ~6 quarters (short hit).
    The quarterly peak sits ~7q back so the LAGGED TTM peak lands ~4-5q back
    — recent enough for the peak-recency ramp, old enough to satisfy the
    'rolled over for >=2 quarters' requirement."""
    vals = np.full(n, base)
    vals[n - 13: n - 7] = np.linspace(base, base * 2.2, 6)
    vals[n - 7:] = np.linspace(base * 2.2, base * 1.1, 7)
    return qseries(vals)


def shape_steady_grower(n: int = 40, base: float = 100.0) -> pd.Series:
    """Compounding at ~3%/quarter, at all-time high: neither side should love it."""
    return qseries(base * 1.03 ** np.arange(n))
