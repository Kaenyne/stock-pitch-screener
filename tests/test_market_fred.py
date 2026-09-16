"""FRED disk-cache tests.

The cache is what makes a replay reproducible: before it, DGS10 and every
PPI/CPI series were re-fetched live on each run, so the same replay date
produced different inputs on different days and could not run offline at all.
The failure modes that matter are therefore (a) silently going to the network
when told not to, and (b) losing cached history on a partial refresh.
"""

import pandas as pd
import pytest

from screener import market


@pytest.fixture
def fred(tmp_path, monkeypatch):
    """Isolated cache file + reset module state, no network."""
    monkeypatch.setattr(market, "FRED_CACHE_PATH", tmp_path / "fred.parquet")
    monkeypatch.setattr(market, "_FRED_DISK", None)
    monkeypatch.setattr(market, "_FRED_DIRTY", set())
    monkeypatch.setattr(market, "_FRED_OFFLINE", False)
    calls = {"api": 0, "csv": 0}

    def _api(sid):
        calls["api"] += 1
        return _series(sid)

    def _csv(sid, extra_params=""):
        calls["csv"] += 1
        return None

    monkeypatch.setattr(market, "_fred_api_series", _api)
    monkeypatch.setattr(market, "_fred_series", _csv)
    return calls


def _series(sid, start="2024-01-01", n=5, base=100.0):
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.Series([base + i for i in range(n)], index=idx, name=sid)


class TestFetchAndPersist:
    def test_fetch_then_serve_from_disk(self, fred, monkeypatch):
        s = market.fred_series_cached("WPU01")
        assert s is not None and len(s) == 5
        market.fred_cache_flush()
        assert market.FRED_CACHE_PATH.exists()
        # fresh file -> second call must not touch the network
        monkeypatch.setattr(market, "_FRED_DISK", None)
        again = market.fred_series_cached("WPU01")
        assert fred["api"] == 1
        # round-tripping through parquet drops the index `freq` attribute;
        # the observations are what matter
        pd.testing.assert_series_equal(s, again, check_names=False,
                                       check_freq=False)

    def test_flush_is_noop_when_nothing_changed(self, fred):
        market.fred_cache_flush()
        assert not market.FRED_CACHE_PATH.exists()

    def test_multiple_series_round_trip(self, fred, monkeypatch):
        for sid in ("WPU01", "WPU02", "DGS10"):
            market.fred_series_cached(sid)
        market.fred_cache_flush()
        monkeypatch.setattr(market, "_FRED_DISK", None)
        disk = market._fred_cache_read()
        assert set(disk) == {"WPU01", "WPU02", "DGS10"}
        assert len(disk["WPU02"]) == 5


class TestOffline:
    def test_offline_serves_cache_without_network(self, fred, monkeypatch):
        market.fred_series_cached("WPU01")
        market.fred_cache_flush()
        before = fred["api"]
        monkeypatch.setattr(market, "_FRED_DISK", None)
        market.set_fred_offline(True)
        try:
            s = market.fred_series_cached("WPU01")
        finally:
            market.set_fred_offline(False)
        assert s is not None and len(s) == 5
        assert fred["api"] == before      # no network call

    def test_offline_returns_none_when_uncached(self, fred):
        assert market.fred_series_cached("NOPE", offline=True) is None
        assert fred["api"] == 0

    def test_offline_ignores_staleness(self, fred, monkeypatch):
        """A stale cache is still the right answer offline — reproducibility
        beats freshness when the whole point is rerunning a past date."""
        market.fred_series_cached("WPU01")
        market.fred_cache_flush()
        monkeypatch.setattr(market, "_fred_cache_age_days", lambda: 999.0)
        assert market.fred_series_cached("WPU01", offline=True) is not None
        assert fred["api"] == 1


class TestStalenessAndMerge:
    def test_stale_cache_refetches(self, fred, monkeypatch):
        market.fred_series_cached("WPU01")
        market.fred_cache_flush()
        monkeypatch.setattr(market, "_fred_cache_age_days", lambda: 999.0)
        market.fred_series_cached("WPU01")
        assert fred["api"] == 2

    def test_refresh_keeps_history_the_fetch_omits(self, fred, monkeypatch):
        """FRED truncates and revises. A refresh returning only recent
        observations must not silently drop the older ones already cached."""
        market.fred_series_cached("WPU01")            # 2024-01 .. 2024-05
        market.fred_cache_flush()
        monkeypatch.setattr(market, "_fred_cache_age_days", lambda: 999.0)
        monkeypatch.setattr(market, "_fred_api_series",
                            lambda sid: _series(sid, start="2024-04-01", n=3,
                                                base=500.0))
        merged = market.fred_series_cached("WPU01")
        assert len(merged) == 6                        # Jan..Jun
        assert merged.loc["2024-01-01"] == 100.0       # old history kept
        assert merged.loc["2024-04-01"] == 500.0       # revision wins

    def test_network_failure_falls_back_to_cache(self, fred, monkeypatch):
        market.fred_series_cached("WPU01")
        market.fred_cache_flush()
        monkeypatch.setattr(market, "_fred_cache_age_days", lambda: 999.0)
        monkeypatch.setattr(market, "_fred_api_series", lambda sid: None)
        monkeypatch.setattr(market, "_fred_series",
                            lambda sid, extra_params="": None)
        s = market.fred_series_cached("WPU01")
        assert s is not None and len(s) == 5

    def test_total_failure_returns_none(self, fred, monkeypatch):
        monkeypatch.setattr(market, "_fred_api_series", lambda sid: None)
        monkeypatch.setattr(market, "_fred_series",
                            lambda sid, extra_params="": None)
        assert market.fred_series_cached("NOPE") is None

    def test_csv_fallback_when_api_returns_none(self, fred, monkeypatch):
        monkeypatch.setattr(market, "_fred_api_series", lambda sid: None)
        monkeypatch.setattr(market, "_fred_series",
                            lambda sid, extra_params="": _series(sid))
        assert len(market.fred_series_cached("WPU01")) == 5


class TestFetchPpiSeries:
    def test_returns_only_resolvable(self, fred, monkeypatch):
        def _api(sid):
            return _series(sid) if sid != "BAD" else None
        monkeypatch.setattr(market, "_fred_api_series", _api)
        monkeypatch.setattr(market, "_fred_series",
                            lambda sid, extra_params="": None)
        out = market.fetch_ppi_series(["WPU01", "BAD", "WPU02"])
        assert set(out) == {"WPU01", "WPU02"}

    def test_persists_after_fetch(self, fred):
        market.fetch_ppi_series(["WPU01", "WPU02"])
        assert market.FRED_CACHE_PATH.exists()
