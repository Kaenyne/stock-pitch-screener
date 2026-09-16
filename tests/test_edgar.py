"""EDGAR catalyst-calendar and point-in-time submissions-parse tests."""

import datetime as dt

import numpy as np
import pandas as pd

from screener.edgar import (SubmissionsInfo, catalyst_calendar, has_8k_item,
                            parse_submissions)


def _periodic(report_ends, lag=40, form="10-Q"):
    """Build periodic_filings (form, filingDate, reportDate) with a fixed lag."""
    out = []
    for re_iso in report_ends:
        rd = pd.Timestamp(re_iso)
        fd = rd + pd.Timedelta(days=lag)
        out.append((form, fd.date().isoformat(), rd.date().isoformat()))
    return out


def _info(periodic, fye="1231"):
    return SubmissionsInfo(cik=1, fiscal_year_end=fye, periodic_filings=periodic)


class TestCatalystCalendar:
    def test_lag_and_projection(self):
        ends = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
        info = _info(_periodic(ends, lag=40))
        asof = pd.Timestamp("2026-02-15")
        cat = catalyst_calendar(info, pd.Timestamp("2025-12-31"), asof)
        assert cat["n_periodic"] == 8
        assert cat["lag_days"] == 40
        # next print ~= 2025-12-31 + 91.25 cadence + 40 lag, from asof 2026-02-15
        assert 70 < cat["days_to_catalyst"] < 100

    def test_guide_reset_true_on_fy_print(self):
        ends = ["2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30",
                "2025-09-30"]
        info = _info(_periodic(ends), fye="1231")
        # latest period end Sep-30 -> next ~Dec-31 == FY end -> guide reset
        cat = catalyst_calendar(info, pd.Timestamp("2025-09-30"),
                                pd.Timestamp("2025-11-01"))
        assert cat["guide_reset_window"] is True

    def test_guide_reset_false_on_q1_print(self):
        ends = ["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30",
                "2025-12-31"]
        info = _info(_periodic(ends), fye="1231")
        # latest Dec-31 -> next ~Mar-31 (a Q1 print), not the FY reset
        cat = catalyst_calendar(info, pd.Timestamp("2025-12-31"),
                                pd.Timestamp("2026-02-20"))
        assert cat["guide_reset_window"] is False

    def test_annual_only_is_guide_reset(self):
        info = _info(_periodic(["2023-12-31", "2024-12-31"], form="20-F"))
        cat = catalyst_calendar(info, pd.Timestamp("2024-12-31"),
                                pd.Timestamp("2025-03-01"), annual_only=True)
        assert cat["guide_reset_window"] is True

    def test_asof_leak_guard_excludes_future_filings(self):
        ends = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
                "2025-03-31"]
        info = _info(_periodic(ends, lag=40))
        asof = pd.Timestamp("2024-12-01")   # before the last two filings
        cat = catalyst_calendar(info, pd.Timestamp("2024-09-30"), asof)
        # only the filings filed on/before asof count
        assert cat["n_periodic"] == 3

    def test_no_anchor_returns_nan(self):
        info = _info([])
        cat = catalyst_calendar(info, None, pd.Timestamp("2026-01-01"))
        assert np.isnan(cat["days_to_catalyst"])


# ---------------------------------------------------------------------------
# Point-in-time submissions parse (the replay lookahead-leak guard).
#
# Before this guard, `_classify` anchored the NT (3y) and 8-K (2y) lookback
# windows on wall-clock today and applied no upper bound at all, so a replay
# as of date D saw every NT/8-K/4.01/4.02 filed between D and today. Measured
# on the live universe at 2025-10-15: 144,674 filings visible that should not
# have been, flipping nt_filer on 52 names, auditor_change on 92 and
# non_reliance on 30 of ~39 — the contamination that made the base-rate table
# unusable for tuning.
# ---------------------------------------------------------------------------

def _subs(rows, former=None, cik=1, **kw):
    """Submissions JSON from (form, filingDate, accn, items) tuples.
    EDGAR returns `recent` newest-first; keep that ordering here."""
    rows = sorted(rows, key=lambda r: r[1], reverse=True)
    d = {"cik": cik, "name": "Test Co", "sic": "2000",
         "sicDescription": "Food", "tickers": ["TEST"], "exchanges": ["NYSE"],
         "fiscalYearEnd": "1231", "formerNames": former or [],
         "filings": {"recent": {
             "form": [r[0] for r in rows],
             "filingDate": [r[1] for r in rows],
             "accessionNumber": [r[2] for r in rows],
             "items": [r[3] for r in rows],
             "reportDate": [kw.get("report_dates", {}).get(r[1], "")
                            for r in rows]}}}
    return d


class TestPointInTimeSubmissions:
    ASOF = "2025-10-15"

    def _mixed(self):
        """Filings straddling ASOF, plus an 8-K old enough that a window
        anchored on *today* would miss it but one anchored on ASOF keeps it."""
        return _subs([
            ("8-K", "2024-03-01", "0000-24-001", "4.02"),   # pre-asof 4.02
            ("NT 10-Q", "2024-05-10", "0000-24-002", ""),   # pre-asof NT
            ("10-K", "2025-02-20", "0000-25-001", ""),      # pre-asof 10-K
            ("10-Q", "2025-08-05", "0000-25-002", ""),      # pre-asof 10-Q
            ("8-K", "2026-01-09", "0000-26-001", "4.01"),   # POST-asof
            ("NT 10-K", "2026-03-02", "0000-26-002", ""),   # POST-asof
            ("10-K", "2026-02-25", "0000-26-003", ""),      # POST-asof
        ])

    def test_future_filings_dropped_and_counted(self):
        info = parse_submissions(self._mixed(), asof=self.ASOF)
        assert info.n_filings_suppressed == 3
        assert all(dte <= self.ASOF for _, dte, _, _ in info.forms)
        assert info.asof_iso == self.ASOF

    def test_no_asof_keeps_todays_view(self):
        info = parse_submissions(self._mixed())
        assert info.n_filings_suppressed == 0
        assert len(info.forms) == 7
        assert info.asof_iso == ""

    def test_post_asof_8k_item_invisible(self):
        """The 4.01 filed after the as-of date must not set auditor_change."""
        info = parse_submissions(self._mixed(), asof=self.ASOF)
        assert has_8k_item(info, "4.01") is False
        assert has_8k_item(info, "4.02") is True   # the pre-asof one survives

    def test_nt_window_anchored_on_asof_not_today(self):
        info = parse_submissions(self._mixed(), asof=self.ASOF)
        assert info.nt_filings_recent == 1        # only the 2024-05-10 NT
        live = parse_submissions(self._mixed())
        assert live.nt_filings_recent >= 1        # today's view sees the 2026 one

    def test_8k_window_slides_back_with_asof(self):
        """A 2024-03-01 8-K is inside a 2y window anchored on 2025-10-15 but
        outside one anchored on today — the leak cut both ways."""
        info = parse_submissions(self._mixed(), asof=self.ASOF)
        dates = [dte for dte, _ in info.recent_8k_items]
        assert "2024-03-01" in dates
        if dt.date.today() > dt.date(2026, 3, 1):
            live = parse_submissions(self._mixed())
            assert "2024-03-01" not in [dte for dte, _ in live.recent_8k_items]

    def test_latest_accessions_are_as_of(self):
        info = parse_submissions(self._mixed(), asof=self.ASOF)
        assert info.latest_10k_accn == "0000-25-001"   # not the 2026 10-K
        assert info.latest_10q_accn == "0000-25-002"
        assert parse_submissions(self._mixed()).latest_10k_accn == "0000-26-003"

    def test_former_name_not_former_until_renamed(self):
        """A rename that happens after the as-of date leaves the old name
        current, so it is not yet a formerName and cannot flag a de-SPAC."""
        d = _subs([("10-K", "2025-02-20", "a", "")],
                  former=[{"name": "Foo Acquisition Corp",
                           "from": "2020-01-01T00:00:00.000Z",
                           "to": "2026-02-01T00:00:00.000Z"}])
        assert parse_submissions(d, asof=self.ASOF).former_names == []
        assert parse_submissions(d, asof=self.ASOF).is_despac_name is False
        assert parse_submissions(d).is_despac_name is True

    def test_former_name_with_missing_to_is_kept(self):
        d = _subs([("10-K", "2025-02-20", "a", "")],
                  former=[{"name": "Old Name Inc", "from": "2015-01-01"}])
        assert parse_submissions(d, asof=self.ASOF).former_names == ["Old Name Inc"]

    def test_timestamp_asof_accepted(self):
        a = parse_submissions(self._mixed(), asof=pd.Timestamp(self.ASOF))
        b = parse_submissions(self._mixed(), asof=self.ASOF)
        assert a.n_filings_suppressed == b.n_filings_suppressed
        assert [f[1] for f in a.forms] == [f[1] for f in b.forms]

    def test_periodic_filings_also_bounded(self):
        """catalyst_calendar filters on asof itself, but the list it reads
        must not carry post-asof rows either."""
        info = parse_submissions(self._mixed(), asof=self.ASOF)
        assert all(filed <= self.ASOF for _, filed, _ in info.periodic_filings)
