"""Quarterlyizer unit tests (audit 1.1 behaviors)."""

import numpy as np
import pandas as pd
import pytest

from screener import config
from screener.ladders import (derive_gross_profit, derive_total_debt,
                              resolve_flow)
from screener.quarterlyize import (QuarterlySeries, build_quarterly,
                                   dedupe_facts, to_ttm, yoy_delta)
from tests.conftest import (facts_json, make_duration_obs, make_instant_obs,
                            qseries)

ALLOWED = config.ALLOWED_FORMS


def _quarterly_from_obs(obs):
    return build_quarterly(dedupe_facts(obs, ALLOWED))


class TestQ4Derivation:
    def test_q4_derived_from_fy_minus_q123(self):
        q = qseries([10, 20, 30, 40, 11, 21, 31, 41])
        obs = make_duration_obs(q)          # 10-K files FY only — no Q4 fact
        got = _quarterly_from_obs(obs)
        assert len(got.values) == 8
        # Q4s recovered exactly
        assert got.values.iloc[3] == pytest.approx(40)
        assert got.values.iloc[7] == pytest.approx(41)
        # and flagged as derived Q4
        assert bool(got.derived_q4.iloc[3])
        assert bool(got.derived_q4.iloc[7])
        # as-filed quarters not flagged
        assert not got.derived.iloc[0]

    def test_ytd_differencing_cash_flow_style(self):
        q = qseries([5, 7, -3, 9, 6, 8, -2, 10])
        obs = make_duration_obs(q, ytd_style=True)  # YTD-only 10-Qs
        got = _quarterly_from_obs(obs)
        assert len(got.values) == 8
        np.testing.assert_allclose(got.values.values,
                                   [5, 7, -3, 9, 6, 8, -2, 10], atol=1e-9)
        # Q2/Q3/Q4 come from differencing -> derived
        assert bool(got.derived.iloc[1])
        assert bool(got.derived.iloc[2])
        assert bool(got.derived_q4.iloc[3])

    def test_fy_sum_validation_nulls_derived_quarters(self):
        """A derived Q4 self-reconciles with its own FY by construction; the
        validation exists for MIXED sources: as-filed discrete quarters that
        disagree with the YTD chain the Q4 was differenced from."""
        ends = qseries([0, 0, 0, 0]).index
        fy_start = ends[0] - pd.Timedelta(days=90)
        M = 1e6  # realistic dollar magnitudes (the tolerance floor is $1M)
        obs = []
        # discrete as-filed Q1..Q3 = 10M, 20M, 30M (sum 60M)
        for j, val in enumerate([10 * M, 20 * M, 30 * M]):
            obs.append({"start": (ends[j] - pd.Timedelta(days=89)).date().isoformat(),
                        "end": ends[j].date().isoformat(), "val": val,
                        "form": "10-Q", "filed": "2025-11-01", "accn": f"q{j}"})
        # 9M YTD mark says 55M (inconsistent with 60M) and FY = 100M
        obs.append({"start": fy_start.date().isoformat(),
                    "end": ends[2].date().isoformat(), "val": 55 * M,
                    "form": "10-Q", "filed": "2025-11-02", "accn": "ytd9m"})
        obs.append({"start": fy_start.date().isoformat(),
                    "end": ends[3].date().isoformat(), "val": 100 * M,
                    "form": "10-K", "filed": "2026-05-01", "accn": "fyk"})
        got = _quarterly_from_obs(obs)
        # Q4 was derived as FY-9M = 45M; 10+20+30+45 = 105M != 100M -> the
        # derived quarter must be NULLED, never interpolated; as-filed stay
        assert len(got.values) == 3
        assert not any(got.derived_q4)
        np.testing.assert_allclose(got.values.values, [10 * M, 20 * M, 30 * M])

    def test_fy_fp_fields_ignored(self):
        # conftest deliberately writes nonsense fy/fp values; if the parser
        # consulted them the periods would misalign
        q = qseries([10, 20, 30, 40])
        got = _quarterly_from_obs(make_duration_obs(q))
        assert list(got.values.values) == [10, 20, 30, 40]


class TestRestatementDedup:
    def test_latest_filed_wins_and_restated_flag(self):
        q = qseries([10, 20, 30, 40])
        obs = make_duration_obs(q)
        # re-file Q1 with a materially different value, later filed date
        orig = obs[0]
        obs.append({**orig, "val": orig["val"] * 1.5,
                    "filed": "2026-05-01", "accn": "amend1",
                    "form": "10-Q/A"})
        got = _quarterly_from_obs(obs)
        assert got.values.iloc[0] == pytest.approx(15.0)
        assert got.restated_any

    def test_small_diff_not_restated(self):
        q = qseries([100, 200, 300, 400])
        obs = make_duration_obs(q)
        orig = obs[0]
        obs.append({**orig, "val": orig["val"] * 1.001,
                    "filed": "2026-05-01", "accn": "amend2",
                    "form": "10-Q/A"})
        got = _quarterly_from_obs(obs)
        assert not got.restated_any

    def test_disallowed_forms_excluded(self):
        q = qseries([10, 20, 30, 40])
        obs = make_duration_obs(q)
        obs.append({"start": "2030-01-01", "end": "2030-03-31", "val": 999.0,
                    "form": "8-K", "filed": "2030-04-15", "accn": "x"})
        got = _quarterly_from_obs(obs)
        assert 999.0 not in set(got.values.values)


class TestPointInTimeReplay:
    def test_filter_facts_asof_drops_later_filings(self):
        """Replay harness (user 2026-07-08): only facts FILED by the cutoff
        are visible; a restatement filed after the cutoff reverts to the
        originally-filed value — true point-in-time behavior."""
        from screener.edgar import filter_facts_asof
        q = qseries([10, 20, 30, 40])
        obs = make_duration_obs(q)
        orig = obs[0]
        obs.append({**orig, "val": orig["val"] * 2.0,
                    "filed": "2026-06-01", "accn": "amend-late",
                    "form": "10-Q/A"})
        fj = facts_json({"Revenues": obs})
        # full view: the amendment wins
        full = _quarterly_from_obs(
            fj["facts"]["us-gaap"]["Revenues"]["units"]["USD"])
        assert full.values.iloc[0] == pytest.approx(20.0)
        # as-of BEFORE the amendment: originally-filed value visible
        pit = filter_facts_asof(fj, "2026-05-01")
        pit_q = _quarterly_from_obs(
            pit["facts"]["us-gaap"]["Revenues"]["units"]["USD"])
        assert pit_q.values.iloc[0] == pytest.approx(10.0)
        assert not pit_q.restated_any

    def test_filter_facts_asof_empty_before_first_filing(self):
        from screener.edgar import filter_facts_asof
        q = qseries([10, 20, 30, 40])
        fj = facts_json({"Revenues": make_duration_obs(q)})
        pit = filter_facts_asof(fj, "2000-01-01")
        assert not pit["facts"]  # nothing was filed yet


class TestDerivedSeries:
    def test_ttm_requires_four_consecutive(self):
        q = qseries([10, 20, 30, 40, 11, 21, 31, 41])
        t = to_ttm(q)
        assert t.iloc[-1] == pytest.approx(11 + 21 + 31 + 41)
        # drop an interior quarter -> TTMs spanning the hole disappear
        q2 = q.drop(q.index[5])
        t2 = to_ttm(q2)
        assert q.index[6] not in t2.index  # window over the hole is invalid

    def test_yoy_delta_fiscal_aligned(self):
        q = qseries([10, 20, 30, 40, 15, 26, 37, 48])
        y = yoy_delta(q)
        np.testing.assert_allclose(y.values, [5, 6, 7, 8], atol=1e-9)


class TestLadders:
    def test_ladder_priority_and_stitch(self):
        """Tag A covers recent periods, tag B older ones -> stitched series."""
        recent = qseries([5, 6, 7, 8], end="2026-03-31")
        old = qseries([1, 2, 3, 4], end="2022-03-31")
        fj = facts_json({
            "RevenueFromContractWithCustomerExcludingAssessedTax":
                make_duration_obs(recent),
            "Revenues": make_duration_obs(old),
        })
        got = resolve_flow(fj, config.TAG_LADDERS["revenue"], ALLOWED)
        assert len(got.values) == 8
        assert got.values.iloc[0] == pytest.approx(1)
        assert got.values.iloc[-1] == pytest.approx(8)

    def test_gross_profit_derived_from_revenue_minus_cogs(self):
        rev = qseries([100, 110, 120, 130])
        cogs = qseries([60, 66, 72, 78])
        fj = facts_json({"Revenues": make_duration_obs(rev),
                         "CostOfRevenue": make_duration_obs(cogs)})
        rev_qs = resolve_flow(fj, config.TAG_LADDERS["revenue"], ALLOWED)
        gp = derive_gross_profit(fj, rev_qs, ALLOWED)
        np.testing.assert_allclose(gp.values.values, [40, 44, 48, 52],
                                   atol=1e-9)
        assert all(gp.derived)

    def test_total_debt_no_double_count(self):
        idx = qseries([1, 1, 1, 1]).index
        ltd_nc = pd.Series([800.0] * 4, index=idx)
        ltd_cur = pd.Series([100.0] * 4, index=idx)
        debt_cur = pd.Series([150.0] * 4, index=idx)  # includes the 100 LTD
        fj = facts_json({
            "LongTermDebtNoncurrent": make_instant_obs(ltd_nc),
            "LongTermDebtCurrent": make_instant_obs(ltd_cur),
            "DebtCurrent": make_instant_obs(debt_cur),
        })
        total = derive_total_debt(fj, ALLOWED)
        # granular tags exist -> DebtCurrent must NOT be added on top
        assert total.iloc[-1] == pytest.approx(900.0)

    def test_total_debt_debtcurrent_fallback(self):
        idx = qseries([1, 1, 1, 1]).index
        fj = facts_json({
            "LongTermDebtNoncurrent": make_instant_obs(
                pd.Series([800.0] * 4, index=idx)),
            "DebtCurrent": make_instant_obs(pd.Series([150.0] * 4, index=idx)),
        })
        total = derive_total_debt(fj, ALLOWED)
        assert total.iloc[-1] == pytest.approx(950.0)

    def test_total_debt_combined_lt_no_double_count(self):
        """LongTermDebt (combined tag) already INCLUDES current maturities:
        the LTD-current piece must not be added on top of it."""
        idx = qseries([1, 1, 1, 1]).index
        fj = facts_json({
            "LongTermDebt": make_instant_obs(pd.Series([900.0] * 4, index=idx)),
            "LongTermDebtCurrent": make_instant_obs(
                pd.Series([100.0] * 4, index=idx)),
            "ShortTermBorrowings": make_instant_obs(
                pd.Series([50.0] * 4, index=idx)),
        })
        total = derive_total_debt(fj, ALLOWED)
        # combined 900 (incl. the 100 current) + true ST borrowings 50
        assert total.iloc[-1] == pytest.approx(950.0)

    def test_gross_profit_cos_only_in_cogsonly_branch(self):
        """CostOfServices adds ONLY when COGS came from CostOfGoodsSold —
        CostOfRevenue already includes services (double-count guard)."""
        rev = qseries([100, 110, 120, 130])
        cor = qseries([60, 66, 72, 78])          # CostOfRevenue (all-in)
        cos = qseries([10, 11, 12, 13])          # CostOfServices
        fj = facts_json({"Revenues": make_duration_obs(rev),
                         "CostOfRevenue": make_duration_obs(cor),
                         "CostOfServices": make_duration_obs(cos)})
        rev_qs = resolve_flow(fj, config.TAG_LADDERS["revenue"], ALLOWED)
        gp = derive_gross_profit(fj, rev_qs, ALLOWED)
        np.testing.assert_allclose(gp.values.values, [40, 44, 48, 52],
                                   atol=1e-9)
        # and WITH the CostOfGoodsSold branch, CoS does add
        fj2 = facts_json({"Revenues": make_duration_obs(rev),
                          "CostOfGoodsSold": make_duration_obs(cor),
                          "CostOfServices": make_duration_obs(cos)})
        rev_qs2 = resolve_flow(fj2, config.TAG_LADDERS["revenue"], ALLOWED)
        gp2 = derive_gross_profit(fj2, rev_qs2, ALLOWED)
        np.testing.assert_allclose(gp2.values.values, [30, 33, 36, 39],
                                   atol=1e-9)
