"""Archetype verification-checklist rendering tests."""

import pandas as pd

from screener.tearsheet import _archetype_checklist_block


def test_checklist_emits_only_fired_tags_with_deep_link():
    row = pd.Series({
        "cik": 887596, "pricing_masking_volume": True,
        "archetype_ran_on_temp_success": False,
        "priced_for_impossible_growth": True})
    details = {"tier2": {"latest_10q_accn": "0001-23-000456",
                         "latest_10k_accn": "0001-23-000111"}}
    text = "\n".join(_archetype_checklist_block(row, details))
    assert "Pricing masking volume" in text
    assert "Reverse-DCF: priced above own history" in text
    # 10-Q link for pmv, 10-K link for the reverse-DCF tag
    assert "sec.gov/Archives/edgar/data/887596/000123000456" in text
    assert "sec.gov/Archives/edgar/data/887596/000123000111" in text
    # a non-fired tag is absent
    assert "Ran on temporary success" not in text


def test_checklist_empty_when_nothing_fires():
    row = pd.Series({"cik": 1, "pricing_masking_volume": False,
                     "multiple_disconnect": False})
    assert _archetype_checklist_block(row, {"tier2": {}}) == []


def test_checklist_survives_missing_link():
    row = pd.Series({"cik": None, "nonop_income_propping": True})
    text = "\n".join(_archetype_checklist_block(row, {"tier2": {}}))
    assert "link unavailable" in text
