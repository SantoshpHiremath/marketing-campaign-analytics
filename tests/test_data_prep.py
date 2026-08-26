import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_prep import load_raw, clean, load_clean, EXPECTED_COLUMNS, MONTH_ORDER


RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bank_marketing_raw.csv")


def test_load_raw_returns_expected_shape():
    df = load_raw(RAW_PATH)
    assert df.shape[0] > 40000
    assert df.shape[1] == 21


def test_load_raw_has_expected_columns():
    df = load_raw(RAW_PATH)
    assert set(EXPECTED_COLUMNS) == set(df.columns)


def test_load_raw_rejects_bad_schema(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        load_raw(str(bad_csv))


def test_clean_adds_converted_column_as_binary():
    df = load_raw(RAW_PATH)
    cleaned = clean(df)
    assert set(cleaned["converted"].unique()) <= {0, 1}
    assert cleaned["converted"].sum() == (df["y"] == "yes").sum()


def test_clean_converted_matches_original_yes_no_counts():
    df = load_raw(RAW_PATH)
    cleaned = clean(df)
    assert cleaned["converted"].sum() == 4640
    assert (cleaned["converted"] == 0).sum() == 36548


def test_clean_previously_contacted_flag_matches_pdays_sentinel():
    df = load_raw(RAW_PATH)
    cleaned = clean(df)
    # pdays == 999 is the dataset's documented "never contacted before" sentinel
    assert (cleaned.loc[cleaned["pdays"] == 999, "previously_contacted"] == False).all()
    assert (cleaned.loc[cleaned["pdays"] != 999, "previously_contacted"] == True).all()


def test_clean_month_is_ordered_categorical_in_calendar_order():
    df = load_raw(RAW_PATH)
    cleaned = clean(df)
    assert list(cleaned["month"].cat.categories) == MONTH_ORDER
    assert cleaned["month"].cat.ordered


def test_clean_campaign_bucket_assigns_all_rows():
    df = load_raw(RAW_PATH)
    cleaned = clean(df)
    assert cleaned["campaign_bucket"].isnull().sum() == 0


def test_clean_campaign_bucket_boundaries():
    df = load_raw(RAW_PATH)
    cleaned = clean(df)
    assert (cleaned.loc[cleaned["campaign"] == 1, "campaign_bucket"] == "1 contact").all()
    assert (cleaned.loc[cleaned["campaign"] == 3, "campaign_bucket"] == "2-3 contacts").all()
    assert (cleaned.loc[cleaned["campaign"] == 6, "campaign_bucket"] == "4-6 contacts").all()
    assert (cleaned.loc[cleaned["campaign"] == 10, "campaign_bucket"] == "7+ contacts").all()


def test_clean_raises_on_null_target():
    df = load_raw(RAW_PATH).copy()
    df.loc[0, "y"] = None
    with pytest.raises(ValueError):
        clean(df)


def test_load_clean_end_to_end():
    cleaned = load_clean(RAW_PATH)
    assert "converted" in cleaned.columns
    assert "previously_contacted" in cleaned.columns
    assert "campaign_bucket" in cleaned.columns
    assert len(cleaned) > 40000


def test_clean_does_not_mutate_input():
    df = load_raw(RAW_PATH)
    original_cols = list(df.columns)
    clean(df)
    assert list(df.columns) == original_cols
