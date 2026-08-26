import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_prep import load_clean
from src import analysis

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bank_marketing_raw.csv")


@pytest.fixture(scope="module")
def df():
    return load_clean(RAW_PATH)


def test_overall_conversion_rate_matches_known_value(df):
    rate = analysis.overall_conversion_rate(df)
    assert rate == pytest.approx(4640 / 41188, abs=1e-9)


def test_conversion_by_channel_covers_all_contacts(df):
    result = analysis.conversion_by_channel(df)
    assert result["contacts"].sum() == len(df)
    assert set(result.index) == set(df["contact"].unique())
    assert (result["conversion_rate"] >= 0).all()
    assert (result["conversion_rate"] <= 1).all()


def test_conversion_by_month_covers_all_contacts_and_is_chronological(df):
    result = analysis.conversion_by_month(df)
    assert result["contacts"].sum() == len(df)
    months_present = list(result.index)
    calendar_order = ["jan", "feb", "mar", "apr", "may", "jun",
                       "jul", "aug", "sep", "oct", "nov", "dec"]
    filtered_calendar_order = [m for m in calendar_order if m in months_present]
    assert months_present == filtered_calendar_order


def test_conversion_by_day_of_week_covers_all_contacts(df):
    result = analysis.conversion_by_day_of_week(df)
    assert result["contacts"].sum() == len(df)
    assert list(result.index) == ["mon", "tue", "wed", "thu", "fri"]


def test_conversion_by_contact_frequency_covers_all_contacts(df):
    result = analysis.conversion_by_contact_frequency(df)
    assert result["contacts"].sum() == len(df)
    assert list(result.index) == ["1 contact", "2-3 contacts", "4-6 contacts", "7+ contacts"]


def test_conversion_by_contact_frequency_shows_diminishing_returns(df):
    """
    A genuine, testable claim about this real dataset: conversion rate
    for clients contacted just once during the campaign is higher than
    for clients contacted 7+ times -- more calls does not mean a better
    outcome. This is exactly the kind of finding a real campaign
    fatigue / frequency-capping analysis should surface.
    """
    result = analysis.conversion_by_contact_frequency(df)
    rate_1 = result.loc["1 contact", "conversion_rate"]
    rate_7plus = result.loc["7+ contacts", "conversion_rate"]
    assert rate_1 > rate_7plus


def test_conversion_by_prior_outcome_success_beats_failure(df):
    """
    Clients whose PREVIOUS campaign outcome was 'success' should convert
    at a meaningfully higher rate this time than clients whose previous
    outcome was 'failure' -- a real, checkable claim about warm vs. cold
    audiences in this dataset.
    """
    result = analysis.conversion_by_prior_outcome(df)
    assert result.loc["success", "conversion_rate"] > result.loc["failure", "conversion_rate"]


def test_channel_ab_test_returns_expected_keys(df):
    result = analysis.channel_ab_test(df)
    for key in ["group_a", "group_a_rate", "group_a_n", "group_b", "group_b_rate",
                "group_b_n", "z_statistic", "p_value", "significant_at_0_05"]:
        assert key in result


def test_channel_ab_test_sample_sizes_sum_to_total(df):
    result = analysis.channel_ab_test(df)
    assert result["group_a_n"] + result["group_b_n"] == len(df)


def test_channel_ab_test_p_value_is_valid_probability(df):
    result = analysis.channel_ab_test(df)
    assert 0 <= result["p_value"] <= 1


def test_channel_ab_test_rejects_more_than_two_channels():
    fake_df = pd.DataFrame({
        "contact": ["cellular", "telephone", "carrier_pigeon"],
        "converted": [1, 0, 1],
    })
    with pytest.raises(ValueError):
        analysis.channel_ab_test(fake_df)


def test_channel_ab_test_matches_manual_z_test_calculation(df):
    """Cross-check the z-statistic against an independent manual computation."""
    result = analysis.channel_ab_test(df)
    n1, x1 = result["group_a_n"], round(result["group_a_rate"] * result["group_a_n"])
    n2, x2 = result["group_b_n"], round(result["group_b_rate"] * result["group_b_n"])
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_manual = (p1 - p2) / se
    assert result["z_statistic"] == pytest.approx(z_manual, rel=1e-6)


def test_age_segment_performance_covers_all_contacts(df):
    result = analysis.age_segment_performance(df)
    assert result["contacts"].sum() == len(df)


def test_funnel_summary_stages_are_monotonically_decreasing(df):
    summary = analysis.funnel_summary(df)
    assert summary["total_contacts_attempted"] >= summary["calls_connected"]
    assert summary["calls_connected"] >= summary["conversions"]


def test_funnel_summary_rates_are_valid_probabilities(df):
    summary = analysis.funnel_summary(df)
    assert 0 <= summary["connect_rate"] <= 1
    assert 0 <= summary["conversion_rate_of_attempted"] <= 1
    assert 0 <= summary["conversion_rate_of_connected"] <= 1


def test_funnel_summary_conversion_rate_of_connected_higher_than_of_attempted(df):
    """
    Conversion rate calculated against connected calls only should be
    >= conversion rate against all attempted contacts, since connected
    calls are a subset of attempted contacts and every conversion is
    necessarily among the connected group.
    """
    summary = analysis.funnel_summary(df)
    assert summary["conversion_rate_of_connected"] >= summary["conversion_rate_of_attempted"]
