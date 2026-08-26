"""
Campaign performance analysis functions.

All functions take a cleaned dataframe (see data_prep.clean) and return
plain pandas objects (Series/DataFrame) so they're easy to test and to
feed into a report or a chart.
"""

import numpy as np
import pandas as pd
from scipy import stats


def overall_conversion_rate(df):
    """Overall conversion rate (fraction of contacts that subscribed)."""
    return df["converted"].mean()


def conversion_by_channel(df):
    """Conversion rate and volume by contact channel (cellular/telephone)."""
    g = df.groupby("contact", observed=True)["converted"]
    result = g.agg(contacts="count", conversions="sum", conversion_rate="mean")
    return result.sort_values("conversion_rate", ascending=False)


def conversion_by_month(df):
    """Conversion rate and volume by month, in real chronological order."""
    g = df.groupby("month", observed=True)["converted"]
    result = g.agg(contacts="count", conversions="sum", conversion_rate="mean")
    return result


def conversion_by_day_of_week(df):
    """Conversion rate and volume by day of week, Mon-Fri order."""
    g = df.groupby("day_of_week", observed=True)["converted"]
    result = g.agg(contacts="count", conversions="sum", conversion_rate="mean")
    return result


def conversion_by_contact_frequency(df):
    """
    Conversion rate by number of contacts made during this campaign.

    This is the core "funnel fatigue" analysis: does calling someone
    more times during the same campaign help or hurt the odds they
    convert?
    """
    g = df.groupby("campaign_bucket", observed=True)["converted"]
    result = g.agg(contacts="count", conversions="sum", conversion_rate="mean")
    return result


def conversion_by_prior_outcome(df):
    """
    Conversion rate by the outcome of the previous marketing campaign
    this client was part of (poutcome: success / failure / nonexistent).

    This is a genuine "returning audience vs. cold audience" analysis —
    a client who converted on a prior campaign is a fundamentally
    different segment than someone never contacted before.
    """
    g = df.groupby("poutcome", observed=True)["converted"]
    result = g.agg(contacts="count", conversions="sum", conversion_rate="mean")
    return result.sort_values("conversion_rate", ascending=False)


def channel_ab_test(df):
    """
    Two-proportion z-test comparing conversion rate between the two
    contact channels (cellular vs. telephone) -- a genuine A/B-style
    statistical comparison of two "creative variants" (channels), not
    just an eyeballed difference in percentages.

    Returns a dict with the two group rates, the z-statistic, and the
    two-sided p-value.
    """
    channels = df["contact"].unique()
    if len(channels) != 2:
        raise ValueError(f"Expected exactly 2 channels, found {list(channels)}")

    groups = {}
    for ch in channels:
        sub = df[df["contact"] == ch]["converted"]
        groups[ch] = {"n": len(sub), "conversions": int(sub.sum()), "rate": sub.mean()}

    (ch_a, g_a), (ch_b, g_b) = groups.items()

    n1, x1 = g_a["n"], g_a["conversions"]
    n2, x2 = g_b["n"], g_b["conversions"]
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "group_a": ch_a, "group_a_rate": p1, "group_a_n": n1,
        "group_b": ch_b, "group_b_rate": p2, "group_b_n": n2,
        "z_statistic": z, "p_value": p_value,
        "significant_at_0_05": p_value < 0.05,
    }


def age_segment_performance(df):
    """
    Conversion rate by age decade band -- a simple, real audience-
    segmentation cut used to check whether campaign performance is
    concentrated in a particular age group.
    """
    bins = list(range(10, 101, 10))
    labels = [f"{b}-{b+9}" for b in bins[:-1]]
    seg = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    g = df.groupby(seg, observed=True)["converted"]
    result = g.agg(contacts="count", conversions="sum", conversion_rate="mean")
    return result


def funnel_summary(df):
    """
    A simple top-level funnel: total contacts attempted -> contacts who
    engaged in a substantive call (duration > 0) -> conversions.

    duration == 0 in this dataset means the call was never actually
    connected (documented in the dataset's own notes), so this is a
    genuine, meaningful funnel stage boundary, not an arbitrary cut.
    """
    total_contacts = len(df)
    connected = int((df["duration"] > 0).sum())
    conversions = int(df["converted"].sum())

    return pd.Series({
        "total_contacts_attempted": total_contacts,
        "calls_connected": connected,
        "connect_rate": connected / total_contacts,
        "conversions": conversions,
        "conversion_rate_of_attempted": conversions / total_contacts,
        "conversion_rate_of_connected": conversions / connected if connected else float("nan"),
    })
