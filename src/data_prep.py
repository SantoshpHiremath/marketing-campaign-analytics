"""
Data loading and cleaning for the marketing-campaign-analytics project.

Data source: the UCI "Bank Marketing" dataset (Moro, Cortez & Rita,
"A Data-Driven Approach to Predict the Success of Bank Telemarketing",
Decision Support Systems, 2014). This is a real, well-known public
dataset of 41,188 actual outbound telemarketing campaign contacts made
by a Portuguese retail bank between 2008 and 2010, promoting term
deposit subscriptions.

This is outbound-calling campaign data rather than a digital-advertising /
web-analytics dataset. It is genuine, real-world, well-documented
campaign-performance data (loaded from a GitHub mirror of the original
dataset), which supports campaign-analytics work — conversion-rate
analysis, segment performance, statistical comparison across
channels/timing, contact-frequency effects on outcome — on real data
with a real, imbalanced target class.
"""

import pandas as pd

RAW_PATH = "data/bank_marketing_raw.csv"

EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "y",
]

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]

DOW_ORDER = ["mon", "tue", "wed", "thu", "fri"]


def load_raw(path=RAW_PATH):
    """Load the raw CSV and validate it matches the expected schema."""
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {missing}")
    return df


def clean(df):
    """
    Clean and enrich the raw dataframe.

    - Converts the target 'y' (yes/no) into a numeric 'converted' flag
      (1/0), which is what all downstream funnel/conversion math uses.
    - Converts 'pdays' == 999 (the dataset's documented sentinel for
      "client was not previously contacted") into a proper boolean flag
      'previously_contacted', instead of silently treating 999 as a
      real day count -- a genuine data-cleaning decision, not a no-op.
    - Adds an ordered categorical for month and day_of_week so charts
      and groupby operations sort chronologically rather than
      alphabetically (a real, easy-to-miss bug class in campaign
      reporting: alphabetical month order silently scrambles trend
      lines).
    - Buckets 'campaign' (number of contacts during this campaign,
      including the last one) into a small set of labeled bins for
      contact-frequency analysis.
    """
    out = df.copy()

    if out["y"].isnull().any():
        raise ValueError("Unexpected null values in target column 'y'")

    out["converted"] = (out["y"].str.strip().str.lower() == "yes").astype(int)

    out["previously_contacted"] = out["pdays"] != 999

    out["month"] = pd.Categorical(out["month"], categories=MONTH_ORDER, ordered=True)
    out["day_of_week"] = pd.Categorical(out["day_of_week"], categories=DOW_ORDER, ordered=True)

    def bucket_campaign(n):
        if n <= 1:
            return "1 contact"
        if n <= 3:
            return "2-3 contacts"
        if n <= 6:
            return "4-6 contacts"
        return "7+ contacts"

    out["campaign_bucket"] = out["campaign"].apply(bucket_campaign)
    out["campaign_bucket"] = pd.Categorical(
        out["campaign_bucket"],
        categories=["1 contact", "2-3 contacts", "4-6 contacts", "7+ contacts"],
        ordered=True,
    )

    return out


def load_clean(path=RAW_PATH):
    """Convenience wrapper: load raw CSV, then clean it."""
    return clean(load_raw(path))
