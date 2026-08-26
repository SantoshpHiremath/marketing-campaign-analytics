"""
End-to-end demo: load the real UCI Bank Marketing dataset, clean it,
run every analysis function, and print a genuine campaign-performance
report to the console.
"""

import pandas as pd

from src.data_prep import load_clean
from src import analysis

pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def main():
    df = load_clean("data/bank_marketing_raw.csv")

    print("=" * 70)
    print("CAMPAIGN PERFORMANCE REPORT")
    print(f"Source: UCI Bank Marketing dataset — {len(df):,} real campaign contacts")
    print("=" * 70)

    print(f"\nOverall conversion rate: {analysis.overall_conversion_rate(df):.4%}")

    print("\n--- Conversion by contact channel ---")
    print(analysis.conversion_by_channel(df))

    print("\n--- Channel A/B test (two-proportion z-test) ---")
    ab = analysis.channel_ab_test(df)
    print(f"{ab['group_a']}: {ab['group_a_rate']:.4%} (n={ab['group_a_n']:,})")
    print(f"{ab['group_b']}: {ab['group_b_rate']:.4%} (n={ab['group_b_n']:,})")
    print(f"z = {ab['z_statistic']:.3f}, p = {ab['p_value']:.2e}, "
          f"significant at 0.05: {ab['significant_at_0_05']}")

    print("\n--- Conversion by month (chronological order) ---")
    print(analysis.conversion_by_month(df))

    print("\n--- Conversion by day of week ---")
    print(analysis.conversion_by_day_of_week(df))

    print("\n--- Conversion by contact frequency this campaign ---")
    print(analysis.conversion_by_contact_frequency(df))

    print("\n--- Conversion by prior campaign outcome ---")
    print(analysis.conversion_by_prior_outcome(df))

    print("\n--- Conversion by age segment ---")
    print(analysis.age_segment_performance(df))

    print("\n--- Funnel summary ---")
    print(analysis.funnel_summary(df))

    print("\n" + "=" * 70)
    print("Note: 'duration' (call length) is known post-hoc — it isn't")
    print("available before a call is placed, so it is reported here for")
    print("funnel description only and deliberately excluded from any")
    print("segment used to inform who to target before the fact.")
    print("=" * 70)


if __name__ == "__main__":
    main()
