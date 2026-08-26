# marketing-campaign-analytics

A real campaign-performance analysis project, built to demonstrate
genuine data-cleaning and marketing-analytics engineering (conversion
funnels, channel comparison, statistical significance testing, segment
performance) against a real, public, well-documented campaign dataset —
built to close a "digital marketing / campaign analytics" evidence gap
identified against a Marketing Intelligence & Technology working-student
posting.

## What this is, precisely

- `data/bank_marketing_raw.csv` — the real UCI "Bank Marketing" dataset
  (Moro, Cortez & Rita, *A Data-Driven Approach to Predict the Success
  of Bank Telemarketing*, Decision Support Systems, 2014): 41,188 real
  outbound telemarketing campaign contacts made by a Portuguese retail
  bank between 2008 and 2010, promoting term-deposit subscriptions.
- `src/data_prep.py` — schema-validated loading and real data cleaning:
  converts the `pdays == 999` sentinel into a proper
  `previously_contacted` boolean instead of treating 999 as a literal
  day count, builds ordered categoricals for month/day-of-week so
  trend analysis sorts chronologically rather than alphabetically, and
  buckets contact frequency into labeled ranges.
- `src/analysis.py` — the actual analysis functions: conversion rate by
  channel, by month, by day of week, by contact frequency, by prior
  campaign outcome, by age segment; a two-proportion z-test comparing
  the two contact channels; and a funnel summary (attempted → connected
  → converted).
- `run_pipeline.py` — runs the full pipeline end to end and prints a
  real report (see "Sample output" below — copied directly from an
  actual run, not written by hand).
- `tests/` — 28 tests, all passing, covering both the data-cleaning
  logic and the analysis functions, including a manual cross-check of
  the z-test statistic against an independent calculation and checks
  that every groupby covers all 41,188 contacts with no rows dropped.

## Honest disclosure — what this is and isn't evidence of

**This is real public data, not DATEV's own data**, and it is outbound
telemarketing campaign data, not web/digital-advertising analytics
data. It was chosen deliberately: a real, genuinely messy, well-
documented, class-imbalanced campaign dataset that could actually be
downloaded in this environment, over a synthetic stand-in. Two more
directly-matching public sources were tried first and found to be
network-blocked from this environment when checked directly —
`archive.ics.uci.edu` (the dataset's original host) and `kaggle.com`
both failed to connect; a GitHub mirror of the identical dataset
(`raw.githubusercontent.com`) was reachable and used instead.

**This project does not demonstrate Adobe Experience Cloud, Databricks,
Power Automate, Power Apps, or SharePoint experience**, and makes no
claim to. Those specific tools were checked directly for reachability
in this sandbox (`community.cloud.databricks.com`,
`www.databricks.com`, and the Adobe Experience Cloud login/product
pages all failed to connect) and were confirmed genuinely not buildable
here — this is stated directly rather than worked around by claiming
unearned experience with them. What this project *does* demonstrate
directly is the underlying analytical skill those tools would be used
to apply: real data cleaning, funnel/conversion analysis, channel
comparison with a proper significance test, and segment performance
reporting — the same category of work, on real data, using Python/
pandas instead of a specific enterprise platform.

**Statistical rigor**: the channel comparison (`channel_ab_test` in
`src/analysis.py`) is a real two-proportion z-test, not just an
eyeballed percentage difference — pooled proportion, standard error,
z-statistic, and a two-sided p-value, cross-checked in the test suite
against an independently written manual calculation
(`test_channel_ab_test_matches_manual_z_test_calculation`).

**The `duration` column is deliberately excluded from any targeting-
relevant segment.** The dataset's own documentation notes that call
duration is only known *after* a call ends, so it can't be used to
decide who to call — it's reported once in the funnel summary purely
as a call-connection indicator, and nowhere else, to avoid the well-
known data-leakage mistake of using a post-hoc variable as if it were
predictive.

## Sample output (from an actual run of `run_pipeline.py`)

```
Overall conversion rate: 11.2654%

--- Conversion by contact channel ---
           contacts  conversions  conversion_rate
contact
cellular      26144         3853           0.1474
telephone     15044          787           0.0523

--- Channel A/B test (two-proportion z-test) ---
telephone: 5.2313% (n=15,044)
cellular: 14.7376% (n=26,144)
z = -29.381, p = 0.00e+00, significant at 0.05: True

--- Conversion by contact frequency this campaign ---
                 contacts  conversions  conversion_rate
campaign_bucket
1 contact           17642         2300           0.1304
2-3 contacts        15911         1785           0.1122
4-6 contacts         5229          444           0.0849
7+ contacts          2406          111           0.0461

--- Conversion by prior campaign outcome ---
             contacts  conversions  conversion_rate
poutcome
success          1373          894           0.6511
failure          4252          605           0.1423
nonexistent     35563         3141           0.0883
```

Three genuine, checkable findings came out of this real data:

1. **Cellular contacts convert at roughly 2.8x the rate of landline
   contacts** (14.7% vs. 5.2%), and the difference is statistically
   significant (p < 0.001 by a two-proportion z-test) — a real channel-
   performance signal, not noise.
2. **More contact attempts within a campaign correlate with a lower
   conversion rate**, not a higher one (13.0% at 1 contact down to 4.6%
   at 7+ contacts) — a real, testable "campaign fatigue" pattern
   (`test_conversion_by_contact_frequency_shows_diminishing_returns`
   asserts this directly against the data).
3. **Clients who converted on a previous campaign convert again at
   65.1%**, versus 8.8% for clients never previously contacted — a
   clear warm-audience effect.

## Verification performed

- `python3 -m pytest tests/ -v` — 28/28 tests pass.
- `python3 run_pipeline.py` — runs end to end against the real 41,188-
  row dataset; the "Sample output" section above is copied directly
  from this run's actual stdout.
- Every groupby-based analysis function is tested to confirm its
  `contacts` column sums to the full dataset size (41,188), so no
  analysis is silently dropping rows.
- The z-test implementation is cross-checked in
  `test_channel_ab_test_matches_manual_z_test_calculation` against an
  independently written manual computation of the same statistic.

## Running it yourself

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 run_pipeline.py
```
