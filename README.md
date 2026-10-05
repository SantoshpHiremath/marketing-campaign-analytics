# marketing-campaign-analytics

A campaign-performance analysis project covering data cleaning and marketing-analytics engineering: conversion funnels, channel comparison, statistical significance testing, and segment performance, on a real, public, well-documented campaign dataset.

## What it does

- `src/data_prep.py`: schema-validated loading and data cleaning. It converts the `pdays == 999` sentinel into a proper `previously_contacted` boolean instead of treating 999 as a literal day count, builds ordered categoricals for month and day of week so trend analysis sorts chronologically rather than alphabetically, and buckets contact frequency into labeled ranges.
- `src/analysis.py`: the analysis functions: conversion rate by channel, by month, by day of week, by contact frequency, by prior campaign outcome, and by age segment; a two-proportion z-test comparing the two contact channels; and a funnel summary (attempted, connected, converted).
- `run_pipeline.py`: runs the full pipeline end to end and prints a report (see "Results" below, copied directly from an actual run).
- `tests/`: 28 tests, all passing, covering both the data-cleaning logic and the analysis functions, including a manual cross-check of the z-test statistic against an independent calculation and checks that every groupby covers all 41,188 contacts with no rows dropped.

## Data

`data/bank_marketing_raw.csv` is the real UCI "Bank Marketing" dataset (Moro, Cortez & Rita, *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems, 2014): 41,188 real outbound telemarketing campaign contacts made by a Portuguese retail bank between 2008 and 2010, promoting term-deposit subscriptions. It is outbound-calling campaign data rather than web advertising data, and a real, messy, class-imbalanced target makes it a good fit for campaign analytics. I loaded it from a GitHub mirror of the original dataset.

**Statistical rigor.** The channel comparison (`channel_ab_test` in `src/analysis.py`) is a two-proportion z-test rather than an eyeballed percentage difference: pooled proportion, standard error, z-statistic, and a two-sided p-value, cross-checked in the test suite against an independently written manual calculation (`test_channel_ab_test_matches_manual_z_test_calculation`).

**The `duration` column is excluded from any targeting-relevant segment.** The dataset's own documentation notes that call duration is only known *after* a call ends, so it cannot be used to decide who to call. It is reported once in the funnel summary purely as a call-connection indicator, and nowhere else, to avoid the data-leakage mistake of using a post-hoc variable as if it were predictive.

## Results (from an actual run of `run_pipeline.py`)

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

Three checkable findings came out of this data:

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

## Tests

- `python3 -m pytest tests/ -v`: 28/28 tests pass.
- `python3 run_pipeline.py` runs end to end against the 41,188-row dataset; the "Results" section above is copied directly from this run's actual stdout.
- Every groupby-based analysis function is tested to confirm its `contacts` column sums to the full dataset size (41,188), so no analysis silently drops rows.
- The z-test implementation is cross-checked in `test_channel_ab_test_matches_manual_z_test_calculation` against an independently written manual computation of the same statistic.

## Project structure

```
data/bank_marketing_raw.csv
src/
  data_prep.py
  analysis.py
tests/
  test_data_prep.py
  test_analysis.py
run_pipeline.py
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 run_pipeline.py
```

## Possible extensions

- Add a channel-by-segment breakdown (for example channel conversion by age band).
- Apply the same funnel and significance tooling to web or digital-advertising campaign data.
- Build a dashboard on top of the analysis functions.
