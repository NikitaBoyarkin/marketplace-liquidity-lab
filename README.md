<!-- ascii-art: pyfiglet + ascii-bin (CC0) -->
```text
._ _  _.._|  __|_._ | _. _ ___|o _.   o _|o_|___| _.|_
| | |(_|| |<(/_|_|_)|(_|(_(/_ ||(_||_||(_|| |_\/|(_||_)
                 |                |           /

╭─────────────────────────────────────────────────────────────────────────╮
│ A synthetic classifieds marketplace with a planted answer key. Every... │
╰─────────────────────────────────────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

# marketplace-liquidity-lab

> **The platform never sees the sale.** Its liquidity metric is a seller's self-report
> (`removal_reason == "sold"`), and this repo measures how far that self-report is from the
> latent truth — per category. A **second, independent leg** asks whether paid visibility creates
> contact demand or relocates it. The two legs answer different questions and share only the log:
> the visibility contrast does not consume the sell-through correction.

A synthetic classifieds marketplace with a planted answer key. Every number below comes from a
command in this repo or from the planted key, and the suite scores the estimators against the key.

**Role:** product analyst. **The decision this case supports:** roll back paid visibility in its
current form, and publish sell-through per category with a correction — never as one platform
number. **The hard part** was not computing either number but showing where each one is wrong: the
metric the platform ships under-reports sales by 6.6 pp, and the paid-visibility readout that looked
like a +113% win is selection.

**Public link:** живого демо нет · [кейс](https://nikitaboyarkin.github.io/projects/marketplace-liquidity/) · [исходники](https://github.com/NikitaBoyarkin/marketplace-liquidity-lab)

## Situation

A two-sided classifieds marketplace (Avito-like) where the transaction closes off-platform. The
platform logs searches, impressions, views, contacts and removals; it never logs `true_sale`. Its
sell-through metric — the share of a publish cohort whose removal reason is `sold` — is
seller-reported and wrong in both directions: sellers mark "sold" on listings that merely lapsed,
and let genuine sales lapse unmarked.

The log holds 6,000 listings and 40,000 search sessions over 90 days, in six regions and five
categories. The analysis cohort is the 1,956 listings published in the first 30 days, each with a
full 30-day observation window. The schema is in `docs/data_dictionary.md`.

**Data.** Synthetic, with a planted answer key: `liquidity/generate.py` (seed 42)
writes `data/raw/*.parquet`; the latent truth lives in `_truth/ground_truth.json`
and is read by tests only (`tests/test_no_truth_leak.py`). `data/raw/` is
gitignored — rebuild with `uv run python -m liquidity.generate`. Not Avito data.

## Task

Two questions, fixed in `docs/preregistration.md` before any analysis ran:

1. **Q1 — can liquidity be measured from what the platform sees?** Audit the visible proxy against
   the latent truth, then build a correction that uses observable signals only.
2. **Q2 — does paid visibility create contact demand, or relocate it?** Contrast the naive
   pre/post read with a randomised holdout, and report guardrails that a primary-only readout
   would hide.

## Actions

- Generated a deterministic event log (`seed 42`) with latent sale state. The answer key is written
  outside `data/`, and analysis modules are statically forbidden to read it (AST scan in
  `tests/test_no_truth_leak.py`).
- Built cohort metrics from observable tables only: cohort fixed by publish time, contact defined
  as a distinct `(buyer, listing)` pair, and every platform rollup returned next to its category
  split (`liquidity/metrics.py`). The SQL layer is cross-checked against pandas to 1e-9
  (`tests/test_metrics.py::test_sql_matches_pandas_on_cohort_contact_share`).
- Estimated latent sell-through with a two-component sold/unsold mixture fitted by EM on
  observable signals — a short reply-to-removal separator, the reply lag, and the contact count
  (`liquidity/bias.py`).
- Ran the monetisation leg twice: naive pre/post on boosted listings, and contacts per session in
  treated sessions versus a 10% randomised holdout, plus unpaid-contact and reply-rate guardrails
  (`liquidity/monetization.py`).
- Parsed the decision rule out of the preregistration document at runtime, so the recommendation
  cannot drift from what was declared (`read_decision_rules`).

## Result

### 1. The platform's own number is 6.6 pp low, and the aggregate is worse than its categories (H1a falsified)

From `uv run python -m liquidity.scoring`; the `true` columns are the planted truth, read by that
scorer and by the tests — never by the analysis modules:

| Category | True | Naive | Naive error | Corrected | Corrected error |
|---|---:|---:|---:|---:|---:|
| goods | 0.4241 | 0.4219 | −0.0021 | 0.4262 | +0.0021 |
| services | 0.2735 | 0.1609 | −0.1126 | 0.2895 | +0.0161 |
| realty | 0.1310 | 0.1310 | 0.0000 | 0.1414 | +0.0103 |
| auto | 0.3093 | 0.1827 | −0.1266 | 0.3253 | +0.0160 |
| jobs | 0.3385 | 0.3077 | −0.0308 | 0.3538 | +0.0154 |
| **platform** | **0.3070** | **0.2413** | **−0.0657** | **0.3185** | **+0.0115** |

Errors are computed from unrounded estimates, so hand-subtracting the displayed columns can differ
in the fourth decimal.

**H1a is FALSIFIED.** The preregistration predicted the platform-level aggregate would be closer to
truth than the average category (specifically `|platform error| < mean(|category error|) / 2`). It
is the opposite: the platform error is 0.0657 while the mean absolute category error is 0.0544 —
the platform-level aggregate is further from truth than the average category, not closer.

**H1b holds — under a separator that the generator also uses, so read it as a specification check,
not as independent validation.** The correction lands within ±0.03 on all five categories, worst
0.0161 (`tests/test_bias_recovery.py`). The sensitivity run is what qualifies that result: drop the
short reply-to-removal separator — the feature the preregistration actually specified — and the
correction misses on all five categories (worst +0.121); push the cutoff to 2 days and it misses
again; it recovers truth only for cutoffs of roughly 4–6 days, which is where the generator places
its own sale rule (`SOLD_REPLY_LAG_MAX = 4.5`, `liquidity/generate.py:290`). So the estimator
demonstrates that a mixture recovers truth *when the separator is correctly specified* — it does not
show that an analyst without the DGP would have found that separator.
`tests/test_separator_sensitivity.py` binds both halves of this claim.

**The `realty` zero is a near-cancellation that this sample resolved exactly.** Its naive estimate
equals the latent truth to six decimals (0.131034 = 38 marked-sold listings out of a 290-listing
cohort). Under the planted parameters the two error directions almost cancel: expected marked share
is `0.70 × 0.14 + 0.10 × 0.86 × 0.45 = 0.1367` against a true 0.14 — about one listing apart in this
cohort (`liquidity/generate.py:43-44`, removal gating at `:63`). The realised count landed on 38
versus 38. A zero error that is one listing wide is not evidence the proxy is sound: the
platform-level error is −6.6 pp regardless.

### 2. Paid visibility: the naive read says +113%, the randomised holdout says −2.5% and cannot prove it

Naive pre/post on the 738 boosted listings: contacts per listing rise from 0.4702 before the
purchase to 1.0027 after — a lift ratio of 2.13. This is selection: boosts are bought
disproportionately by weak organic listings
(`tests/test_monetization.py::test_naive_prepost_is_positive_because_of_selection`).

The causal contrast, contacts per search session, treated versus randomised holdout
(`uv run python -m liquidity.monetization`):

| Metric | Value |
|---|---:|
| Contacts/session, treated | 0.4581 |
| Contacts/session, holdout | 0.4697 |
| Incrementality | −0.0116 (−2.47%) |
| Welch t | −1.02 |
| Welch p | 0.308 |
| 95% bootstrap CI | (−0.0346, +0.0084) |
| Standard error | 0.0114 |
| **MDE at 80% power** | **0.0319 (6.8% of the holdout rate)** |
| Sample ratio: observed / planned holdout | 10.02% / 10% (χ² = 0.023, p = 0.88) |

The incrementality is negative, but the Welch p-value is 0.308 and the interval includes zero. The
MDE is the sentence that matters: **the observed 2.47% effect is less than half of what this design
can detect (6.8%)**, so "negative but not significant" is a property of the sample size, not a
finding about the product. The honest read is "no proven gain in contacts per session, with a
negative point estimate consistent with displacement" — not "paid visibility destroys contacts", and
not "no effect". The sample-ratio check is reported for the same reason: a silent drift in the 10%
holdout would widen every interval above without failing anything (`sample_ratio_check`).

### 3. The guardrail is dirty — and 42% of the drop is a slot count

Two contrasts, kept apart because they support different claims. Only the holdout ones can carry a
causal reading; the paid-vs-unpaid one compares different listings and is labelled as such.

| Contrast (treated vs randomised holdout) | Treated | Holdout | Change | 95% CI |
|---|---:|---:|---:|---|
| Organic contacts per session | 0.4184 | 0.4697 | −10.9% | (−7.3%, −2.8%) |
| Organic impressions per session | 11.496 | 12.071 | −4.8% | (−0.685, −0.464) |
| Contacts per organic impression | 0.03639 | 0.03891 | −6.5% | (−0.0042, −0.0006) |
| Reply rate on **organic** contacts | 0.6431 | 0.6480 | −0.5 pp | (−2.7, +1.9) pp |

| Descriptive, different listings | Paid | Unpaid | Change |
|---|---:|---:|---:|
| Seller reply rate | 0.5662 | 0.6436 | −7.7 pp |

**The reply-rate guardrail does not survive as causal evidence.** On the same organic listings,
suppressing promoted slots moves the reply rate by −0.5 pp with zero inside the interval. The −7.7 pp
that a paid-vs-unpaid readout reports is listing selection: a boosted listing is a weak listing by
construction, and weak listings get worse replies.

**The displacement is real, but a promoted slot also displaces a slot.** Organic contacts per session
fall 10.9%, which decomposes into 4.8% fewer organic impressions per session (a promoted slot
occupies one) and 6.5% lower conversion per organic impression. Slot arithmetic accounts for ~42% of
the drop; the remaining ~58% is a conversion effect and is the part a product decision should be
about. The position mix shows where it lands — organic impressions in positions 1–3 fall **21.6%**
per session while positions 4–20 move by 1.5–2.3%, so the top of the SERP is what the promoted slots
eat (`liquidity/monetization.py::guardrails`).

Under the preregistered rule — parsed from `docs/preregistration.md`, not restated in code — the
decision keys on organic contacts falling, and they do fall, so the rule still binds: **roll back,
whatever H2b says**; and because H2b is negative, **price paid visibility as a tax on organic reach
and cap slots per SERP** (`liquidity/monetization.py::decision`,
`tests/test_monetization.py::test_decision_rolls_back_on_a_dirty_guardrail`). What changed is the
evidence behind the verdict, not the verdict: the cap argument now rests on the conversion slope and
the top-of-SERP mix, not on the reply-rate delta.

The recommendation, its rollback thresholds and what is deliberately not measured are collected in
`docs/decision-memo.md` — including the fact that **no slot limit is measured here**: the experiment
compares "all promoted slots suppressed" with "shown", never 1, 2 or 3 slots.

The suite pins the estimators rather than every cell of the tables: the corrected errors against
their tolerance, the treated/holdout rates, the guardrail shares and the separator sensitivity above.
Table values not asserted by a test are reproducible from the commands in Quick start —
`uv run pytest` → 115 passed.

### 4. Two invariants that used to be decoration

Both are mechanisms now, and both are worth reading before trusting the numbers above.

- **Every quality gate can fail.** `liquidity/quality.py` runs eight raw-layer gates, and
  `tests/test_quality_gates.py` injects one targeted defect per gate and requires it to go red. The
  two gates this replaced could not fail: one asserted `distinct_pairs <= rows`, true for any frame,
  and one compared two config constants with each other. A gate that has never failed is
  indistinguishable from a gate that cannot fail.
- **One source of truth for every window and threshold.** `metrics` used to keep its own
  `COHORT_WINDOW_DAYS = 30.0` next to `config.COHORT_DAYS = 30`, and `monetization` ran 1 000
  bootstrap draws at seed 7 while config said 2 000 at seed 42. Nothing crashed; the two halves of
  the report simply stopped describing the same dataset. `tests/test_config_single_source.py` fails
  if a module redefines a config constant with a literal, or if a config key is named only once —
  which is how the dead `SHORT_WINDOW_DAYS` was caught before it was wired into the W=7 readout.

## What this proves

- **The preregistration is executed, not cited.** The decision rule is parsed out of
  `docs/preregistration.md` at runtime, and the suite fails if the code and the document diverge —
  `liquidity/monetization.py::decision`, `tests/test_monetization.py`
  (`test_decision_rules_come_from_the_document_not_the_code`,
  `test_preregistration_declares_the_guardrail_condition`).
- **A seller self-report can be corrected, and the correction is scored against the key.** The EM
  correction lands within ±0.03 on all five categories (worst 0.0161) and the bootstrap interval
  covers the truth in at least four of them — `liquidity/bias.py`,
  `tests/test_bias_recovery.py` (`test_every_category_recovers_within_tolerance`,
  `test_bootstrap_interval_covers_the_truth_in_at_least_four_categories`), scored against
  `_truth/ground_truth.json` — the same key `uv run python -m liquidity.scoring` prints next to
  every estimate. The scorer is the only module in the package that opens the key, and
  `tests/test_no_truth_leak.py` asserts that reader set is exactly `{scoring.py}`; adding an
  import elsewhere turns the suite red.
- **The validity band is a tested artifact, not a caveat.** Remove the reply-to-removal separator
  or move the cutoff outside roughly 4–6 days and the correction misses on all five categories —
  `tests/test_separator_sensitivity.py` (`test_without_the_separator_the_correction_fails`,
  `test_the_separator_cutoff_is_not_generic`, `test_recovery_holds_across_the_working_band`).
- **The full report is part of the repository, not an appendix.** `reports/report.md` carries the
  liquidity and demand-sufficiency blocks (cohort contact share, dead listings, time to first
  contact, zero-result by population tier, the W=7 vs W=30 comparison), ten numbered limitations and
  a claims-to-artefact table — `tests/test_report_claims.py` fails if its numbers drift from the code.
- **The five-minute check and the transferable piece are one run.** `uv run pytest` → 115 passed
  re-derives every asserted cell (corrected errors, treated/holdout rates, guardrail shares,
  separator sensitivity) from `data/raw/`; `liquidity/monetization.py` is the part that moves to
  another "did the paid surface create demand or relocate it" question, because it contrasts the
  naive pre/post read (`+113%`, selection) against a 10% randomised holdout (`−0.0116` contacts
  per session, Welch p 0.308) and prints the unpaid-contact and reply-rate guardrails in the same
  output. `data/raw/` is gitignored, so a fresh clone needs `make all` before the suite can score
  anything.

## Quick start

```bash
uv sync --all-groups
make all                              # generates data/raw (seconds), then pytest + ruff
uv run python -m liquidity.scoring    # estimates next to the planted truth
uv run python -m liquidity.bias       # observable-only audit table, no truth
```

`liquidity.generate` is deterministic — regenerating from seed 42 on a clean copy reproduced all
eight parquet files byte-for-byte — and fast (seconds, not minutes). `data/raw/` is gitignored, so
a clone starts empty: run `make all` before `pytest`, or the suite fails on missing parquet rather
than skipping. The determinism claim was checked on a clean copy, not asserted by the suite.
Optional: `uv run python -m liquidity.monetization` for the visibility leg.

## Layout

```
marketplace-liquidity-lab/
├── liquidity/
│   ├── generate.py       # seeded DGP: raw tables + _truth/ground_truth.json
│   ├── etl.py            # raw -> staging -> DuckDB marts
│   ├── quality.py        # eight raw-layer gates, each with a fault-injection test
│   ├── metrics.py        # cohort liquidity (W=7 and W=30), demand sufficiency, zero-result
│   ├── bias.py           # proxy audit, EM correction, bootstrap CI
│   ├── scoring.py        # scores the estimators against the key (only truth reader)
│   ├── monetization.py   # naive pre/post vs holdout, guardrails, decision
│   └── config.py         # windows, seeds, tolerances (single source)
├── sql/                  # cohort_liquidity.sql, zero_result.sql
├── docs/                 # preregistration.md, decision-memo.md, data_dictionary.md, brief.md
├── reports/report.md     # the full evidence-classed report, including the limitations
├── LICENSE               # MIT
├── tests/                # 115 tests, including the no-leak, must-fail and separator-sensitivity guards
├── _truth/               # frozen answer key — read by liquidity/scoring.py and tests only
├── data/                 # raw/, staging/, marts/ (gitignored, regenerable)
└── scripts/generate_data.py
```

## How this differs from `serp-rank-lab`

Both repositories end with a negative point estimate and a breached guardrail, and that resemblance
is worth naming rather than leaving for a reader to notice.

| | `serp-rank-lab` | this repo |
|---|---|---|
| Subject | which ranking policy to ship | whether the shipped metric measures the thing |
| Primary artefact | experiment on a ranking policy, revenue per search | audit of a proxy against a latent truth |
| The "second leg" | the experiment itself | a method demonstration: naive pre/post vs holdout |
| Status of the finding | the result | the first leg is the finding; the second is underpowered (MDE 6.8% vs a 2.5% effect) |

What transfers out of this repo is the audit shape: take the metric a team actually reports, score
it against something outside itself, and price the decision on the corrected read. The visibility
contrast is here because the same log answers it, not because it is the contribution.

## What this is not

- **Not Avito data.** The log is synthetic, produced by a documented DGP with planted ground
  truth. It is calibrated to public order-of-magnitude numbers only; it is evidence about a
  measurement mechanism, not about the real platform.
- **`_truth/ground_truth.json` is the test answer key, not a dataset.** No analysis module reads
  it (`tests/test_no_truth_leak.py`); the tests score the estimator against it.
- **Not an A/B-test portfolio piece.** A randomised holdout appears as one leg; the subject is
  measurement validity, not experiment execution.
- **Out of scope:** RFM, churn prediction, GMV root-cause decomposition, retention modelling,
  fraud classification, a dashboard for its own sake. Trust and safety is not analysed.
