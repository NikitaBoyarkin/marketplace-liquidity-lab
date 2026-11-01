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
> latent truth — per category — then prices the paid-visibility decision on the corrected read.

A synthetic classifieds marketplace with a planted answer key. Every number below comes from a
command in this repo or from the planted key, and the suite scores the estimators against the key.

## Situation

A two-sided classifieds marketplace (Avito-like) where the transaction closes off-platform. The
platform logs searches, impressions, views, contacts and removals; it never logs `true_sale`. Its
sell-through metric — the share of a publish cohort whose removal reason is `sold` — is
seller-reported and wrong in both directions: sellers mark "sold" on listings that merely lapsed,
and let genuine sales lapse unmarked.

The log holds 6,000 listings and 40,000 search sessions over 90 days, in six regions and five
categories. The analysis cohort is the 1,956 listings published in the first 30 days, each with a
full 30-day observation window. The schema is in `docs/data_dictionary.md`.

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

### 1. The sell-through proxy is biased by category — H1a is falsified

From `uv run python -m liquidity.bias`; true values are the planted truth (the test answer key,
never read by analysis):

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

### 2. Paid visibility: naive +113%, holdout −0.0116 contacts per session

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
| Sessions (treated / holdout) | 34,498 / 3,843 |

The incrementality is negative, but the Welch p-value is 0.308 and the interval includes zero.
**The sign is suggestive, not stable:** a 1,000-draw bootstrap puts 84.9% of resamples below zero,
so roughly one in seven flips the sign. The honest read is "no proven gain in contacts per session,
with a negative point estimate consistent with displacement" — not "paid visibility destroys
contacts".

### 3. H2c is the decisive guardrail: roll back / cap slots per SERP

| Guardrail | Treated / paid | Holdout / unpaid | Change |
|---|---:|---:|---:|
| Unpaid contacts per session | 0.4184 | 0.4697 | −10.9% |
| Seller reply rate | 0.5662 | 0.6436 | −7.7 pp |

Promoted slots push organic listings down the SERP: contacts on **unpaid** listings fall 10.9% in
treated sessions, and paid contacts convert into a seller reply less often than unpaid ones (56.6%
vs 64.4%). Under the preregistered rule — parsed from `docs/preregistration.md`, not restated in
code — the guardrail dominates: **roll back, whatever H2b says**; and because H2b is negative,
**price paid visibility as a tax on organic reach and cap slots per SERP**
(`liquidity/monetization.py::decision`,
`tests/test_monetization.py::test_decision_rolls_back_on_a_dirty_guardrail`).

The suite pins the estimators rather than every cell of the tables: the corrected errors against
their tolerance, the treated/holdout rates, the guardrail shares and the separator sensitivity above.
Table values not asserted by a test are reproducible from the commands in Quick start —
`uv run pytest` → 61 passed.

## Quick start

```bash
uv sync --all-groups
uv run python -m liquidity.generate   # rewrites data/raw/*.parquet — takes minutes
uv run pytest
uv run python -m liquidity.bias
```

`liquidity.generate` is deterministic — regenerating from seed 42 on a clean copy reproduced all
eight parquet files byte-for-byte — but slow. `data/raw/` is gitignored, so a clone starts empty and
`make all` generates the log before testing; the determinism claim was checked on a clean copy, not
asserted by the suite.
Optional: `uv run python -m liquidity.monetization` for the visibility leg.

## Layout

```
marketplace-liquidity-lab/
├── liquidity/
│   ├── generate.py       # seeded DGP: raw tables + _truth/ground_truth.json
│   ├── etl.py            # raw -> staging -> DuckDB marts
│   ├── quality.py        # five raw-layer quality gates
│   ├── metrics.py        # cohort liquidity, demand sufficiency, zero-result
│   ├── bias.py           # proxy audit, EM correction, bootstrap CI
│   ├── monetization.py   # naive pre/post vs holdout, guardrails, decision
│   └── config.py         # windows, seeds, tolerances (single source)
├── sql/                  # cohort_liquidity.sql, zero_result.sql
├── docs/                 # preregistration.md, data_dictionary.md, brief.md
├── tests/                # 61 tests, including the no-leak, must-fail and separator-sensitivity guards
├── _truth/               # frozen answer key — read by tests only
├── data/                 # raw/, staging/, marts/ (gitignored, regenerable)
└── scripts/generate_data.py
```

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
