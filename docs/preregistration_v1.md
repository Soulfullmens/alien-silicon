# Pre-registration v1 (DRAFT, not frozen)

Follows §26 of `master_plan.md`. When it's final: `sha256sum docs/preregistration_v1.md`, post the hash on OSF and in a public commit. It is updated once, in Week 17 (v2), then frozen.

Status: **draft**. Items marked TODO must be filled in before the hash is posted.

## 1. Research questions and hypotheses

| RQ | Question | Hypothesis |
|---|---|---|
| RQ1 | Do agents do worse on alien chips than on familiar ones, relative to the known optimum? | H1: The gap to the optimum is clearly larger on alien chips. Most wins on familiar chips are Level A or recalled patterns. |
| RQ2 | Can a pre-FA4 model invent FA4-style tricks from raw chip specs, with no hints? | H2a: Modern models sometimes find Level-B tricks. H2b: The pinned pre-FA4 model rarely or never finds them. Either result is reported. |
| RQ3 | Does the self-improving loop (vK) beat v0 on unseen chips at the same budget? | H3: vK > v0 on `frac_opt` at matched dollars and time. Most of the gain comes from self-built tools. |
| RQ4 | How often do agents cheat, bloat, or fail to stop, and do the session rules reduce it? | H4: Hack attempts and bloat rise with session length. Consolidated restarts reduce them. |
| RQ5 | Does the simulator predict real hardware? | H5: Spearman ρ ≥ 0.8 between simulated and real rankings on T4 and one modern GPU. |

Minimum publishable result: RQ1, RQ2 and RQ5 answered cleanly.

## 2. Metrics

As defined in §24 of the master plan, unchanged. Primary metrics: `frac_opt`, `discovery_rate_B`, `rho_sim_real`, `hack_rate`, `false_consensus_rate`.

## 3. Correctness rule

A kernel is `correct` only if it passes the static checks, all D tests, all S tests and the white-box adversarial tests (§14.1). Tolerance **τ = 1e-3**, relative plus absolute, defined per task. TODO: the exact per-task formula.

## 4. Chip generator

TODO: generator version (git hash of `aliensilicon/chips/`), and the seeds for the train, test and sealed splits.

## 5. Models

TODO: exact model IDs, providers, access dates and temperatures. Time machine: original DeepSeek-R1 (Jan 2025) and DeepSeek-R1-Distill-Qwen-32B, pinned by Hugging Face commit SHA.

## 6. Time machine and contamination

TODO: checkpoint SHAs, the contamination probe questions, and the rule for rejecting a contaminated model.

## 7. Budgets

TODO: dollars per run, and the global cap (see §29).

## 8. Analysis plan

As in §25: the unit is the (chip, task) pair. Paired Wilcoxon signed-rank tests, median difference with a bootstrap 95% CI (10,000 resamples), Wilson CIs for rates, and Holm correction across the 5 primary hypotheses. Analysis code is frozen and hashed in Week 18, before the main runs.

## 9. Success and failure per hypothesis

TODO: one line per hypothesis giving the exact threshold that counts as support, refutation or inconclusive.
