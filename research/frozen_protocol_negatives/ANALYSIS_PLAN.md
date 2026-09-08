# Frozen experimental protocol

Written: 2026-07-28, while the run (research/frozen_protocol_negatives/run.py) was in
progress. Status of the data at time of writing: 8 of 40 seed-results existed (seeds
0–7 on the amc/01013500 cell only). The other three cells, amc/07067000 and the entire
recession arm (11143000, 01466500), had zero observed results.

The negative-control experiment uses the same frozen experimental protocol as the positive
snow-prior experiment. No incoming negative-arm value was used to modify the experimental
procedure.

Disclosure. Per-seed dNSE values from the in-progress run were incidentally visible in
results/frozen_protocol_negatives/run.log during routine health checks before this plan
was fixed. The amc/01013500 cell was therefore partially observed at plan-lock. The
amc/07067000 cell and the entire recession arm were fully unobserved at lock.

Amendment A2 (2026-07-28, applied at plan-lock). The manuscript will disclose that the
amc/01013500 cell was partially observed at plan-lock: 8 of 10 seed results existed and
per-seed values for two seeds (6–7) had been incidentally seen in the run log. The
amc/07067000 cell and the entire recession arm were fully unobserved, so 3 of 4
experimental cells were cleanly frozen before any result was observed.

## 1. Experimental question

The boundary thesis concerns two physics priors whose reconstructed state memory does
not exceed the model's 30-day input window: antecedent soil moisture (AMC,
rc_ratio = 20) and the anchored master recession (k = 0.94).

The negative-control experiment evaluates these priors under the identical frozen training,
evaluation, masking, and seeding protocol used for the positive snow-prior experiment.

The physics prior is the intended difference between the prior and carried-control arms.
The training protocol, model configuration, data handling, masking procedure, and
carried-control construction remain unchanged.

## 2. Experimental cells

The experiment contains four prior–basin cells:

| Prior | Basin | Prior setting | Input-window relation |
|---|---|---|---|
| AMC | 01013500 | `rc_ratio = 20` | State memory <= 30 days |
| AMC | 07067000 | `rc_ratio = 20` | State memory <= 30 days |
| Anchored recession | 11143000 | `k = 0.94` | State memory <= 30 days |
| Anchored recession | 01466500 | `k = 0.94` | State memory <= 30 days |


Each prior arm is compared with its carried control under the identical frozen experimental
procedure.

## 3. Statistical procedure

**Per cell** (prior × gauge), on the paired per-seed differences
d_s = metric(prior, seed s) − metric(control, seed s), s = 0…9, at mask 1.0:

1. Compute the mean difference and its **90 % t-interval** [L, U]:
   mean(d) ± t₀.₉₅,₉ · sd(d)/√10, with t₀.₉₅,₉ = 1.8331 (sd with ddof = 1). Containment of
   the 90 % CI in (−ε, +ε) is exactly the two-one-sided-tests (TOST) procedure at α = 0.05.
2. Map [L, U] to a cell verdict:

   | condition                  | verdict            | reading |
   |----------------------------|--------------------|---------|
   | −ε < L **and** U < +ε      | **EQUIVALENT**     | no meaningful effect either way; confirms specificity. If additionally L > 0, label "significant-but-negligible" (as in ledger row FP-01), still confirms. |
   | U < +ε **and** L ≤ −ε      | **NO-BENEFIT**     | benefit confidently below ε; possible harm — confirms specificity (the thesis claims *no gain*, not exact zero); report the harm bound separately if U < 0. |
   | L < +ε ≤ U                 | **INCONCLUSIVE**   | cannot claim specificity for this cell; report as underpowered, no confirmation language. |
   | L ≥ +ε                     | **BENEFIT — FAIL** | confident, meaningful gain from a prior predicted null; contradicts the boundary thesis for this prior. |

3. **Aggregation** is conjunctive, never pooled: seeds aggregate within a cell (step 1);
   an **arm confirms** iff *both* of its gauges are EQUIVALENT or NO-BENEFIT; the **study
   confirms specificity** iff *both* arms confirm. Seeds are never pooled across gauges
   (basins are not exchangeable draws).

4. **Sensitivity (reported, not decisive):** the same containment rule using the 95 % CI
   already emitted by `assemble.py` (`dNSE_ci95_lo/hi`), i.e. TOST at α = 0.025.

## 4. Replication structure

Each cell is evaluated with 10 seeds, numbered 0–9.

The prior and carried-control arms are paired by seed. Seeds are not pooled across basins;
each basin constitutes its own experimental cell.

Completeness requires 10/10 seed results in every cell. The assembly step
(assemble.py) hard-fails otherwise.

No extension of the seed count or modification of the experimental procedure is made after
observing negative-arm results unless documented by a written, dated amendment fixed before
the additional results are observed.

## 5. Frozen protocol

The negative-control experiment uses the same frozen protocol as the positive snow-prior
experiment.

The following remain fixed between prior and carried-control arms:

training procedure;
model configuration;
input-window length of 30 days;
discharge-masking procedure;
evaluation masks;
seed structure;
carried-control construction;
evaluation metrics;
data preprocessing and handling;
prior-arm implementation and state reconstruction procedure.

The AMC prior uses rc_ratio = 20. The anchored recession prior uses k = 0.94.
Neither value is adjusted using negative-arm results.

## 6. Reproducibility and data-validity conditions

The negative-control run must reproduce the established prior-arm implementation before its
results are treated as valid experimental output.

Fresh prior-arm seeds 0–2 are compared with the legacy Study 6 arrays. The maximum absolute
deviation must be no greater than 1e-4 across all metrics and masks, using the same
tolerance as the FP-001 carry gate; the comparison is printed by assemble.py.

A failure of this reproducibility condition is treated as a protocol/data-validity failure
and triggers diagnosis or rerun rather than interpretation as an experimental finding.

Completeness also requires 10/10 seeds in every cell. The assembly step (assemble.py)
hard-fails when a cell is incomplete.

## 7. Experimental outputs and provenance
Positive reference: results/frozen_protocol/summary.csv +
results/frozen_protocol/frozen_protocol.json (FP-001, reproducibility gate 0.0).
Negative data: results/frozen_protocol_negatives/negative_runs.json, assembled by
research/frozen_protocol_negatives/assemble.py into
frozen_protocol_negatives.json + summary.csv.
The negative-control output schema is identical to the frozen positive summary, with
per-seed arrays retained for reproducibility.
8. Protocol lock and amendment record
date	entry
2026-07-28	Architect sign-off (FP-006 review): negative-control experimental design, four cells, seed structure, masks, and frozen protocol approved.
2026-07-28	A2 (Disclosure): manuscript-disclosure commitment for the partially observed amc/01013500 cell.
2026-07-28	Plan lock (FP-007): confirmatory evaluator frozen (evaluate.py, fixture-tested only, no real data read); SHA-256 hashes recorded below the marker. Any subsequent edit to this file or to evaluate.py requires a new dated entry in this log before the edit.

Verification rule. The plan hash covers every line of this file above the
<!-- PLAN-LOCK --> marker line (marker excluded):

awk '/^<!-- PLAN-LOCK -->$/{exit} {print}' ANALYSIS_PLAN.md | sha256sum


The evaluate.py and test_evaluate.py hashes cover the whole file:

sha256sum <file>

<!-- PLAN-LOCK -->

Lock record (2026-07-28T01:22:19-04:00).

ANALYSIS_PLAN.md (content above the marker): sha256 =
e4b69b1722d12b1f233d883fb953e8fe078ff4714fe167c175327fe20162c17c
evaluate.py: sha256 =
c1d3a3f3923ce5b6864f031ef32c0742285693578e1e271b6cfe301a7c1c60c6
test_evaluate.py (10/10 fixture tests passing at lock): sha256 =
10d27a1b84187af3f0f657537e5c7e956a4200be5e164c7839838ecd072a1b18
