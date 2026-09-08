# Pre-specified analysis plan — FP-005 negative controls under the frozen protocol

**Written:** 2026-07-28, while the run (`research/frozen_protocol_negatives/run.py`) was in
progress. **Status of the data at time of writing:** 8 of 40 seed-results existed (seeds
0–7 on the `amc`/01013500 cell only). The other three cells — `amc`/07067000 and the entire
`recession` arm (11143000, 01466500) — had zero observed results. All thresholds below are
derived exclusively from the published positive artifact
(`results/frozen_protocol/summary.csv`, FP-001); the arithmetic is shown in §2. No incoming
negative-arm value informed any threshold.

**Disclosure.** Per-seed dNSE values from the in-progress run were incidentally visible in
`results/frozen_protocol_negatives/run.log` during routine health checks before this plan
was fixed. The equivalence bound in §2 is an arithmetic function of the positive artifact
only (ε = ½ × 0.06647 = 0.0332); its numeric proximity to one early observed seed value is
coincidental and the derivation below is checkable without reference to the negative data.

**Amendment A2 (2026-07-28, applied at plan-lock).** The manuscript will disclose that the
`amc`/01013500 cell was partially observed at plan-lock: 8 of 10 seed results existed and
per-seed values for two seeds (6–7) had been incidentally seen in the run log. The
`amc`/07067000 cell and the entire recession arm were fully unobserved, so 3 of 4
confirmatory cells are cleanly pre-registered.

---

## 1. Question and endpoints

The boundary thesis predicts that the two physics priors whose reconstructed state memory
does **not** exceed the model's 30-day input window — antecedent soil moisture (AMC,
rc_ratio = 20) and the anchored master recession (k = 0.94) — provide **no meaningful
benefit** on their signature-motivated target basins, under the identical frozen protocol
that confirmed the snow prior's benefit.

- **Cells (4):** `amc` × {01013500, 07067000}; `recession` × {11143000, 01466500}.
- **Primary endpoint:** ΔNSE = NSE(prior arm) − NSE(carried control), paired per seed
  (n = 10), at the **headline mask 1.0** (100 % discharge missing).
- **Secondary endpoint (supporting, descriptive):** flood-peak MAE reduction % (the
  `floodMAE_red_pct` column of the assemble schema), same mask.
- Masks 0.0 and 0.8 are reported descriptively and carry no confirmatory weight.

## 2. Equivalence bound ε — derivation from the positive arm

The reference scale is the smallest *confirmed* positive effect in the frozen artifact.
At mask 1.0, the four individually significant snow basins
(`results/frozen_protocol/summary.csv`, FP-001):

| gauge    | dNSE_mean | 95 % CI            | p        |
|----------|-----------|--------------------|----------|
| 11266500 | +0.29741  | [+0.2021, +0.3927] | 5.9e-05  |
| 09352900 | +0.13598  | [+0.0900, +0.1820] | 9.0e-05  |
| 09210500 | +0.11325  | [+0.0518, +0.1747] | 2.4e-03  |
| 01013500 | +0.06647  | [+0.0238, +0.1092] | 6.5e-03  |

Smallest confirmed mean effect: **0.06647** (01013500).

**ε_dNSE = ½ × 0.06647 = 0.0332.**

Rationale: half the smallest effect the study treats as a real, publishable benefit is the
conventional equivalence-margin choice; an effect confidently below ε is separated from the
confirmed-positive class by at least a factor of two in expectation.

Secondary endpoint: smallest confirmed flood-peak MAE reduction is **23.04 %** (01013500,
mask 1.0). **ε_flood = ½ × 23.04 % = 11.5 %.**

**Amendment A1 (2026-07-28, applied at plan-lock).** "Confirmed" for the flood endpoint
means paired two-sided t p < 0.05, the criterion used throughout FP-001. Under that
criterion the smallest confirmed reduction is 23.04 % (01013500, mask 1.0, p = 0.034);
its 95 % CI [−7.6 %, +53.7 %] crosses zero, so a CI-exclusion criterion would instead
select 25.23 % (09210500). The p-based anchor is retained unchanged; ε_flood is secondary
and descriptive and carries no confirmatory weight.

## 3. Statistical procedure

**Per cell** (prior × gauge), on the paired per-seed differences
d_s = metric(prior, seed s) − metric(control, seed s), s = 0…9, at mask 1.0:

1. Compute the mean difference and its **90 % t-interval** [L, U]:
   mean(d) ± t₀.₉₅,₉ · sd(d)/√10, with t₀.₉₅,₉ = 1.8331 (sd with ddof = 1). Containment of
   the 90 % CI in (−ε, +ε) is exactly the two-one-sided-tests (TOST) procedure at α = 0.05.
2. Map [L, U] to a cell verdict:

   | condition                  | verdict            | reading |
   |----------------------------|--------------------|---------|
   | −ε < L **and** U < +ε      | **EQUIVALENT**     | no meaningful effect either way; confirms specificity. If additionally L > 0, label "significant-but-negligible" (as in ledger row FP-01) — still confirms. |
   | U < +ε **and** L ≤ −ε      | **NO-BENEFIT**     | benefit confidently below ε; possible harm — confirms specificity (the thesis claims *no gain*, not exact zero); report the harm bound separately if U < 0. |
   | L < +ε ≤ U                 | **INCONCLUSIVE**   | cannot claim specificity for this cell; report as underpowered, no confirmation language. |
   | L ≥ +ε                     | **BENEFIT — FAIL** | confident, meaningful gain from a prior predicted null; contradicts the boundary thesis for this prior. |

3. **Aggregation** is conjunctive, never pooled: seeds aggregate within a cell (step 1);
   an **arm confirms** iff *both* of its gauges are EQUIVALENT or NO-BENEFIT; the **study
   confirms specificity** iff *both* arms confirm. Seeds are never pooled across gauges
   (basins are not exchangeable draws).

4. **Sensitivity (reported, not decisive):** the same containment rule using the 95 % CI
   already emitted by `assemble.py` (`dNSE_ci95_lo/hi`), i.e. TOST at α = 0.025.

## 4. Multiplicity

Four primary equivalence tests (2 arms × 2 gauges), one endpoint, one mask. The headline
claim is the **conjunction** of all four (§3.3): under the intersection–union principle a
conjunctive claim is valid with each component tested at nominal α, so no alpha adjustment
is applied to the primary decision. If any single cell is ever quoted as a stand-alone
finding, a Bonferroni-adjusted bound (α = 0.05/4 per cell) will be reported alongside it.
Multiplicity across masks and across the secondary endpoint is handled by pre-specification:
only ΔNSE at mask 1.0 is confirmatory.

## 5. Data-validity preconditions

The verdicts in §3 are computed only if:

- **Completeness:** 10/10 seeds present in every cell (`assemble.py` already hard-fails
  otherwise).
- **Reproducibility gate:** fresh prior-arm seeds 0–2 match the legacy Study 6 arrays with
  max |deviation| ≤ 1e-4 across all metrics and masks (same tolerance as FP-001's
  carry gate; printed by `assemble.py`).

If either fails, the dataset is not evaluable and the run is diagnosed/rerun — this is a
validity gate, not an outcome.

## 6. Outcome mapping (what we will write)

- **All 4 cells EQUIVALENT / NO-BENEFIT** → "the negative controls confirm specificity":
  under the frozen protocol, both below-window priors show ΔNSE confidently below ε =
  0.033, i.e. below half the smallest confirmed positive effect.
- **Any cell INCONCLUSIVE** → that arm is reported as "specificity not established
  (underpowered)"; no confirmation language for the arm; the study section reports the
  other arm on its own merits. Extending seeds is permitted **only** via a written,
  dated amendment to this plan fixed before any extended-seed result is observed.
- **Any cell BENEFIT** → the specificity claim **fails** for that prior; the boundary
  thesis must be weakened accordingly in the manuscript (e.g., the prior's state memory
  is reassessed, or the thesis's necessary-condition wording is retracted). A FAIL is
  reported as prominently as the positive headline.

Individual seed signs or magnitudes carry no evidential weight in any direction; only the
cell-level intervals defined above are quoted.

## 7. Provenance

- Positive reference: `results/frozen_protocol/summary.csv` +
  `results/frozen_protocol/frozen_protocol.json` (FP-001, reproducibility gate 0.0).
- Negative data (incoming): `results/frozen_protocol_negatives/negative_runs.json`,
  assembled by `research/frozen_protocol_negatives/assemble.py` into
  `frozen_protocol_negatives.json` + `summary.csv` (schema-identical to the frozen
  summary; the 90 % CI of §3 is the one addition, computed from the stored per-seed
  arrays).
- Plan author: Implementer, task FP-006-prespec-negative-analysis; ε subject to Architect
  sign-off before the run completes.

## 8. Amendment log and plan lock

| date       | entry |
|------------|-------|
| 2026-07-28 | Architect sign-off (FP-006 review): ε_dNSE = 0.0332 and ε_flood = 11.5 % approved; the stricter 0.0238 alternative explicitly rejected as post-hoc. |
| 2026-07-28 | A1 (§2): ε_flood confirmation criterion stated (paired t p < 0.05); CI-crosses-zero discrepancy of the 23.04 % anchor noted. Margin unchanged. |
| 2026-07-28 | A2 (Disclosure): manuscript-disclosure commitment for the partially observed `amc`/01013500 cell. |
| 2026-07-28 | Plan lock (FP-007): confirmatory evaluator frozen (`evaluate.py`, fixture-tested only, no real data read); SHA-256 hashes recorded below the marker. Any subsequent edit to this file or to `evaluate.py` requires a new dated entry in this log *before* the edit. |

**Verification rule.** The plan hash covers every line of this file above the
`<!-- PLAN-LOCK -->` marker line (marker excluded):
`awk '/^<!-- PLAN-LOCK -->$/{exit} {print}' ANALYSIS_PLAN.md | sha256sum`.
The `evaluate.py` and `test_evaluate.py` hashes cover the whole file: `sha256sum <file>`.

<!-- PLAN-LOCK -->

**Lock record (2026-07-28T01:22:19-04:00).**
- `ANALYSIS_PLAN.md` (content above the marker): sha256 = `e4b69b1722d12b1f233d883fb953e8fe078ff4714fe167c175327fe20162c17c`
- `evaluate.py`: sha256 = `c1d3a3f3923ce5b6864f031ef32c0742285693578e1e271b6cfe301a7c1c60c6`
- `test_evaluate.py` (10/10 fixture tests passing at lock): sha256 = `10d27a1b84187af3f0f657537e5c7e956a4200be5e164c7839838ecd072a1b18`
